"""Build results/fingpt/dashboard/compare.json: FinGPT vs FinBERT on the panel.

Three scopes, because the two runs do not cover the same years:
  head_to_head  2016-2023  both news tracks warmed up and both runs present
  warmup        2015       FinGPT news still inside its 250-day z-score warm-up
  extension     2024-2025  FinGPT only (FNSPID, and so FinBERT, ends in 2023)

The news specialist's vote and reasoning are not in the provenance files, so
both panel runs are REPLAYED from the LLM cache with the API blocked. Every
call must hit the cache; a miss aborts rather than silently spending money.
"""
import asyncio
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src import agents, config, detector, features as F, ingest, labels, scoring
from src.detector import STATES
from src.memory import build_history

RES = config.paths()["results"]
FG = RES / "fingpt"
OUT = FG / "dashboard" / "compare.json"
SCOPES = {"head_to_head": ("2016-01-01", "2023-12-31"),
          "warmup": ("2015-01-01", "2015-12-31"),
          "extension": ("2024-01-01", "2025-12-31")}


def _blocked():
    raise SystemExit("cache miss during replay -- refusing to call the API")


agents._client = _blocked
agents._save = lambda k, v: None


def replay(arm_b, feats, cfg, gate, sent, mkt) -> dict:
    """Full graph state (opinions + critic) for every gated day, from cache."""
    views = agents.build_agent_views(feats, cfg, sent, mkt)
    hist = build_history(arm_b["label"], arm_b[[f"p_{s}" for s in STATES]],
                         feats["hmm"], cfg)
    nd = sent["article_count"].fillna(0) > 0
    md = mkt["market_articles"].fillna(0) > 0 if mkt is not None else None
    days = list(arm_b.index[gate.reindex(arm_b.index, fill_value=False)])
    res = asyncio.run(agents._run_days(days, agents.build_graph(cfg), views,
                                       arm_b, hist, cfg, asyncio.Semaphore(16),
                                       nd, md))
    return {d: r for d, r in res}


def metrics(arm, target, mask) -> dict:
    r = scoring.score_arm(arm, target, {}, mask=mask)
    keep = ["n", "accuracy", "brier", "log_loss", "calibration_error",
            "kappa_vs_target"]
    return {k: (None if r.get(k) is None or pd.isna(r.get(k)) else round(float(r[k]), 4))
            for k in keep}


def main() -> None:
    cfg = config.load()
    blocks = ingest.build_panel(cfg=cfg)
    feats = F.build_features(blocks, cfg)
    # As src/contest.py: forward realised-vol terciles of the index close.
    target = labels.build_target(blocks["index"]["close"], cfg)

    b25 = pd.read_parquet(FG / "arm_b_hmm_2025.parquet")
    b23 = pd.read_parquet(RES / "arm_b_hmm.parquet")
    c_bert = pd.read_parquet(RES / "arm_c_agentic.parquet")
    c_gpt = pd.read_parquet(FG / "arm_c_fingpt.parquet")
    gate = detector.candidate_days(b25, cfg)

    s_bert = pd.read_parquet(RES / "sentiment_daily.parquet")
    m_bert = pd.read_parquet(RES / "market_sentiment_daily.parquet")
    s_gpt = pd.read_parquet(RES / "fingpt_sentiment_daily.parquet")

    print("replaying FinBERT panel from cache ...", flush=True)
    g_bert = detector.candidate_days(b23, cfg) & (b23.index >= "2015-01-01")
    rb = replay(b23, feats, cfg, g_bert, s_bert, m_bert)
    print("replaying FinGPT panel from cache ...", flush=True)
    cfg_g = json.loads(json.dumps(cfg))
    cfg_g["agents"]["news_track"] = "fingpt"
    g_gpt = gate & (b25.index >= "2015-01-01")
    rg = replay(b25, feats, cfg_g, g_gpt, s_gpt, None)

    # ---------------------------------------------------------- scores by scope
    arms = {"detector": b25, "finbert": c_bert, "fingpt": c_gpt}
    scores = {}
    for scope, (a, z) in SCOPES.items():
        win = (b25.index >= a) & (b25.index <= z)
        win = pd.Series(win, index=b25.index)
        scores[scope] = {"all_days": {}, "gated_days": {},
                         "n_gated": int((gate & win).sum())}
        for name, arm in arms.items():
            if not ((arm.index >= a) & (arm.index <= z)).any():
                continue
            scores[scope]["all_days"][name] = metrics(arm, target, win)
            scores[scope]["gated_days"][name] = metrics(arm, target, gate & win)

    a, z = SCOPES["head_to_head"]
    m = gate & (b25.index >= a) & (b25.index <= z)
    idx = b25.index[m]
    lb = scoring.pointwise_loss(c_bert.loc[idx, scoring.PROB_COLS], target)
    lg = scoring.pointwise_loss(c_gpt.loc[idx, scoring.PROB_COLS], target)
    dm = scoring.diebold_mariano(lg, lb, block=int(cfg["target"]["forward_window"]))
    dm = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dm.items()}

    # ------------------------------------------------- per gated day, both runs
    def side(r, arm, day):
        if r is None:
            return None
        o = r["opinions"]["news_sentiment"]
        f = r["final"]
        return {"label": str(arm.loc[day, "label"]),
                "conf": round(float(arm.loc[day, "confidence"]), 3),
                "action": f["action"], "critic": f["critic_verdict"],
                "news_vote": o["state"], "news_informative": bool(o.get("informative")),
                "news_conf": round(float(o.get("confidence", 0)), 2),
                "news_reason": o.get("reasoning", ""),
                "votes": f["panel_votes"], "unanimous_against": f["unanimous_against"]}

    days = sorted(set(rb) | set(rg))
    gated = []
    for d in days:
        gated.append({
            "date": str(d.date()),
            "detector": str(b25.loc[d, "label"]),
            "detector_conf": round(float(b25.loc[d, "confidence"]), 3),
            "target": None if pd.isna(target.get(d)) else str(target.get(d)),
            "finbert": side(rb.get(d), c_bert, d) if d in c_bert.index else None,
            "fingpt": side(rg.get(d), c_gpt, d) if d in c_gpt.index else None})

    def behaviour(run: str, a: str, z: str) -> dict:
        rows = [g[run] for g in gated if g[run] and a <= g["date"] <= z]
        if not rows:
            return {}
        det = {g["date"]: g["detector"] for g in gated}
        dates = [g["date"] for g in gated if g[run] and a <= g["date"] <= z]
        inf = [r for r in rows if r["news_informative"]]
        return {"n": len(rows),
                "actions": pd.Series([r["action"] for r in rows]).value_counts().to_dict(),
                "critic": pd.Series([r["critic"] for r in rows]).value_counts().to_dict(),
                "news_informative_share": round(len(inf) / len(rows), 3),
                "news_vote_vs_detector_disagree": round(np.mean(
                    [r["news_vote"] != det[d] for r, d in zip(rows, dates)
                     if r["news_informative"]]) if inf else 0, 3),
                "news_votes": pd.Series([r["news_vote"] for r in inf]).value_counts().to_dict(),
                "news_conf_mean": round(float(np.mean([r["news_conf"] for r in inf])), 3) if inf else None,
                "labels_changed": sum(r["label"] != det[d] for r, d in zip(rows, dates)),
                "mean_conf": round(float(np.mean([r["conf"] for r in rows])), 4)}

    behav = {scope: {run: behaviour(run, a, z) for run in ("finbert", "fingpt")}
             for scope, (a, z) in SCOPES.items()}

    # ------------------------------------------------------------ daily series
    span = pd.date_range("2015-01-01", "2025-12-31", freq="B")
    vb = agents.build_agent_views(feats, cfg, s_bert, m_bert)["news_sentiment"]
    vg = agents.build_agent_views(feats, cfg_g, s_gpt, None)["news_sentiment"]

    def ser(s, end=None):
        s = s.reindex(b25.loc["2015":].index)
        if end is not None:                 # FNSPID stops; anything later is decay
            s[s.index > end] = np.nan
        return [None if pd.isna(v) else round(float(v), 3) for v in s]

    cal = b25.loc["2015":].index
    daily = {
        "dates": [str(d.date()) for d in cal],
        "z_finbert_single": ser(vb.get("mean_finbert_ewma"), "2023-12-31"),
        "z_finbert_market": ser(vb.get("market_finbert_ewma"), "2023-12-31"),
        "z_fingpt": ser(vg.get("mean_fingpt_ewma")),
        "raw_fingpt": ser(s_gpt["mean_fingpt_ewma"]),
        "raw_finbert_single": ser(s_bert["mean_finbert_ewma"], "2023-12-31"),
        "raw_finbert_market": ser(m_bert["market_finbert_ewma"], "2023-12-31"),
        "labels": {
            "detector": [str(v) for v in b25.loc[cal, "label"]],
            "finbert": [str(c_bert.loc[d, "label"]) if d in c_bert.index else None for d in cal],
            "fingpt": [str(c_gpt.loc[d, "label"]) if d in c_gpt.index else None for d in cal],
            "target": [None if pd.isna(target.get(d)) else str(target.get(d)) for d in cal]},
        "gated": [bool(gate.get(d, False)) for d in cal],
    }

    # ------------------------------------------------------------ corpus stats
    sg = s_gpt.loc["2015":]
    sb = s_bert.loc["2015":]
    mb = m_bert.loc["2015":]
    corpus = []
    for y in range(2015, 2026):
        ys = str(y)
        corpus.append({
            "year": y,
            "fingpt_articles": int(sg.loc[ys, "article_count"].sum()),
            "finbert_single_articles": int(sb.loc[ys, "article_count"].sum()) if ys in sb.index.year.astype(str) else 0,
            "finbert_market_articles": int(mb.loc[ys, "market_articles"].sum()) if ys in mb.index.year.astype(str) else 0,
            "fingpt_days_with_news": round(float((sg.loc[ys, "article_count"] > 0).mean()), 3),
            "finbert_days_with_news": round(float(((sb.loc[ys, "article_count"] > 0) | (mb.loc[ys, "market_articles"] > 0).reindex(sb.loc[ys].index, fill_value=False)).mean()), 3) if ys in sb.index.year.astype(str) else None})
    body = pd.read_parquet(config.ROOT / "data/alpaca/fingpt/fingpt_scores.parquet",
                           columns=["trading_day", "has_body", "label_mass"])
    by = body.groupby(body["trading_day"].dt.year)
    for c in corpus:
        g = by.get_group(c["year"])
        c["fingpt_body_share"] = round(float(g["has_body"].mean()), 3)
        c["fingpt_confident_share"] = round(float((g["label_mass"] >= 0.5).mean()), 3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
               "model": cfg["agents"]["model"], "scopes": SCOPES,
               "scores": scores, "dm_head_to_head": dm, "behaviour": behav,
               "gated": gated, "daily": daily, "corpus": corpus}
    OUT.write_text(json.dumps(payload, default=str))
    print(f"written: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps({"scores": scores["head_to_head"], "dm": dm,
                      "behaviour": behav["head_to_head"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
