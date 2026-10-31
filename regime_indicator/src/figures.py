"""Everything the deck draws, extracted once into a single JSON payload.

The deck is a self-contained page: no data fetches, no build step. So the
series it plots have to be small enough to inline, which means deciding here
what is signal and what is resolution nobody can see on a chart 900 px wide.

Weekly sampling of the label tracks is the compromise. Regime episodes have a
median duration of 21 days for Arm B and 5 for Arm A, so a weekly grid keeps
every episode of consequence while cutting the payload roughly fivefold. The
metrics tables and the episode lead times are exact -- those are the numbers a
reader will quote, and they are small.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, labels as L, scoring as S
from .detector import STATES, candidate_days

ARMS = {"A_hamilton": "arm_a_hamilton.parquet",
        "B_hmm": "arm_b_hmm.parquet",
        "C_agentic": "arm_c_agentic.parquet"}
CODE = {"bull_quiet": 0, "bull_volatile": 1, "bear_stress": 2}


def _codes(s: pd.Series) -> list:
    return [int(CODE[v]) if v in CODE else None for v in s]


def build(cfg: dict) -> dict:
    res = config.paths()["results"]
    blocks = __import__("src.ingest", fromlist=["x"]).build_panel(cfg=cfg)
    close = blocks["index"]["close"]

    arms = {k: pd.read_parquet(res / v) for k, v in ARMS.items()
            if (res / v).exists()}
    target = L.build_target(close, cfg)
    contest = json.load(open(res / "contest.json"))

    a, b = cfg["sample"]["score_start"], cfg["sample"]["score_end"]
    idx = target.dropna().loc[a:b].index
    for df in arms.values():
        idx = idx.intersection(df.index)

    # Weekly grid for the tracks; exact numbers everywhere it matters.
    wk = pd.Series(1, index=idx).resample("W").last().dropna().index
    wk = idx[idx.searchsorted(wk).clip(0, len(idx) - 1)].unique()

    out = {
        "meta": {
            "index": cfg["universe"]["index_name"],
            "score_start": a, "score_end": b,
            "n_scored": int(len(idx)),
            "model": cfg["agents"]["model"],
            "news_tickers": cfg["universe"]["news_tickers"],
        },
        "dates": [d.strftime("%Y-%m-%d") for d in wk],
        "close": [round(float(v), 1) for v in close.reindex(wk)],
        "target": _codes(target.reindex(wk)),
        "tracks": {k: _codes(df["label"].reindex(wk)) for k, df in arms.items()},
        "confidence": {k: [round(float(v), 3) if pd.notna(v) else None
                           for v in df["confidence"].reindex(wk)]
                       for k, df in arms.items()},
        "scopes": contest.get("full", {}),
        "scopes_gated": contest.get("gated", {}),
        "scopes_news": contest.get("news_live", {}),
        "tests": contest.get("tests", {}),
        "episodes": cfg["ground_truth_episodes"],
    }

    # Lead time, exact, per arm.
    out["lead"] = {}
    for k, df in arms.items():
        lt = S.transition_lead_time(df["label"], cfg)
        out["lead"][k] = json.loads(lt.to_json(orient="records", date_format="iso"))

    # What the panel did, straight from the provenance.
    p = res / "arm_c_provenance.json"
    if p.exists():
        prov = json.load(open(p))
        acts, crit = {}, {}
        for v in prov.values():
            acts[v["action"]] = acts.get(v["action"], 0) + 1
            crit[v["critic_verdict"]] = crit.get(v["critic_verdict"], 0) + 1
        vetoes = [{"date": d, "from": None, "to": v["label"],
                   "votes": v["panel_votes"],
                   "objection": v["objection"]}
                  for d, v in prov.items() if v["action"] == "veto"]
        for r in vetoes:
            r["from"] = str(arms["B_hmm"].loc[pd.Timestamp(r["date"]), "label"])
        out["panel"] = {"actions": acts, "critic": crit, "vetoes": vetoes,
                        "n_gated": len(prov), "n_calls": len(prov) * 5}

    # Confidence saturation, the mechanism behind C beating B.
    gate = candidate_days(arms["B_hmm"], cfg).reindex(idx, fill_value=False)
    out["calibration"] = {}
    for k in ("B_hmm", "C_agentic"):
        if k in arms:
            c = arms[k]["confidence"].reindex(idx)[gate]
            hist, edges = np.histogram(c.dropna(), bins=20, range=(0, 1))
            out["calibration"][k] = {
                "hist": [int(x) for x in hist],
                "edges": [round(float(x), 3) for x in edges],
                "mean": round(float(c.mean()), 4),
                "above_99": round(float((c > 0.99).mean()), 4)}
    return out


if __name__ == "__main__":
    cfg = config.load()
    data = build(cfg)
    dest = config.paths()["results"] / "deck_data.json"
    with open(dest, "w") as fh:
        json.dump(data, fh, separators=(",", ":"), default=str)
    kb = dest.stat().st_size / 1024
    print(f"written: {dest}  ({kb:.0f} KB)")
    print(f"  weekly points: {len(data['dates'])}")
    print(f"  arms: {list(data['tracks'])}")
    print(f"  vetoes: {len(data.get('panel', {}).get('vetoes', []))}")


# ---------------------------------------------------------------- dashboard
def build_dashboard(cfg: dict) -> dict:
    """Daily payload for the player. Deck data is weekly; playback needs days.

    Only the two arms the audience is being shown -- Hamilton and the agentic
    panel -- plus the ground truth. The three-state HMM is the agentic arm's
    own detector, so it appears in the panel view as "what the detector
    proposed", not as a third contestant.
    """
    res = config.paths()["results"]
    blocks = __import__("src.ingest", fromlist=["x"]).build_panel(cfg=cfg)
    close = blocks["index"]["close"]

    A = pd.read_parquet(res / "arm_a_hamilton.parquet")
    B = pd.read_parquet(res / "arm_b_hmm.parquet")
    C = pd.read_parquet(res / "arm_c_agentic.parquet")
    target = L.build_target(close, cfg)

    a, b = cfg["sample"]["score_start"], cfg["sample"]["score_end"]
    idx = target.dropna().loc[a:b].index
    for df in (A, B, C):
        idx = idx.intersection(df.index)

    gate = candidate_days(B, cfg).reindex(idx, fill_value=False)
    prov = {}
    p = res / "arm_c_provenance.json"
    if p.exists():
        prov = json.load(open(p))

    out = {
        "meta": {
            "index": cfg["universe"]["index_name"],
            "start": str(idx[0].date()), "end": str(idx[-1].date()),
            "n": int(len(idx)), "n_gated": int(gate.sum()),
            "model": cfg["agents"]["model"],
            "states": list(STATES),
            "display": cfg["regimes"]["display"],
        },
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "close": [round(float(v), 1) for v in close.reindex(idx)],
        "truth": _codes(target.reindex(idx)),
        "hamilton": _codes(A["label"].reindex(idx)),
        "agentic": _codes(C["label"].reindex(idx)),
        "detector": _codes(B["label"].reindex(idx)),
        "conf_h": [round(float(v), 3) for v in A["confidence"].reindex(idx)],
        "conf_c": [round(float(v), 3) for v in C["confidence"].reindex(idx)],
        "gated": [int(v) for v in gate],
        "episodes": cfg["ground_truth_episodes"],
    }

    # The panel record, keyed by position so the player can look it up in O(1)
    # while scrubbing rather than searching a date list every frame.
    pos = {d: i for i, d in enumerate(out["dates"])}
    panel = {}
    for d, v in prov.items():
        i = pos.get(str(d)[:10])
        if i is None:
            continue
        panel[i] = {
            "action": v["action"], "critic": v["critic_verdict"],
            "votes": v["panel_votes"], "informative": v.get("panel_informative", {}),
            "objection": v.get("objection", ""), "falsifier": v.get("falsifier", ""),
            "recency": v.get("recency_bias_risk", ""),
            "unanimous": bool(v.get("unanimous_against")),
        }
    out["panel"] = panel

    contest = json.load(open(res / "contest.json"))
    out["scopes"] = contest.get("full", {})
    out["scopes_gated"] = contest.get("gated", {})
    out["tests"] = contest.get("tests", {})
    out["lead"] = {}
    for k, df in (("hamilton", A), ("agentic", C)):
        lt = S.transition_lead_time(df["label"], cfg)
        out["lead"][k] = json.loads(lt.to_json(orient="records",
                                               date_format="iso"))

    # Every prompt and reply, recovered from the LLM cache. Roughly 2.3 MB, so
    # it doubles the page; worth it, because it is the difference between
    # asserting the panel reasoned and letting a reader check.
    try:
        out["tx"] = build_transcripts(cfg)
    except Exception as exc:                                   # noqa: BLE001
        print(f"transcripts unavailable ({type(exc).__name__}: {exc}) -- "
              f"the tab will be hidden")
        out["tx"] = None
    return out


# --------------------------------------------------------------- transcripts
def build_transcripts(cfg: dict, limit: int | None = None) -> dict:
    """Recover every prompt and every reply from the LLM cache.

    The provenance file records what the panel DECIDED; it does not record what
    each specialist was shown or what it said. Both are in the disk cache, and
    the cache key is a pure function of (model, role, system, payload, effort) --
    so rebuilding the payload exactly as the run built it regenerates the key
    and retrieves the call. Nothing is re-requested from the API.

    Stored column-wise: the system prompt is identical across all 582 sittings
    for a given role, so it is kept once per role rather than 582 times, and the
    z-scores become a plain matrix in a fixed feature order instead of repeating
    every key name five times a day. That is the difference between a ~4 MB
    payload and a ~1 MB one.
    """
    from . import agents as AG
    from .memory import build_history

    res = config.paths()["results"]
    blocks = __import__("src.ingest", fromlist=["x"]).build_panel(cfg=cfg)
    feats = __import__("src.features", fromlist=["x"]).build_features(blocks, cfg)

    B = pd.read_parquet(res / "arm_b_hmm.parquet")
    C = pd.read_parquet(res / "arm_c_agentic.parquet")
    sent = None
    if (res / "sentiment_daily.parquet").exists():
        sent = pd.read_parquet(res / "sentiment_daily.parquet")
    market = None
    if (res / "market_sentiment_daily.parquet").exists():
        market = pd.read_parquet(res / "market_sentiment_daily.parquet")

    views = AG.build_agent_views(feats, cfg, sent, market)
    history = build_history(B["label"], B[[f"p_{s}" for s in STATES]],
                            feats["hmm"], cfg)
    news_days = None
    if sent is not None and "article_count" in sent:
        news_days = sent["article_count"].fillna(0) > 0
    market_days = None
    if market is not None and "market_articles" in market:
        market_days = market["market_articles"].fillna(0) > 0

    gate = candidate_days(B, cfg)
    days = list(B.index[gate.reindex(B.index, fill_value=False)])
    if limit:
        days = days[:limit]

    model = cfg["agents"]["model"]
    effort = cfg["agents"].get("effort", "low")
    roles = list(AG.AGENT_INPUTS)

    out = {
        "roles": roles,
        "systems": {r: AG.SYSTEM.format(role=r, question=AG.AGENT_QUESTION[r])
                    for r in roles},
        "features": {r: list(views[r].columns) for r in roles},
        "periods": ["t-4", "t-3", "t-2", "t-1", "t"],
        "calls": {}, "missing": 0,
    }
    out["systems"]["critic"] = AG.CRITIC_SYSTEM

    # Index positions so the dashboard can key on the same day index it uses
    # everywhere else.
    a, b = cfg["sample"]["score_start"], cfg["sample"]["score_end"]
    scored = [d for d in C.loc[a:b].index]
    posmap = {d: i for i, d in enumerate(scored)}

    for day in days:
        proposal = {
            "label": str(B.loc[day, "label"]),
            "confidence": round(float(B.loc[day, "confidence"]), 4),
            "probabilities": {s: round(float(B.loc[day, f"p_{s}"]), 4)
                              for s in STATES},
            "state_meaning": {
                "bear_stress": "highest mean drawdown from the running peak in "
                               "the fit window",
                "bull_quiet": "of the two remaining states, the lower mean "
                              "realized volatility",
                "bull_volatile": "the remaining state"},
        }
        mem = history.get(day, {})
        payloads = {r: AG._window(v, day, anonymise=True) for r, v in views.items()}
        if "news_sentiment" in payloads and news_days is not None:
            tk = bool(news_days.get(day, False))
            mk = (bool(market_days.get(day, False))
                  if market_days is not None else False)
            payloads["news_sentiment"] = {
                "z_scores": payloads["news_sentiment"],
                "single_stock_articles_today": tk,
                "market_wide_articles_today": mk,
                "note": "mean_* are single-stock headlines (7 mega-caps); "
                        "market_* are market-wide wire copy plus SPY and QQQ. "
                        "Where a track's flag is false, its values are a "
                        "decayed echo of older headlines rather than today's "
                        "news; weight that track accordingly. If BOTH flags "
                        "are false, report informative=false unless a decayed "
                        "level is itself extreme.",
            }

        rec, opinions = {}, {}
        for r in roles:
            sys_p = out["systems"][r]
            pay = {"your_evidence": payloads[r], "recent_history": mem}
            hit = AG._load(AG._key(model, r, {"s": sys_p, "p": pay, "e": effort}))
            if hit is None:
                out["missing"] += 1
                continue
            opinions[r] = hit
            zsrc = (payloads[r]["z_scores"] if r == "news_sentiment"
                    else payloads[r])
            cols = out["features"][r]
            mat = [[zsrc.get(p, {}).get(c) for c in cols] for p in out["periods"]]
            rec[r] = {"z": mat, "state": hit.get("state"),
                      "conf": hit.get("confidence"),
                      "adj": hit.get("confidence_adjustment"),
                      "inf": hit.get("informative"),
                      "why": hit.get("reasoning", "")}
            if r == "news_sentiment":
                rec[r]["news_today"] = (payloads[r]["single_stock_articles_today"]
                                        or payloads[r]["market_wide_articles_today"])
                rec[r]["stock_news"] = payloads[r]["single_stock_articles_today"]
                rec[r]["market_news"] = payloads[r]["market_wide_articles_today"]

        cpay = {"detector_proposal": proposal,
                "specialist_opinions": {k: {kk: vv for kk, vv in o.items()
                                            if not kk.startswith("_")}
                                        for k, o in opinions.items()},
                "recent_history": mem}
        ch = AG._load(AG._key(model, "critic",
                              {"s": AG.CRITIC_SYSTEM, "p": cpay, "e": effort}))
        if ch is not None:
            rec["critic"] = {"verdict": ch.get("verdict"),
                             "final": ch.get("final_state"),
                             "mult": ch.get("confidence_multiplier"),
                             "recency": ch.get("recency_bias_risk"),
                             "obj": ch.get("objection", ""),
                             "fals": ch.get("falsifier", "")}
        else:
            out["missing"] += 1

        rec["proposal"] = proposal
        rec["memory"] = mem
        rec["emitted"] = str(C.loc[day, "label"])
        rec["date"] = str(day.date())
        out["calls"][str(posmap.get(day, -1))] = rec

    return out
