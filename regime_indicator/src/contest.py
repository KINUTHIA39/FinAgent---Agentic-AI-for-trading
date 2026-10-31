"""The contest: score every available arm on one path and run the paired tests.

This is the plan's step-5 hard stop. Reaching it with Arms A and B scored means
a complete, defensible result exists whether or not the news and agent layers
ever ship.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import labels as L, scoring as S
from .detector import STATES, candidate_days

KEYS = ["n", "accuracy", "kappa_vs_target", "kappa_two_state", "brier",
        "bss_vs_persistence", "bss_vs_climatology", "log_loss",
        "calibration_error", "reliability", "resolution",
        "n_episodes", "median_duration", "false_alarms_per_year"]


def build_baselines(target: pd.Series, cfg: dict) -> dict[str, pd.DataFrame]:
    """Persistence and climatology, the two bars every arm must clear."""
    return {"persistence": L.persistence_forecast(target, cfg),
            "climatology": L.climatology_forecast(target, cfg)}


def run(arms: dict[str, pd.DataFrame], target: pd.Series, cfg: dict,
        gate_mask: pd.Series | None = None,
        secondary: pd.Series | None = None,
        news_mask: pd.Series | None = None) -> dict:
    """Score every arm at each scope, then the paired tests.

    Three scopes, and the third exists because of what the corpus turned out to
    be. FNSPID's ticker-tagged news is concentrated in 2020-2023 while 84% of
    gated days fall before 2020, so on the headline window Arm C minus Arm B
    largely measures the LLM panel with the news agent dark. `news_live`
    restricts to gated days that actually carry news, which is the only scope
    where that difference isolates the sentiment layer.
    """
    base = build_baselines(target, cfg)
    block = int(cfg["target"]["forward_window"])
    res = {"full": {}, "gated": {}, "news_live": {}, "tests": {},
           "secondary": {}}
    live = None
    if news_mask is not None:
        live = news_mask if gate_mask is None else (
            news_mask.reindex(gate_mask.index).fillna(False) & gate_mask)

    scored = dict(arms)
    # The baselines are scored as arms too, so they sit in the same table
    # rather than in a footnote nobody reads.
    for name, df in base.items():
        scored[f"Z_{name}"] = df

    for name, df in scored.items():
        res["full"][name] = S.score_arm(df, target, cfg, base)
        if gate_mask is not None:
            res["gated"][name] = S.score_arm(df, target, cfg, base, gate_mask)
        if live is not None:
            res["news_live"][name] = S.score_arm(df, target, cfg, base, live)
        if secondary is not None:
            res["secondary"][name] = {
                "accuracy": S.accuracy(df["label"], secondary),
                "brier": S.brier_score(df[S.PROB_COLS], secondary)}

    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            la = S.pointwise_loss(arms[a][S.PROB_COLS], target, "brier")
            lb = S.pointwise_loss(arms[b][S.PROB_COLS], target, "brier")
            res["tests"][f"{a}_vs_{b}"] = {
                "dm_brier": S.diebold_mariano(la, lb, block),
                "kappa_between_arms": S.cohens_kappa(arms[a]["label"],
                                                     arms[b]["label"]),
                "kappa_two_state": S.cohens_kappa(
                    S.to_two_state(arms[a]["label"]),
                    S.to_two_state(arms[b]["label"])),
                "brier_diff_ci": S.stationary_bootstrap_diff(
                    la, lb, lambda v: float(np.mean(v)), mean_block=block,
                    seed=int(cfg["run"]["seed"]) % 2**31),
            }
    return res


def report(arms: dict[str, pd.DataFrame], target: pd.Series, cfg: dict,
           gate_mask: pd.Series | None = None,
           secondary: pd.Series | None = None,
           news_mask: pd.Series | None = None) -> dict:
    from . import config as _cfg

    out = run(arms, target, cfg, gate_mask, secondary, news_mask)
    print("=" * 104)
    print("BUILD STEP 5 - THE CONTEST")
    print("=" * 104)

    notes = {
        "gated": "   (days the panel may act; off-gate Arm C IS Arm B)",
        "news_live": "   (gated days that CARRY news; the only scope where "
                     "C-B isolates sentiment)"}
    for scope in ("full", "gated", "news_live"):
        if not out[scope]:
            continue
        note = notes.get(scope, "")
        print(f"\n{'=' * 104}\nSCOPE: {scope.upper()}{note}\n{'=' * 104}")
        print(pd.DataFrame(out[scope]).reindex(KEYS).round(4).to_string())

    print("\n-- how to read the skill scores --")
    print("   BSS > 0 beats that baseline. Persistence is the honest bar for")
    print("   autocorrelated labels; climatology is the one studies forget.")
    print(f"   Above {cfg['scoring']['leakage_alarm_bss']} against persistence, treat it as a")
    print("   LEAKAGE ALARM and re-run the look-ahead checklist.")

    print("\n-- Murphy decomposition (Brier = reliability - resolution + uncertainty) --")
    print("   reliability low is good; resolution high is good. A perfectly")
    print("   calibrated forecast that never leaves the base rate has zero")
    print("   resolution and is useless.")

    print("\n-- paired tests, blocked at the target's forward window --")
    for k, v in out["tests"].items():
        dm, ci = v["dm_brier"], v["brier_diff_ci"]
        print(f"\n   {k}")
        print(f"      kappa between arms: {v['kappa_between_arms']:+.3f}   "
              f"(two-state {v['kappa_two_state']:+.3f})")
        print(f"      DM on Brier: stat={dm['dm_stat']:+.2f}  p={dm['p_value']:.4f}  "
              f"n={dm['n']}  n_eff~{dm.get('n_effective', 0)}")
        print(f"      mean Brier difference: {ci['diff']:+.4f}  "
              f"95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")

    print("\n-- transition lead time vs hand-labelled episodes --")
    for name, df in arms.items():
        lt = S.transition_lead_time(df["label"], cfg)
        got = int(lt["lead_days"].notna().sum())
        n_ep = len(lt)
        print(f"\n   {name}: reached {got} of {n_ep} onsets")
        print("      " + lt.to_string(index=False).replace("\n", "\n      "))

    if out["secondary"]:
        print("\n-- robustness: the drawdown-state target --")
        print("   NEAR-CIRCULAR for Arm B, which has drawdown as an input feature.")
        print(pd.DataFrame(out["secondary"]).round(4).to_string())

    res_dir = _cfg.paths()["results"]
    with open(res_dir / "contest.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nwritten: {res_dir / 'contest.json'}")
    return out


if __name__ == "__main__":
    from . import config, ingest

    cfg = config.load()
    res_dir = config.paths()["results"]
    blocks = ingest.build_panel(cfg=cfg)
    close = blocks["index"]["close"]

    target = L.build_target(close, cfg)
    secondary = L.build_secondary_target(close, cfg)

    arms = {}
    for key, fname in (("A_hamilton", "arm_a_hamilton.parquet"),
                       ("B_hmm", "arm_b_hmm.parquet"),
                       ("C_agentic", "arm_c_agentic.parquet")):
        p = res_dir / fname
        if p.exists():
            arms[key] = pd.read_parquet(p)

    gate = candidate_days(arms["B_hmm"], cfg) if "B_hmm" in arms else None

    # Absent a news corpus the news_live scope is simply skipped, so the
    # contest still runs on Arms A and B alone.
    news_mask = None
    try:
        from . import news as N
        news_mask = N.coverage_mask(cfg, close.index)
    except FileNotFoundError:
        print("no news corpus yet -- skipping the news_live scope")

    report({k: arms[k] for k in sorted(arms)}, target, cfg,
           gate_mask=gate, secondary=secondary, news_mask=news_mask)
