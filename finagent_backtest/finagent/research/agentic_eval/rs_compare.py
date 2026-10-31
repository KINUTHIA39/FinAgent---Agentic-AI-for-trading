"""Phase 4: compare the four arms of the agentic-RS experiment.

The question is not whether the agent beats the unmodified model. That is too low
a bar, because a two-line deterministic rule already does. The question is
whether it beats *both* controls:

  * the rule, which sees exactly the same diagnostics and applies fixed logic; and
  * random selection from the same parameter space, which is what "the agent is
    just picking numbers" would look like.

Beating the rule is evidence of judgement. Failing to is evidence that the
judgement is decorative, which is a reportable result rather than a failure.

Run:
    python -m finagent.research.agentic_eval.rs_compare
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "agentic_eval" / "rs_agent"


def _stats(r: pd.Series) -> dict:
    w = (1 + r).cumprod()
    return {"months": len(r), "final": float(w.iloc[-1]),
            "cagr": float(w.iloc[-1] ** (12 / len(r)) - 1) * 100,
            "vol": float(r.std() * np.sqrt(12)) * 100,
            "sharpe": float(r.mean() / r.std() * np.sqrt(12)),
            "maxdd": float((w / w.cummax() - 1).min()) * 100}


def load_arms() -> dict[str, pd.DataFrame]:
    arms: dict[str, pd.DataFrame] = {}
    for label, fname in [("RS fixed (classical)", "path_fixed.parquet"),
                         ("RS + deterministic rule", "path_rule.parquet"),
                         ("RS + agent", "path_agent.parquet")]:
        f = OUT / fname
        if f.exists():
            arms[label] = pd.read_parquet(f)
    rnd = sorted(OUT.glob("path_random_s*.parquet"))
    for f in rnd:
        arms[f"RS + random (seed {f.stem.split('_s')[-1]})"] = pd.read_parquet(f)
    return arms


def main() -> int:
    arms = load_arms()
    if "RS + agent" not in arms:
        print("agent arm not found; run --mode agent first")
        return 1

    # Compare every arm on the window the agent actually covered.
    idx = arms["RS + agent"].index
    print(f"common window: {idx[0]:%Y-%m} .. {idx[-1]:%Y-%m}  ({len(idx)} months)\n")

    hdr = f"{'arm':<28}{'final':>9}{'CAGR':>8}{'vol':>8}{'Sharpe':>9}{'maxDD':>9}"
    print(hdr)
    print("-" * len(hdr))
    rand_sharpe, rand_final = [], []
    for label, df in arms.items():
        s = _stats(df.loc[df.index.intersection(idx), "ret"])
        print(f"{label:<28}{s['final']:>8.1f}x{s['cagr']:>7.1f}%{s['vol']:>7.1f}%"
              f"{s['sharpe']:>9.2f}{s['maxdd']:>8.1f}%")
        if "random" in label:
            rand_sharpe.append(s["sharpe"])
            rand_final.append(s["final"])

    ag = arms["RS + agent"]
    a_s = _stats(ag.loc[idx, "ret"])
    r_s = _stats(arms["RS + deterministic rule"].loc[idx, "ret"])
    f_s = _stats(arms["RS fixed (classical)"].loc[idx, "ret"])

    print("\nverdict")
    beat_fixed = a_s["sharpe"] > f_s["sharpe"]
    beat_rule = a_s["sharpe"] > r_s["sharpe"]
    beat_rand = (a_s["sharpe"] > np.mean(rand_sharpe)) if rand_sharpe else None
    print(f"  beats classical RS on Sharpe : {beat_fixed} "
          f"({a_s['sharpe']:.2f} vs {f_s['sharpe']:.2f})")
    print(f"  beats deterministic rule      : {beat_rule} "
          f"({a_s['sharpe']:.2f} vs {r_s['sharpe']:.2f})")
    if rand_sharpe:
        print(f"  beats random mean             : {beat_rand} "
              f"({a_s['sharpe']:.2f} vs {np.mean(rand_sharpe):.2f}, "
              f"n={len(rand_sharpe)})")
    if beat_rule and beat_rand:
        print("  -> the agentic layer adds something a fixed rule does not.")
    elif beat_fixed:
        print("  -> the agent improves on the classical model, but not on a simple "
              "rule using the same information. The layer is not earning its cost.")
    else:
        print("  -> no evidence the agentic layer helps.")

    print("\nagent behaviour")
    print(f"  deviated from defaults : {int(ag['deviated'].sum())}/{len(ag)} "
          f"({ag['deviated'].mean()*100:.0f}%)")
    print(f"  schema rejections      : {int((~ag['accepted']).sum())}")
    print(f"  gamma  : mean {ag['gamma'].mean():.1f}  "
          f"range {ag['gamma'].min():.1f}-{ag['gamma'].max():.1f}")
    print(f"  cap    : mean {ag['cap'].mean():.2f}  "
          f"months capped {int((ag['cap'] < 1.0).sum())}")
    print(f"  blend  : {int((ag['treatment'] == 'probability_blend').sum())} months")

    by_reg = ag.groupby("regime").agg(n=("gamma", "size"), gamma=("gamma", "mean"),
                                      cap=("cap", "mean"))
    print("\n  by regime:")
    for reg, row in by_reg.iterrows():
        print(f"    {reg:<5} n={int(row['n']):<4} mean gamma {row['gamma']:.1f}  "
              f"mean cap {row['cap']:.2f}")

    # Did the months it chose to deviate actually do better than the classical
    # model in those same months? This is the sharpest test of selectivity.
    base = arms["RS fixed (classical)"].loc[idx, "ret"]
    dev = ag["deviated"]
    if dev.any() and (~dev).any():
        d_gain = (ag.loc[dev, "ret"] - base[dev]).mean() * 100
        n_gain = (ag.loc[~dev, "ret"] - base[~dev]).mean() * 100
        print(f"\n  months it deviated   : {int(dev.sum()):>3}  "
              f"mean edge over classical {d_gain:+.3f}%/mo")
        print(f"  months it did not    : {int((~dev).sum()):>3}  "
              f"mean edge over classical {n_gain:+.3f}%/mo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
