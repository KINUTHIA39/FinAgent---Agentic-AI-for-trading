"""Contamination probes for Arm C (spec 5).

Two independent tests, because they fail in different ways:

  1. DATE-GUESSING PROBE. Show the model an anonymised window in exactly the
     format the specialists see and ask it what year it is. If it can tell --
     especially if it clusters on 2008 and 2020 -- the anonymisation failed and
     any Arm C advantage may be recall rather than inference.

  2. DATES-ON ABLATION. Re-run a sample of gated days with calendar dates
     restored and compare. The gap is what being told the date is worth. A
     large gap means the anonymised run was leaving recall on the table, which
     is the outcome you want; a small gap means dates add nothing and the
     anonymisation was not doing much work either way.

Neither probe can prove the absence of leakage. They bound it, and reporting
what they found is the point -- most LLM-in-finance demos skip this entirely.

    python scripts/contamination_ablation.py [n_probe] [n_ablation]
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import warnings
from typing import Literal

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src import agents, config, detector, features as F, ingest

DATE_SYSTEM = """You are shown a window of standardized financial market
indicators for four currency pairs, exactly as a forecasting model would see
them. Values are z-scores against an expanding history; periods are labelled
t, t-1, t-2 relative to the most recent observation.

Identify when this window is from. Guess even if you are unsure -- an honest
guess with low confidence is more useful here than a refusal. This is a
calibration check on an anonymisation scheme, not a trick."""


class DateGuess(BaseModel):
    year: int = Field(ge=1990, le=2030, description="Best guess at the calendar year.")
    confidence: float = Field(ge=0.0, le=1.0)
    recognized_episode: str = Field(
        description="Named market episode you think this is, or 'none'.")
    reasoning: str = Field(description="One or two sentences.")


async def _probe(days, views, cfg, sem):
    async def one(day):
        payload = {r: agents._window(v, day) for r, v in views.items()}
        out = await agents._ask(DATE_SYSTEM, payload, DateGuess,
                                "date_probe", cfg, sem)
        return day, out

    return await asyncio.gather(*[one(d) for d in days])


def main() -> None:
    n_probe = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_abl = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    cfg = config.load()
    blocks = ingest.build_panel(cfg=cfg)
    feats = F.build_features(blocks, cfg)
    arm_b = pd.read_parquet("results/arm_b_hmm.parquet")
    gate = detector.candidate_days(arm_b, cfg)
    views = agents.build_agent_views(feats, cfg)

    res_dir = config.paths()["results"]
    rng = np.random.default_rng(int(cfg["run"]["seed"]))

    # ------------------------------------------------------------- probe 1
    print("=" * 92)
    print("PROBE 1 - date guessing from anonymised windows")
    print("=" * 92)
    pool = arm_b.index
    days = sorted(pd.DatetimeIndex(rng.choice(pool, size=n_probe, replace=False)))
    sem = asyncio.Semaphore(int(cfg["agents"].get("concurrency", 8)))
    guesses = asyncio.run(_probe(days, views, cfg, sem))

    rows = []
    for day, g in guesses:
        rows.append({"true_year": day.year, "guess": g["year"],
                     "err": abs(g["year"] - day.year),
                     "confidence": g["confidence"],
                     "episode": g.get("recognized_episode", "")})
    df = pd.DataFrame(rows)
    exact = float((df["err"] == 0).mean())
    within2 = float((df["err"] <= 2).mean())
    baseline = 1.0 / arm_b.index.year.nunique()

    print(f"\nn={len(df)}   exact-year accuracy={exact * 100:.1f}%   "
          f"within +/-2y={within2 * 100:.1f}%")
    print(f"random-guess baseline over {arm_b.index.year.nunique()} years = "
          f"{baseline * 100:.1f}%")
    print(f"median absolute error: {df['err'].median():.1f} years")
    verdict = ("ANONYMISATION HOLDS" if exact < baseline * 3
               else "LEAKAGE - the model can date these windows")
    print(f"\n   VERDICT: {verdict}")

    print("\n-- does it cluster on the crisis years? --")
    for y in (2008, 2020):
        guessed = int((df["guess"] == y).sum())
        actual = int((df["true_year"] == y).sum())
        print(f"   guessed {y}: {guessed:3d} times   (actually {y}: {actual})")
    print("\n-- most-guessed years --")
    print(df["guess"].value_counts().head(5).to_string())

    named = df[df["episode"].str.lower().str.strip().ne("none")]
    print(f"\n-- named a specific episode on {len(named)}/{len(df)} windows --")
    if len(named):
        for _, r in named.head(8).iterrows():
            hit = "CORRECT" if r["err"] <= 1 else "wrong"
            print(f"   true {r['true_year']}, guessed {r['guess']} ({hit}): "
                  f"{r['episode'][:60]}")

    df.to_csv(res_dir / "contamination_date_probe.csv", index=False)

    # ------------------------------------------------------------- probe 2
    print("\n" + "=" * 92)
    print("PROBE 2 - dates-on ablation")
    print("=" * 92)
    gated_days = arm_b.index[gate]
    sample = sorted(pd.DatetimeIndex(
        rng.choice(gated_days, size=min(n_abl, len(gated_days)), replace=False)))
    # Pass the FULL arm_b with a sparse mask, not a 60-row slice. The rolling
    # memory buffer is replayed from whatever frame it is given, so slicing
    # first would hand the panel a fabricated history -- and the anonymised
    # half would then miss the main run's cache and be paid for twice.
    sparse = pd.Series(False, index=arm_b.index)
    sparse.loc[sample] = True

    anon_cfg = copy.deepcopy(cfg)
    anon_cfg["agents"]["anonymise_dates"] = True
    dated_cfg = copy.deepcopy(cfg)
    dated_cfg["agents"]["anonymise_dates"] = False

    print(f"\nrunning {len(sample)} gated days twice "
          f"({len(sample) * 10} calls; anonymised half is cache-warm)")
    out_anon = agents.walk_forward(arm_b, feats, anon_cfg, sparse)
    out_dated = agents.walk_forward(arm_b, feats, dated_cfg, sparse)

    agree = float((out_anon.loc[sample, "label"] == out_dated.loc[sample, "label"]).mean())
    conf_a = float(out_anon.loc[sample, "confidence"].mean())
    conf_d = float(out_dated.loc[sample, "confidence"].mean())
    pa = pd.Series([v["action"] for v in out_anon.attrs["provenance"].values()])
    pd_ = pd.Series([v["action"] for v in out_dated.attrs["provenance"].values()])

    print(f"\n   label agreement anonymised vs dated: {agree * 100:.1f}%")
    print(f"   mean confidence   anonymised {conf_a:.4f}   dated {conf_d:.4f}   "
          f"delta {conf_d - conf_a:+.4f}")
    print(f"\n   actions (anonymised):\n{pa.value_counts().to_string()}")
    print(f"\n   actions (dated):\n{pd_.value_counts().to_string()}")

    if agree > 0.95 and abs(conf_d - conf_a) < 0.02:
        print("\n   VERDICT: dates change essentially nothing. Either the panel "
              "is not\n   using recall, or the features already identify the "
              "period. Read this\n   together with probe 1, not on its own.")
    else:
        print(f"\n   VERDICT: dates change the output on "
              f"{(1 - agree) * 100:.0f}% of days. Quantify this in the write-up "
              f"and\n   treat the anonymised run as the headline.")

    json.dump({"date_probe": {"exact": exact, "within2": within2,
                              "baseline": baseline, "n": len(df)},
               "dates_on_ablation": {"label_agreement": agree,
                                     "conf_anon": conf_a, "conf_dated": conf_d,
                                     "n": len(sample)}},
              open(res_dir / "contamination_summary.json", "w"), indent=1)
    print(f"\nwritten: {res_dir / 'contamination_summary.json'}")


if __name__ == "__main__":
    main()
