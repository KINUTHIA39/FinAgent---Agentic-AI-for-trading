"""Freeze the regime series into a consumable artifact.

Writes three things to the destination directory:

    regime_daily_v1.parquet   the series, all three arms, long format
    manifest.json             provenance, so a backtest can record its vintage
    regime_reader.py          a pandas-only reader the consumer drops in

The reader ships WITH the data rather than living in the consuming repo,
because the two must not drift apart: the reader's guards depend on the
manifest's score_end, and a reader that has fallen behind its data is worse
than no reader at all.

WHY AN ARTIFACT AND NOT A LIVE IMPORT. Arm C's labels come from ~2,910 LLM
calls. Regenerating them per backtest would make results non-reproducible,
cost money every run, and -- worst -- invite look-ahead, since asking a live
pipeline for "the regime on 2008-10-15" is one careless line from computing it
with data through today. Frozen artifacts make all three impossible.

    python -m src.export_regime --to ../finagent_backtest/data/regime
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

ARMS = {"C_agentic": "arm_c_agentic.parquet",     # the product
        "B_hmm": "arm_b_hmm.parquet",             # benchmark
        "A_hamilton": "arm_a_hamilton.parquet"}   # benchmark

COLS = ["label", "p_bull_quiet", "p_bull_volatile", "p_bear_stress",
        "p_bear_raw", "confidence"]

READER = '''"""Read the frozen regime artifact. Pandas only -- no other dependencies.

Shipped alongside regime_daily_v1.parquet by the regime indicator's exporter.
Do not edit here; edit it there and re-export, or the two will drift.

    from regime_reader import RegimeReader
    r = RegimeReader("data/regime")
    r.on("2020-09-03")                 # one day, raises outside the window
    r.month_summary("2020-09-30")      # what a monthly rebalance should read
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

STATES = ("bull_quiet", "bull_volatile", "bear_stress")


class RegimeOutOfRange(KeyError):
    """Asked for a date the frozen artifact does not cover."""


class RegimeReader:
    """Frozen S&P 500 regime series. See CONTRACT.md in the producing repo."""

    def __init__(self, directory: str | Path, arm: str = "C_agentic"):
        d = Path(directory)
        self.manifest = json.loads((d / "manifest.json").read_text())
        df = pd.read_parquet(d / "regime_daily_v1.parquet")
        if arm not in set(df["arm"]):
            raise ValueError(f"arm {arm!r} not in artifact; "
                             f"have {sorted(set(df['arm']))}")
        self.arm = arm
        self.df = df[df["arm"] == arm].drop(columns="arm").sort_index()
        self.start = self.df.index.min()
        self.end = self.df.index.max()

    # ------------------------------------------------------------------ core
    def on(self, date) -> pd.Series:
        """The regime row for one date.

        Raises rather than returning the last known row. This is the whole
        point of the class: `df.asof(date)` and `reindex(method="ffill")` both
        walk silently past the end of the frozen window and hand back a
        confident, stale answer. A backtest that does that looks fine and is
        wrong.
        """
        ts = pd.Timestamp(date)
        if ts < self.start or ts > self.end:
            raise RegimeOutOfRange(
                f"{ts.date()} is outside the frozen window "
                f"{self.start.date()}..{self.end.date()}. The artifact does "
                f"not extrapolate; re-export from the regime repo to extend it."
            )
        if ts not in self.df.index:
            # a real date inside the window but not a trading session
            prior = self.df.index[self.df.index <= ts]
            if not len(prior):
                raise RegimeOutOfRange(f"no session at or before {ts.date()}")
            ts = prior[-1]
        return self.df.loc[ts]

    def between(self, start, end) -> pd.DataFrame:
        """Every session in [start, end]. Clipped to the frozen window."""
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        if b < self.start or a > self.end:
            raise RegimeOutOfRange(
                f"{a.date()}..{b.date()} does not overlap the frozen window "
                f"{self.start.date()}..{self.end.date()}")
        return self.df.loc[max(a, self.start):min(b, self.end)]

    # -------------------------------------------------------------- monthly
    def month_summary(self, rebalance_date, lookback_start=None) -> dict:
        """What a monthly rebalance should read -- the month, not one day.

        A point sample at month-end is not enough. 71% of regime episodes are
        shorter than a month, and on 2007-10-31 the month-end reading was
        bull_quiet even though the detector had flagged stress on 2007-10-19.
        `regime_changed` and `worst_regime` are what carry that fact.
        """
        end = pd.Timestamp(rebalance_date)
        start = (pd.Timestamp(lookback_start) if lookback_start is not None
                 else end - pd.DateOffset(months=1))
        seg = self.between(start, end)
        cur = self.on(end)

        order = {s: i for i, s in enumerate(STATES)}
        worst = max(seg["label"], key=lambda s: order.get(s, -1))

        # how long the current state has run, looking back through the series
        lab = self.df.loc[:seg.index[-1], "label"]
        run = int((lab != lab.shift()).cumsum().iloc[-1])
        days_in_state = int((lab.groupby(
            (lab != lab.shift()).cumsum()).size()).loc[run])

        return {
            "as_of": str(seg.index[-1].date()),
            "arm": self.arm,
            "label": str(cur["label"]),
            "p_bull_quiet": float(cur["p_bull_quiet"]),
            "p_bull_volatile": float(cur["p_bull_volatile"]),
            "p_bear_stress": float(cur["p_bear_stress"]),
            "confidence": float(cur["confidence"]),
            "days_in_state": days_in_state,
            "regime_changed": bool(seg["label"].nunique() > 1),
            "worst_regime": str(worst),
            "mean_p_bear_stress": float(seg["p_bear_stress"].mean()),
            "max_p_bear_stress": float(seg["p_bear_stress"].max()),
            "sessions": int(len(seg)),
        }

    def __repr__(self):
        return (f"RegimeReader(arm={self.arm!r}, "
                f"{self.start.date()}..{self.end.date()}, "
                f"n={len(self.df):,}, built={self.manifest.get('built_at')})")
'''


def build(dest: Path, cfg: dict) -> None:
    res = config.paths()["results"]
    a, z = cfg["sample"]["score_start"], cfg["sample"]["score_end"]

    frames = []
    for arm, fname in ARMS.items():
        p = res / fname
        if not p.exists():
            print(f"   skip {arm}: {fname} not found")
            continue
        df = pd.read_parquet(p).loc[a:z]
        keep = [c for c in COLS if c in df.columns]
        out = df[keep].copy()
        for c in COLS:
            if c not in out.columns:
                out[c] = pd.NA          # p_bear_raw exists only for Arm A
        out = out[COLS]
        out.insert(0, "arm", arm)
        frames.append(out)
        print(f"   {arm:12s} {len(out):,} rows  "
              f"{out.index[0].date()} -> {out.index[-1].date()}")

    if not frames:
        raise SystemExit("no arm outputs found -- run the pipeline first")

    series = pd.concat(frames).sort_index()
    dest.mkdir(parents=True, exist_ok=True)
    series.to_parquet(dest / "regime_daily_v1.parquet")

    prod = series[series["arm"] == "C_agentic"]
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    manifest = {
        "artifact": "regime_daily_v1",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": "Agentic_Regime_Indicator",
        "product_arm": "C_agentic",
        "benchmark_arms": [k for k in ARMS if k != "C_agentic"],
        "index": cfg["universe"]["index_name"],
        "score_start": str(prod.index.min().date()),
        "score_end": str(prod.index.max().date()),
        "n_days": int(len(prod)),
        "states": list(cfg["regimes"]["states"]),
        "refit_frequency": cfg["sample"]["refit_frequency"],
        "probabilities": "filtered",
        "config_hash": cfg_hash,
        "agent_model": cfg["agents"]["model"],
        "note": "Frozen snapshot. Does not extrapolate. See CONTRACT.md in the "
                "producing repo before consuming.",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (dest / "regime_reader.py").write_text(READER, encoding="utf-8")

    # The contract travels WITH the data. A consumer that has to go looking in
    # the producing repo for it will not, and the misuse cases in section 8 are
    # exactly the ones that fail silently.
    src_contract = config.ROOT / "CONTRACT.md"
    files = ["regime_daily_v1.parquet", "manifest.json", "regime_reader.py"]
    if src_contract.exists():
        (dest / "CONTRACT.md").write_text(
            src_contract.read_text(encoding="utf-8"), encoding="utf-8")
        files.append("CONTRACT.md")
    else:
        print("   WARNING: CONTRACT.md not found in the producing repo; "
              "the consumer will have no guidance")

    print(f"\n   written to {dest}")
    for f in files:
        print(f"      {f:26s} {(dest / f).stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True, help="destination directory")
    args = ap.parse_args()
    build(Path(args.to), config.load())
