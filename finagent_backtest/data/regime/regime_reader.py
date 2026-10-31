"""Read the frozen regime artifact. Pandas only -- no other dependencies.

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
