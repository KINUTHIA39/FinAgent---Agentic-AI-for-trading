"""Fixed-threshold 2x2 market-regime classifier — Bull/Bear x Calm/Stress.

Reads the dissertation's VIX and SPY series (READ-ONLY) and labels each trading
day on two independent axes:

  - trend       = sign of SPY's trailing ``trend_window``-day return
                  -> "bull" (>= 0) | "bear" (< 0)
  - volatility  = VIX vs a fixed threshold
                  -> "stress" (VIX >= threshold) | "calm" (VIX < threshold)

Composite regime = "{trend}-{volatility}" in
  {bull-calm, bull-stress, bear-calm, bear-stress}.

Fixed thresholds (not in-sample terciles) avoid the look-ahead bias of
percentile cuts and keep the classifier simple, reproducible, and interpretable.
Inputs:
  data/dissertation/vix.parquet            (column: vix_close, DatetimeIndex 'date')
  data/dissertation/benchmark/SPY.parquet  (column: close,     DatetimeIndex 'timestamp')
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
_VIX_PATH = REPO_ROOT / "data" / "dissertation" / "vix.parquet"
_SPY_PATH = REPO_ROOT / "data" / "dissertation" / "benchmark" / "SPY.parquet"

# Defaults — fixed, documented, look-ahead-free.
DEFAULT_TREND_WINDOW = 63      # ~1 quarter of trading days
DEFAULT_VIX_THRESHOLD = 20.0   # calm < 20 <= stress (long-run VIX median ~17-18)

REGIMES = ("bull-calm", "bull-stress", "bear-calm", "bear-stress")


@dataclass(frozen=True)
class RegimeLabel:
    """A single day's regime classification."""
    date: pd.Timestamp
    trend: str          # "bull" | "bear"
    volatility: str     # "calm" | "stress"
    spy_return: float   # trailing trend-window return
    vix: float

    @property
    def composite(self) -> str:
        return f"{self.trend}-{self.volatility}"


def classify_regimes(
    trend_window: int = DEFAULT_TREND_WINDOW,
    vix_threshold: float = DEFAULT_VIX_THRESHOLD,
    vix_path: Optional[Path] = None,
    spy_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return a daily regime table indexed by date.

    Columns: close, vix, spy_ret, trend, volatility, regime.
    """
    vix = pd.read_parquet(vix_path or _VIX_PATH)
    spy = pd.read_parquet(spy_path or _SPY_PATH)

    vix = vix.rename(columns={vix.columns[0]: "vix"})[["vix"]].copy()
    vix.index = pd.to_datetime(vix.index).normalize()

    spy = spy[["close"]].copy()
    spy.index = pd.to_datetime(spy.index).normalize()

    df = spy.join(vix, how="inner").sort_index()
    df["spy_ret"] = df["close"].pct_change(trend_window)
    df = df.dropna(subset=["spy_ret", "vix"])

    df["trend"] = df["spy_ret"].apply(lambda r: "bull" if r >= 0 else "bear")
    df["volatility"] = df["vix"].apply(
        lambda v: "stress" if v >= vix_threshold else "calm")
    df["regime"] = df["trend"] + "-" + df["volatility"]
    return df[["close", "vix", "spy_ret", "trend", "volatility", "regime"]]


def latest_regime(
    trend_window: int = DEFAULT_TREND_WINDOW,
    vix_threshold: float = DEFAULT_VIX_THRESHOLD,
) -> RegimeLabel:
    """Classify the most recent available day."""
    df = classify_regimes(trend_window, vix_threshold)
    row = df.iloc[-1]
    return RegimeLabel(
        date=df.index[-1], trend=row["trend"], volatility=row["volatility"],
        spy_return=float(row["spy_ret"]), vix=float(row["vix"]),
    )


def regime_distribution(
    trend_window: int = DEFAULT_TREND_WINDOW,
    vix_threshold: float = DEFAULT_VIX_THRESHOLD,
) -> pd.Series:
    """Fraction of OOS days in each regime — for the methodology chapter."""
    return classify_regimes(trend_window, vix_threshold)["regime"].value_counts(
        normalize=True)


def sample_dates_per_regime(
    n_per_regime: int = 5,
    trend_window: int = DEFAULT_TREND_WINDOW,
    vix_threshold: float = DEFAULT_VIX_THRESHOLD,
    seed: int = 42,
) -> dict:
    """Sample real historical dates for each regime — for ecological-validity runs."""
    df = classify_regimes(trend_window, vix_threshold)
    out: dict = {}
    for regime in REGIMES:
        days = df[df["regime"] == regime]
        if len(days):
            out[regime] = list(days.sample(min(n_per_regime, len(days)),
                                           random_state=seed).index)
    return out


__all__ = [
    "RegimeLabel", "classify_regimes", "latest_regime", "regime_distribution",
    "sample_dates_per_regime", "REGIMES",
    "DEFAULT_TREND_WINDOW", "DEFAULT_VIX_THRESHOLD",
]
