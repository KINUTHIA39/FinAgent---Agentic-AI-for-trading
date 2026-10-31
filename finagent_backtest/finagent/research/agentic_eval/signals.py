"""Phase 3: 6-factor Alpha158 signal blend -> per-stock monthly score in [FLOOR, 1].

Six real Alpha158 factors (one per family), each oriented so "higher = more
attractive", cross-sectionally z-scored across the 8 stocks, equal-weighted into a
composite, then min-max rescaled to [FLOOR, 1] so all 8 names stay eligible:

  Momentum            ROC60                 (low ROC = riser)   -> -ROC60
  Trend quality       RSQR60 x sign(uptrend)                    -> RSQR60 * up
  Low volatility      STD20                                     -> -STD20
  Proximity to highs  MAX60                 (low = near high)   -> -MAX60
  Breadth / RSI       SUMD60                                    -> +SUMD60
  Volume confirmation CORR60                                    -> +CORR60

Point-in-time: every factor at a month-end uses only daily OHLCV up to that date
(no look-ahead). The real Alpha158 vector is computed by the production module
``alpha158`` (vendored beside this file from the FinAgent alpha pool) and the six
factors are selected by name.

Reads market_data/stocks_daily; writes market_data/signals_monthly.parquet.

Run:
    python -m finagent.research.agentic_eval.signals
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from finagent.research.agentic_eval.alpha158 import (  # noqa: E402
    compute_alpha158, get_alpha158_names)

MD = REPO / "data" / "agentic_eval" / "market_data"
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "XOM", "JNJ"]
FLOOR = 0.05
_FACTORS = ["ROC60", "RSQR60", "STD20", "MAX60", "SUMD60", "CORR60"]
_NAMES = get_alpha158_names()
_IDX = {f: _NAMES.index(f) for f in _FACTORS}


def _load_daily() -> dict[str, pd.DataFrame]:
    out = {}
    for t in TICKERS:
        df = pd.read_parquet(MD / "stocks_daily" / f"{t}.parquet").rename(columns=str.lower)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        out[t] = df[["open", "high", "low", "close", "volume"]]
    return out


def _oriented_factors(df: pd.DataFrame, date: pd.Timestamp):
    """The 6 factors (oriented higher=better) for one stock as of `date`, point-in-time."""
    sub = df.loc[:date]
    if len(sub) < 70:
        return None
    vec = compute_alpha158(sub.tail(120))
    if vec is None:
        return None
    roc = vec[_IDX["ROC60"]]
    up = 1.0 if roc < 1.0 else -1.0          # riser if ROC60 < 1 (price up over 60d)
    return np.array([
        -roc,                                 # momentum
        vec[_IDX["RSQR60"]] * up,             # trend quality, in the trend direction
        -vec[_IDX["STD20"]],                  # low volatility
        -vec[_IDX["MAX60"]],                  # proximity to highs (low MAX = near high)
        vec[_IDX["SUMD60"]],                  # breadth / RSI
        vec[_IDX["CORR60"]],                  # volume confirmation
    ])


def build_signals() -> pd.DataFrame:
    daily = _load_daily()
    months = pd.read_parquet(MD / "stocks_monthly.parquet").index
    if getattr(months, "tz", None) is not None:
        months = months.tz_localize(None)
    rows: dict = {}
    for d in months:
        raw = {t: _oriented_factors(daily[t], d) for t in TICKERS}
        if any(v is None for v in raw.values()):
            continue
        M = np.vstack([raw[t] for t in TICKERS])          # (8 stocks, 6 factors)
        z = (M - M.mean(0)) / (M.std(0) + 1e-12)          # cross-sectional z per factor
        comp = z.mean(1)                                   # equal-weight composite
        lo, hi = comp.min(), comp.max()
        sc = (np.full(8, (FLOOR + 1) / 2) if hi - lo < 1e-12
              else FLOOR + (1 - FLOOR) * (comp - lo) / (hi - lo))
        rows[d] = dict(zip(TICKERS, sc))
    sig = pd.DataFrame.from_dict(rows, orient="index")[TICKERS]
    sig.index.name = "date"
    return sig


def main() -> int:
    sig = build_signals()
    sig.to_parquet(MD / "signals_monthly.parquet")
    print(f"signals: {sig.shape[0]} months x {sig.shape[1]} stocks  "
          f"{sig.index.min().date()} -> {sig.index.max().date()}  (score in [{FLOOR}, 1])")
    print("\nSample (last 3 months):")
    print(sig.tail(3).round(2).to_string())
    print("\nMean score by stock (how often each name is near the top):")
    print(sig.mean().round(3).sort_values(ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
