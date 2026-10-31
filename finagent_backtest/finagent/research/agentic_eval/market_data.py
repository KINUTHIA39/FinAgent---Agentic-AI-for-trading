"""Collect real market data for the agentic backtest (Phase 1).

Sources (yfinance):
  - ^GSPC  : S&P 500 index -> monthly returns 1990-2025 (Hamilton regime estimation).
  - ^IRX   : 13-week T-bill yield -> monthly risk-free rate (Sharpe + cash leg).
  - 8 stocks: daily OHLCV 2005-2025 (adjusted) + monthly returns (backtest universe).

Writes to data/agentic_eval/market_data/ only. Read-only w.r.t. the frozen
dissertation assets (data/dissertation/ is never touched).

Run:
    python -m finagent.research.agentic_eval.market_data
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "agentic_eval" / "market_data"

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "XOM", "JNJ"]
# GOOGL class-A shares only trade from the 2014 split; GOOG is the continuous
# series back to the 2004 IPO. Fall back to GOOG if GOOGL lacks early history.
ALT_TICKER = {"GOOGL": "GOOG"}

REGIME_START = "1990-01-01"   # long S&P history for a stable Hamilton fit
STOCK_START = "2005-01-01"    # all 8 names fully live (GOOGL IPO Aug 2004)
END = "2026-01-01"            # exclusive; covers through Dec 2025


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _dl(ticker: str, start: str, end: str, interval: str = "1d", tries: int = 3) -> pd.DataFrame:
    for k in range(tries):
        df = yf.download(ticker, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=True)
        if df is not None and len(df):
            return _flat(df)
        time.sleep(2)
    raise RuntimeError(f"no data returned for {ticker}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stocks_daily").mkdir(exist_ok=True)

    # 1) S&P 500 monthly returns (regime estimation)
    spx = _dl("^GSPC", REGIME_START, END)
    spx_m = spx["Close"].resample("ME").last().to_frame("close")
    spx_m["ret"] = spx_m["close"].pct_change()
    spx_m.index.name = "date"
    spx_m.to_parquet(OUT / "spx_monthly.parquet")

    # 2) risk-free monthly: ^IRX is an annualised discount rate in % -> monthly decimal
    try:
        irx = _dl("^IRX", REGIME_START, END)
        rf_m = (irx["Close"].resample("ME").last() / 100.0 / 12.0).to_frame("rf")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] risk-free fetch failed ({exc}); using 0.", file=sys.stderr)
        rf_m = pd.DataFrame({"rf": 0.0}, index=spx_m.index)
    rf_m.index.name = "date"
    rf_m.to_parquet(OUT / "rf_monthly.parquet")

    # 3) 8 stocks: daily OHLCV + monthly returns
    monthly: dict[str, pd.Series] = {}
    meta: list[tuple] = []
    for t in TICKERS:
        d = _dl(t, STOCK_START, END)
        note = ""
        if t in ALT_TICKER and d.index.min() > pd.Timestamp("2005-06-01"):
            alt = ALT_TICKER[t]
            d = _dl(alt, STOCK_START, END)
            note = f"(used {alt} for full history)"
        d.index.name = "date"
        d.to_parquet(OUT / "stocks_daily" / f"{t}.parquet")
        monthly[t] = d["Close"].resample("ME").last().pct_change()
        meta.append((t, str(d.index.min().date()), str(d.index.max().date()), len(d), note))

    mret = pd.DataFrame(monthly)
    mret.index.name = "date"
    mret.to_parquet(OUT / "stocks_monthly.parquet")

    # ---- report ----
    print("=" * 64)
    print(f"wrote market data -> {OUT}")
    print("=" * 64)
    print(f"S&P 500 monthly : {len(spx_m):4d} rows  {spx_m.index.min().date()} -> {spx_m.index.max().date()}")
    print(f"risk-free month : {len(rf_m):4d} rows  (mean {rf_m['rf'].mean()*12*100:.2f}% p.a.)")
    print(f"stocks monthly  : {mret.shape[0]} months x {mret.shape[1]} stocks  "
          f"{mret.index.min().date()} -> {mret.index.max().date()}")
    print("daily OHLCV per stock:")
    for t, s, e, n, note in meta:
        print(f"  {t:6s} {n:5d} rows  {s} -> {e}  {note}")
    print("monthly-return coverage (first valid month):")
    for t in TICKERS:
        first = mret[t].dropna().index.min()
        print(f"  {t:6s} {first.date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
