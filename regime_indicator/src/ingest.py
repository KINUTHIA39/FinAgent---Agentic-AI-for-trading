"""Data loaders. One function per source, each returning a tidy frame.

Rules:
  - Immutable parquet per pull into data/raw/, keyed by as_of.
  - No forward-fill on the index. A fabricated flat day reads to the HMM as a
    calm regime, which is the single easiest way to manufacture a fake result.
  - Weekly and survey series carry a PUBLICATION lag, not their reference date.

Every function that touches time takes `as_of` and must not read past it.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from . import config

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv(config.ROOT / ".env")
_FRED = None


def _fred():
    """Lazy client, so importing this module never requires a key."""
    global _FRED
    if _FRED is None:
        from fredapi import Fred

        key = os.getenv("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED_API_KEY not found in .env")
        _FRED = Fred(api_key=key)
    return _FRED


# --------------------------------------------------------------------------- io
def _cache_path(source: str, name: str, as_of: pd.Timestamp) -> Path:
    safe = name.replace("^", "").replace("=", "_").replace(".", "_")
    return config.paths()["raw"] / f"{source}__{safe}__{as_of.date()}.parquet"


def _cached(source: str, name: str, as_of: pd.Timestamp, fetch) -> pd.Series:
    """Immutable point-in-time pull. One parquet per (source, name, as_of)."""
    path = _cache_path(source, name, as_of)
    if path.exists():
        return pd.read_parquet(path)["value"]
    s = fetch()
    s = s[s.index <= as_of]
    pd.DataFrame({"value": s}).to_parquet(path)
    return s


def _yahoo(ticker: str, as_of: pd.Timestamp, start: str,
           field: str = "Close") -> pd.Series:
    def fetch():
        import yfinance as yf

        df = yf.download(ticker, start=start, interval="1d", progress=False,
                         auto_adjust=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df[field].dropna().astype(float)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        return s.rename(ticker)

    return _cached("yahoo", f"{ticker}_{field}", as_of, fetch)


def _fred_series(sid: str, as_of: pd.Timestamp, start: str) -> pd.Series:
    def fetch():
        s = _fred().get_series(sid, observation_start=start)
        s.index = pd.DatetimeIndex(s.index)
        return s.dropna().astype(float).rename(sid)

    return _cached("fred", sid, as_of, fetch)


def _lag_to_publication(s: pd.Series, calendar: pd.DatetimeIndex,
                        bdays: int) -> pd.Series:
    """Shift a series forward to when it was actually published, then hold.

    A weekly figure stamped with its reference week is not knowable during that
    week. Shifting by the publication lag and holding flat is the honest
    treatment; interpolating would imply foresight.
    """
    shifted = s.copy()
    shifted.index = shifted.index + pd.tseries.offsets.BDay(bdays)
    joined = shifted.reindex(shifted.index.union(calendar)).ffill()
    return joined.reindex(calendar)


# ------------------------------------------------------------------- the index
def load_index(as_of: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Index close and volume proxy. Never forward-filled.

    Volume comes from SPY rather than the index itself -- index volume is
    unreliable and, before ETF inception, meaningless.
    """
    u, d = cfg["universe"], cfg["data"]
    px = _yahoo(u["index"], as_of, d["start"])
    out = pd.DataFrame({"close": px})
    try:
        out["volume"] = _yahoo(cfg["sources"]["yahoo"]["spy"], as_of,
                               d["start"], field="Volume")
    except Exception:
        out["volume"] = np.nan

    if d.get("drop_incomplete_days", True):
        out = out.dropna(subset=["close"])
    out = out[out.index <= as_of]
    assert not out["close"].isna().any(), "index close must contain no NaNs"
    return out


def load_volatility(as_of: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Implied-volatility surface inputs.

    The VIX term-structure ratio the plan specified is absent: Yahoo's ^VIX3M,
    ^VIX9D and ^VIX6M each return a single row, and FRED's VXVCLS starts
    2007-12-04 -- after the scoring window opens. VVIX and SKEW are kept as
    agent inputs, where a missing early value is reported uninformative rather
    than silently imputed.
    """
    y, d = cfg["sources"]["yahoo"], cfg["data"]
    out = {"VIX": _yahoo(y["vix"], as_of, d["start"])}
    for key, tick in (("VVIX", y.get("vvix")), ("SKEW", y.get("skew"))):
        if not tick:
            continue
        try:
            out[key] = _yahoo(tick, as_of, d["start"])
        except Exception:
            pass
    return pd.DataFrame(out).sort_index()


def load_conditions(as_of: pd.Timestamp, cfg: dict,
                    calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Credit and financial-conditions context, on the index calendar.

    ICE BofA option-adjusted spreads are licence-truncated to a rolling three
    years on FRED, so BAA10Y (Moody's Baa less the 10-year Treasury, daily from
    1990) carries the credit signal instead. NFCI is weekly and is lagged to
    publication.
    """
    f, d = cfg["sources"]["fred"], cfg["data"]
    lags = cfg.get("publication_lags", {})
    out = {}
    for key, sid in (("credit_spread", f["credit_spread"]),
                     ("term_spread", f["term_spread"])):
        s = _fred_series(sid, as_of, d["start"])
        out[key] = s.reindex(s.index.union(calendar)).ffill().reindex(calendar)
    nfci = _fred_series(f["nfci"], as_of, d["start"])
    out["nfci"] = _lag_to_publication(nfci, calendar, int(lags.get("nfci", 5)))
    return pd.DataFrame(out)


def load_breadth(as_of: pd.Timestamp, cfg: dict,
                 calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Equal-weight versus cap-weight, the cheapest breadth proxy available.

    RSP/SPY falling means the average stock is lagging the index -- leadership
    narrowing, which is a classic late-cycle signature. RSP starts 2003.
    """
    y, d = cfg["sources"]["yahoo"], cfg["data"]
    try:
        rsp = _yahoo(y["rsp"], as_of, d["start"])
        spy = _yahoo(y["spy"], as_of, d["start"])
    except Exception:
        return pd.DataFrame({"breadth_ratio": np.nan}, index=calendar)
    ratio = (rsp / spy).reindex(calendar)
    return pd.DataFrame({"breadth_ratio": ratio})


def load_put_call(as_of: pd.Timestamp, cfg: dict,
                  calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Cboe put/call ratio. Optional -- Cboe reorganised these archives in 2019.

    Returns an all-NaN column rather than raising when unavailable, so the
    market-sentiment agent can report it uninformative instead of the pipeline
    failing on a third-party CSV move.
    """
    url = cfg.get("sources", {}).get("cboe", {}).get("equity_put_call")
    if not url:
        return pd.DataFrame({"put_call": np.nan}, index=calendar)
    def fetch():
        import requests

        r = requests.get(url, timeout=45)
        r.raise_for_status()
        from io import StringIO

        raw = pd.read_csv(StringIO(r.text), skiprows=2)
        date_col = next(c for c in raw.columns if "date" in c.lower())
        pc_col = next(c for c in raw.columns if "ratio" in c.lower())
        s = pd.Series(pd.to_numeric(raw[pc_col], errors="coerce").values,
                      index=pd.to_datetime(raw[date_col], errors="coerce"))
        return s.dropna().sort_index()

    try:
        # Cached like every other source, so a replay does not depend on the
        # archive still being where it was.
        s = _cached("cboe", "equity_put_call", as_of, fetch)
        return pd.DataFrame({"put_call": s.reindex(calendar)})
    except Exception as exc:                                   # noqa: BLE001
        print(f"   put/call unavailable ({type(exc).__name__}); "
              f"continuing without it")
        return pd.DataFrame({"put_call": np.nan}, index=calendar)


# ------------------------------------------------------------------------ panel
def build_panel(as_of: pd.Timestamp | str | None = None,
                cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Assemble every source onto the index calendar.

    The index defines the calendar and is never filled. Context series are
    reindexed onto it with a bounded fill, since they follow their own release
    schedules -- but none of them reach the HMM feature vector, which is built
    from price and VIX alone.
    """
    cfg = cfg or config.load()
    as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp(cfg["data"]["end"])
    limit = int(cfg["data"].get("cross_asset_ffill_limit", 5))

    index = load_index(as_of, cfg)
    cal = index.index

    vol = load_volatility(as_of, cfg).reindex(cal).ffill(limit=limit)
    cond = load_conditions(as_of, cfg, cal)
    breadth = load_breadth(as_of, cfg, cal).ffill(limit=limit)
    pc = load_put_call(as_of, cfg, cal).ffill(limit=limit)

    return {"index": index, "volatility": vol, "conditions": cond,
            "breadth": breadth, "put_call": pc}


# -------------------------------------------------------------------- checkpoint
def report(blocks: dict[str, pd.DataFrame]) -> None:
    """Build-step-1 checkpoint: head/tail, gaps, and fill detection."""
    idx = blocks["index"]
    bar = "=" * 88
    print(bar)
    print(f"INDEX PANEL  {idx.index[0].date()} -> {idx.index[-1].date()}   "
          f"n={len(idx)}")
    print(bar)
    print("\nhead:\n" + idx.head(3).round(2).to_string())
    print("\ntail:\n" + idx.tail(3).round(2).to_string())

    print("\n-- calendar steps --")
    gaps = idx.index.to_series().diff().dt.days.dropna()
    for d, n in gaps.value_counts().head(5).items():
        kind = {1: "normal", 3: "weekend"}.get(int(d), "holiday / long weekend")
        print(f"   {int(d):3d}-day step: {n:5d}   {kind}")

    print("\n-- per-block integrity --")
    for name, df in blocks.items():
        nan_pct = df.isna().mean().mean() * 100
        flat = float(np.mean([(df[c].diff() == 0).mean() for c in df.columns
                              if df[c].notna().any()])) * 100
        print(f"   {name:14s} cols={df.shape[1]:2d}  NaN={nan_pct:6.2f}%  "
              f"flat days={flat:5.1f}%")

    print("\n-- index zero-return check (must be near 0%; higher means fill) --")
    z = (idx["close"].pct_change() == 0).mean() * 100
    print(f"   close  {z:5.2f}%{'   <-- CHECK' if z > 1.0 else ''}")

    print("\n-- coverage of every series --")
    for name, df in blocks.items():
        for c in df.columns:
            s = df[c].dropna()
            span = (f"{s.index[0].date()} -> {s.index[-1].date()}"
                    if len(s) else "EMPTY")
            flag = ""
            if len(s) and s.index[0] > pd.Timestamp("2005-01-01"):
                flag = "   <-- starts after the scoring window opens"
            print(f"   {name + '.' + c:28s} n={len(s):5d}  {span}{flag}")


if __name__ == "__main__":
    report(build_panel())
