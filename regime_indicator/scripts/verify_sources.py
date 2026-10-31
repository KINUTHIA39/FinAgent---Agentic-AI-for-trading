"""Fetch one row from every source in spec section 1 and print what is live.

Run this BEFORE writing loaders. Several series in the spec are flagged as
possibly discontinued; this decides the shape of ingest.py rather than
discovering it three hours in.

    python scripts/verify_sources.py
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

START = "2000-01-01"
STALE_DAYS = 45
rows: list[dict] = []


def record(group, name, ident, series, note=""):
    if series is None or len(series) == 0:
        rows.append(dict(group=group, name=name, id=ident, status="EMPTY",
                         first="-", last="-", n=0, last_val="-", note=note))
        return
    s = series.dropna()
    if len(s) == 0:
        rows.append(dict(group=group, name=name, id=ident, status="ALL-NaN",
                         first="-", last="-", n=0, last_val="-", note=note))
        return
    last_dt = pd.Timestamp(s.index[-1]).tz_localize(None)
    age = (pd.Timestamp.today().normalize() - last_dt).days
    status = "OK" if age <= STALE_DAYS else f"STALE({age}d)"
    rows.append(dict(group=group, name=name, id=ident, status=status,
                     first=str(pd.Timestamp(s.index[0]).date()),
                     last=str(last_dt.date()), n=len(s),
                     last_val=f"{float(s.iloc[-1]):.4f}", note=note))


def fail(group, name, ident, exc, note=""):
    rows.append(dict(group=group, name=name, id=ident, status="FAIL",
                     first="-", last="-", n=0, last_val="-",
                     note=(note + " " + f"{type(exc).__name__}: {exc}").strip()[:70]))


# ------------------------------------------------------------------ Yahoo
import yfinance as yf

YAHOO = [
    ("fx",          "EURUSD",       "EURUSD=X"),
    ("fx",          "GBPUSD",       "GBPUSD=X"),
    ("fx",          "USDJPY",       "USDJPY=X"),
    ("fx",          "USDCAD",       "USDCAD=X"),
    ("vol",         "VIX",          "^VIX"),
    ("vol",         "MOVE",         "^MOVE"),
    ("vol",         "EVZ (FX vol)", "^EVZ"),
    ("cross_asset", "S&P 500",      "^GSPC"),
    ("cross_asset", "Dollar index", "DX-Y.NYB"),
    ("cross_asset", "Oil",          "CL=F"),
    ("cross_asset", "Gold",         "GC=F"),
    ("cross_asset", "Copper",       "HG=F"),
]

print("Probing Yahoo Finance ...", flush=True)
for group, name, tkr in YAHOO:
    try:
        df = yf.download(tkr, start=START, interval="1d", progress=False,
                         auto_adjust=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        record(group, name, tkr, df["Close"] if "Close" in df else None)
    except Exception as e:
        fail(group, name, tkr, e)

# ------------------------------------------------------------------ FRED
from fredapi import Fred

key = os.getenv("FRED_API_KEY")
if not key:
    print("FRED_API_KEY not set — skipping FRED probes", file=sys.stderr)
else:
    fred = Fred(api_key=key)
    FREDS = [
        ("carry",       "US fed funds (eff)",   "DFF",              "policy, daily"),
        ("carry",       "ECB deposit facility", "ECBDFR",           "policy, daily"),
        ("carry",       "UK SONIA",             "IUDSOIA",          "policy proxy"),
        ("carry",       "BoE Bank Rate",        "BOERUKM",          "alt UK"),
        ("carry",       "JP policy rate",       "IRSTCB01JPM156N",  "monthly"),
        ("carry",       "CA policy rate",       "IRSTCB01CAM156N",  "monthly"),
        ("carry",       "CA overnight target",  "IRSTCI01CAM156N",  "alt CA"),
        ("carry_2y",    "US 2y",                "DGS2",             "upside tier"),
        ("carry_2y",    "CA 2y",                "IRLTCT01CAM156N",  "upside tier"),
        ("vol",         "VIX (FRED)",           "VIXCLS",           "cross-check"),
        ("cross_asset", "Broad dollar",         "DTWEXBGS",         ""),
        ("cross_asset", "HY OAS",               "BAMLH0A0HYM2",     ""),
        ("cross_asset", "IG OAS",               "BAMLC0A0CM",       ""),
        ("fx_xcheck",   "USD/EUR (noon)",       "DEXUSEU",          "noon ET - do not mix"),
        ("fx_xcheck",   "USD/UK (noon)",        "DEXUSUK",          "noon ET - do not mix"),
        ("fx_xcheck",   "JPY/USD (noon)",       "DEXJPUS",          "noon ET, inverted"),
        ("fx_xcheck",   "CAD/USD (noon)",       "DEXCAUS",          "noon ET, inverted"),
    ]
    print("Probing FRED ...", flush=True)
    for group, name, sid, note in FREDS:
        try:
            s = fred.get_series(sid, observation_start=START)
            record(group, name, sid, s, note)
        except Exception as e:
            fail(group, name, sid, e, note)

# ------------------------------------------------------------------ report
out = pd.DataFrame(rows)
order = {"fx": 0, "carry": 1, "carry_2y": 2, "vol": 3, "cross_asset": 4, "fx_xcheck": 5}
out = out.sort_values(["group", "name"], key=lambda c: c.map(order).fillna(9) if c.name == "group" else c)

pd.set_option("display.width", 200, "display.max_colwidth", 52)
print("\n" + "=" * 118)
print("SOURCE VERIFICATION —", datetime.now().strftime("%Y-%m-%d %H:%M"))
print("=" * 118)
for g, sub in out.groupby("group", sort=False):
    print(f"\n[{g}]")
    print(sub[["name", "id", "status", "first", "last", "n", "last_val", "note"]]
          .to_string(index=False))

bad = out[~out.status.str.startswith("OK")]
print("\n" + "-" * 118)
print(f"{len(out) - len(bad)}/{len(out)} sources live.")
if len(bad):
    print("NEEDS ATTENTION:")
    for _, r in bad.iterrows():
        print(f"  {r['status']:12s} {r['name']} ({r['id']}) {r['note']}")

os.makedirs("results", exist_ok=True)
out.to_csv("results/source_verification.csv", index=False)
print("\nwritten: results/source_verification.csv")
