"""Alpaca (Benzinga) news for the index-news track: SPY and QQQ, 2015 onward.

Replaces the FNSPID index extract. Alpaca carries no symbol for the indices
themselves -- ^GSPC, GSPC, SPX, ^IXIC, IXIC, NDX, $SPX and I:SPX all return
nothing -- so index news arrives through the SPY and QQQ tags, as it did with
FNSPID. The archive opens on 2015-01-01; nothing earlier exists.

Bodies are requested, but most items have none. Measured on SPY+QQQ, 6-32% of
articles carry a body depending on the year; the rest are newsdesk snaps and
data prints ("NY Fed 1-Year Consumer Inflation Expectations 3.1% Vs 3.0%
Prior") where the headline is the whole item. The summary field was empty on
every bodyless article sampled, so there is no middle tier.

Fetched month by month. Each month is written to
data/alpaca/news/<SYMBOLS>/YYYY-MM.parquet as soon as it completes, and a
re-run skips months already on disk, so an interrupted pull resumes where it
stopped; only the month in flight is refetched.

Raw only: no trading-day assignment and no scoring. Timestamps stay in UTC and
the 16:00 ET rule is applied downstream, as for FNSPID.

Keys come from ALPACA_API_KEY / ALPACA_SECRET_KEY (or APCA_API_KEY_ID /
APCA_API_SECRET_KEY) in the environment, else from this repo's .env, else from
--env-file. Their values are never printed.

Usage:
    python -m src.alpaca_news
    python -m src.alpaca_news --env-file path/to/.env
    python -m src.alpaca_news --start 2015-01 --end 2025-12 --symbols SPY,QQQ
    python -m src.alpaca_news --combine-only
"""

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "data" / "alpaca"
BASE = "https://data.alpaca.markets/v1beta1/news"

PAGE = 50                  # the endpoint's hard maximum
MAX_PER_MIN = 180          # free plan allows 200; keep headroom
RETRIES = 6
BODY_MIN_WORDS = 20        # below this a "body" is a dateline or a stray tag

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

COLUMNS = ["id", "created_at", "updated_at", "headline", "summary", "author",
           "source", "url", "symbols", "n_symbols", "content_html",
           "content_text", "body_words", "has_body"]


# ------------------------------------------------------------------ credentials
def _read_env_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def credentials(env_file: str | None) -> tuple[str, str]:
    pools = [dict(os.environ), _read_env_file(REPO / ".env")]
    if env_file:
        pools.append(_read_env_file(Path(env_file)))
    for pool in pools:
        key = pool.get("ALPACA_API_KEY") or pool.get("APCA_API_KEY_ID")
        sec = pool.get("ALPACA_SECRET_KEY") or pool.get("APCA_API_SECRET_KEY")
        if key and sec:
            return key, sec
    sys.exit("no Alpaca keys found: set ALPACA_API_KEY / ALPACA_SECRET_KEY, "
             "add them to this repo's .env, or pass --env-file")


# ------------------------------------------------------------------ transport
class Client:
    """Rate-limited, retrying GET against the news endpoint."""

    def __init__(self, key: str, secret: str):
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                        "accept": "application/json"}
        self.gap = 60.0 / MAX_PER_MIN
        self.last = 0.0
        self.calls = 0

    def get(self, params: dict) -> dict:
        url = BASE + "?" + urllib.parse.urlencode(params)
        for attempt in range(RETRIES):
            wait = self.gap - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.calls += 1
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                # 429 and 5xx are transient; anything else is a real error.
                if e.code != 429 and e.code < 500:
                    raise
                why = f"HTTP {e.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                why = type(e).__name__
            back = 2 ** (attempt + 1)
            print(f"      {why}, retrying in {back}s ({attempt + 1}/{RETRIES})",
                  flush=True)
            time.sleep(back)
        raise RuntimeError(f"gave up after {RETRIES} attempts: {url}")


# ------------------------------------------------------------------ parsing
def _text(raw: str | None) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", raw or ""))).strip()


def to_frame(items: list[dict]) -> pd.DataFrame:
    rows = []
    for a in items:
        body = _text(a.get("content"))
        words = len(body.split())
        syms = a.get("symbols") or []
        rows.append({
            "id": int(a["id"]),
            "created_at": a.get("created_at"),
            "updated_at": a.get("updated_at"),
            "headline": (a.get("headline") or "").strip(),
            "summary": _text(a.get("summary")),
            "author": a.get("author") or "",
            "source": a.get("source") or "",
            "url": a.get("url") or "",
            "symbols": ",".join(syms),
            "n_symbols": len(syms),
            "content_html": a.get("content") or "",
            "content_text": body,
            "body_words": words,
            "has_body": words > BODY_MIN_WORDS,
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("created_at", "updated_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["id"] = df["id"].astype("int64")
    df["n_symbols"] = df["n_symbols"].astype("int64")
    df["body_words"] = df["body_words"].astype("int64")
    df["has_body"] = df["has_body"].astype(bool)
    return df


# ------------------------------------------------------------------ fetch
def fetch_month(client: Client, month: pd.Period, symbols: str) -> pd.DataFrame:
    start = month.start_time.strftime("%Y-%m-%dT00:00:00Z")
    end = month.end_time.strftime("%Y-%m-%dT23:59:59Z")
    items, token = [], None
    while True:
        q = {"start": start, "end": end, "symbols": symbols, "limit": PAGE,
             "sort": "asc", "include_content": "true"}
        if token:
            q["page_token"] = token
        d = client.get(q)
        items.extend(d.get("news", []))
        token = d.get("next_page_token")
        if not token:
            break
    df = to_frame(items)
    # An item tagged with both symbols must not appear twice.
    return df.drop_duplicates("id").sort_values("created_at").reset_index(drop=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def cache_dir(symbols: str) -> Path:
    return OUTDIR / "news" / symbols.replace(",", "_")


def run(client: Client, months: pd.PeriodIndex, symbols: str) -> None:
    d = cache_dir(symbols)
    d.mkdir(parents=True, exist_ok=True)
    todo = [m for m in months if not (d / f"{m}.parquet").exists()]
    print(f"{len(months)} months in range, {len(months) - len(todo)} already on "
          f"disk, {len(todo)} to fetch", flush=True)
    t0 = time.monotonic()
    for i, m in enumerate(todo, 1):
        ts = time.monotonic()
        df = fetch_month(client, m, symbols)
        _write(df, d / f"{m}.parquet")
        body = df["has_body"].mean() * 100 if len(df) else 0.0
        el = time.monotonic() - t0
        eta = el / i * (len(todo) - i)
        print(f"  {m}  {len(df):5d} articles  {body:4.0f}% with body  "
              f"{time.monotonic() - ts:5.1f}s   [{i}/{len(todo)}  "
              f"elapsed {el / 60:4.1f} min, ~{eta / 60:4.1f} min left]",
              flush=True)


def combine(months: pd.PeriodIndex, symbols: str) -> Path:
    d = cache_dir(symbols)
    missing = [str(m) for m in months if not (d / f"{m}.parquet").exists()]
    if missing:
        sys.exit(f"cannot combine, {len(missing)} month(s) not fetched: "
                 f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    frames = [pd.read_parquet(d / f"{m}.parquet") for m in months]
    df = (pd.concat([f for f in frames if len(f)], ignore_index=True)
          .drop_duplicates("id").sort_values("created_at").reset_index(drop=True))
    out = OUTDIR / f"news_{symbols.replace(',', '_').lower()}.parquet"
    _write(df, out)

    yr = df["created_at"].dt.year
    days = pd.Series({y: (pd.Timestamp(f"{y + 1}-01-01") - pd.Timestamp(f"{y}-01-01")).days
                      for y in sorted(yr.unique())})
    summ = pd.DataFrame({
        "articles": df.groupby(yr).size(),
        "per_day": df.groupby(yr).size() / days,
        "with_body_%": df.groupby(yr)["has_body"].mean() * 100,
        "median_body_words": df[df.has_body].groupby(yr[df.has_body])["body_words"].median(),
        "<=4_symbols_%": (df["n_symbols"] <= 4).groupby(yr).mean() * 100,
    }).round(1)
    print("\n" + summ.to_string())
    print(f"\n{len(df):,} articles  {df.created_at.min():%Y-%m-%d} -> "
          f"{df.created_at.max():%Y-%m-%d}   {out.stat().st_size / 1e6:.1f} MB")
    print("wrote", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2015-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2025-12", help="last month, YYYY-MM")
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--env-file", default=None,
                    help="a .env holding ALPACA_API_KEY / ALPACA_SECRET_KEY")
    ap.add_argument("--combine-only", action="store_true",
                    help="skip fetching; rebuild the combined file from disk")
    a = ap.parse_args()

    symbols = ",".join(s.strip().upper() for s in a.symbols.split(",") if s.strip())
    months = pd.period_range(a.start, a.end, freq="M")

    if not a.combine_only:
        client = Client(*credentials(a.env_file))
        run(client, months, symbols)
        print(f"\n{client.calls} requests made", flush=True)
    combine(months, symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
