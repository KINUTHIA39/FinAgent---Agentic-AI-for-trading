"""FNSPID: stream, filter to our tickers, and align to trading days.

THE BIGGEST LOOK-AHEAD RISK IN THE STUDY. Getting the timestamp-to-trading-day
mapping wrong by one day produces a spectacular and entirely fake result, so
the rules are spelled out here rather than buried in a lambda.

What the probe found in the actual file (23.23 GB, 12 columns):

  Date            "2023-12-16 23:00:00 UTC"  -- UTC, stated explicitly.
                  16:00 ET is 20:00 UTC in summer and 21:00 in winter, so a
                  fixed offset is wrong twice a year. Converted with a real
                  timezone, never by subtracting hours.

  00:00:00 UTC    8.7% of rows. These are date-only records whose time was
                  unknown and defaulted to midnight. Treating midnight as a
                  real timestamp would place them at 19:00 the PREVIOUS
                  evening in ET, and they would be credited to a day before
                  the one they are stamped with. They get their own rule.

  sort order      by ticker, then date descending. Our eight names sit in
                  contiguous blocks scattered through the file, so there is no
                  early exit -- the whole 23 GB is streamed once.

  columns         no bundled sentiment column exists in this file, so the
                  plan's warning about ChatGPT labels is moot here. The price
                  archive is a separate file and is not touched.

Only the headline is kept. Full bodies are present but out of scope: they blow
past FinBERT's 512 tokens and are inconsistently populated across sources.
"""
from __future__ import annotations

import csv
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

USECOLS = ["Date", "Article_title", "Stock_symbol", "Publisher"]
ET = "America/New_York"


# ---------------------------------------------------------------- alignment
def assign_trading_day(dates: pd.Series, calendar: pd.DatetimeIndex,
                       close_hhmm: str = "16:00") -> pd.Series:
    """Map article timestamps to the trading day that could first trade on them.

    Three cases:
      published before the close on a trading day  -> that day
      published at or after the close              -> the NEXT trading day
      time unknown (midnight UTC)                  -> the NEXT trading day

    The unknown-time case is deliberately conservative. The article appeared
    somewhere in that calendar day and we cannot tell whether it beat the close,
    so it is credited to the next session. Assuming the favourable case would
    hand the news arm up to a day of free look-ahead on 8.7% of the corpus.
    """
    ts = pd.to_datetime(dates, utc=True, errors="coerce")
    unknown = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)

    local = ts.dt.tz_convert(ET)
    hh, mm = (int(x) for x in close_hhmm.split(":"))
    after_close = (local.dt.hour > hh) | ((local.dt.hour == hh) & (local.dt.minute >= mm))

    # For a known time use the ET calendar date; for an unknown one the stated
    # UTC date is the only meaningful thing about it.
    base = np.where(unknown, ts.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None),
                    local.dt.normalize().dt.tz_localize(None))
    base = pd.Series(pd.to_datetime(base), index=dates.index)
    push = (after_close | unknown).fillna(False)

    cal = pd.DatetimeIndex(calendar).sort_values()
    # searchsorted "right" lands on the first session strictly after the date
    # when pushing, and "left" on the first session on or after it otherwise.
    side = np.where(push.to_numpy(), "right", "left")
    pos = np.empty(len(base), dtype=int)
    r = cal.searchsorted(base.to_numpy(), side="right")
    l = cal.searchsorted(base.to_numpy(), side="left")
    pos = np.where(side == "right", r, l)
    pos = np.clip(pos, 0, len(cal) - 1)

    out = pd.Series(cal[pos], index=dates.index)
    out[base.isna() | (pos >= len(cal))] = pd.NaT
    return out.rename("trading_day")


# ------------------------------------------------------------------ streaming
# A record begins at a newline followed by the float row index and then the
# timestamp. Article bodies contain newlines of their own, so this anchor is
# what lets us land on a true record boundary from an arbitrary byte offset.
_REC = re.compile(rb"\n\d*\.?\d*,\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC,")
_FILE_BYTES = 23_232_979_597

# THE FILE IS THREE TABLES CONCATENATED, not one. Probing byte offsets end to
# end found:
#
#   [0, 17.50 GB)        rows with a numeric row index, sorted by ticker A->Z
#                        and by date DESCENDING within each ticker. Ends at
#                        ZYME. This is the block the first run read.
#   [17.50, 17.84 GB)    rows with an EMPTY row-index field, sorted by ticker
#                        A->Z again. Same schema otherwise. This is where the
#                        OLDER per-ticker news lives -- AAPL's part-1 block
#                        stops at 2022-06, and its part-2 block carries the
#                        history before that.
#   [17.84, 23.23 GB)    rows with an empty index AND an empty symbol: general
#                        market news, date-sorted from 2000 on. Not attributable
#                        to any ticker, so out of scope here.
#
# Missing the second table is what capped the first fetch at 22.3% day coverage
# with nothing before 2019. The empty index is also why the original anchor
# regex (which required \d+ before the comma) found no record boundaries in the
# final 5.8 GB and reported the region as unparseable.
#
# The tables are told apart by the ROW-INDEX COLUMN, not by byte offset. Offsets
# are only a coarse hint: the boundary is not exactly 17.50 GB, and a part-2
# scan that begins a few MB early lands inside part 1's trailing ZYME block.
# ZYME sorts above every ticker we want, so the block-ended test fires at once
# and the scan returns nothing. Keying on the index column makes each scan
# ignore rows belonging to the other table, so the ranges may overlap freely.
#   (lo, hi, indexed)
_TABLES = ((0, 17_600_000_000, True),
           (17_400_000_000, 17_900_000_000, False))

# A row is only trustworthy if its date field still looks like a date and its
# symbol field still looks like a symbol. Anything else means the CSV parse
# desynchronised inside an article body and the columns have shifted.
_DATE_OK = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
_SYM_OK = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

# Consecutive trustworthy rows sorting above the target before we accept that
# the block has ended. One row is not evidence: the first run of this fetch
# stopped AAPL at 2022-06 -- 18 months of a 24-year block -- because a single
# malformed record put article text in the symbol column, and "The..." > "AAPL".
_PAST_CONFIRM = 25

# Some bodies exceed the 128 KB default. Set at import so a direct fetch_ticker
# call behaves the same as one made through stream_filter.
csv.field_size_limit(1 << 24)


def _get(url: str, lo: int, hi: int) -> bytes:
    import requests

    r = requests.get(url, headers={"Range": f"bytes={lo}-{hi}"},
                     timeout=(30, 300))
    r.raise_for_status()
    return r.content


# Sentinels for "this offset is outside the table we are searching". They sort
# below and above every real ticker, so the binary search can move on them
# exactly as it would on a symbol.
_BELOW, _ABOVE = "", "￿"


def _symbol_at(url: str, off: int, span: int = 400_000,
               indexed: bool | None = None) -> str | None:
    """Ticker of the first trustworthy record at or after `off`.

    `indexed` selects which table to read: True wants rows with a numeric row
    index, False wants rows with an empty one.

    Returning a bare None when the window holds no row of the wanted kind is not
    enough, because None conflates two opposite situations -- the probe is
    BEFORE our table, or PAST it -- and the search has to move in opposite
    directions for each. Conflating them cost real time: searching part 2 for
    XOM, every probe in the untagged region came back None, the search fell back
    to stepping right 4 MB at a time, and one binary search turned into ~60
    sequential requests. So report which side we are on instead.
    """
    buf = _get(url, off, min(off + span, _FILE_BYTES - 1))
    recs, _ = _records(buf)
    other = None
    for row in recs:
        if len(row) < 4 or not _DATE_OK.match(row[1] or ""):
            continue
        is_indexed = bool(row[0].strip())
        if indexed is not None and is_indexed != indexed:
            # The indexed table comes first, the untagged rows last.
            other = _BELOW if is_indexed else _ABOVE
            continue
        if _SYM_OK.match(row[3] or ""):
            return row[3]
        if not (row[3] or "").strip():
            other = _ABOVE          # untagged row: past the tagged block
    return other


def _find_block_start(url: str, ticker: str, lo: int = 0,
                      hi: int | None = None, precision: int = 4_000_000,
                      indexed: bool | None = None) -> int:
    """Binary-search for the first byte whose record has symbol >= ticker.

    Valid because the file is sorted by ticker -- verified by sampling 17 evenly
    spaced offsets and confirming the symbols increase monotonically
    (A, AMZN, BRK, COCP, ... VB, ZIONL).

    This is the whole reason the download is minutes rather than hours. Reading
    the file end to end meant parsing 15.7M records, each carrying a 2-5 KB
    article body that is then discarded; ~20 ranged requests find the block
    instead.
    """
    hi = hi if hi is not None else _FILE_BYTES - 1
    floor = lo                              # never back off out of this table
    while hi - lo > precision:
        mid = (lo + hi) // 2
        s = _symbol_at(url, mid, indexed=indexed)
        if s is None:                       # genuinely unparseable: step right
            lo = mid + precision
            continue
        # s may be a real ticker or one of the _BELOW / _ABOVE sentinels; all
        # three compare correctly against the target.
        if s < ticker:
            lo = mid
        else:
            hi = mid
    return max(lo - precision, floor)       # back off so we cannot overshoot


def _records(buf: bytes) -> tuple[list[list[str]], int]:
    """Split a byte window at true record starts and parse each one alone.

    Parsing the window as one CSV document was the mistake. Article bodies run
    to thousands of characters and some contain unbalanced quotes; one of those
    puts csv.reader out of step and every row after it has shifted columns, so
    row[3] stops being a ticker and starts being prose. Anchoring on _REC and
    handing csv.reader one record at a time confines that damage to the single
    bad record.

    Returns the parsed rows and the offset of the last boundary found -- the
    trailing record is truncated by the window edge, so the next request
    restarts there rather than losing it.
    """
    cuts = [m.start() for m in _REC.finditer(buf)]
    if len(cuts) < 2:
        return [], 0
    out = []
    for a, b in zip(cuts, cuts[1:]):
        try:
            out.append(next(csv.reader(io.StringIO(
                buf[a + 1:b].decode("utf-8", "replace")))))
        except Exception:                                  # noqa: BLE001
            continue                                       # this record only
    return out, cuts[-1]


def _trustworthy(row: list[str]) -> bool:
    """Do the date and symbol columns still hold a date and a symbol?"""
    return (len(row) > 3
            and bool(_DATE_OK.match(row[1] or ""))
            and bool(_SYM_OK.match(row[3] or "")))


def _scan_block(url: str, ticker: str, lo: int, hi: int, indexed: bool,
                chunk: int = 32 << 20) -> list[dict]:
    """One ticker's records from one of the sorted tables in [lo, hi)."""
    start = _find_block_start(url, ticker, lo, hi, indexed=indexed)
    rows, pos, seen_it, past_run = [], start, False, 0
    t0, read0 = time.time(), start
    # ~1.5 MB/s from this host, so a block is minutes. Cap the search so a
    # ticker whose block never terminates cannot run forever.
    max_bytes = 3 << 30
    while pos < hi and (pos - start) < max_bytes:
        if (pos - read0) > (256 << 20):
            read0 = pos
            print(f"      {ticker}: {(pos-start)/1e6:6.0f} MB scanned, "
                  f"{len(rows):6,d} kept, {(time.time()-t0)/60:4.1f} min",
                  flush=True)
        buf = _get(url, pos, min(pos + chunk, hi - 1))
        if not buf:
            break
        recs, last = _records(buf)
        if not recs:
            pos += max(len(buf) - 4096, 1)   # no boundary here; step over it
            continue
        pos += last

        stop = False
        for row in recs:
            if not _trustworthy(row):
                continue          # a mangled record is never break evidence
            if bool(row[0].strip()) != indexed:
                continue          # belongs to the other table; not evidence
            sym = row[3]
            if sym == ticker:
                seen_it, past_run = True, 0
                rows.append({c: row[i] for c, i in
                             (("Date", 1), ("Article_title", 2),
                              ("Stock_symbol", 3), ("Publisher", 5))
                             if i < len(row)})
            elif sym > ticker:
                # Deliberately NOT conditioned on having seen the ticker. The
                # table is sorted, so symbols above the target prove we are
                # past its slot whether or not it was ever there -- and a
                # ticker absent from the file is the common case worth exiting
                # fast. Requiring `seen_it` here made an absent ticker scan the
                # full 3 GB cap: GOOGL ran 1.2 GB and kept nothing.
                past_run += 1
                if past_run >= _PAST_CONFIRM:
                    stop = True
                    break
            else:
                past_run = 0
        if stop:
            break
    print(f"      {ticker}: {'block' if seen_it else 'ABSENT'} at "
          f"{start/1e9:.2f} GB -> {(pos-start)/1e6:.0f} MB, "
          f"{len(rows):,} records, {(time.time()-t0)/60:.1f} min", flush=True)
    return rows


def fetch_ticker(url: str, ticker: str, chunk: int = 32 << 20) -> pd.DataFrame:
    """All of one ticker's records, from BOTH ticker-sorted tables.

    Reading only the first table is what truncated the first run: it holds
    2022-2023 for the mega-caps and nothing earlier.
    """
    rows = []
    for lo, hi, indexed in _TABLES:
        rows += _scan_block(url, ticker, lo, hi, indexed, chunk)
    return pd.DataFrame(rows)


def stream_filter(cfg: dict, limit_rows: int | None = None,
                  max_retries: int = 4) -> Path:
    """Fetch each ticker's block by binary search, rather than reading 23 GB.

    An end-to-end stream was tried first and abandoned: pandas.read_csv over the
    URL yielded zero chunks before the connection dropped at 4.35 GB, and a
    hand-rolled csv.reader managed under a million rows in ten minutes because
    every record carries an article body it then throws away. Seeking is roughly
    a tenth of the bytes and a fraction of the parsing.
    """
    n = cfg["news"]
    url = n["url"]
    out_dir = config.ROOT / "data" / "fnspid"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "articles_filtered.parquet"
    csv.field_size_limit(1 << 24)

    # Per-ticker cache. Each block costs minutes of ranged requests, so a rerun
    # after a code fix should not refetch what already landed.
    cache = out_dir / "by_ticker"
    cache.mkdir(exist_ok=True)

    # Index-ETF blocks get the extra noise filter; single names do not need it.
    index_tickers = set(cfg["universe"].get("index_news_tickers", []))
    tickers = sorted(set(cfg["universe"]["news_tickers"]) | index_tickers)

    frames, missing, t0 = [], [], time.time()
    for tk in tickers:
        cp = cache / f"{tk}.parquet"
        if cp.exists():
            df = pd.read_parquet(cp)
            print(f"   {tk:6s} {len(df):7,d} articles   (cached)", flush=True)
            if len(df):
                frames.append(df)
            else:
                missing.append(tk)
            continue
        for attempt in range(max_retries):
            try:
                df = fetch_ticker(url, tk)
                if tk in index_tickers and len(df):
                    raw = len(df)
                    df = drop_index_noise(df)
                    print(f"        kept {len(df):,} of {raw:,} after the "
                          f"index-ETF filter", flush=True)
                print(f"   {tk:6s} {len(df):7,d} articles   "
                      f"{(time.time()-t0)/60:5.1f} min", flush=True)
                df.to_parquet(cp)
                if len(df):
                    frames.append(df)
                else:
                    # Not an error. Some names simply are not in the tagged
                    # tables; the study skips them and says so.
                    missing.append(tk)
                break
            except Exception as exc:                       # noqa: BLE001
                print(f"   {tk:6s} attempt {attempt+1} failed "
                      f"({type(exc).__name__}: {str(exc)[:60]})", flush=True)
                time.sleep(3 * (attempt + 1))
        if frames:
            pd.concat(frames).to_parquet(dest)

    if missing:
        print(f"\n   no news found for: {', '.join(missing)}  -- skipped")
    if not frames:
        raise RuntimeError("no matching articles found")
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out.to_parquet(dest)
    print(f"total {len(out):,} articles in {(time.time()-t0)/60:.1f} min")
    print(f"written: {dest}")
    return dest


def load_filtered(cfg: dict) -> pd.DataFrame:
    p = config.ROOT / "data" / "fnspid" / "articles_filtered.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run `python -m src.news` to build it")
    return pd.read_parquet(p)


# ------------------------------------------------------- index-ETF filtering
# FNSPID carries no index symbols at all -- SPX, GSPC, NDX and IXIC are absent
# from both tagged tables (checked, and the neighbours around where each would
# sort confirm it is not a naming-variant miss). Index-level news reaches us
# only through ETF tickers, and those blocks carry two kinds of article that
# must not enter a sentiment series.
#
# FUND PLUMBING is noise: fund launches, expense ratios, flows, "best ETF"
# listicles. It says nothing about the market.
#
# PRICE RESTATEMENTS are worse than noise, because they look like signal. A
# headline that reports what the index just did carries the same information
# as the day's return, which the detector already has from price. Feeding them
# to the news specialist turns the sentiment channel into a second, noisier
# copy of the return channel -- and if Arm C then beat Arm B, the honest
# reading would be "the panel was told the day's move twice", not "news carries
# information". Since the study's headline finding is that the news layer adds
# nothing, admitting a price echo could reverse that result artificially.
#
# Only the AUTO-GENERATED templated series are matched. Human-written headlines
# that mention a move alongside a cause ("FOMC Minutes Send US Equities Lower")
# are kept: the cause is genuine news, and hand-adjudicating every such headline
# would be a judgement call this filter has no business making.
_FUND_PLUMBING = [
    r"\bBe on Your Investing Radar\b",
    r"\ba Strong ETF Right Now\b",
    r"\bETFs? to Watch\b",
    r"\bMost Active for\b",
    r"\bETF (?:In|Out)flow Alert\b",
    r"\bMost-Heavily Traded ETFs\b",
    r"\bBest ETFs\b",
    r"\bETFs to Buy\b",
    r"\bon ETF Edge\b",
    r"\bImplied .{0,24}Analyst Target Price\b",
    r"^This Week in ETFs\b",
    r"\bFunds? Debut\b",
]
_PRICE_RESTATEMENT = [
    r"\bPrice Check For Major Averages\b",
    r"^Market Update:",
    r"\bEquity Markets are (?:Green|Red|Mixed|Flat)\b",
    r"\bFutures [+-]\s*\d",
    r"^US indices are (?:lower|higher)\b",
    r"^US stocks close (?:lower|higher)\b",
]
_NOISE_RE = {
    "fund_plumbing": re.compile("|".join(_FUND_PLUMBING), re.I),
    "price_restatement": re.compile("|".join(_PRICE_RESTATEMENT), re.I),
}


def classify_index_noise(titles: pd.Series) -> pd.Series:
    """Label each headline 'fund_plumbing', 'price_restatement' or 'keep'."""
    out = pd.Series("keep", index=titles.index, dtype="object")
    t = titles.astype(str)
    for name, rx in _NOISE_RE.items():
        hit = t.str.contains(rx, na=False) & (out == "keep")
        out[hit] = name
    return out


def drop_index_noise(df: pd.DataFrame, report: bool = True) -> pd.DataFrame:
    """Strip fund plumbing and price restatements from an index-ETF block."""
    if not len(df):
        return df
    cls = classify_index_noise(df["Article_title"])
    if report:
        for name in _NOISE_RE:
            n = int((cls == name).sum())
            print(f"        dropped {n:6,d}  {name}", flush=True)
    return df[cls == "keep"].copy()


# ------------------------------------------------- market-wide (untagged) news
# The final 5.39 GB carries general news with NO ticker column. It was skipped
# at first because articles cannot be attributed to a ticker -- but for an
# INDEX-level sentiment track that is exactly the constraint that does not
# apply. Market-wide news with no ticker is still market-wide news, and this is
# the only part of FNSPID that reaches 2005-2010, where the ticker-tagged
# corpus has nothing at all.
#
# The region is NOT ticker-sorted, so binary search cannot help: it is a linear
# stream, roughly an hour at the ~1.5 MB/s this host sustains.
#
# Filtering went through three iterations, measured each time on 90 sampled
# windows by reading the surviving headlines by hand:
#     v1  match any market keyword          ~1,099,000 articles, ~27% precision
#     v2  require a US-equity anchor          ~125,700 articles, ~70% precision
#     v3  five leak fixes                      ~92,300 articles, ~78% precision
# Tuning stopped at v3. The residual errors are semantic rather than lexical --
# "Wall Street" means the market, the industry and analyst consensus, and no
# regex separates those three. Crucially the residual noise is near-neutral in
# tone (personnel notices, calendar listings), so it dilutes the daily mean
# rather than biasing it. That is a far better failure mode than v1's, where
# foreign market moves would have injected real directional signal from the
# wrong market.
_MKT_LO, _MKT_HI = 17_840_000_000, _FILE_BYTES

_PUBLISHERS = {"reuters.com", "bloomberg.com", "benzinga.com",
               "seekingalpha.com", "zacks.com", "investors.com",
               "news.investors.com", "gurufocus.com", "marketwatch.com",
               "cnbc.com", "fool.com", "barrons.com"}

# Reuters desk slugs. Anchored to ^ because they are PREFIXES -- leaving this
# unanchored let "Wall St" match mid-headline and granted strong-anchor status
# to "European shares hit by Wall St decline", overriding the foreign-venue
# exclusion entirely. That was the single largest source of bad rows in v2.
_MKT_STRONG = re.compile(
    r"^(?:RPT-|REFILE-|UPDATE \d+-|WRAPUP \d+-)?"
    r"(US STOCKS|U\.S\. STOCKS|MONEY MARKETS|TREASURIES|WALL ST WEEK)[-\s]",
    re.I)

_MKT_ANCHOR = re.compile(
    r"\b(?:"
    r"US stocks|U\.S\. stocks|Wall St(?:reet)?|"
    r"S&P ?500|SPX|"
    r"Dow Jones|Dow industrials|Dow 30|the Dow\b|"
    r"Nasdaq (?:composite|100|index|falls|rises|gains|drops|jumps|slides)|"
    r"US equit|U\.S\. equit|American equit|"
    r"FOMC|Federal Reserve|Fed(?:'s)? (?:rate|policy|minutes|chair|cut|hike|meeting)|"
    r"VIX|"
    r"US (?:market|indices|shares)|U\.S\. (?:market|indices|shares)"
    r")", re.I)

# Adjectival forms matter: \bJapan\b never matches "Japanese", so
# "Japanese stocks tumble on Fed rate cut" survived v2's foreign filter.
_MKT_FOREIGN = re.compile(
    r"\b(Toronto|Canad\w+|Mexic\w+|Brazil\w*|Bovespa|"
    r"Hong Kong|Hang Seng|Shanghai|China\w*|Chinese|Japan\w*|Nikkei|Tokyo|"
    r"Europe\w*|Euro ?zone|Bunds|EURO GOVT|"
    r"UK stocks|Britain|British|London|FTSE|DAX|CAC|"
    r"India\w*|Sensex|Australia\w*|ASX|Korea\w*|Kospi|Singapore|Russia\w*|"
    r"Ukrain\w*|emerging (?:debt|market)|Latin America|Africa\w*|Israel\w*|"
    r"Turkey|Turkish|Greek|Greece|Spain|Spanish|Italy|Italian|German\w*|"
    r"French|France)\b", re.I)

_MKT_BOILER = re.compile(
    r"\b("
    r"announce\w*|presents?|present at|to Present|webcast|conference call|"
    r"declares?|appoints?|names? .{0,20}(?:CEO|CFO|President|Director)|"
    r"files? (?:to|for)|completes?|prices? offering|awarded|unveil\w*|"
    r"introduc\w*|launch\w*|receives?|wins?|acquires?|to acquire|"
    r"survey says|study (?:finds|shows)|"
    r"Fitch|Moody's|"
    r"S&P;? (?:Says|Ratings|assigns|affirms|downgrades|eyes|cuts|raises|revises)|"
    r"affirms|assigns|rating action|"
    r"REG-|REG |TEXT-|TEXT:|BRIEF-|DIARY-|PRECIOUS-|Research and Markets|"
    r"Form 8\.3|Net Asset Value|Holding\(s\) in Company|Director/PDMR|"
    r"annual meeting|shareholder\w* (?:meeting|approve)|"
    r"quarterly (?:results|report|dividend)|1st qtr|2nd qtr|3rd qtr|4th qtr|"
    r"net (?:income|profit|loss) (?:up|down|rises|falls)|"
    r"Wall St(?:reet)? Journal|PRESS DIGEST|HEADLINE STOCKS|TraderNotes|"
    r"BeaconEquity|Analyst Blog|Issues? Trader|Component Changes|"
    r"beats? Wall St(?:reet)?|miss(?:es)? Wall St(?:reet)?|"
    r"exceeds? Wall St(?:reet)?|"
    r"Wall St(?:reet)? (?:Expects|estimates|consensus|forecasts?|view|"
    r"analysts? expect|firms?|bonus)|"
    # The industry sense, where the words need not be adjacent:
    # "Wall Street will only add 3,800 jobs by 2013".
    r"Wall St(?:reet)?\b.{0,40}\b(jobs|hiring|headcount|layoffs?|bonuses|"
    r"subpoena\w*|prosecutor\w*|regulat\w*|lobby\w*)|"
    r"\([A-Z]{1,5}\) (?:Reports|Stock|Shares|Gains|Q\d)"
    r")\b", re.I)

_MKT_PRODUCT = re.compile(
    r"\b(?:Global|Worldwide|Regional|Domestic)\s+[\w\s,-]{0,45}?\bMarket\b"
    r"|\bMarket (?:to Be Worth|Size|Research|Report|Forecast|Outlook|Share)\b"
    r"|\bWorth US\$|\bBillion by 20\d\d"
    r"|\b(?:carbon|grain|labou?r|housing|job|energy|oil|commodity) market\b"
    r"|\bTargets? U\.?S\.? Market", re.I)


def is_market_news(title: str, host: str) -> bool:
    """Does this untagged headline belong in the market-wide track?"""
    if host not in _PUBLISHERS:
        return False
    strong = bool(_MKT_STRONG.search(title))
    if not strong and not _MKT_ANCHOR.search(title):
        return False
    if _MKT_PRODUCT.search(title) or _MKT_BOILER.search(title):
        return False
    # A desk slug overrides the foreign filter, so "US STOCKS-Wall St falls on
    # Europe fears" survives while "Stocks in Europe Drop" does not.
    if not strong and _MKT_FOREIGN.search(title):
        return False
    return True


def fetch_market_news(url: str, chunk: int = 32 << 20,
                      checkpoint_mb: int = 512) -> pd.DataFrame:
    """Linear stream of the untagged region, keeping market-wide headlines.

    Checkpointed: an hour-long stream that dies at 90% must not lose
    everything, so partial results are written every `checkpoint_mb` and the
    scan resumes from the recorded offset.
    """
    from urllib.parse import urlparse

    out_dir = config.ROOT / "data" / "fnspid"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "_MARKET.partial.parquet"
    state = out_dir / "_MARKET.offset"

    rows, pos = [], _MKT_LO
    if ckpt.exists() and state.exists():
        rows = pd.read_parquet(ckpt).to_dict("records")
        pos = int(state.read_text().strip())
        print(f"   resuming at {pos/1e9:.2f} GB with {len(rows):,} rows",
              flush=True)

    t0, seen, last_ck, fails = time.time(), 0, pos, 0
    while pos < _MKT_HI:
        try:
            buf = _get(url, pos, min(pos + chunk, _MKT_HI - 1))
            fails = 0
        except Exception as exc:                              # noqa: BLE001
            fails += 1
            # _FILE_BYTES is a recorded constant, and the server disagreeing
            # with it by a few bytes makes the final ranged request a 416.
            # Retrying that forever is the failure mode this guard exists for:
            # the loop never advances `pos`, so a PERSISTENT error at the
            # boundary spins indefinitely rather than terminating. Near the end
            # of the file, treat repeated failure as end-of-file; elsewhere,
            # give up after enough tries rather than hanging.
            near_end = (_MKT_HI - pos) < (64 << 20)
            if fails >= (2 if near_end else 6):
                print(f"   stopping at {pos/1e9:.2f} GB after {fails} failed "
                      f"reads ({type(exc).__name__})"
                      + ("  -- treated as end of file" if near_end else ""),
                      flush=True)
                break
            print(f"   read failed at {pos/1e9:.2f} GB "
                  f"({type(exc).__name__}); retry {fails}", flush=True)
            time.sleep(4)
            continue
        if not buf:
            break
        recs, last = _records(buf)
        if not recs:
            pos += max(len(buf) - 4096, 1)
            continue
        for r in recs:
            if len(r) < 3 or not _DATE_OK.match(r[1] or ""):
                continue
            seen += 1
            title = str(r[2])
            host = ""
            if len(r) > 4 and str(r[4]).startswith("http"):
                host = urlparse(str(r[4])).netloc.replace("www.", "")
            if is_market_news(title, host):
                rows.append({"Date": r[1], "Article_title": title,
                             "Stock_symbol": "_MARKET", "Publisher": host})
        pos += last

        if pos - last_ck >= (checkpoint_mb << 20):
            last_ck = pos
            pd.DataFrame(rows).to_parquet(ckpt)
            state.write_text(str(pos))
            done = (pos - _MKT_LO) / (_MKT_HI - _MKT_LO)
            el = (time.time() - t0) / 60
            print(f"   {done*100:5.1f}%  {(pos-_MKT_LO)/1e9:.2f} GB  "
                  f"{seen:,} scanned  {len(rows):,} kept  "
                  f"{el:.0f} min elapsed  ~{el/max(done,1e-9)-el:.0f} min left",
                  flush=True)

    df = pd.DataFrame(rows).drop_duplicates(subset=["Date", "Article_title"])
    dest = out_dir / "by_ticker" / "_MARKET.parquet"
    dest.parent.mkdir(exist_ok=True)
    df.to_parquet(dest)
    for p in (ckpt, state):
        if p.exists():
            p.unlink()
    print(f"   scanned {seen:,} untagged rows, kept {len(df):,} "
          f"in {(time.time()-t0)/60:.0f} min", flush=True)
    print(f"   written: {dest}", flush=True)
    return df


def coverage_mask(cfg: dict, calendar: pd.DatetimeIndex) -> pd.Series:
    """Trading days on which the news agent actually has something to read.

    The corpus is heavily skewed to 2020-2023, while most of the days the panel
    convenes are earlier. Scored over the whole window, Arm C minus Arm B is
    therefore mostly a test of the LLM panel, not of sentiment -- the news agent
    is dark on the majority of gated days. This mask marks the subset where the
    sentiment layer is genuinely switched on, so the contest can report that
    comparison separately from the headline.

    Deliberately day-level rather than a date range: a threshold picked to carve
    out a tidy era would be an arbitrary choice defended after the fact, whereas
    "did this day have any news at all" is a fact about the corpus.
    """
    cal = pd.DatetimeIndex(calendar)
    days: set = set()

    df = load_filtered(cfg)
    days |= set(assign_trading_day(df["Date"], cal,
                                   cfg["news"]["market_close_et"]).dropna())

    # The market-wide track counts too: it is real news the specialist reads,
    # and it reaches 2007-2010 where the single-stock corpus has nothing. A
    # mask built from the ticker corpus alone would mark the whole GFC as
    # "no news" when the panel could in fact see wire copy there.
    root = config.ROOT / "data" / "fnspid" / "by_ticker"
    for src in ("_MARKET", "SPY", "QQQ"):
        p = root / f"{src}.parquet"
        if p.exists():
            m = pd.read_parquet(p, columns=["Date"])
            days |= set(assign_trading_day(m["Date"], cal,
                                           cfg["news"]["market_close_et"]
                                           ).dropna())

    return pd.Series(cal.isin(days), index=cal, name="news_live")


def align(df: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Attach the trading day and drop anything unusable."""
    out = df.copy()
    out["trading_day"] = assign_trading_day(
        out["Date"], calendar, cfg["news"]["market_close_et"])
    out = out.dropna(subset=["trading_day", "Article_title"])
    out = out[out["Article_title"].astype(str).str.len() > 10]
    return out.sort_values("trading_day").reset_index(drop=True)


# ------------------------------------------------------------------ checkpoint
def report(df: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict) -> dict:
    """Build-step-6 checkpoint: is the trading-day assignment right?"""
    res = {}
    aligned = align(df, calendar, cfg)
    ts = pd.to_datetime(aligned["Date"], utc=True, errors="coerce")
    local = ts.dt.tz_convert(ET)

    print("=" * 92)
    print("BUILD STEP 6 CHECKPOINT - FNSPID news")
    print("=" * 92)
    print(f"\narticles kept: {len(aligned):,}")
    print(f"span {aligned['trading_day'].min().date()} -> "
          f"{aligned['trading_day'].max().date()}")
    res["n_articles"] = len(aligned)

    print("\n-- per ticker --")
    for t, c in aligned["Stock_symbol"].value_counts().items():
        sub = aligned[aligned["Stock_symbol"] == t]
        print(f"   {t:6s} {c:7,d}   {sub['trading_day'].min().date()} -> "
              f"{sub['trading_day'].max().date()}")

    print("\n-- articles per year --")
    yr = aligned["trading_day"].dt.year.value_counts().sort_index()
    for y, c in yr.items():
        print(f"   {y}  {c:7,d}  {'#' * min(int(c / max(yr.max(), 1) * 50), 50)}")

    unknown = ((ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0))
    hh = int(cfg["news"]["market_close_et"].split(":")[0])
    after = (local.dt.hour >= hh) & ~unknown
    res["pct_unknown_time"] = float(unknown.mean())
    res["pct_after_close"] = float(after.mean())
    print("\n-- timestamp handling --")
    print(f"   unknown time (midnight UTC): {unknown.mean() * 100:5.1f}%  "
          f"-> pushed to the NEXT session")
    print(f"   at or after the 16:00 ET close: {after.mean() * 100:5.1f}%  "
          f"-> pushed to the NEXT session")
    print(f"   before the close: {(~after & ~unknown).mean() * 100:5.1f}%  "
          f"-> same session")

    print("\n-- SPOT CHECK: 12 articles across the three cases --")
    print(f"   {'published (UTC)':24s} {'-> ET':22s} {'case':14s} {'trading day'}")
    show = pd.concat([
        aligned[after.values].head(4),
        aligned[unknown.values].head(4),
        aligned[(~after & ~unknown).values].head(4)])
    for i, row in show.iterrows():
        t_utc = pd.to_datetime(row["Date"], utc=True)
        t_et = t_utc.tz_convert(ET)
        mid = t_utc.hour == 0 and t_utc.minute == 0 and t_utc.second == 0
        case = ("unknown time" if mid
                else "after close" if t_et.hour >= hh else "before close")
        print(f"   {str(t_utc)[:19]:24s} {str(t_et)[:19]:22s} {case:14s} "
              f"{row['trading_day'].date()}")

    bad = aligned[pd.to_datetime(aligned["Date"], utc=True).dt.tz_convert(ET)
                  .dt.normalize().dt.tz_localize(None) > aligned["trading_day"]]
    res["n_backwards"] = len(bad)
    print(f"\n   articles credited to a session BEFORE their ET date: {len(bad)}")
    print("   (must be zero -- anything else is a look-ahead)")

    counts = aligned.groupby("trading_day").size()
    cal_in = pd.DatetimeIndex(calendar)
    cov = counts.reindex(cal_in).fillna(0)
    print(f"\n-- coverage over the scored window --")
    scored = cov.loc[cfg["sample"]["score_start"]:cfg["sample"]["score_end"]]
    print(f"   trading days with at least one article: "
          f"{(scored > 0).mean() * 100:.1f}%")
    print(f"   median articles per day: {scored.median():.0f}   "
          f"mean: {scored.mean():.1f}   max: {scored.max():.0f}")
    res["day_coverage"] = float((scored > 0).mean())

    out = config.paths()["results"] / "news_aligned.parquet"
    aligned.to_parquet(out)
    print(f"\nwritten: {out}")
    return res


if __name__ == "__main__":
    import sys

    from . import ingest

    cfg = config.load()
    blocks = ingest.build_panel(cfg=cfg)
    cal = blocks["index"].index

    p = config.ROOT / "data" / "fnspid" / "articles_filtered.parquet"
    if "--market" in sys.argv:
        # The untagged region: a linear hour, so it is its own command rather
        # than something a routine --stream re-run pays for.
        fetch_market_news(cfg["news"]["url"])
        raise SystemExit
    if "--stream" in sys.argv or not p.exists():
        limit = 2_000_000 if "--sample" in sys.argv else None
        stream_filter(cfg, limit_rows=limit)
    report(load_filtered(cfg), cal, cfg)
