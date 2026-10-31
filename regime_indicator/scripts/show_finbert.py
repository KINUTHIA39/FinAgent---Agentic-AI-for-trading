"""Print FinBERT headline scores for one trading day, from the cached run.

Reads results/articles_detail.parquet (per-headline scores, as produced by
src.sentiment) and results/sentiment_daily.parquet (the daily features the news
specialist actually sees), and renders a single day: every headline with its
FinBERT tone, ordered most negative to most positive, then the aggregates.

No model is loaded and no API is called; this is the stored output of the run.
To score a fresh headline with the live model instead, use:
    python -m src.sentiment --score "Stocks plunge as recession fears mount"

Usage:
    python scripts/show_finbert.py 2018-12-03
    python scripts/show_finbert.py 2018-12-03 --no-color
    python scripts/show_finbert.py 2018-12-03 --top 8
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252, which cannot print the transcript text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
DETAIL = REPO / "results" / "articles_detail.parquet"
DAILY = REPO / "results" / "sentiment_daily.parquet"
WIDTH = 96


class Ink:
    def __init__(self, on):
        self.on = on

    def _w(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def bold(self, s):  return self._w("1", s)
    def dim(self, s):   return self._w("2", s)
    def cyan(self, s):  return self._w("36", s)
    def red(self, s):   return self._w("31", s)
    def green(self, s): return self._w("32", s)
    def amber(self, s): return self._w("33", s)


def band(score):
    """The reading a score corresponds to, matching the worked example."""
    if score <= -0.5:  return "strongly negative"
    if score <= -0.15: return "negative"
    if score < 0.15:   return "neutral"
    if score < 0.5:    return "positive"
    return "strongly positive"


def bar(score, ink, width=18):
    """A centred bar: negative left of the midpoint, positive right."""
    half = width // 2
    n = int(round(abs(score) * half))
    if score < 0:
        cell = " " * (half - n) + ink.red("█" * n) + ink.dim("│") + " " * half
    else:
        cell = " " * half + ink.dim("│") + ink.green("█" * n) + " " * (half - n)
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("date", help="trading day, e.g. 2018-12-03")
    ap.add_argument("--top", type=int, default=None,
                    help="show only the N most extreme headlines")
    ap.add_argument("--features", action="store_true",
                    help="also print the stored daily feature row")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()

    for f in (DETAIL, DAILY):
        if not f.exists():
            sys.exit(f"not found: {f}")

    det = pd.read_parquet(DETAIL)
    det["trading_day"] = det["trading_day"].astype(str)
    day = det[det.trading_day == a.date].copy()
    if day.empty:
        near = sorted(det[det.trading_day.str.startswith(a.date[:7])].trading_day.unique())
        sys.exit(f"no scored headlines on {a.date}. "
                 f"That month has: {', '.join(near[:12]) or 'none'}")

    ink = Ink(not a.no_color and sys.stdout.isatty())
    day = day.sort_values("finbert_score")
    shown = day
    if a.top and len(day) > a.top:
        h = a.top // 2
        shown = pd.concat([day.head(h), day.tail(a.top - h)])

    print()
    print(ink.dim("═" * WIDTH))
    print(ink.bold(f"  FINBERT HEADLINE SCORES — {a.date}")
          + ink.dim(f"    {len(day)} headline(s) scored"))
    print(ink.dim("═" * WIDTH))
    print(ink.dim("\n  score = P(positive) - P(negative),  range -1 to +1"))
    print(ink.dim(f"  {'':22s}{'negative':>9s}  0  {'positive':<9s}"))
    print(ink.dim("  " + "-" * (WIDTH - 4)))

    for _, r in shown.iterrows():
        sc = float(r.finbert_score)
        colour = ink.red if sc < -0.15 else (ink.green if sc > 0.15 else ink.amber)
        print(f"\n  {colour(f'{sc:+.4f}')}  {bar(sc, ink)}  "
              + ink.dim(f"{band(sc)}"))
        title = str(r.Article_title).strip()
        for ln in textwrap.wrap(title, WIDTH - 8) or [""]:
            print("        " + ln)
        print(ink.dim(f"        {r.Stock_symbol}  ·  {str(r.Publisher).strip()[:40]}"
                      + f"  ·  LM net tone {float(r.lm_net_tone):+.2f}"))

    if a.top and len(day) > a.top:
        print(ink.dim(f"\n  ... {len(day) - a.top} further headline(s) omitted "
                      f"(--top {a.top})"))

    # The stored daily feature file is indexed on the news date, while the
    # headlines above are grouped by the trading day they inform, so the two do
    # not reconcile. The aggregate below is therefore computed over exactly the
    # headlines displayed; the stored feature row is opt-in via --features.
    nd = pd.to_datetime(day["Date"]).dt.strftime("%Y-%m-%d")
    print("\n" + ink.dim("-" * WIDTH))
    print(ink.cyan(ink.bold("  AGGREGATE OVER THESE HEADLINES"))
          + ink.dim("   (point-in-time: news published before the session)"))
    print(ink.dim("-" * WIDTH))
    print(f"    {'headlines':26s} {len(day)}")
    print(f"    {'published':26s} {nd.min()} to {nd.max()}")
    print(f"    {'mean FinBERT tone':26s} {day.finbert_score.mean():+.4f}")
    print(f"    {'dispersion (std)':26s} {day.finbert_score.std():+.4f}")
    print(f"    {'mean LM net tone':26s} {day.lm_net_tone.mean():+.4f}")
    print(ink.dim("\n    the news specialist reads the EWMA-smoothed and expanding-"
                  "standardised forms"))
    print(ink.dim("    of these features, never the raw headline text"))

    if a.features:
        dly = pd.read_parquet(DAILY)
        dly.index = pd.to_datetime(dly.index).strftime("%Y-%m-%d")
        print("\n" + ink.dim("-" * WIDTH))
        print(ink.cyan(ink.bold("  STORED DAILY FEATURE ROW"))
              + ink.dim("   (indexed on the news date, not the trading day)"))
        print(ink.dim("-" * WIDTH))
        if a.date in dly.index:
            row = dly.loc[a.date]
            for label, col in [("mean FinBERT tone", "mean_finbert"),
                               ("mean LM net tone", "mean_lm_tone"),
                               ("LM negative %", "lm_negative_pct"),
                               ("LM uncertainty %", "lm_uncertainty"),
                               ("article count", "article_count"),
                               ("sentiment dispersion", "sentiment_disp")]:
                if col in row:
                    print(f"    {label:26s} {float(row[col]):+.4f}")
            print(ink.dim("\n    this row aggregates the news published ON this date,"))
            print(ink.dim("    a different set of articles from the ones listed above"))
        else:
            print(ink.dim("    (no row for this date)"))
    print(ink.dim("═" * WIDTH))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
