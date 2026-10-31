"""Headlines extracted from FNSPID for the Nasdaq and the S&P 500, by year.

The Nasdaq track is the QQQ ticker-tagged extract plus the market-desk headlines
that name the Nasdaq; the S&P track is SPY plus the desk headlines that name the
S&P 500. A headline naming both indices counts once on each side (172 do).

Drawn from files already on disk:
    data/fnspid/by_ticker/{QQQ,SPY,_MARKET}.parquet

Usage:
    python scripts/plot_nasdaq_vs_spx.py
    python scripts/plot_nasdaq_vs_spx.py --table     # print the counts instead
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TICK = REPO / "data" / "fnspid" / "by_ticker"
OUT = REPO / "results" / "nasdaq_vs_spx.png"

# Palette: validated with the dataviz six-checks script (light surface).
# #1668B0 / #C0851C -> chroma floor PASS, CVD dE 26.0 protan, normal dE 30.6.
SPX_C, NDX_C = "#1668B0", "#C0851C"
INK, MUTED, LINE, SURFACE = "#1A1A2E", "#5B6472", "#D5DAE1", "#FCFCFB"

# The desk track stops here; everything after is ticker-tagged only, so the
# chart is not comparable across this line without saying so.
DESK_END = 2020

_NDX = re.compile(r"\bNasdaq\b|\bNDX\b|\bIXIC\b", re.I)
_SPX = re.compile(r"S&P ?500|\bSPX\b|\bGSPC\b|S ?& ?P ?500", re.I)


def headline_counts():
    """Headlines per year on each track, as a two-column frame."""
    mkt = pd.read_parquet(TICK / "_MARKET.parquet")
    title = mkt.Article_title.astype(str)
    mkt = mkt.assign(
        yr=pd.to_datetime(mkt.Date, utc=True).dt.year,
        is_ndx=title.str.contains(_NDX),
        is_spx=title.str.contains(_SPX),
    )

    def tagged(tkr):
        d = pd.read_parquet(TICK / f"{tkr}.parquet")
        return pd.to_datetime(d.Date, utc=True).dt.year.value_counts()

    ndx = tagged("QQQ").add(mkt[mkt.is_ndx].groupby("yr").size(), fill_value=0)
    spx = tagged("SPY").add(mkt[mkt.is_spx].groupby("yr").size(), fill_value=0)
    out = pd.concat({"Nasdaq": ndx, "S&P 500": spx}, axis=1).fillna(0.0)
    return out.sort_index().astype(int)


def draw(cnt):
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9.4, length=3, width=0.9)
    ax.grid(axis="y", color=LINE, lw=0.7, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)

    yrs = cnt.index.to_numpy()
    w = 0.40
    gap = 0.012  # ~2px of surface between the paired bars
    tot = cnt.sum()
    for off, col, colour in ((-w / 2 - gap, "Nasdaq", NDX_C),
                             (+w / 2 + gap, "S&P 500", SPX_C)):
        ax.bar(yrs + off, cnt[col].to_numpy(), width=w, color=colour,
               zorder=3, label=f"{col}   {tot[col]:,} headlines", linewidth=0)

    top = cnt.to_numpy().max()
    ax.set_ylim(0, top * 1.14)

    # The one bar that needs explaining gets a label; the rest do not.
    spike = int(cnt["S&P 500"].idxmax())
    ax.annotate(f"{cnt.loc[spike, 'S&P 500']:,}",
                (spike + w / 2 + gap, cnt.loc[spike, "S&P 500"]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9.0, color=INK, fontweight="bold")

    ax.axvline(DESK_END + 0.5, color=MUTED, lw=1.1, ls=(0, (3, 3)), zorder=2)
    ax.annotate("market-desk headlines stop" + chr(10) +
                "mid-2020; later years are" + chr(10) + "ticker-tagged only",
                (DESK_END + 0.64, top * 1.10), fontsize=8.6, color=MUTED,
                va="top", ha="left", style="italic")

    ax.set_ylabel("Headlines extracted", fontsize=10.0, color=MUTED, labelpad=8)
    ax.text(0, 1.195, "Nasdaq and S&P 500 headlines extracted from FNSPID",
            transform=ax.transAxes, fontsize=14.0, color=INK,
            fontweight="bold", va="top", ha="left")
    ax.text(0, 1.105,
            f"{tot['Nasdaq']:,} on the Nasdaq track against "
            f"{tot['S&P 500']:,} on the S&P 500 track. The 2018 S&P spike is "
            f"SPY ticker-tagging starting," + chr(10) +
            "not a real surge in coverage; before Aug 2017 the S&P side rests "
            "entirely on the market desk.",
            transform=ax.transAxes, fontsize=9.6, color=MUTED,
            va="top", ha="left")

    ax.set_xticks(yrs)
    ax.set_xticklabels([str(y) for y in yrs])
    ax.legend(frameon=False, loc="upper left", fontsize=9.6, labelcolor=INK,
              handlelength=1.1, borderpad=0.1, bbox_to_anchor=(0.015, 0.97))
    ax.margins(x=0.015)

    fig.text(0.5, 0.005,
             "Source: FNSPID nasdaq_exteral_data.csv - QQQ and SPY "
             "ticker-tagged extracts plus the market-desk track. A headline "
             "naming both indices counts on each side (172 do).",
             ha="center", fontsize=8.2, color=MUTED)

    fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--table", action="store_true",
                    help="print the underlying counts instead of drawing")
    a = ap.parse_args()

    cnt = headline_counts()
    if a.table:
        tot = cnt.sum()
        print(cnt.to_string())
        print("-" * 25)
        print(f"{'total':<4}{tot['Nasdaq']:>8,}{tot['S&P 500']:>9,}")
        return 0

    draw(cnt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
