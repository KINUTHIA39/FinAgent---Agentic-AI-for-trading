"""Figures for the agentic backtest: 3-panel wealth / drawdown / relative-strength.

Reads a completed backtest path (data/agentic_eval/backtest/path0.parquet) plus the
S&P monthly series, and renders one stacked figure that makes the agent's *risk-adjusted*
edge legible (the raw log-wealth curve hides it because Agent and Equal-weight nearly
overlap):

  Panel 1  Log wealth         Agent vs Equal-weight-8 vs S&P 500 (shaded = detected bear)
  Panel 2  Drawdown underwater Agent vs Equal-weight-8 (where the -36% vs -44% gap shows)
  Panel 3  Relative strength   Agent / Equal-weight-8 cumulative (step-ups during bears)

All three series are rebuilt from monthly returns with a common $1 start so they are
directly comparable. Equal-weight is the mean of the eight realized stock returns stored
in the path (identical universe to the agent) -> the *fair* benchmark.

Run:
    python -m finagent.research.agentic_eval.make_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BT = REPO / "data" / "agentic_eval" / "backtest"
MD = REPO / "data" / "agentic_eval" / "market_data"
ANALYSIS_FIG = REPO / "data" / "agentic_eval" / "analysis" / "figures"
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "XOM", "JNJ"]

_AGENT = "#27408b"   # dark blue
_EW = "#2e8b57"      # sea green
_SPX = "#9a9a9a"     # grey
_BEAR = "#d62728"    # crisis shading


def _wealth(ret: pd.Series) -> pd.Series:
    """$1 invested the month before the first return -> compounded NAV."""
    w = (1.0 + ret).cumprod()
    t0 = ret.index[0] - pd.offsets.MonthEnd(1)
    return pd.concat([pd.Series([1.0], index=[t0]), w])


def _drawdown(wealth: pd.Series) -> pd.Series:
    return wealth / wealth.cummax() - 1.0


def _bear_spans(regime: pd.Series):
    """Contiguous (start, end) month-end spans where the detected regime is bear."""
    is_bear = (regime == "bear").values
    spans, i, n = [], 0, len(is_bear)
    idx = regime.index
    while i < n:
        if is_bear[i]:
            j = i
            while j + 1 < n and is_bear[j + 1]:
                j += 1
            lo = idx[i] - pd.offsets.MonthEnd(1)
            spans.append((lo, idx[j]))
            i = j + 1
        else:
            i += 1
    return spans


def build(path: Path = BT / "path0.parquet", cost_bps: int = 10) -> Path:
    df = pd.read_parquet(path)
    idx = df.index

    agent_ret = df["net_ret"]
    ew_ret = df[[f"r_{t}" for t in TICKERS]].mean(axis=1)        # fair benchmark, same universe
    spx_ret = pd.read_parquet(MD / "spx_monthly.parquet")["ret"].reindex(idx)

    w_agent, w_ew, w_spx = _wealth(agent_ret), _wealth(ew_ret), _wealth(spx_ret)
    dd_agent, dd_ew = _drawdown(w_agent), _drawdown(w_ew)
    rel = (w_agent / w_ew)                                       # relative strength, common t0=1
    spans = _bear_spans(df["regime"])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.2, 1.2], "hspace": 0.12})

    def shade(ax, label=False):
        for k, (lo, hi) in enumerate(spans):
            ax.axvspan(lo, hi, color=_BEAR, alpha=0.10,
                       label="Detected bear" if (label and k == 0) else None)

    # Panel 1 — log wealth
    shade(ax1, label=True)
    ax1.plot(w_agent.index, w_agent.values, color=_AGENT, lw=1.8, label="Agent (LLM)")
    ax1.plot(w_ew.index, w_ew.values, color=_EW, lw=1.4, label="Equal-weight 8")
    ax1.plot(w_spx.index, w_spx.values, color=_SPX, lw=1.3, label="S&P 500")
    ax1.set_yscale("log")
    ax1.set_ylabel("Wealth (log, $1 start)")
    ax1.set_title(f"Agentic backtest 2005-2025 ({cost_bps} bps) — "
                  f"Agent {w_agent.iloc[-1]:.0f}x  vs  Equal-weight {w_ew.iloc[-1]:.0f}x  "
                  f"vs  S&P {w_spx.iloc[-1]:.0f}x")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, which="both", alpha=0.15)

    # Panel 2 — drawdown underwater (the risk-adjusted edge becomes visible here)
    shade(ax2)
    ax2.fill_between(dd_ew.index, dd_ew.values * 100, 0, color=_EW, alpha=0.30,
                     label=f"Equal-weight 8  (maxDD {dd_ew.min()*100:.0f}%)")
    ax2.fill_between(dd_agent.index, dd_agent.values * 100, 0, color=_AGENT, alpha=0.45,
                     label=f"Agent  (maxDD {dd_agent.min()*100:.0f}%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.15)

    # Panel 3 — relative strength Agent / Equal-weight
    shade(ax3)
    ax3.plot(rel.index, rel.values, color=_AGENT, lw=1.6)
    ax3.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax3.fill_between(rel.index, rel.values, 1.0, where=(rel.values >= 1.0),
                     color=_AGENT, alpha=0.18)
    ax3.set_ylabel("Agent / Equal-weight")
    ax3.set_xlabel("")
    ax3.annotate(f"final {rel.iloc[-1]:.2f}x", xy=(rel.index[-1], rel.iloc[-1]),
                 xytext=(-4, 6), textcoords="offset points", ha="right",
                 fontsize=9, color=_AGENT)
    ax3.grid(True, alpha=0.15)

    out = BT / "figures" / "wealth_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    print(f"  Agent final {w_agent.iloc[-1]:.1f}x  maxDD {dd_agent.min()*100:.1f}%")
    print(f"  EW    final {w_ew.iloc[-1]:.1f}x  maxDD {dd_ew.min()*100:.1f}%")
    print(f"  S&P   final {w_spx.iloc[-1]:.1f}x")
    print(f"  relative strength: min {rel.min():.2f}x  final {rel.iloc[-1]:.2f}x")
    return out


def returns_series(path: Path = BT / "path0.parquet", cost_bps: int = 10) -> Path:
    """Monthly net-return time series, bars colored by detected regime, bears shaded.

    Written into the dissertation analysis figure directory for \\includegraphics.
    """
    df = pd.read_parquet(path)
    ret = df["net_ret"] * 100.0
    is_bear = (df["regime"] == "bear").values
    colors = np.where(is_bear, _BEAR, _AGENT)
    spans = _bear_spans(df["regime"])

    fig, ax = plt.subplots(figsize=(11, 3.4))
    for lo, hi in spans:
        ax.axvspan(lo, hi, color=_BEAR, alpha=0.08)
    ax.bar(ret.index, ret.values, width=24, color=colors, alpha=0.85, linewidth=0)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("Monthly net return (%)")
    ax.set_title(f"Agent monthly portfolio returns, 2005-2025 ({cost_bps} bps); "
                 f"red = month in a detected bear regime")
    ax.margins(x=0.01)
    ax.grid(True, axis="y", alpha=0.15)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=_AGENT, label="bull-regime month"),
                       Patch(color=_BEAR, label="bear-regime month")],
              loc="lower left", framealpha=0.9, fontsize=9)

    ANALYSIS_FIG.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_FIG / "returns_timeseries.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    pos = (ret > 0).mean() * 100
    print(f"wrote {out}  (best {ret.max():+.1f}%, worst {ret.min():+.1f}%, "
          f"{pos:.0f}% of months positive)")
    return out


if __name__ == "__main__":
    build()
    returns_series()
