"""Dump the *complete* prompt the backtest's LLM planner sent to Claude.

The dissertation appendix previously showed only the SESSION.md tail. The full query
Claude actually receives is three blocks (see dag_planner._build_prompt_parts):

    [system]   _SYSTEM_PROMPT          - planner role + strict JSON schema + rules   (cached)
    [cached]   ## Agent Registry       - the agent roster                            (cached)
    [volatile] ## Session State ...     - the per-month SESSION.md + "## Output JSON" (volatile)

The cache breakpoint (cache_control: ephemeral) sits between the cached blocks and the
volatile tail, so the system prompt and registry are byte-stable and served from cache on
every cycle after the first.

To reproduce a *real-run* query faithfully, the per-month SESSION.md depends on the
endogenous memory ledger, whose state is path-dependent. We therefore replay the recorded
run (path0.parquet stores w_*, nav, r_* per month) to rebuild the exact ledger, portfolio
state, and recall block at the chosen month, then assemble the identical three blocks.

Writes UTF-8 text files under data/agentic_eval/analysis/full_query/ for \\lstinputlisting.

Run:
    python -m finagent.research.agentic_eval.dump_prompt              # default bull + bear
    python -m finagent.research.agentic_eval.dump_prompt --month 2008-11 --label bear
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from finagent.core.dag_planner import _build_prompt_parts
from finagent.core.prompt_builder import SessionWriter
from finagent.research.agentic_eval.backtest import (
    MemoryLedger, REGISTRY, TICKERS, load_inputs, w_prev_cash, _wr)

REPO = Path(__file__).resolve().parents[3]
PATH0 = REPO / "data" / "agentic_eval" / "backtest" / "path0.parquet"
OUT = REPO / "data" / "agentic_eval" / "analysis" / "full_query"


def _replay_to(path: pd.DataFrame, j: int):
    """Rebuild (ledger, nav_before, last_ret, w_prev) as of decision month index ``j``.

    Row i of ``path`` (indexed by the *realised* month) corresponds to the decision taken
    at the j-th backtest month, holding weights ``w_*`` and realising returns ``r_*``.
    """
    ledger = MemoryLedger()
    nav_before = 1.0
    for i in range(j):
        row = path.iloc[i]
        for t in TICKERS:
            w = float(row[f"w_{t}"])
            if w > 1e-6:
                ledger.record(t, w * nav_before * 1_000_000 * float(row[f"r_{t}"]))
        nav_before = float(row["nav"])
    if j == 0:
        return ledger, 1.0, 0.0, {t: 0.0 for t in TICKERS}
    prev = path.iloc[j - 1]
    nav_j = float(prev["nav"])
    last_ret = float(prev["net_ret"])
    gr = float(prev["gross_ret"])
    w_prev = {t: float(prev[f"w_{t}"]) * (1 + float(prev[f"r_{t}"])) / ((1 + gr) or 1.0)
              for t in TICKERS}
    return ledger, nav_j, last_ret, w_prev


def dump_month(month: str, label: str) -> Path:
    spx, rf, stk, sig, reg, vol, params = load_inputs()
    path = pd.read_parquet(PATH0)

    months = [d for d in sig.index if d in stk.index and d in reg.index]
    months = sorted(d for d in months if d <= stk.index[-2])
    target = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    if target not in months:
        # snap to the nearest available month-end in the backtest
        target = min(months, key=lambda d: abs((d - target).days))
    j = months.index(target)

    ledger, nav, last_ret, w_prev = _replay_to(path, j)
    d = target
    scores = {t: float(sig.loc[d, t]) for t in TICKERS}
    rg = reg.month(d)
    reg_label, p_bull = rg.label, rg.p_bull

    portfolio = {"nav": round(nav * 1_000_000, 2),
                 "cash": round(w_prev_cash(w_prev) * nav * 1_000_000, 2),
                 "position_count": int(sum(v > 1e-6 for v in w_prev.values())),
                 "daily_pnl_pct": round(last_ret, 4)}
    regime = {"tag": reg_label, "prob_bull": round(p_bull, 2),
              "prob_bear": round(1 - p_bull, 2), "prob_sideways": 0.0}
    candidates = [{"symbol": t, "signal": "BUY", "score": round(scores[t], 3),
                   "rsi": 0.0, "win_rate": _wr(ledger, t)} for t in TICKERS]
    recall = ledger.recall_block()

    sw_dir = tempfile.mkdtemp(prefix="dump_session_")
    writer = SessionWriter(config_dir=sw_dir)
    session_md = Path(writer.write(portfolio_state=portfolio, regime=regime,
                                   memory_recall=recall, candidates=candidates,
                                   cycle_id=f"bt_{d.strftime('%Y%m')}")).read_text(encoding="utf-8")

    system, cached, volatile = _build_prompt_parts(session_md, REGISTRY)
    full = (
        "================ BLOCK 1/3: SYSTEM PROMPT (cached) ================\n"
        f"{system}\n"
        "================ BLOCK 2/3: AGENT REGISTRY (cached) ===============\n"
        f"{cached}"
        "<<< CACHE BOUNDARY: blocks above are byte-stable and served from cache >>>\n"
        "============== BLOCK 3/3: SESSION STATE (volatile) ===============\n"
        f"{volatile}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{label}.txt"
    out.write_text(full, encoding="utf-8")
    print(f"wrote {out}  ({d.date()}, regime={reg_label}, "
          f"{len([1 for t in TICKERS if len(ledger.rec[t])>=3])} symbols in memory, "
          f"{len(full)} chars)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="YYYY-MM; omit to dump default bull + bear pair")
    ap.add_argument("--label", default="query")
    args = ap.parse_args()
    if args.month:
        dump_month(args.month, args.label)
    else:
        dump_month("2017-06", "bull")     # deep calm bull, rich memory, no crisis
        dump_month("2008-11", "bear")     # GFC trough, rich memory + flagged losers
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
