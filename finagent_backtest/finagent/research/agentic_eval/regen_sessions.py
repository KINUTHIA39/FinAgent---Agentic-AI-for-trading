"""Regenerate the SESSION.md briefing for every month of the archived backtest.

The backtest wrote each cycle's SESSION.md to a temp dir that was overwritten on
the next cycle, so the per-month briefings were never kept. They are, however,
fully recoverable: ``SessionWriter`` is deterministic, and every input it needs
(regime, portfolio state, endogenous memory, candidate scores) is reproducible
by replaying the recorded run.

This script replays the exact backtest loop but substitutes the *recorded*
weights from ``path0.parquet`` instead of calling the planner. Because the
memory ledger, NAV and drifted weights all evolve from those weights, the state
at every month matches the original run, and the regenerated briefings are the
ones the model actually read. No LLM calls, no cost.

Run:
    python -m finagent.research.agentic_eval.regen_sessions
    # -> data/agentic_eval/sessions/session_YYYY-MM.md  (+ index.parquet)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from finagent.core.prompt_builder import SessionWriter
from finagent.research.agentic_eval.backtest import (
    OUTDIR,
    TICKERS,
    MemoryLedger,
    _wr,
    load_inputs,
    w_prev_cash,
)

SESSIONS_DIR = OUTDIR.parent / "sessions"

def _out_suffix(regime: str | None) -> str:
    """Hamilton keeps the original paths so archived artefacts stay untouched."""
    import os
    name = (regime or os.getenv("FINAGENT_REGIME") or "hamilton").lower()
    return "" if name in ("hamilton", "ham") else f"_{name}"



def regenerate(path_file: str = "path0.parquet",
               regime: str | None = None) -> pd.DataFrame:
    spx, rf, stk, sig, reg, vol, params = load_inputs(regime)
    out_dir = SESSIONS_DIR.with_name(SESSIONS_DIR.name + _out_suffix(regime))

    rec = pd.read_parquet(OUTDIR / path_file)
    rec.index = pd.to_datetime(rec.index)

    # The recorded row is stamped with the month the return was REALISED (t+1);
    # the weights on that row were decided at the preceding decision month.
    months = [d for d in sig.index if d in stk.index and d in reg.index]
    months = [d for d in months if d <= stk.index[-2]]
    months.sort()

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SessionWriter(config_dir=tempfile.mkdtemp(prefix="regen_session_"))

    ledger = MemoryLedger()
    nav, last_ret = 1.0, 0.0
    w_prev = {t: 0.0 for t in TICKERS}
    rows = []

    for i, d in enumerate(months):
        nxt = stk.index[stk.index.get_loc(d) + 1]
        if nxt not in rec.index:
            break

        scores = {t: float(sig.loc[d, t]) for t in TICKERS}
        rg = reg.month(d)
        reg_label, p_bull = rg.label, rg.p_bull

        # --- the briefing, built exactly as backtest.py builds it ---
        portfolio = {
            "nav": round(nav * 1_000_000, 2),
            "cash": round(w_prev_cash(w_prev) * nav * 1_000_000, 2),
            "position_count": int(sum(v > 1e-6 for v in w_prev.values())),
            "daily_pnl_pct": round(last_ret, 4),
        }
        regime_ctx = rg.briefing_ctx()
        candidates = [{"symbol": t, "signal": "BUY", "score": round(scores[t], 3),
                       "rsi": 0.0, "win_rate": _wr(ledger, t)} for t in TICKERS]
        cycle_id = f"bt_{d.strftime('%Y%m')}"
        session_md = Path(writer.write(
            portfolio_state=portfolio, regime=regime_ctx,
            memory_recall=ledger.recall_block(), candidates=candidates,
            cycle_id=cycle_id)).read_text(encoding="utf-8")

        (out_dir / f"session_{d.strftime('%Y-%m')}.md").write_text(
            session_md, encoding="utf-8")

        # --- replay the recorded decision so state evolves identically ---
        row = rec.loc[nxt]
        w = {t: float(row[f"w_{t}"]) for t in TICKERS}
        stock_ret = {t: float(row[f"r_{t}"]) for t in TICKERS}
        gross_ret = float(row["gross_ret"])
        net_ret = float(row["net_ret"])

        for t in TICKERS:
            if w[t] > 1e-6:
                ledger.record(t, w[t] * nav * 1_000_000 * stock_ret[t])

        rows.append({
            "decision_date": d, "realised_date": nxt, "cycle_id": cycle_id,
            "regime": reg_label, "p_bull": p_bull,
            "nav_at_decision": nav,
            "gross_exposure": 1.0 - float(row["w_cash"]),
            "w_cash": float(row["w_cash"]),
            "turnover": float(row["turnover"]),
            "net_ret_next": net_ret,
            "used_fallback": bool(row["used_fallback"]),
            "n_dropped": int(sum(1 for t in TICKERS if w[t] <= 1e-9)),
            "dropped": ",".join(t for t in TICKERS if w[t] <= 1e-9),
            "session_file": f"session_{d.strftime('%Y-%m')}.md",
            **{f"w_{t}": w[t] for t in TICKERS},
            **{f"score_{t}": scores[t] for t in TICKERS},
        })

        denom = (1 + gross_ret) or 1.0
        w_prev = {t: w[t] * (1 + stock_ret[t]) / denom for t in TICKERS}
        nav, last_ret = nav * (1 + net_ret), net_ret

    idx = pd.DataFrame(rows)
    idx.to_parquet(out_dir / "index.parquet", index=False)
    idx.attrs["out_dir"] = str(out_dir)
    idx.attrs["regime"] = reg.name
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--regime", choices=["hamilton", "arm_c"], default=None,
                    help="regime signal to brief the agent with "
                         "(default: hamilton, or $FINAGENT_REGIME)")
    a = ap.parse_args()
    idx = regenerate(regime=a.regime)
    print(f"regenerated {len(idx)} briefings -> {idx.attrs['out_dir']}")
    print(f"  regime    : {idx.attrs['regime']}")
    print(f"  span      : {idx.decision_date.min():%Y-%m} .. {idx.decision_date.max():%Y-%m}")
    print(f"  final NAV : {idx.nav_at_decision.iloc[-1]:.2f}x (at last decision)")
    print(f"  fallbacks : {int(idx.used_fallback.sum())}")
    print(f"  months with a name dropped: {int((idx.n_dropped > 0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
