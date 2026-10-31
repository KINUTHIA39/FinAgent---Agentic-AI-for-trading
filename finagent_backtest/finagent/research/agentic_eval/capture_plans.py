"""Capture the planner's real output for every month of the archived backtest.

The backtest kept each cycle's *weights* but discarded the DAG the planner
returned, so there is no record of what the model actually wrote in response to
each SESSION.md. This script recovers that layer.

It replays the recorded run exactly as ``regen_sessions`` does — substituting
the archived weights from ``path0.parquet`` so the memory ledger, NAV and
drifted weights evolve identically — which means the SESSION.md handed to the
model each month is byte-identical to the one the original run produced. The
only difference is that here the planner is actually called, and everything it
returns is written to disk: raw response text, parsed DAG, execution stages,
extracted risk posture, token usage and latency.

Two things this is NOT:

  * It is not a reconstruction of the archived plans. Sampling is fresh, so
    these are *new, representative* plans for the same briefings — not the
    specific DAGs that produced the 87.4x NAV path.
  * It does not re-run the backtest. Weights, returns and NAV all come from the
    recorded path; nothing here changes the reported performance.

Costs money. At the observed ~854 in / ~1400 out tokens per call on
claude-sonnet-4-6 ($3/$15 per MTok) the full 248 months runs about $5.85 and
70 minutes. Start with ``--limit`` to sanity-check the output shape, then run
the rest — completed months are skipped on re-run, so an interrupted capture
resumes rather than re-paying.

Run:
    python -m finagent.research.agentic_eval.capture_plans --limit 3   # smoke test
    python -m finagent.research.agentic_eval.capture_plans             # full run
    # -> data/agentic_eval/plans/plan_YYYY-MM.json  (+ index.parquet)
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import pandas as pd

from finagent.core.prompt_builder import SessionWriter
from finagent.research.agentic_eval.backtest import (
    OUTDIR,
    REGISTRY,
    TICKERS,
    MemoryLedger,
    _load_env,
    _wr,
    extract_posture,
    load_inputs,
    w_prev_cash,
)

PLANS_DIR = OUTDIR.parent / "plans"

def _out_suffix(regime: str | None) -> str:
    """Hamilton keeps the original paths so archived artefacts stay untouched."""
    import os
    name = (regime or os.getenv("FINAGENT_REGIME") or "hamilton").lower()
    return "" if name in ("hamilton", "ham") else f"_{name}"


# Per-MTok list prices. Keyed by the model the client is actually configured
# with, so the running cost line stays honest if that default ever changes.
PRICING = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _price(model: str, tok_in: int, tok_out: int, cache_read: int = 0) -> float:
    """List-price cost of one call. Cache reads bill at ~0.1x input."""
    p_in, p_out = PRICING.get(model, (3.00, 15.00))
    return (tok_in * p_in + cache_read * p_in * 0.1 + tok_out * p_out) / 1e6


class RecordingClient:
    """Transparent proxy over ``ClaudeClient`` that keeps the raw response.

    The planner strips code fences and parses JSON before returning a DAG, so
    the model's actual text is lost by the time ``plan()`` returns. Wrapping the
    client instead of reimplementing the parse keeps this capture faithful to
    the production path — same prompt construction, same fallback handling.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.last_raw: str | None = None
        self.last_usage: dict | None = None
        self.last_latency_ms: float | None = None

    @property
    def model(self) -> str:
        return getattr(self._inner, "_model", "unknown")

    def complete_structured(self, system: str, cached: str, volatile: str) -> str:
        t0 = time.perf_counter()
        raw = self._inner.complete_structured(system, cached, volatile)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_raw = raw
        self.last_usage = dict(getattr(self._inner, "last_usage", None) or {})
        return raw

    def complete(self, prompt: str) -> str:  # pragma: no cover — not the live path
        return self._inner.complete(prompt)


def _dag_to_dict(dag) -> dict:
    """Serialise a validated DAG, including its parallelisable stages."""
    try:
        # to_execution_plan() yields stages of DAGNode objects; the dashboard
        # only needs the ids, and ids keep the JSON flat.
        stages = [[getattr(n, "id", str(n)) for n in stage]
                  for stage in dag.to_execution_plan()]
    except Exception:  # noqa: BLE001 — a malformed DAG should not lose the capture
        stages = []
    return {
        "source": getattr(dag, "source", "unknown"),
        "nodes": [
            {"id": n.id, "type": n.type.value, "agent": n.agent, "params": n.params}
            for n in dag.nodes
        ],
        "edges": [{"src": e.src, "dst": e.dst} for e in dag.edges],
        "stages": stages,
    }


def capture(path_file: str = "path0.parquet", limit: int | None = None,
            overwrite: bool = False,
            regime: str | None = None) -> pd.DataFrame:
    spx, rf, stk, sig, reg, vol, params = load_inputs(regime)
    out_dir = PLANS_DIR.with_name(PLANS_DIR.name + _out_suffix(regime))

    rec = pd.read_parquet(OUTDIR / path_file)
    rec.index = pd.to_datetime(rec.index)

    months = [d for d in sig.index if d in stk.index and d in reg.index]
    months = [d for d in months if d <= stk.index[-2]]
    months.sort()

    out_dir.mkdir(parents=True, exist_ok=True)

    from finagent.core.claude_client import ClaudeClient
    from finagent.core.dag_planner import LLMDAGPlanner

    _load_env()
    client = RecordingClient(ClaudeClient())
    planner = LLMDAGPlanner(client=client)
    model = client.model

    writer = SessionWriter(config_dir=tempfile.mkdtemp(prefix="capture_session_"))

    ledger = MemoryLedger()
    nav, last_ret = 1.0, 0.0
    w_prev = {t: 0.0 for t in TICKERS}
    rows: list[dict] = []
    spend, called, skipped = 0.0, 0, 0
    t_start = time.perf_counter()

    for d in months:
        loc = stk.index.get_loc(d)
        nxt = stk.index[loc + 1]
        if nxt not in rec.index:
            break

        scores = {t: float(sig.loc[d, t]) for t in TICKERS}
        rg = reg.month(d)
        reg_label, p_bull = rg.label, rg.p_bull
        stamp = d.strftime("%Y-%m")
        out_file = out_dir / f"plan_{stamp}.json"

        # --- rebuild the briefing exactly as the backtest built it ---
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

        # --- plan, unless this month is already captured ---
        if out_file.exists() and not overwrite:
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            skipped += 1
        else:
            if limit is not None and called >= limit:
                break
            client.last_raw = None
            client.last_usage = None
            client.last_latency_ms = None

            result = planner.plan(session_md, REGISTRY)
            posture = extract_posture(result.dag)

            usage = client.last_usage or {}
            tok_in = int(usage.get("input_tokens") or 0)
            tok_out = int(usage.get("output_tokens") or 0)
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
            cost = _price(model, tok_in, tok_out, cache_read)
            spend += cost
            called += 1

            payload = {
                "decision_date": stamp,
                "cycle_id": cycle_id,
                "model": model,
                "regime": reg_label,
                "p_bull": p_bull,
                "captured_at": pd.Timestamp.utcnow().isoformat(),
                "used_fallback": bool(result.used_fallback),
                "fallback_reason": result.reason or "",
                "raw_response": client.last_raw or "",
                "dag": _dag_to_dict(result.dag),
                "posture": {**posture, "flagged": sorted(posture.get("flagged") or set())},
                "usage": {
                    "input_tokens": tok_in,
                    "output_tokens": tok_out,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": cache_creation,
                    "latency_ms": round(client.last_latency_ms or 0.0, 1),
                    "cost_usd": round(cost, 6),
                },
                "session_md": session_md,
            }
            out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            if called % 10 == 0 or called == 1:
                el = time.perf_counter() - t_start
                print(f"  [{called:>3}] {stamp} {reg_label:<4} "
                      f"nodes={len(payload['dag']['nodes'])} "
                      f"${spend:.2f} ({el:.0f}s)", flush=True)

        dag = payload["dag"]
        post = payload["posture"]
        use = payload["usage"]
        rows.append({
            "decision_date": d,
            "cycle_id": cycle_id,
            "regime": reg_label,
            "p_bull": p_bull,
            "used_fallback": bool(payload["used_fallback"]),
            "source": dag.get("source", ""),
            "n_nodes": len(dag["nodes"]),
            "n_edges": len(dag["edges"]),
            "n_stages": len(dag.get("stages") or []),
            "n_risk_nodes": sum(1 for n in dag["nodes"] if n["type"] == "risk"),
            "cap_pos": post.get("cap_pos"),
            "sizing": post.get("sizing"),
            "reduce_only": bool(post.get("reduce_only")),
            "halt": bool(post.get("halt")),
            "n_flagged": len(post.get("flagged") or []),
            "flagged": ",".join(post.get("flagged") or []),
            "input_tokens": use.get("input_tokens"),
            "output_tokens": use.get("output_tokens"),
            "cache_read_input_tokens": use.get("cache_read_input_tokens"),
            "latency_ms": use.get("latency_ms"),
            "cost_usd": use.get("cost_usd"),
            "plan_file": out_file.name,
        })

        # --- replay the recorded decision so state evolves identically ---
        row = rec.loc[nxt]
        w = {t: float(row[f"w_{t}"]) for t in TICKERS}
        stock_ret = {t: float(row[f"r_{t}"]) for t in TICKERS}
        gross_ret = float(row["gross_ret"])
        net_ret = float(row["net_ret"])

        for t in TICKERS:
            if w[t] > 1e-6:
                ledger.record(t, w[t] * nav * 1_000_000 * stock_ret[t])

        denom = (1 + gross_ret) or 1.0
        w_prev = {t: w[t] * (1 + stock_ret[t]) / denom for t in TICKERS}
        nav, last_ret = nav * (1 + net_ret), net_ret

    idx = pd.DataFrame(rows)
    if not idx.empty:
        idx.to_parquet(out_dir / "index.parquet", index=False)

    idx.attrs["called"] = called
    idx.attrs["skipped"] = skipped
    idx.attrs["spend"] = spend
    idx.attrs["elapsed"] = time.perf_counter() - t_start
    idx.attrs["model"] = model
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N new API calls (smoke test); already-captured "
                         "months are replayed from disk and do not count")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-call the planner for months already captured")
    ap.add_argument("--path-file", default="path0.parquet")
    args = ap.parse_args()

    idx = capture(path_file=args.path_file, limit=args.limit, overwrite=args.overwrite)
    if idx.empty:
        print("no months captured")
        return 1

    called = idx.attrs["called"]
    remaining = 248 - len(idx)
    print(f"\ncaptured {len(idx)} plans -> {out_dir}")
    print(f"  model       : {idx.attrs['model']}")
    print(f"  new calls   : {called}   (replayed from disk: {idx.attrs['skipped']})")
    print(f"  span        : {idx.decision_date.min():%Y-%m} .. {idx.decision_date.max():%Y-%m}")
    print(f"  spend       : ${idx.attrs['spend']:.2f}   elapsed {idx.attrs['elapsed']:.0f}s")
    print(f"  fallbacks   : {int(idx.used_fallback.sum())}/{len(idx)}")
    print(f"  nodes       : mean {idx.n_nodes.mean():.1f}  range {idx.n_nodes.min()}-{idx.n_nodes.max()}")
    print(f"  risk nodes  : mean {idx.n_risk_nodes.mean():.2f}")
    if called:
        live = idx[idx.cost_usd.notna() & (idx.cost_usd > 0)]
        if not live.empty:
            print(f"  tokens/call : in {live.input_tokens.mean():.0f}  out {live.output_tokens.mean():.0f}"
                  f"  cache_read {live.cache_read_input_tokens.mean():.0f}")
            print(f"  cost/call   : ${live.cost_usd.mean():.4f}")
            if remaining > 0:
                print(f"  projected   : ${live.cost_usd.mean() * remaining:.2f} and "
                      f"{live.latency_ms.mean() * remaining / 60000:.0f} min for the "
                      f"remaining {remaining} months")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
