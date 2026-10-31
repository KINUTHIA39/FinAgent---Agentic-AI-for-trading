"""The planning harness: run a Scenario through the real LLM planner K times,
logging the DAG features + reliability/cost telemetry.

Uses the production SESSION.md format (``SessionWriter``) and the production
planner (``LLMDAGPlanner`` + ``ClaudeClient``), so the experiment evaluates the
*actual* agent, not a reimplementation.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from finagent.core.prompt_builder import SessionWriter

from .features import dag_features
from .scenarios import Scenario

# Agent registry handed to the planner (the tools it may compose).
REGISTRY = (
    "- DataAgent: fetches OHLCV and market state\n"
    "- FeatureAgent: computes RSI/SMA indicators\n"
    "- AlphaAgent: generates BUY/SELL signals\n"
    "- RiskAgent: VaR + exposure checks\n"
    "- PortfolioAgent: position sizing\n"
    "- ExecutionAgent: places orders\n"
    "- AuditAgent: records cycle outcome\n"
)


def render_session(scenario: Scenario, cycle_id: str = "exp") -> tuple[str, int]:
    """Render the scenario into a production-format SESSION.md string.

    Returns (session_md, n_candidates).
    """
    inp = scenario.to_session_inputs()
    tmp = tempfile.mkdtemp(prefix="agentic_eval_")
    path = SessionWriter(config_dir=tmp).write(
        portfolio_state=inp["portfolio_state"],
        regime=inp["regime"],
        memory_recall=inp["memory_recall"],
        candidates=inp["candidates"],
        cycle_id=cycle_id,
    )
    return Path(path).read_text(encoding="utf-8"), len(inp["candidates"])


def run_scenario(scenario: Scenario, planner: Any, k: int,
                 registry: str = REGISTRY) -> List[Dict[str, Any]]:
    """Plan ``scenario`` K times; return one row dict per run."""
    session_md, n_cand = render_session(
        scenario, cycle_id=f"exp_{scenario.regime}_{scenario.exposure}")
    client = getattr(planner, "_client", None)
    rows: List[Dict[str, Any]] = []

    for i in range(k):
        if client is not None:
            client.last_usage = None
        t0 = time.perf_counter()
        result = planner.plan(session_md, registry)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        usage = getattr(client, "last_usage", None) or {}
        feats = dag_features(result.dag, n_cand)
        rows.append({
            "scenario": scenario.label,
            "regime": scenario.regime,
            "exposure": scenario.exposure,
            "recall_profile": scenario.recall_profile,
            "ablate": "+".join(scenario.ablate),
            "note": scenario.note,
            "run_idx": i,
            "used_fallback": bool(result.used_fallback),
            "reason": result.reason or "",
            "llm_success": (not result.used_fallback) and feats["source"] == "llm",
            **feats,
            "tokens_in": usage.get("input_tokens"),
            "tokens_out": usage.get("output_tokens"),
            "tokens_cache_read": usage.get("cache_read_input_tokens"),
            "tokens_cache_creation": usage.get("cache_creation_input_tokens"),
            "latency_ms": round(latency_ms, 1),
        })
    return rows


__all__ = ["run_scenario", "render_session", "REGISTRY"]
