"""Deterministic fallback DAG for Phase 4.

The LLM planner can fail in several ways — malformed JSON, validation
errors, timeouts, no API key configured. When that happens the
orchestrator must still execute *something*, and it must be a known-good
pipeline. This module hand-codes that pipeline so behaviour is identical
to the pre-Phase-4 scan cycle when the planner is unavailable.
"""

from __future__ import annotations

from .dag_schema import DAG, DAGEdge, DAGNode, NodeType


def build_safe_default_dag() -> DAG:
    """Return a DAG that mirrors the legacy scan-cycle pipeline.

    Stages (one node each, linear for clarity — the orchestrator does
    per-symbol fan-out inside each stage):

      data_fetch → features → signal → risk → portfolio → execution → audit
    """
    nodes = [
        DAGNode(id="data_fetch",  type=NodeType.DATA_FETCH, agent="DataAgent",
                params={"feed": "alpaca", "window": 200}),
        DAGNode(id="features",    type=NodeType.FEATURE,    agent="FeatureAgent",
                params={"indicators": ["rsi", "sma_50"]}),
        DAGNode(id="signal",      type=NodeType.SIGNAL,     agent="AlphaAgent",
                params={"strategy": "default"}),
        DAGNode(id="risk",        type=NodeType.RISK,       agent="RiskAgent",
                params={"checks": ["var", "exposure"]}),
        DAGNode(id="portfolio",   type=NodeType.PORTFOLIO,  agent="PortfolioAgent",
                params={"method": "equal_weight"}),
        DAGNode(id="execution",   type=NodeType.EXECUTION,  agent="ExecutionAgent",
                params={"broker": "alpaca", "order_type": "market"}),
        DAGNode(id="audit",       type=NodeType.AUDIT,      agent="AuditAgent",
                params={}),
    ]
    edges = [
        DAGEdge(src="data_fetch", dst="features"),
        DAGEdge(src="features",   dst="signal"),
        DAGEdge(src="signal",     dst="risk"),
        DAGEdge(src="risk",       dst="portfolio"),
        DAGEdge(src="portfolio",  dst="execution"),
        DAGEdge(src="execution",  dst="audit"),
    ]
    dag = DAG(nodes=nodes, edges=edges, source="safe_default")
    dag.validate()
    return dag


__all__ = ["build_safe_default_dag"]
