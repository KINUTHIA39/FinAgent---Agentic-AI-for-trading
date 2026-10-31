"""Agentic-AI evaluation harness (dissertation).

Empirically characterises the LLM-orchestrated trading agent along the three
axes the literature flags as open — autonomy (plan adaptation), explainability
(provenance), and trustworthiness (reliability/safety + cost) — on an equities
substrate.

Read-only with respect to the frozen dissertation assets (`data/dissertation/`,
the `dissertation` Neo4j DB). All experiment output goes to a separate results
directory.
"""
