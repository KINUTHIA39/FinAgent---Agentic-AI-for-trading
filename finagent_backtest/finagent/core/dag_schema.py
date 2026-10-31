"""Typed DAG schema for the Phase 4 LLM-driven planner.

The planner reads the compressed ``SESSION.md`` tail and emits a DAG of
typed nodes. This module defines the data model, validates structural
invariants (no cycles, referenced nodes exist, at least one terminal), and
linearises the DAG into an execution order the orchestrator can consume.

Nothing in this module performs I/O — it is pure data + validation so it
can be unit-tested without a live LLM or Neo4j connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class NodeType(str, Enum):
    """Kinds of work the planner may schedule.

    Ordering here is the canonical pipeline stage ordering used by the
    safe-default DAG and by validation when checking edge direction.
    """

    DATA_FETCH = "data_fetch"
    FEATURE = "feature"
    SIGNAL = "signal"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    EXECUTION = "execution"
    AUDIT = "audit"


_TERMINAL_KINDS = {NodeType.EXECUTION, NodeType.AUDIT}


class DAGValidationError(ValueError):
    """Raised when a DAG fails structural validation."""


@dataclass(frozen=True)
class DAGNode:
    """A single unit of work in the planned cycle."""

    id: str
    type: NodeType
    agent: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise DAGValidationError("DAGNode.id must be non-empty")
        if not self.agent:
            raise DAGValidationError(f"DAGNode({self.id}).agent must be non-empty")
        if not isinstance(self.type, NodeType):
            raise DAGValidationError(
                f"DAGNode({self.id}).type must be a NodeType, got {type(self.type).__name__}"
            )


@dataclass(frozen=True)
class DAGEdge:
    """A directed dependency: ``src`` must complete before ``dst`` runs."""

    src: str
    dst: str

    def __post_init__(self) -> None:
        if self.src == self.dst:
            raise DAGValidationError(f"Self-loop edge on node {self.src!r}")
        if not self.src or not self.dst:
            raise DAGValidationError("DAGEdge endpoints must be non-empty")


@dataclass
class DAG:
    """A validated DAG the orchestrator can execute topologically."""

    nodes: list[DAGNode]
    edges: list[DAGEdge]
    source: str = "unknown"  # "llm", "safe_default", ...

    def validate(self) -> None:
        """Raise :class:`DAGValidationError` if the DAG is malformed."""
        if not self.nodes:
            raise DAGValidationError("DAG must contain at least one node")

        ids = [n.id for n in self.nodes]
        seen: set[str] = set()
        for nid in ids:
            if nid in seen:
                raise DAGValidationError(f"Duplicate node id: {nid!r}")
            seen.add(nid)

        node_by_id = {n.id: n for n in self.nodes}
        for edge in self.edges:
            if edge.src not in node_by_id:
                raise DAGValidationError(f"Edge references unknown src: {edge.src!r}")
            if edge.dst not in node_by_id:
                raise DAGValidationError(f"Edge references unknown dst: {edge.dst!r}")

        if not any(n.type in _TERMINAL_KINDS for n in self.nodes):
            raise DAGValidationError(
                "DAG must contain at least one EXECUTION or AUDIT terminal node"
            )

        # Kahn's algorithm — also detects cycles.
        indegree: dict[str, int] = {nid: 0 for nid in ids}
        adj: dict[str, list[str]] = {nid: [] for nid in ids}
        for edge in self.edges:
            indegree[edge.dst] += 1
            adj[edge.src].append(edge.dst)

        queue = [nid for nid, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            nid = queue.pop(0)
            visited += 1
            for nxt in adj[nid]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if visited != len(ids):
            raise DAGValidationError("DAG contains a cycle")

    def to_execution_plan(self) -> list[list[DAGNode]]:
        """Linearise the DAG into parallelisable stages.

        Each returned inner list is a set of nodes whose dependencies are
        all satisfied by earlier stages — the orchestrator can execute
        them concurrently.
        """
        self.validate()
        node_by_id = {n.id: n for n in self.nodes}
        indegree: dict[str, int] = {n.id: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            indegree[edge.dst] += 1
            adj[edge.src].append(edge.dst)

        stages: list[list[DAGNode]] = []
        remaining = dict(indegree)
        while remaining:
            ready = sorted(nid for nid, deg in remaining.items() if deg == 0)
            if not ready:
                raise DAGValidationError("DAG contains a cycle")  # defensive
            stages.append([node_by_id[nid] for nid in ready])
            for nid in ready:
                for nxt in adj[nid]:
                    remaining[nxt] -= 1
                del remaining[nid]
        return stages

    def node_ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def kinds(self) -> set[NodeType]:
        return {n.type for n in self.nodes}

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "unknown") -> "DAG":
        """Build a DAG from a plain dict (e.g. parsed LLM JSON).

        Raises :class:`DAGValidationError` on any structural problem —
        this is the single coercion boundary for untrusted planner output.
        """
        if not isinstance(data, dict):
            raise DAGValidationError(f"DAG payload must be a dict, got {type(data).__name__}")

        raw_nodes = data.get("nodes")
        raw_edges = data.get("edges", [])
        if not isinstance(raw_nodes, list):
            raise DAGValidationError("DAG payload missing 'nodes' list")
        if not isinstance(raw_edges, list):
            raise DAGValidationError("DAG payload 'edges' must be a list")

        nodes: list[DAGNode] = []
        for i, raw in enumerate(raw_nodes):
            if not isinstance(raw, dict):
                raise DAGValidationError(f"node[{i}] must be a dict")
            try:
                ntype = NodeType(raw["type"])
            except KeyError as exc:
                raise DAGValidationError(f"node[{i}] missing 'type'") from exc
            except ValueError as exc:
                raise DAGValidationError(
                    f"node[{i}] has unknown type {raw.get('type')!r}"
                ) from exc
            nodes.append(
                DAGNode(
                    id=str(raw.get("id", "")),
                    type=ntype,
                    agent=str(raw.get("agent", "")),
                    params=dict(raw.get("params", {}) or {}),
                )
            )

        edges: list[DAGEdge] = []
        for i, raw in enumerate(raw_edges):
            if not isinstance(raw, dict):
                raise DAGValidationError(f"edge[{i}] must be a dict")
            try:
                edges.append(DAGEdge(src=str(raw["src"]), dst=str(raw["dst"])))
            except KeyError as exc:
                raise DAGValidationError(f"edge[{i}] missing {exc.args[0]!r}") from exc

        dag = cls(nodes=nodes, edges=edges, source=source)
        dag.validate()
        return dag

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "nodes": [
                {"id": n.id, "type": n.type.value, "agent": n.agent, "params": dict(n.params)}
                for n in self.nodes
            ],
            "edges": [{"src": e.src, "dst": e.dst} for e in self.edges],
        }


def terminal_kinds() -> Iterable[NodeType]:
    """Expose the terminal-kind set for tests and planners."""
    return frozenset(_TERMINAL_KINDS)


__all__ = [
    "DAG",
    "DAGEdge",
    "DAGNode",
    "DAGValidationError",
    "NodeType",
    "terminal_kinds",
]
