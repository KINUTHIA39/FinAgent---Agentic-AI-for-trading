"""LLM-driven DAG planner for the Phase 4 scan cycle.

The planner feeds the compressed ``SESSION.md`` tail plus an agent
registry description to Claude and asks for a DAG in a strict JSON
shape. The response is parsed, validated, and — on any failure — the
deterministic safe-default DAG is returned instead, so the trading loop
never stalls on a bad plan.

This module intentionally does not import the Anthropic SDK at module
scope; the client is injected via :class:`LLMDAGPlanner`'s constructor
so the planner stays unit-testable and works when no API key is set.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from .dag_schema import DAG, DAGValidationError
from .safe_default_dag import build_safe_default_dag

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Minimum surface the planner needs from whatever Claude wrapper is injected.

    Implementations may optionally expose ``complete_structured(system,
    cached_context, volatile_context)`` to enable prompt caching on the
    stable prefix — the planner prefers that path when available.
    """

    def complete(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class PlannerResult:
    """Outcome of a planning attempt."""

    dag: DAG
    used_fallback: bool
    reason: str = ""  # populated when ``used_fallback`` is True


_SYSTEM_PROMPT = (
    "You are the DAG planner for an agentic hedge-fund trading loop. "
    "Read the compressed session state and output a directed-acyclic-graph "
    "describing which agents to run this cycle, in what order, and with what "
    "parameters.\n\n"
    "Return ONE JSON object and nothing else, matching exactly this shape:\n"
    "{\n"
    '  "nodes": [{"id": "<str>", "type": "<data_fetch|feature|signal|risk|portfolio|execution|audit>",'
    ' "agent": "<AgentClassName>", "params": {}}],\n'
    '  "edges": [{"src": "<node_id>", "dst": "<node_id>"}]\n'
    "}\n\n"
    "Rules:\n"
    "- Every DAG must contain at least one `execution` or `audit` terminal node.\n"
    "- No cycles. Every edge must reference declared node ids.\n"
    "- Use agent class names from the provided registry.\n"
    "- Prefer the canonical stage ordering: data_fetch → feature → signal → risk → portfolio → execution → audit.\n"
)


def _build_prompt_parts(session_md: str, registry_desc: str) -> tuple[str, str, str]:
    """Split the planning prompt into (system, cacheable, volatile) blocks.

    ``system`` and ``cacheable`` are byte-stable across cycles, so a
    structured client can place a cache breakpoint between them and the
    volatile block. The legacy ``complete(prompt)`` path simply
    concatenates them.
    """
    system = _SYSTEM_PROMPT
    cached = f"## Agent Registry\n{registry_desc.strip()}\n"
    volatile = (
        f"## Session State (compressed)\n{session_md.strip()}\n\n"
        f"## Output JSON\n"
    )
    return system, cached, volatile


def _render_prompt(session_md: str, registry_desc: str) -> str:
    system, cached, volatile = _build_prompt_parts(session_md, registry_desc)
    return f"{system}\n{cached}\n{volatile}"


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull the first ``{...}`` block out of the model's reply.

    Claude often wraps JSON in fences or preamble; we accept either.
    """
    if raw is None:
        raise DAGValidationError("LLM returned no content")
    txt = raw.strip()
    # Strip code fences if present.
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?", "", txt).rstrip("`").strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(txt)
        if not m:
            raise DAGValidationError("No JSON object found in LLM reply")
        return json.loads(m.group(0))


class LLMDAGPlanner:
    """Plan a per-cycle DAG using Claude with deterministic fallback."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        on_fallback: Optional[Callable[[str, Exception | None], None]] = None,
    ) -> None:
        """
        Args:
            client: Any object exposing ``complete(prompt) -> str``. If ``None``
                the planner *always* falls back to the safe default — useful
                when no API key is configured.
            on_fallback: Optional hook fired whenever the safe default is used.
                Receives ``(reason, exception_or_None)``. Exceptions inside the
                hook are swallowed so planning is never destabilised.
        """
        self._client = client
        self._on_fallback = on_fallback

    def plan(self, session_md: str, registry_desc: str) -> PlannerResult:
        """Return a validated DAG for this cycle.

        Never raises. Any failure — missing client, JSON parse error,
        validation error, client exception — yields the safe-default DAG
        with ``used_fallback=True`` and a populated ``reason``.
        """
        if self._client is None:
            return self._fallback("no_client_configured", None)

        system, cached, volatile = _build_prompt_parts(session_md, registry_desc)
        structured = getattr(self._client, "complete_structured", None)
        try:
            if callable(structured):
                raw = structured(system, cached, volatile)
            else:
                raw = self._client.complete(f"{system}\n{cached}\n{volatile}")
        except Exception as exc:  # noqa: BLE001 — planner must never break the loop
            return self._fallback(f"client_error: {type(exc).__name__}", exc)

        try:
            payload = _extract_json(raw)
            dag = DAG.from_dict(payload, source="llm")
        except DAGValidationError as exc:
            return self._fallback(f"validation_error: {exc}", exc)
        except Exception as exc:  # noqa: BLE001
            return self._fallback(f"parse_error: {type(exc).__name__}: {exc}", exc)

        return PlannerResult(dag=dag, used_fallback=False)

    def _fallback(self, reason: str, exc: Exception | None) -> PlannerResult:
        logger.info("dag_planner falling back to safe default: %s", reason)
        if self._on_fallback is not None:
            try:
                self._on_fallback(reason, exc)
            except Exception as hook_exc:  # noqa: BLE001
                logger.debug("on_fallback hook raised (ignored): %s", hook_exc)
        return PlannerResult(
            dag=build_safe_default_dag(),
            used_fallback=True,
            reason=reason,
        )


__all__ = ["LLMClient", "LLMDAGPlanner", "PlannerResult"]
