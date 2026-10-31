"""Anthropic Claude client wrapper for the Phase 4 DAG planner.

The planner talks to Claude via any object satisfying
``LLMClient`` (see :mod:`finagent.core.dag_planner`). This module provides
the production implementation: a thin wrapper around the Anthropic SDK
that applies **prompt caching** to the two stable parts of the planning
prompt — the system instructions and the agent-registry description —
so only the volatile compressed SESSION.md is billed at full rate each
cycle.

Two code paths are exposed:

* ``complete(prompt)`` — single concatenated string, legacy path, no caching.
* ``complete_structured(system, cached_context, volatile_context)`` —
  structured path that places a cache breakpoint after ``cached_context``
  so the stable prefix is read from Anthropic's cache on every cycle
  after the first.

The planner will prefer the structured path when the client exposes it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
# DAGs with rich per-node risk-posture params run ~1500 tokens, and verbose
# cycles exceeded the old 2048 budget — truncating the JSON mid-structure and
# forcing a safe-default fallback. 4096 gives ~2.7x headroom over the largest
# observed plan.
DEFAULT_MAX_TOKENS = 4096


class ClaudeClient:
    """Production LLM client with ephemeral prompt caching on stable blocks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 30.0,
    ) -> None:
        """
        Args:
            api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY``.
            model: Claude model id. Defaults to Sonnet 4.6 — a strong
                price/performance point for structured planning.
            max_tokens: Response budget. Planning outputs are small.
            timeout: Per-request timeout in seconds.

        Raises:
            RuntimeError: when neither ``api_key`` nor ``ANTHROPIC_API_KEY``
                is set. The orchestrator checks for this before constructing
                the client and falls back to the safe-default DAG path
                instead of surfacing the error.
        """
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ClaudeClient requires ANTHROPIC_API_KEY (pass api_key or set env var)"
            )

        # Lazy import so the module is importable even in environments
        # where the SDK is not installed — callers get a clear error only
        # if they actually try to construct the client.
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "anthropic SDK not installed; `pip install anthropic`"
            ) from exc

        self._client = Anthropic(api_key=resolved_key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        # Token usage from the most recent call (input/output/cache_*), for cost
        # telemetry in the agentic-eval harness. None until the first call.
        self.last_usage: Optional[dict] = None

    # ------------------------------------------------------------------
    # Primary interface — structured call with prompt caching.
    # ------------------------------------------------------------------
    def complete_structured(
        self,
        system: str,
        cached_context: str,
        volatile_context: str,
    ) -> str:
        """Call Claude with an explicit cache breakpoint after ``cached_context``.

        Layout:
            system blocks:
                [0] system instructions (cacheable)
                [1] cached_context (agent registry — cache_control: ephemeral)
            messages:
                [0] user: volatile_context (compressed SESSION.md + output ask)

        Anthropic's cache boundary includes everything up to and including
        the marked block, so both the system prompt and the registry are
        served from cache after the first call.
        """
        system_blocks = [
            {"type": "text", "text": system},
            {
                "type": "text",
                "text": cached_context,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": volatile_context}],
            )
        except Exception as exc:  # noqa: BLE001 — planner handles failure
            logger.warning("ClaudeClient.complete_structured failed: %s", exc)
            raise

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            }
            logger.debug(
                "Claude tokens — input=%s, cache_read=%s, cache_creation=%s, output=%s",
                self.last_usage["input_tokens"],
                self.last_usage["cache_read_input_tokens"],
                self.last_usage["cache_creation_input_tokens"],
                self.last_usage["output_tokens"],
            )
        return self._join_text(response)

    # ------------------------------------------------------------------
    # Legacy single-prompt path — no caching.
    # ------------------------------------------------------------------
    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._join_text(response)

    # ------------------------------------------------------------------
    @staticmethod
    def _join_text(response: Any) -> str:
        """Concatenate all text blocks from a Messages API response."""
        content = getattr(response, "content", None) or []
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


__all__ = ["ClaudeClient", "DEFAULT_MODEL", "DEFAULT_MAX_TOKENS"]
