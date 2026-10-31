"""Cache-aware system-prompt builder for Claude planning calls.

The fund mandate lives in ``config/FUND.md`` (stable, cacheable).
Strategy rules live in ``config/strategies/*.md`` (slow-moving, cacheable
per strategy). Session context is written fresh each 15-minute cycle via
:class:`SessionWriter` and is the only uncached prompt tail.

The builder returns an Anthropic-compatible system value as a list of
content blocks. The cacheable prefix is marked with ``cache_control
{type: ephemeral}`` so Anthropic's prompt cache can reuse it across
cycles.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

CACHE_BOUNDARY = "__CACHE_BOUNDARY__"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class PromptBuilder:
    """Build system-prompt content blocks with Anthropic cache boundary.

    Layers (matches the blueprint):
      1. Priority resolver   — strategy override > coordinator default
      2. Content factory     — FUND.md + STRATEGY.md
      3. Section registry    — memoised (stable) vs volatile (per-cycle)
      4. Scope loader        — fund > strategy > symbol hierarchy
      5. Memory injector     — SESSION.md written each cycle
      6. Cache boundary split — global prefix | session tail
    """

    def __init__(self, config_dir: Optional[Path | str] = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self._fund_cache: Optional[str] = None
        self._strategy_cache: dict[str, str] = {}

    # ── Layer 4: Scope loader ────────────────────────────────────────
    def _load_fund(self) -> str:
        if self._fund_cache is None:
            path = self.config_dir / "FUND.md"
            self._fund_cache = path.read_text(encoding="utf-8")
        return self._fund_cache

    def _load_strategy(self, strategy: str) -> str:
        if strategy not in self._strategy_cache:
            path = self.config_dir / "strategies" / f"{strategy}.md"
            if path.exists():
                self._strategy_cache[strategy] = path.read_text(encoding="utf-8")
            else:
                logger.debug("PromptBuilder: strategy file missing: %s", path)
                self._strategy_cache[strategy] = ""
        return self._strategy_cache[strategy]

    # ── Layer 5: Memory injector ─────────────────────────────────────
    def _load_session(self) -> str:
        path = self.config_dir / "SESSION.md"
        if not path.exists():
            return "## Session Context\nNo session context available.\n"
        return path.read_text(encoding="utf-8")

    # ── Layer 6: Cache boundary split ────────────────────────────────
    def build_for_claude(
        self,
        strategy: str = "rsi_reversion",
        extra_prefix: Optional[str] = None,
        skills: Optional[list[Any]] = None,
    ) -> list[dict[str, Any]]:
        """Return Anthropic system content blocks with cache boundary.

        If ``skills`` is provided (a list of :class:`finagent.core.skill_loader.Skill`
        instances, or any object with ``.name`` and ``.body``), those bodies
        are concatenated in order and replace the legacy ``strategy`` lookup.
        """
        fund_md = self._load_fund()
        session_md = self._load_session()

        if CACHE_BOUNDARY in fund_md:
            global_prefix, _ = fund_md.split(CACHE_BOUNDARY, 1)
        else:
            global_prefix = fund_md

        prefix_sections: list[str] = [global_prefix.strip()]
        if skills:
            for skill in skills:
                body = getattr(skill, "body", "").strip()
                if body:
                    prefix_sections.append(body)
        else:
            strategy_md = self._load_strategy(strategy)
            if strategy_md.strip():
                prefix_sections.append(strategy_md.strip())
        if extra_prefix:
            prefix_sections.append(extra_prefix.strip())
        cacheable = "\n\n".join(prefix_sections)

        return [
            {
                "type": "text",
                "text": cacheable,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": session_md.strip() or "## Session Context\n(no data)",
            },
        ]

    def build_plain_text(
        self,
        strategy: str = "rsi_reversion",
        skills: Optional[list[Any]] = None,
    ) -> str:
        """Flatten to a single string for legacy callers that can't send blocks."""
        blocks = self.build_for_claude(strategy=strategy, skills=skills)
        return "\n\n".join(b["text"] for b in blocks)

    def invalidate(self) -> None:
        """Drop the in-memory fund/strategy caches (e.g. after config edits)."""
        self._fund_cache = None
        self._strategy_cache.clear()


def _within_month_lines(regime: Mapping[str, Any]) -> list[str]:
    """Optional within-month regime context.

    A regime signal sampled once at month-end hides everything that happened in
    between. Where the provider can describe the whole month it supplies these
    keys and they are rendered; a month-end series such as the Hamilton filter
    supplies none, in which case nothing is emitted and the block renders
    exactly as it always has. That conditionality is deliberate: it keeps
    previously archived briefings reproducible byte for byte.
    """
    out: list[str] = []
    if regime.get("worst_regime") is not None:
        out.append(f"- Worst regime within the month: {regime['worst_regime']}")
    if regime.get("regime_changed") is not None:
        out.append("- Regime changed within the month: "
                   f"{'yes' if regime['regime_changed'] else 'no'}")
    if regime.get("max_prob_bear") is not None:
        out.append("- Highest P(bear) within the month: "
                   f"{_fmt_pct(regime['max_prob_bear'])}")
    if regime.get("days_in_state") is not None:
        out.append(f"- Sessions in current state: {int(regime['days_in_state'])}")
    return out


class SessionWriter:
    """Writes SESSION.md at the start of each scan cycle."""

    def __init__(self, config_dir: Optional[Path | str] = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR

    def write(
        self,
        portfolio_state: Mapping[str, Any],
        regime: Mapping[str, Any],
        memory_recall: str,
        candidates: Iterable[Mapping[str, Any]],
        cycle_id: Optional[str] = None,
        compressed_context: Optional[Any] = None,
    ) -> Path:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cycle_id = cycle_id or f"cycle_{int(time.time())}"

        lines: list[str] = [
            "## Current Session Context",
            f"- Cycle ID: {cycle_id}",
            f"- Timestamp (UTC): {now_iso}",
            "",
            "### Portfolio State",
            f"- NAV: ${_fmt_float(portfolio_state.get('nav'))}",
            f"- Cash: ${_fmt_float(portfolio_state.get('cash'))}",
            f"- Open positions: {int(portfolio_state.get('position_count', 0) or 0)}",
            f"- Daily P&L: {_fmt_pct(portfolio_state.get('daily_pnl_pct'))}",
            "",
            "### Market Regime",
            f"- Regime: {regime.get('tag', 'unknown')}",
            f"- P(bull):     {_fmt_pct(regime.get('prob_bull'))}",
            f"- P(bear):     {_fmt_pct(regime.get('prob_bear'))}",
            f"- P(sideways): {_fmt_pct(regime.get('prob_sideways'))}",
            *_within_month_lines(regime),
            "",
            "### Memory Recall",
            memory_recall.strip() or "(no recall)",
            "",
            "### Current Candidates (first 10)",
        ]

        candidate_rows = list(candidates)[:10]
        if not candidate_rows:
            lines.append("- (no candidates)")
        else:
            for c in candidate_rows:
                symbol = c.get("symbol", "?")
                signal = c.get("signal", "?")
                rsi = c.get("rsi")
                score = c.get("score")
                win_rate = c.get("win_rate")
                lines.append(
                    f"- {symbol}: {signal} | RSI={_fmt_float(rsi, 1)} | "
                    f"Score={_fmt_float(score, 2)} | WinRate={_fmt_pct(win_rate)}"
                )

        if compressed_context is not None:
            render = getattr(compressed_context, "to_markdown", None)
            if callable(render):
                lines.append("")
                lines.append(render().rstrip())

        path = self.config_dir / "SESSION.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return str(value)


__all__ = ["PromptBuilder", "SessionWriter", "CACHE_BOUNDARY", "DEFAULT_CONFIG_DIR"]
