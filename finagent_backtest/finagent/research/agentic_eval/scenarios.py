"""Equities scenarios for the agentic-eval harness.

A Scenario is one cell of the experiment grid: a (regime x exposure x recall)
configuration that renders into a SESSION.md the LLM planner reads. The grid is
the input space for RQ1 (adaptation); ablation/adversarial variants drop or
corrupt one channel for RQ2/RQ3.

Substrate = S&P 500 equities (matching the dissertation data + literature).
The regime axis uses the fixed-threshold 2x2 labels from ``regime_2x2``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .regime_2x2 import REGIMES

# Fixed candidate universe (representative S&P 500). Held constant across cells
# so plan variation is attributable to regime/exposure/recall, not candidates.
UNIVERSE: List[Tuple[str, float]] = [
    ("AAPL", 0.41), ("MSFT", 0.33), ("NVDA", 0.29), ("JPM", 0.22),
    ("AMZN", 0.18), ("GOOGL", 0.14), ("XOM", 0.09), ("JNJ", 0.05),
]

# Real per-stock technicals computed from data/dissertation/sp500_daily (read-only),
# as of 2025-02-28. rsi = 14-day RSI; win_rate = trailing 252-day fraction of up-days.
# These are HELD CONSTANT across all cells (a control, not a manipulated variable),
# so they do not affect the regime/exposure/recall attribution; they replace the earlier
# neutral 0.0 placeholders so the candidate block reflects real market data.
_REAL_TECH = {
    "AAPL":  {"rsi": 70.7, "win_rate": 0.579},
    "MSFT":  {"rsi": 37.9, "win_rate": 0.536},
    "NVDA":  {"rsi": 45.0, "win_rate": 0.536},
    "JPM":   {"rsi": 37.2, "win_rate": 0.583},
    "AMZN":  {"rsi": 27.9, "win_rate": 0.524},
    "GOOGL": {"rsi": 24.0, "win_rate": 0.567},
    "XOM":   {"rsi": 60.5, "win_rate": 0.524},
    "JNJ":   {"rsi": 78.5, "win_rate": 0.532},
}

EXPOSURES = ("low", "mid", "near-full")
RECALL_PROFILES = ("winners", "losers", "mixed", "none")

# Direction priors per composite regime (volatility is carried in the tag).
_REGIME_PROBS = {
    "bull-calm":   {"prob_bull": 0.85, "prob_bear": 0.05, "prob_sideways": 0.10},
    "bull-stress": {"prob_bull": 0.70, "prob_bear": 0.20, "prob_sideways": 0.10},
    "bear-calm":   {"prob_bull": 0.10, "prob_bear": 0.80, "prob_sideways": 0.10},
    "bear-stress": {"prob_bull": 0.05, "prob_bear": 0.90, "prob_sideways": 0.05},
}

_EXPOSURE_PORTFOLIO = {
    "low":       {"nav": 1_000_000, "cash": 700_000, "position_count": 5,  "daily_pnl_pct": 0.012},
    "mid":       {"nav": 1_000_000, "cash": 350_000, "position_count": 15, "daily_pnl_pct": 0.000},
    "near-full": {"nav": 1_000_000, "cash": 30_000,  "position_count": 28, "daily_pnl_pct": -0.021},
}


def _recall_text(profile: str) -> str:
    """Render an equities memory-recall block matching the production format."""
    if profile == "none":
        return ""
    header = "=== HISTORICAL CONTEXT FROM KNOWLEDGE GRAPH ===\n\n## Symbol History:"
    end = "\n=== END HISTORICAL CONTEXT ==="
    if profile == "winners":
        lines = [
            "  - AAPL: | 30 trades | WR: 73% (profitable) | Avg P&L: $+1,240 | Slip: 4.1bps | Conf: 1.18x",
            "  - MSFT: | 28 trades | WR: 68% (profitable) | Avg P&L: $+980 | Slip: 3.7bps | Conf: 1.12x",
            "  - NVDA: | 22 trades | WR: 71% (profitable) | Avg P&L: $+2,310 | Slip: 6.0bps | Conf: 1.15x",
        ]
        return header + "\n" + "\n".join(lines) + end
    if profile == "losers":
        lines = [
            "  - AAPL: | 26 trades | WR: 14% (LOSING) | Avg P&L: $-870 | Slip: 9.2bps | Conf: 0.42x",
            "  - MSFT: | 24 trades | WR: 17% (LOSING) | Avg P&L: $-640 | Slip: 7.8bps | Conf: 0.45x",
            "  - NVDA: | 20 trades | WR: 12% (LOSING) | Avg P&L: $-1,520 | Slip: 11.0bps | Conf: 0.39x",
            "\n!! LOW CONFIDENCE (consider skipping): AAPL, MSFT, NVDA",
        ]
        return header + "\n" + "\n".join(lines) + end
    # mixed
    lines = [
        "  - AAPL: | 30 trades | WR: 70% (profitable) | Avg P&L: $+1,100 | Conf: 1.10x",
        "  - MSFT: | 24 trades | WR: 18% (LOSING) | Avg P&L: $-640 | Conf: 0.45x",
        "  - NVDA: | 22 trades | WR: 55% (profitable) | Avg P&L: $+220 | Conf: 0.95x",
        "\n!! LOW CONFIDENCE (consider skipping): MSFT",
    ]
    return header + "\n" + "\n".join(lines) + end


@dataclass(frozen=True)
class Scenario:
    """One experiment cell."""
    regime: str                 # composite: bull-calm | bull-stress | bear-calm | bear-stress
    exposure: str               # low | mid | near-full
    recall_profile: str         # winners | losers | mixed | none
    ablate: Tuple[str, ...] = ()  # channels to drop: "regime","recall","portfolio"
    note: str = ""              # adversarial/probe annotation

    @property
    def label(self) -> str:
        base = f"{self.regime}|{self.exposure}|{self.recall_profile}"
        if self.ablate:
            base += "|ablate=" + "+".join(self.ablate)
        if self.note:
            base += f"|{self.note}"
        return base

    def to_session_inputs(self) -> Dict[str, Any]:
        """Build the (portfolio_state, regime, memory_recall, candidates) the
        SessionWriter needs, applying any ablations."""
        portfolio = dict(_EXPOSURE_PORTFOLIO[self.exposure])
        regime = {"tag": self.regime, **_REGIME_PROBS[self.regime]}
        recall = _recall_text(self.recall_profile)
        candidates = [
            {"symbol": s, "signal": "BUY", "score": sc,
             "rsi": _REAL_TECH[s]["rsi"], "win_rate": _REAL_TECH[s]["win_rate"]}
            for s, sc in UNIVERSE
        ]
        if "portfolio" in self.ablate:
            portfolio = {}
        if "regime" in self.ablate:
            regime = {}
        if "recall" in self.ablate:
            recall = ""
        return {
            "portfolio_state": portfolio,
            "regime": regime,
            "memory_recall": recall,
            "candidates": candidates,
        }


def build_grid() -> List[Scenario]:
    """Full RQ1 grid: 4 regimes x 3 exposures x 4 recall = 48 cells."""
    return [
        Scenario(regime=r, exposure=e, recall_profile=rc)
        for r, e, rc in itertools.product(REGIMES, EXPOSURES, RECALL_PROFILES)
    ]


def build_pilot() -> List[Scenario]:
    """Three contrasting cells for a quick end-to-end sanity check."""
    return [
        Scenario("bull-calm", "low", "winners"),
        Scenario("bear-stress", "near-full", "losers"),
        Scenario("bear-stress", "near-full", "none"),
    ]


def build_surface(recall_profiles: Tuple[str, ...] = ("none", "losers")) -> List[Scenario]:
    """4 regimes x 3 exposures x given recall layers — enough for a populated
    adaptation surface without the full 48-cell grid."""
    return [
        Scenario(regime=r, exposure=e, recall_profile=rc)
        for r in REGIMES for e in EXPOSURES for rc in recall_profiles
    ]


def build_ablation(base: Scenario) -> List[Scenario]:
    """RQ2: the base cell plus one with each context channel dropped."""
    return [base] + [
        Scenario(base.regime, base.exposure, base.recall_profile, ablate=(ch,))
        for ch in ("regime", "recall", "portfolio")
    ]


__all__ = [
    "Scenario", "build_grid", "build_pilot", "build_ablation",
    "UNIVERSE", "EXPOSURES", "RECALL_PROFILES",
]
