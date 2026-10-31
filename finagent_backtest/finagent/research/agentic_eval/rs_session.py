"""Phase 1: the monthly briefing handed to the RS agent.

Everything here is information the closed-form optimiser structurally cannot
consume. The QP sees a regime label and a set of point-in-time CAPM estimates;
it has no channel for whether the filter has been unstable, whether the two
states are even far enough apart for the regime call to matter, how well the
betas are currently fitting, or how the model's own recent decisions turned out.

That gap is the whole justification for putting an agent here. If the briefing
only restated what the optimiser already uses, the agent could add nothing but
noise.

Rendering is a pure function of the passed context, so briefings regenerate
byte-identically without re-running anything.
"""

from __future__ import annotations

import numpy as np


def render(ctx: dict) -> str:
    """Render one month's briefing. ``ctx`` is built by ``rs_agentic``."""
    pb = ctx["p_bull_path"]                      # oldest -> newest, includes now
    labels = ctx["label_path"]
    flips = sum(1 for a, b in zip(labels, labels[1:]) if a != b)

    e_bull, e_bear = ctx["e_bull"], ctx["e_bear"]
    sd_bull, sd_bear = ctx["sd_bull"], ctx["sd_bear"]
    sep = (e_bull - e_bear) * 100

    lines = [
        f"## RS allocation cycle {ctx['date']:%Y-%m}",
        "",
        "You are setting the decision parameters for a regime-switching optimal",
        "portfolio (Markose & Yang). The optimiser itself is fixed and will run",
        "whatever you specify. Your job is to judge how far the model should be",
        "trusted this month, given the diagnostics below.",
        "",
        "### Regime filter",
        f"- Label this month: **{ctx['label']}**",
        f"- P(bull) now: {pb[-1]:.2f}",
        "- P(bull) last 6 months (oldest first): "
        + ", ".join(f"{v:.2f}" for v in pb),
        f"- Label changes in that window: {flips}",
        "",
        "### Conditional moments the optimiser would use (Eqs. 15-16)",
        f"- If bull: mean {e_bull*100:+.3f}%/mo, sd {sd_bull*100:.2f}%/mo",
        f"- If bear: mean {e_bear*100:+.3f}%/mo, sd {sd_bear*100:.2f}%/mo",
        f"- Separation between the two states: {sep:.3f} percentage points of "
        "monthly mean",
        "",
        "### Estimation health (CAPM, point-in-time)",
        f"- Months of history used: {ctx['n_obs']}",
        f"- Mean R-squared across the eight stocks: {ctx['mean_r2']:.2f}",
        f"- Beta range: {ctx['beta_min']:.2f} to {ctx['beta_max']:.2f}",
        f"- Largest beta move since last month: {ctx['beta_move_name']} "
        f"{ctx['beta_move']:+.2f}",
        "",
        "### What the optimiser wants before any cap",
        f"- Largest single position: {ctx['top_name']} at "
        f"{ctx['top_weight']*100:.0f}%",
        f"- Gross exposure: {ctx['uncon_gross']*100:.0f}%",
        f"- Names above 20%: {ctx['n_above_20']}",
        "",
    ]

    mem = ctx.get("memory")
    if mem and mem["n"] > 0:
        lines += [
            "### How this model's recent decisions turned out",
            f"- Last {mem['n']} months: mean {mem['mean']*100:+.2f}%/mo, "
            f"{mem['wins']} of {mem['n']} positive",
            f"- Most recent month: {mem['last']*100:+.2f}%",
            "",
        ]

    lines += [
        "### Decide",
        "Set `gamma` (higher is more risk averse), `regime_treatment`",
        "(`hard_label` follows the filter, `probability_blend` weights both",
        "states by P(bull)), and `max_position_pct` (a fraction; 1.00 is no cap).",
        "Reply with the JSON object only.",
    ]
    return "\n".join(lines)


def memory_block(returns: list[float], window: int = 12) -> dict:
    """Summarise the model's own recent realised outcomes."""
    r = returns[-window:]
    if not r:
        return {"n": 0}
    return {"n": len(r), "mean": float(np.mean(r)),
            "wins": int(sum(1 for x in r if x > 0)), "last": float(r[-1])}


__all__ = ["render", "memory_block"]
