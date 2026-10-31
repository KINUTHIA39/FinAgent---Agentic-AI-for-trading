---
name: theory_driven
description: Theory-driven alpha signals (momentum + mean-reversion hybrid)
  from the existing alpha_agent_pool theory_driven agents.
version: 1.0
tokens_estimate: 900
tools_required: [alpha_agent, risk_agent, backtest_agent]
regime_tags: [bull, sideways, bear]
last_updated: 2026-04-22
---

# Theory-Driven Composite Strategy

## Signal Logic
Composite of the pool's built-in theory-driven agents:
- `momentum_agent`          (20/60 bar lookback)
- `mean_reversion_agent`    (RSI + z-score of residuals)
Signal is the sign-weighted sum of active sub-signals, normalised to
[-1, +1].

## Position Sizing
- Base size: 2% NAV.
- Additional scaling: `confidence_multiplier` from Neo4j recall (capped 0.5–1.5).

## Risk Rules
- Only trade if at least two sub-signals agree.
- Skip if the memory agent reports `win_rate < 0.45` for the symbol.

## Regime Suitability
- Broad — suitable across bull/sideways/bear but expect lower Sharpe in
  transition regimes.

## Exit Rules
- Inherit from whichever sub-strategy's signal currently dominates.
