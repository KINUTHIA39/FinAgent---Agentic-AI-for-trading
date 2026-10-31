---
name: momentum
description: Rolling momentum signal using 20/60 bar lookbacks.
  Use when momentum-labelled candidates are present in the scan.
version: 1.0
tokens_estimate: 1100
tools_required: [alpha_agent, risk_agent, backtest_agent]
regime_tags: [bull, bear]
last_updated: 2026-04-22
---

# Momentum Strategy

## Signal Logic
- MOMENTUM_LONG:  close > max(close, 20 bars) AND 60-bar slope > 0
- MOMENTUM_SHORT: close < min(close, 20 bars) AND 60-bar slope < 0
- `signal_strength = tanh( (close / SMA(60)) - 1 )`

## Position Sizing
- Base size: 3% NAV for longs, 2% NAV for shorts.
- Scale by `abs(signal_strength)`, cap at 8% NAV.

## Risk Rules
- Require 60 bars of clean data; otherwise skip.
- Do not trade if realised vol (20-bar) exceeds 40% annualised.
- Always pair with a trailing stop: 3× ATR(20).

## Regime Suitability
- Preferred regimes: bull (longs), bear (shorts).
- Avoid: sideways — momentum strategies underperform in chop.

## Exit Rules
- Trailing stop: 3× ATR(20) from most favourable excursion.
- Hard stop: signal flips (long→short candidate or vice versa).
