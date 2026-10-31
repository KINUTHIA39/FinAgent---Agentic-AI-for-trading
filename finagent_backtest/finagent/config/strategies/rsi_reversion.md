---
name: rsi_reversion
description: RSI(14) oversold/overbought mean-reversion signal.
  Use when RSI-based BUY or SELL candidates are present in the scan.
version: 1.0
tokens_estimate: 1200
tools_required: [alpha_agent, risk_agent, backtest_agent]
regime_tags: [bull, sideways]
last_updated: 2026-04-22
---

# RSI Mean-Reversion Strategy

## Signal Logic
- BUY:  RSI(14) < 30 AND price > SMA(50)
- SELL: RSI(14) > 70
- `signal_strength = (RSI - 50) / 50`  — normalised to [-1, +1]

## Position Sizing
- Base size: 2% NAV
- Scale by `abs(signal_strength)`, clamped so that no single position
  exceeds 10% NAV.

## Risk Rules
- Do not trade if portfolio VaR(95%) > 4.5% (leaves buffer before the 5%
  hard limit).
- Do not trade if annualised portfolio vol > 22% (22/25 buffer).
- Do not enter if correlated exposure (|ρ| > 0.7) to an existing position
  would push sector concentration above 30%.

## Regime Suitability
- Preferred regimes: bull, sideways.
- Avoid: confirmed bear (`prob_bear > 0.6`).

## Exit Rules
- Stop-loss: 2× ATR(14) on entry bar.
- Take-profit: mean reversion to SMA(20) OR RSI crosses back through 50.
- Time stop: exit after 20 bars if neither target nor stop triggered.
