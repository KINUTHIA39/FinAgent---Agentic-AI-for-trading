# FinAgent: an LLM-planned 8-stock portfolio

Each month from May 2005 to December 2025, an LLM planner (Claude Sonnet 4.6)
reads a one-page briefing about the market and the fund and returns a plan.
A fixed, deterministic mapping turns the plan into portfolio weights across
eight US large caps: AAPL, MSFT, NVDA, JPM, AMZN, GOOGL, XOM and JNJ. The
study asks whether the agent adds anything over simple, transparent
alternatives, and whether it is autonomous, explainable and reliable while
doing it.

## How one month works

1. **Briefing.** `SESSION.md` is written from point-in-time inputs only:
   - the market regime from a Hamilton (1989) two-state Markov switching
     filter, fitted on 1990–2004 and applied causally;
   - a per-stock score from six Alpha158 factors (ROC60, RSQR60, STD20,
     MAX60, SUMD60, CORR60);
   - the portfolio's NAV, cash and positions;
   - a memory of how each stock's past calls turned out.
2. **Plan.** The planner returns a DAG of agents (data, feature, alpha, risk,
   portfolio, execution, audit) with risk parameters: position cap, sizing,
   flagged names, and reduce-only or halt switches. The DAG is
   schema-validated. Any failure falls back to a safe default plan, and every
   fallback is counted.
3. **Weights.** `posture_to_weights` maps the plan's risk posture and the
   factor scores to target weights.
4. **Outcome.** The portfolio is rebalanced with turnover costs (0, 10 or
   20 bp). The next month's return is realised and each stock's result goes
   into the memory.

The fund mandate and strategy notes the planner reads are in
`finagent/config/`. The full prompt is dumped by
`python -m finagent.research.agentic_eval.dump_prompt`.

## Results

| May 2005 to Dec 2025, 10 bp costs | CAGR | Volatility | Sharpe | Max drawdown | $1 grows to |
|---|---|---|---|---|---|
| FinAgent | 24.1% | 18.0% | 1.21 | −36.1% | $87.4 |
| Equal-weight 8 stocks | 23.7% | 18.6% | 1.15 | −43.8% | $80.5 |
| S&P 500 | 9.0% | 14.9% | 0.54 | −52.6% | $5.9 |

The planner fell back to the default plan in 0 of 248 months.

**Experiment 1: an agent over the regime-switching optimal portfolio**
(216 months, 2008–2025). The classical Markose–Yang model fixes three things
by assumption: risk aversion, a hard 0.5 threshold on the regime probability,
and no concentration limit. Here an agent sets those three each month from
diagnostics about how far the model can be trusted. It is compared with the
untouched model, a deterministic rule that sees the same diagnostics, fixed
caps, and five random-parameter seeds.

| | Sharpe | Max drawdown |
|---|---|---|
| Agent | 1.34 | −19.1% |
| Deterministic rule | 1.29 | −23.0% |
| Constant cap 0.20 | 1.21 | −25.5% |
| Classical regime-switching | 1.17 | −29.3% |
| Random parameters (5 seeds) | 1.00–1.26 | |

The agent's 213 calls cost $1.09 in total.

**Experiment 2: swapping the regime signal** (225 months, May 2005 to Jan
2024). The Hamilton filter was replaced by the Agentic Regime Indicator from
`../regime_indicator`, collapsed to bull or bear, with everything else held
fixed.

| | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| LLM planner with Hamilton regime | 23.2% | 1.16 | −36.1% |
| LLM planner with Agentic AI regime | 22.0% | 1.12 | −37.0% |

The swap did not help. Tab 7 of the portfolio dashboard breaks the result
down.

**Adaptation grid.** The planner was run 25 times on each cell of a
regime × exposure × memory grid, to measure whether its risk posture responds
to market state above run-to-run noise. Results are in
`data/agentic_eval/runs.parquet` and `analysis/`, and are shown in the
adaptation dashboard.

## Dashboards

```bash
pip install -r requirements.txt
python -m finagent.research.agentic_eval.dashboard_portfolio    # http://127.0.0.1:8060
python -m finagent.research.agentic_eval.dashboard_rs_agentic   # http://127.0.0.1:8070
python -m finagent.research.agentic_eval.dashboard              # http://127.0.0.1:8050
```

| Dashboard | Tabs |
|---|---|
| Portfolio (8060) | 1 In action: month scrubber with Play, the briefing the model read and the plan it returned · 2 Autonomy · 3 Trust: reliability, turnover, cost · 4 Performance · 5 AI vs traditional and regime-switching · 6 Planner anatomy · 7 Regime swap |
| Regime-switching agent (8070) | 1 The experiment · 2 Is the edge real? (controls and bootstrap) · 3 What the agent did · 4 Month by month |
| Adaptation (8050) | 3D adaptation surface, heatmap, reproducibility across runs, posture decomposition, reliability and cost |

Run from this folder. All three read only the files in `data/`.

## Replay

None of these needs an API key or costs anything. Each reproduces the shipped
files: the 248 briefings except for their timestamp line, and every number to
floating-point precision.

```bash
M=finagent.research.agentic_eval      # shorthand used below

python -m $M.signals                            # Alpha158 scores    -> market_data/signals_monthly.parquet
python -m $M.regime_hamilton                    # Hamilton filter    -> market_data/regime_hamilton.parquet
python -m $M.backtest --planner mock            # rule-based planner, Hamilton regime
python -m $M.backtest --planner llm --regime arm_c   # LLM planner, Agentic AI regime (225 cached decisions)
python -m $M.regen_sessions                     # rebuild all 248 SESSION.md briefings
python -m $M.rs_optimal                         # classical regime-switching portfolio
python -m $M.rs_agentic --mode fixed            # also: rule, random --seed 0..4, agent (216 cached decisions)
python -m $M.rs_compare                         # compare the arms
python -m $M.analysis                           # adaptation-grid statistics
python -m $M.make_figures                       # wealth / drawdown / relative-strength figure
```

On Windows PowerShell, write the module path out in full instead of using
`$M`.

**What costs money** (needs `ANTHROPIC_API_KEY` in `.env`):

- `backtest --planner llm` with the Hamilton regime re-asks the planner every
  month: about $6 and an hour. The archived run (`backtest/path0.parquet`)
  stored its weights but not its decisions. A new run is a fresh sample and
  will not match it exactly; the dashboards always use the archived run.
- `capture_plans` asks the planner again on the archived briefings (about
  $6). The shipped `plans/` came from this script. They are representative
  plans for each month's briefing, not the exact plans behind `path0`.
- `runner --set grid --k 25` reruns the adaptation grid: 1,200 planner calls,
  roughly $30 at Sonnet 4.6 prices. Start small, and pass `--out`, because
  the default output overwrites the published `runs.parquet`:
  `--set pilot --k 2 --out data/agentic_eval/runs_pilot.parquet`. (`regime_2x2.py`'s classifier
  functions read a VIX/SPY file from an earlier dataset that is not included;
  the grid uses only its regime names.)
- Any `--fresh` or `--overwrite` run ignores the cache by design.

`market_data` re-downloads prices from Yahoo Finance and overwrites the
shipped files. See `../DATA_SOURCES.md`.

## Files

| Path | Contents |
|---|---|
| `finagent/core/` | `dag_planner.py` (LLM planner with safe fallback), `dag_schema.py`, `safe_default_dag.py`, `prompt_builder.py` (writes `SESSION.md`), `claude_client.py` (Anthropic SDK with prompt caching) |
| `finagent/research/agentic_eval/` | `backtest.py`, `regime_hamilton.py`, `regime_provider.py`, `signals.py` and `alpha158.py`, `rs_*.py` (Experiment 1), `dashboard_*.py`, adaptation grid (`scenarios.py`, `harness.py`, `runner.py`, `features.py`, `analysis.py`) |
| `data/agentic_eval/market_data/` | daily OHLCV for the 8 stocks, monthly returns, S&P 500, T-bill rate, Hamilton regime, factor scores |
| `data/agentic_eval/backtest/` | `path0.parquet` (archived LLM run), regime-swap runs, cached LLM decisions, wealth figure |
| `data/agentic_eval/sessions/`, `sessions_arm_c/` | the briefing for every month, under each regime signal |
| `data/agentic_eval/plans/` | the planner's full response for each month: raw text, DAG, posture, tokens |
| `data/agentic_eval/rs_agent/` | Experiment 1 paths and the agent's cached decisions |
| `data/agentic_eval/analysis/` | adaptation-grid statistics, figures, example DAGs and briefings |
| `data/regime/` | the frozen regime-indicator output this folder reads. `CONTRACT.md` documents it, and `regime_reader.py` loads it. |

## Notes

- Only the parts of the original FinAgent codebase that this study uses are
  included. The live trading system (agent pools, MCP servers, Neo4j memory)
  is not.
- `alpha158.py` is a standalone implementation of Qlib's Alpha158 factors that
  runs directly on OHLCV data. The backtest uses six of them.
- The backtest's memory is in-process and resets each run.
