# Agentic AI in Trading: FinAgent and the Agentic Regime Indicator

Two research systems that test what large language model (LLM) agents add to
quantitative investing. Both ship with their data, their cached LLM responses
and their dashboards, so anyone can replay the published results on their own
PC without an API key.

| Folder | What it is | Dashboards |
|---|---|---|
| [`finagent_backtest/`](finagent_backtest/README.md) | **FinAgent**: an LLM planner (Claude) that runs an 8-stock US equity portfolio month by month, 2005–2025, plus two controlled experiments (an agent over the regime-switching optimal portfolio, and a swap of the regime signal) | Three Dash apps on `localhost` |
| [`regime_indicator/`](regime_indicator/README.md) | **Agentic Regime Indicator**: a daily S&P 500 regime classifier in which a panel of LLM specialists and a critic review a hidden Markov model, scored against Hamilton (1989); plus a FinBERT vs FinGPT news-sentiment comparison | Static HTML dashboard and deck, plus a FinGPT vs FinBERT dashboard |

The two are linked: FinAgent's regime-swap experiment reads the regime
indicator's output (`finagent_backtest/data/regime/`, described by its
`CONTRACT.md`).

## Headline results

**FinAgent, 8 stocks, 248 monthly decisions (May 2005 to Dec 2025, 10 bp costs)**

| | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| FinAgent (LLM planner) | 24.1% | 1.21 | −36.1% |
| Equal-weight 8 stocks | 23.7% | 1.15 | −43.8% |
| S&P 500 | 9.0% | 0.54 | −52.6% |

- Agent over the regime-switching optimal portfolio (216 months, 2008–2025):
  Sharpe 1.34 against 1.17 for the classical model and 1.29 for a
  deterministic rule. Max drawdown −19% against −29%. The whole run cost $1.09
  in API calls.
- Swapping FinAgent's Hamilton regime signal for the Agentic Regime Indicator
  (225 shared months) lowered Sharpe slightly, from 1.16 to 1.12.

The full numbers, with the tests behind them, are in each project's dashboards.

## Quick start

Python 3.11 or newer. Each project has its own requirements file; a separate
virtual environment per project keeps them apart.

```bash
git clone https://github.com/KINUTHIA39/FinAgent---Agentic-AI-for-trading.git
cd FinAgent---Agentic-AI-for-trading
```

**FinAgent dashboards**

```bash
cd finagent_backtest
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m finagent.research.agentic_eval.dashboard_portfolio    # http://127.0.0.1:8060
python -m finagent.research.agentic_eval.dashboard_rs_agentic   # http://127.0.0.1:8070
python -m finagent.research.agentic_eval.dashboard              # http://127.0.0.1:8050
```

**Regime indicator dashboards**

```bash
cd regime_indicator
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# The contest dashboard and deck are single self-contained HTML files.
# Open them straight in a browser:
#   deck/dashboard.html   day-by-day replay: Hamilton vs the agent panel
#   deck/deck.html        the results deck

python scripts/serve_fingpt.py        # FinGPT vs FinBERT, http://localhost:8050
```

The FinAgent adaptation dashboard and the FinGPT dashboard both default to
port 8050, so run one at a time or pass another port
(`python scripts/serve_fingpt.py 8080`).

## Replaying the results

Every LLM call behind the published numbers is cached in the repository, so
a replay costs nothing and needs no API key. A replay that strays off the
cache stops with a message rather than calling the API.

- FinAgent: see [Replay](finagent_backtest/README.md#replay).
- Regime indicator: see [Replay](regime_indicator/README.md#replay).

The replays were checked from a clean install against the shipped outputs.
Labels match exactly and probabilities agree to about 1e-10. A few binned
calibration statistics can differ in the 4th decimal place across library
versions.

## Tools and technologies

| Area | Used |
|---|---|
| Language and numerics | Python 3.12, pandas, NumPy, SciPy, statsmodels, scikit-learn |
| Regime models | Hamilton (1989) Markov switching (filter, smoother and EM written from scratch, plus statsmodels `MarkovRegression`), Gaussian HMM (hmmlearn), Markose–Yang regime-switching optimal portfolio |
| LLM agents | Anthropic Claude API (`anthropic` SDK): Claude Sonnet 4.6 for the FinAgent planner and the regime-switching agent, Claude Sonnet 5 for the regime panel. Prompt caching, JSON-schema structured output, deterministic fallbacks |
| Agent orchestration | LangGraph (four specialists plus a critic), a DAG planner with schema validation |
| NLP sentiment | FinBERT (`ProsusAI/finbert`) via Hugging Face Transformers and PyTorch; Loughran–McDonald dictionary (`pysentiment2`); FinGPT (`FinGPT/fingpt-sentiment_llama2-13b_lora` on `NousResearch/Llama-2-13b-hf`, merged with PEFT, bf16) |
| Factors | Qlib Alpha158 (standalone implementation), six factors blended per stock |
| Evaluation | Brier score, log loss, Cohen's kappa, calibration and reliability, Diebold–Mariano tests, stationary-bootstrap confidence intervals, Sharpe, drawdown, transaction-cost sensitivity |
| Dashboards | Dash and Plotly; static HTML with inline SVG; vanilla JavaScript with Chart.js |
| Compute | FinGPT scoring on an NVIDIA B200 (192 GB) GPU VM, driven by bash/ssh scripts and `uv`; everything else runs on a laptop CPU |
| Data | Yahoo Finance (`yfinance`), FRED (`fredapi`), Cboe, FNSPID (Hugging Face), Alpaca News API (Benzinga) |

## API keys

None are needed to view the dashboards or replay the results. They are needed
only to make new LLM calls or to re-download source data. Copy the example
file in each project and fill in what you need:

```bash
cp finagent_backtest/.env.example finagent_backtest/.env
cp regime_indicator/.env.example regime_indicator/.env
```

| Key | Needed for | Cost |
|---|---|---|
| `ANTHROPIC_API_KEY` | new agent runs (cache misses) | pay as you go |
| `FRED_API_KEY` | re-downloading FRED series | free |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | downloading the SPY/QQQ news articles | free (paper account) |

## Data

All data used is included except the Alpaca/Benzinga news article text, which
cannot be redistributed. The FinGPT scores are shipped keyed by Alpaca article
ID, and the articles can be downloaded with a **free Alpaca API key**
(`python -m src.alpaca_news` in `regime_indicator/`). Sources, licences and
re-download commands are in [DATA_SOURCES.md](DATA_SOURCES.md).

## Repository layout

```
FinAgent---Agentic-AI-for-trading/
├── README.md, DATA_SOURCES.md, LICENSE
├── finagent_backtest/
│   ├── finagent/core/                 LLM planner, DAG schema, SESSION.md writer
│   ├── finagent/config/               fund mandate and strategy notes the planner reads
│   ├── finagent/research/agentic_eval/ backtest, regimes, signals, experiments, dashboards
│   └── data/
│       ├── agentic_eval/              market data, briefings, plans, results, LLM caches
│       └── regime/                    frozen regime-indicator output (see CONTRACT.md)
└── regime_indicator/
    ├── src/                           pipeline: ingest, features, Hamilton, HMM, agents, sentiment, scoring
    ├── scripts/                       FinGPT run, comparison export, inspection tools
    ├── infra/hostinger/               GPU scoring scripts for FinGPT
    ├── deck/, dist/                   dashboard and deck (self-contained HTML)
    ├── data/                          price/macro cache, FNSPID headlines, LLM cache, FinGPT scores
    └── results/                       every arm's output, transcripts, FinGPT dashboard
```

## Licence

Code is released under the MIT licence (see [LICENSE](LICENSE)). Data keeps
the terms of its source. In particular, the FNSPID headlines, and the files
derived from them, are CC BY-NC 4.0 (non-commercial use only). See
[DATA_SOURCES.md](DATA_SOURCES.md).

This is research code, not investment advice.
