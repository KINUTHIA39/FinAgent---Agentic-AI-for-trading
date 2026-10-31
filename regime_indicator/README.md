# Agentic Regime Indicator

A daily S&P 500 market-regime classifier, 2005–2023, built three ways and
scored on one path:

- **Arm A, Hamilton (1989).** A two-state Markov switching model on index
  returns (statsmodels `MarkovRegression`), refit annually and read from its
  filtered probabilities only. This is the benchmark to beat.
- **Arm B, HMM detector.** A three-state Gaussian hidden Markov model
  (hmmlearn) on daily return, 20-day realised volatility, drawdown from the
  252-day high and the variance risk premium, walk-forward with annual refits
  and fixed seeds. Its states are `bull_quiet`, `bull_volatile`
  and `bear_stress`.
- **Arm C, agentic panel.** On the 582 days where the detector looks
  unsettled (the gate), four LLM specialists each give a view: volatility,
  market sentiment, news sentiment and credit conditions. A critic then
  reviews the panel (Claude Sonnet 5, low effort, orchestrated with
  LangGraph). The detector's label changes only if every informative
  specialist agrees and the critic signs off; otherwise the panel can only
  adjust confidence. Prompts carry no calendar dates, so the model cannot
  recall what happened next.

The **target** is known only in hindsight: realised volatility over the next
20 trading days, cut into terciles using expanding (point-in-time)
boundaries. Every arm is scored on Brier score, log loss, accuracy, Cohen's
kappa and calibration. Each is compared against persistence and climatology
baselines, with Diebold–Mariano tests and stationary-bootstrap confidence
intervals.

## Results

| 2005–2023, all 4,781 days | Accuracy | Brier (lower is better) |
|---|---|---|
| A: Hamilton | 59.8% | 0.568 |
| B: HMM detector | 56.0% | 0.853 |
| C: HMM + agent panel | 56.0% | 0.834 |
| Persistence | 58.8% | 0.574 |
| Climatology | 44.0% | 0.648 |

On the 582 gated days, the panel improved the detector's Brier score from
0.870 to 0.708 by lowering its confidence. It changed the label on 2 days. The
binding constraint is the unanimity rule, not the model: most sittings have
at least one dissenting specialist.

### FinBERT vs FinGPT news sentiment

The news specialist originally read FinBERT and Loughran–McDonald tone on
FNSPID headlines. The extension replaces that with FinGPT
(`FinGPT/fingpt-sentiment_llama2-13b_lora`, a LoRA on Llama-2 13B) scoring
Alpaca/Benzinga news tagged SPY or QQQ, 2015–2025.

- **Corpus.** 67,624 articles after removing multi-ticker roundups and
  reposts: 9.8M tokens, with full article bodies where Alpaca has them.
- **Scoring.** One forward pass per article in bf16 on an NVIDIA B200, taking
  753 s at 13,000 tokens/s. The score is P(positive) − P(negative) over the
  three answer words.
  - Pilot check: 95.7% agreement with the model's own greedy answer on
    256 articles.
- **Daily series.** Weighted within each day by how much probability the
  model put on the three answer words, so an article the model could not
  label counts for less.
- **Rerun.** The panel was rerun over 2015–2025 with only the news slice
  changed.

| Gated days, 2016–2023 (n = 289) | Brier |
|---|---|
| Detector | 0.873 |
| Panel with FinBERT news | 0.736 |
| Panel with FinGPT news | 0.733 |

The Diebold–Mariano test gives p = 0.38, so there is no significant
difference. The news votes differed on 137 of 341 days, but both runs changed
the same single label. A bigger news model does not move the result while the
unanimity rule holds.

## Dashboards

| What | How to open |
|---|---|
| Contest dashboard: day-by-day replay of Hamilton vs the agent panel, with transcripts | Open `deck/dashboard.html` in a browser. It is self-contained. |
| Results deck | Open `deck/deck.html` |
| FinGPT vs FinBERT comparison | `python scripts/serve_fingpt.py`, then http://localhost:8050 (port as the first argument) |

`dist/` holds the same two pages wrapped for static hosting.
`python -m src.build_deck` rebuilds `deck/` and `dist/` from `results/`.

## Replay

Run from this folder after `pip install -r requirements.txt`. No API key is
needed. Every LLM call is served from `data/llm_cache/`, and every price
series from `data/raw/`.

```bash
python -m src.ingest            # load the point-in-time panel from data/raw
python -m src.features          # features + validation report
python -m src.sentiment         # FinBERT / LM daily tone (scores cached in data/fnspid)
python -m src.benchmark         # Arm A  -> results/arm_a_hamilton.parquet
python -m src.detector          # Arm B  -> results/arm_b_hmm.parquet
python -m src.agents            # Arm C  -> results/arm_c_agentic.parquet (2,910 cached calls)
python -m src.contest           # score everything -> results/contest.json
python -m src.export            # flat record of every prompt and response
python -m src.build_deck        # rebuild the dashboard and deck

python scripts/run_arm_c_fingpt.py        # FinGPT panel, 2015-2025 (cached)
python scripts/export_fingpt_compare.py   # FinGPT vs FinBERT dashboard data
python -m src.export_regime --to ../finagent_backtest/data/regime   # the artifact FinAgent reads

pytest tests                    # look-ahead checks: HMM, gate, panel rules
```

Each step reproduces the shipped file. Labels match exactly and probabilities
agree to about 1e-10. A few binned calibration figures in `contest.json` can
move in the 4th decimal place with library versions.

Inspection tools:

```bash
python scripts/show_sitting.py 2020-09-03    # one panel sitting, as the agents saw it
python scripts/show_finbert.py 2020-03-16    # FinBERT scores for one day's headlines
python -m src.sentiment --score "Stocks slide as yields jump"   # score your own headline (needs requirements-models.txt)
```

### What needs keys or a GPU

- **New LLM calls** (`ANTHROPIC_API_KEY`). Any change to prompts, features or
  config misses the cache, and so does `scripts/contamination_ablation.py`
  (the date-guessing and dates-on probes). With no key, a cache miss stops
  the run with a message; nothing is written to the cache.
- **Fresh market data** (`FRED_API_KEY`). Delete `data/raw/` and run
  `python -m src.ingest`. `scripts/verify_sources.py` probes every source live.
- **FNSPID from source.** `python -m src.news --stream` re-filters the 23 GB
  file from Hugging Face. It streams the file and never downloads it whole.
- **Rescoring FinBERT.** `pip install -r requirements-models.txt`; a CPU is
  fine.
- **Rescoring FinGPT.** This needs the article text, which is not shipped:
  1. `python -m src.alpaca_news`, with a free Alpaca key
  2. `python -m src.fingpt_prep`
  3. `bash infra/hostinger/run.sh user@host [port] [ssh-key]` on a GPU
     machine running Ubuntu with a recent NVIDIA driver. 13B in bf16 needs
     about 26 GB for the weights, and the default batch size peaked at 56 GB.
     For a smaller card, add `--tokens-per-batch 65536` to the two
     `score_fingpt.py` calls in `pipeline.sh`.
  4. `python -m src.fingpt_daily`

## Files

| Path | Contents |
|---|---|
| `config.yaml` | every parameter: sample, features, HMM, gate, agent model and effort, target, scoring |
| `CONTRACT.md` | the exported regime artifact: schema, guarantees, how to read it safely |
| `src/ingest.py`, `features.py`, `labels.py` | data panel, features, target and baselines |
| `src/benchmark.py`, `detector.py`, `agents.py`, `memory.py` | Arms A, B and C |
| `src/news.py`, `sentiment.py` | FNSPID filtering and trading-day alignment, FinBERT and Loughran–McDonald |
| `src/alpaca_news.py`, `fingpt_prep.py`, `fingpt_daily.py` | the FinGPT news pipeline |
| `src/scoring.py`, `contest.py` | metrics and paired tests |
| `src/figures.py`, `build_deck.py`, `export.py`, `export_regime.py` | dashboards, records and the frozen artifact |
| `infra/hostinger/` | GPU scoring: setup, pipeline, scorer and the local driver script |
| `data/raw/` | Yahoo, Cboe and FRED series as of 2025-12-31 |
| `data/fnspid/` | FNSPID headlines for the study's tickers, plus their scores |
| `data/alpaca/fingpt/` | FinGPT scores by article ID, plus run summaries (no article text) |
| `data/llm_cache/` | every Claude response, keyed by a hash of the prompt |
| `results/` | each arm's output, the contest, transcripts and provenance, daily sentiment, `fingpt/` |

Data sources and their terms are in `../DATA_SOURCES.md`.

## Notes

- Earlier, the project classified G7 currency regimes. That version is not
  included, and look-ahead tests written for it were removed from `tests/`.
- Coverage limits what the news specialist can know. FNSPID has almost no
  headlines for these tickers before 2011, and the Alpaca archive starts in 2015, so FinGPT news is blank until
  its 250-day z-score window fills (late 2015).
