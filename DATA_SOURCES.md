# Data sources and terms

Everything the two projects read is in this repository except the
Alpaca/Benzinga news article text. Each source keeps its owner's terms. The
MIT licence covers the code only.

## Market and macro data

| Source | Series | Files | Terms |
|---|---|---|---|
| Yahoo Finance via `yfinance` | S&P 500 (`^GSPC`), SPY close and volume, RSP close | `regime_indicator/data/raw/yahoo__*.parquet` | Yahoo terms of service. Included for research reproducibility only. |
| Yahoo Finance via `yfinance` | Daily OHLCV for AAPL, MSFT, NVDA, JPM, AMZN, GOOGL, XOM, JNJ; monthly S&P 500 and 13-week T-bill (`^IRX`) | `finagent_backtest/data/agentic_eval/market_data/` | As above |
| Cboe (indices via Yahoo Finance) | VIX, VVIX, SKEW | `regime_indicator/data/raw/yahoo__VIX_*`, `__VVIX_*`, `__SKEW_*` | Cboe index data; included for research reproducibility only |
| Cboe | Equity put/call ratio, 2003–2012 archive | `regime_indicator/data/raw/cboe__equity_put_call__*.parquet` | As above |
| FRED, Federal Reserve Bank of St. Louis | `T10Y3M` (10-year minus 3-month Treasury spread) | `regime_indicator/data/raw/fred__T10Y3M__*.parquet` | Public domain (Board of Governors of the Federal Reserve System) |
| FRED | `NFCI` (Chicago Fed National Financial Conditions Index) | `regime_indicator/data/raw/fred__NFCI__*.parquet` | Federal Reserve Bank of Chicago; free to use with attribution |
| FRED | `BAA10Y` (Moody's Baa yield minus 10-year Treasury) | `regime_indicator/data/raw/fred__BAA10Y__*.parquet` | Moody's data redistributed by FRED with permission; fair use only |

Every file is a point-in-time pull as of 2025-12-31. The pipelines read the
cached copy whenever it exists, which is what makes the replays exact.
Re-downloading gives the current vendor history, which can differ slightly
(for example after a data correction):

```bash
cd regime_indicator && python -m src.ingest                                # needs FRED_API_KEY
cd finagent_backtest && python -m finagent.research.agentic_eval.market_data
```

Delete the cached files first if you want fresh copies. Re-downloading the
FinAgent market data overwrites the shipped files.

## News

### FNSPID headlines (included)

Dong, Fan and Peng (2024), *FNSPID: A Comprehensive Financial News Dataset in
Time Series*, [arXiv:2402.06698](https://arxiv.org/abs/2402.06698), available
on [Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID).

**Licence: CC BY-NC 4.0. Non-commercial use only, with attribution.** The
same terms apply to the files here that contain FNSPID headlines, and to the
sentiment series derived from them:

- `regime_indicator/data/fnspid/`: headlines filtered to the study's tickers, with their FinBERT and Loughran–McDonald scores
- `regime_indicator/results/articles_detail.parquet` and `news_aligned.parquet`
- `regime_indicator/results/sentiment_daily.parquet` and `market_sentiment_daily.parquet`

The agents never saw headline text, only the daily sentiment z-scores, so the
LLM cache and transcripts quote no articles.

The full 23 GB source file is not needed. `python -m src.news --stream`
rebuilds the filtered extract from it if you want to check the filter.

### Alpaca (Benzinga) news, SPY and QQQ, 2015–2025 (article text not included)

The FinGPT run scored 67,624 Alpaca news articles. Article text is Benzinga
content licensed to Alpaca's users, so this repository ships **scores only**,
keyed by Alpaca article ID:

- `regime_indicator/data/alpaca/fingpt/fingpt_scores.parquet`: `id`, timestamps, trading day, symbols, `has_body`, FinGPT probabilities, `label_mass` and `score`
- `regime_indicator/results/fingpt_sentiment_daily.parquet`: the daily series built from them

**The articles can be downloaded with a free Alpaca API key.** Sign up for an
Alpaca paper-trading account, which is free, create API keys, add them to
`regime_indicator/.env`, then:

```bash
cd regime_indicator
python -m src.alpaca_news      # ~3 minutes; writes data/alpaca/news_spy_qqq.parquet
python -m src.fingpt_prep      # rebuilds the FinGPT prompts; join to the scores on `id`
```

The download is not redistributed when you push changes: `.gitignore`
excludes it. Alpaca can edit or withdraw articles, so a fresh pull may differ
from the 2026 pull by a handful of items.

## Model and LLM outputs (included)

| What | Files | Notes |
|---|---|---|
| Claude responses, regime panel | `regime_indicator/data/llm_cache/` (5,196 files) | One JSON file per call, keyed by a hash of model, role, prompt and effort level |
| Claude responses, FinAgent planner | `finagent_backtest/data/agentic_eval/plans/`, `backtest/cache_llm_arm_c/` | Per-month plans and postures |
| Claude responses, regime-switching agent | `finagent_backtest/data/agentic_eval/rs_agent/agent_calls/` | Per-month decisions, with the briefing each one answered |
| FinBERT and FinGPT scores | `regime_indicator/data/fnspid/articles_scored.parquet`, `data/alpaca/fingpt/` | Model weights are not included; they download from Hugging Face on first use |
