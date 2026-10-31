# Regime Indicator — Consumer Contract

**For a reader who knows nothing about this repository.** You are almost certainly
here because a portfolio system needs a market-regime signal. This file tells you
what the signal is, which file to read, what the numbers mean, what guarantees
hold, and — importantly — what will silently give you a wrong answer.

Everything below is measured from this repo's own output, not asserted.

---

## 1. What this produces

A daily classification of the **S&P 500** into one of three regimes, with a
probability vector and a confidence, for every trading day from **2005-01-03 to
2023-12-29** (4,781 days).

It is a **regime detector only**. There is no portfolio, no position sizing, no
return series, and nothing in this repo has ever been backtested as a strategy.
Every number it emits is a forecast-quality measure.

**The product is Arm C, the agentic detector.** Two further detectors were built
purely as benchmarks, to test whether the agentic approach made sense. They ship
alongside it as reference columns so the comparison travels with the data.

| Arm | Role | What it is |
|---|---|---|
| **C — Agentic** | **the input you consume** | Three-state HMM, plus an LLM agent panel that may damp its confidence or override its label |
| B — HMM | benchmark | The same three-state HMM, without the panel. Isolates what the panel contributes. |
| A — Hamilton | benchmark | Hamilton (1989) two-state Markov switching on S&P 500 log returns. The classical baseline. |

---

## 2. Consume Arm C — and know what the benchmarks said

**Arm C is the designated input.** Two reasons, and the second is the load-bearing
one:

1. **It is the only arm with three genuine states.** Hamilton is a two-state
   model. Its `bull_quiet` / `bull_volatile` split is an artefact of making it
   comparable — the low-volatility mass is divided evenly, so those two columns
   are *identical by construction*:

   ```
   Arm A, 2005-01-03:   p_bull_quiet 0.4930   p_bull_volatile 0.4930
   ```

   If your optimiser wants different allocations for a quiet bull versus a
   choppy one, Arm A structurally cannot express that and Arm C can.

2. **It is far better calibrated than the detector it wraps.** On the days the
   panel acts, reliability improves 0.3015 → 0.1303, and the share of days
   claiming above 99% confidence falls from 21.6% to zero. For an optimiser
   sizing positions off *probabilities*, that matters: Arm B would routinely
   hand you 0.99-confidence calls that are wrong.

### What the benchmarks found — state this, do not hide it

The contest was run to test whether the agentic approach made sense. On
forecast-quality scoring, **it did not win**:

| Metric (full sample, n=4,781) | **C — Agentic** | B — HMM | A — Hamilton |
|---|---|---|---|
| Brier score (lower better) | 0.8336 | 0.8533 | **0.5679** |
| Skill vs persistence | −0.4389 | −0.4730 | **+0.0198** |
| Skill vs climatology | −0.2602 | −0.2901 | **+0.1415** |
| Accuracy | 0.5597 | 0.5597 | **0.5980** |
| Reliability (lower better) | 0.2851 | 0.3062 | **0.0539** |

Hamilton is the only arm that beats both naive baselines. Arm C is worse than
predicting the historical base rate every day, and the gap to Hamilton is
statistically significant (Diebold–Mariano −6.15, p < 0.0001, blocked at the
20-day forward window).

**What Arm C does beat is Arm B**, significantly (DM +6.61, p < 0.0001) — but
agreement between them is κ = 0.999 and it changed only 2 labels out of 582
eligible days. The entire gain is calibration, not information. Reliability
improved while resolution did not (0.0680 → 0.0639). The panel does not know
more than the detector; it knows when the detector does not.

**Consequences for a consumer:**

- Treat Arm C's probabilities as **usable for sizing** (well calibrated) but
  **weak for direction** (worse than base rate on Brier).
- If you want the best directional call available in this repo, `A_hamilton`
  ships alongside — use its `p_bear_raw` column, which is Hamilton's genuine
  two-state high-volatility probability. Consider it as a cross-check.
- Do not represent Arm C as having outperformed the classical benchmark. It did
  not, and the numbers are in this file.

---

## 3. Schema

### Read it with the reader that ships beside it

`regime_reader.py` sits in this same directory. Use it rather than loading the
parquet directly — its guards are the point:

```python
from regime_reader import RegimeReader

r = RegimeReader("data/regime")          # defaults to the C_agentic product
r.on("2020-09-03")                       # one session; RAISES outside the window
r.between("2020-09-01", "2020-09-30")    # a span, clipped and checked
r.month_summary("2020-09-30")            # what a monthly rebalance should read
r.manifest                               # vintage, record this with results
RegimeReader("data/regime", arm="A_hamilton")   # the benchmark cross-check
```

`month_summary()` returns `label`, the probability vector, `confidence`,
`days_in_state`, `regime_changed`, `worst_regime`, `mean_p_bear_stress` and
`max_p_bear_stress` — the month, not one day of it. See §6 for why that matters.

Pandas only, no other dependencies. It is generated by the exporter, so edit it
in the producing repo and re-export rather than editing it here.

### The underlying file

`regime_daily_v1.parquet`, long format — one row per (trading day, arm),
indexed by `Date` (naive timestamp, US/Eastern trading calendar).

| Column | Type | Meaning |
|---|---|---|
| `Date` | datetime64 (index) | Trading day. Real NYSE sessions only — no weekends, holidays, or synthetic rows. |
| `arm` | string | `C_agentic` (the input), plus `B_hmm` and `A_hamilton` as reference rows. |
| `label` | string | `bull_quiet`, `bull_volatile`, or `bear_stress` — the argmax of the probability vector. |
| `p_bull_quiet` | float | Filtered probability of the quiet state. Genuine for Arm C. **Artefact for Arm A — see §2.** |
| `p_bull_volatile` | float | Filtered probability of the volatile state. Genuine for Arm C. **Artefact for Arm A — see §2.** |
| `p_bear_stress` | float | Filtered probability of the stressed state. |
| `p_bear_raw` | float | **Arm A rows only** (null elsewhere). Hamilton's true two-state high-volatility probability — the benchmark cross-check. |
| `confidence` | float | `max()` of the probability vector. How concentrated the call is — *not* how bearish it is. |

Probabilities sum to 1.0 within each row.

---

## 4. What the state labels mean

In economic terms:

| Label | Meaning |
|---|---|
| `bull_quiet` | Low realized volatility, near the running highs, orderly trading. |
| `bull_volatile` | Still broadly rising, but volatility is elevated and the tape is choppy. |
| `bear_stress` | High volatility together with a meaningful drawdown from the peak. |

### The subtlety that will mislead you if you skip it

**These are relative rankings within a fit window, not absolute thresholds.**

The HMM finds three clusters with no names. A rule then assigns the names by
ranking the fitted state means:

1. highest mean drawdown from the 252-day high → `bear_stress`
2. of the remaining two, lower mean realized volatility → `bull_quiet`
3. the last one → `bull_volatile`

So `bear_stress` means *"whichever of these three clusters has the highest
average drawdown"* — **not** "drawdown worse than 10%". If a fit window contained
three deeply stressed clusters, the worst of them would still be the one called
`bear_stress`.

Do not map these labels onto fixed drawdown or volatility thresholds. They are
ordinal, and their calibration shifts at each annual refit.

### Confidence is a separate axis from the label

`confidence` is `max(p)`. It says how sure the model is, not which way it leans.
A 0.95-confidence `bull_quiet` and a 0.95-confidence `bear_stress` are equally
certain about opposite things. Never use `confidence` as a risk signal.

---

## 5. Point-in-time guarantee

This is the property most easily destroyed by careless consumption, so it is
stated precisely.

**What is guaranteed for every row at date `t`:**

- **Filtered probabilities only.** Computed by forward recursion using data up to
  and including `t`. Smoothed probabilities — which condition on the whole sample
  including the future — are forbidden in code by an assertion, not by convention.
- **Expanding-window standardization.** Every feature z-score at `t` uses the mean
  and standard deviation of history up to `t` only. No full-sample statistics.
- **Walk-forward fits.** Model parameters at `t` come from a fit window that ends
  before the scoring year begins.
- **In-sample-only state naming.** The label-assignment rule (§4) uses statistics
  computed inside the fit window, never the full sample.
- **Publication lags applied.** Weekly series such as NFCI are aligned to their
  real publication date, not their reference date.
- **News aligned conservatively.** A headline published at or after the 16:00 ET
  close is credited to the *next* session. Headlines with an unknown timestamp
  (midnight UTC, ~8.7% of the corpus) are also pushed to the next session.

**What is NOT guaranteed:**

- The file is a **frozen snapshot**. It does not update itself. See §7.
- Nothing beyond `2023-12-29` exists. Reading past it is an error, not a gap.
- Arm C's labels came from LLM calls and are **not bit-reproducible** on a re-run.
  This is why the artifact is frozen rather than regenerated per backtest.

---

## 6. Refit cadence

- **Annual.** Each scoring year uses a model fitted on data ending before that
  year starts. 19 fits across the sample, logged in
  `results/arm_a_fit_log.json` and `results/arm_b_fit_log.json`.
- Every row carries `model_id`, `fit_start` and `fit_end` in the source parquet
  so you can trace which fit produced it.
- The regime series itself is **daily**. Do not downsample it to your rebalance
  frequency — 71% of Arm A's regime episodes are shorter than one month, and
  month-end sampling makes 78% of its transitions invisible. Notably, Hamilton
  flagged the GFC on 2007-10-19 while the 2007-10-31 month-end reading was
  `bull_quiet`. Read the whole month, not one day of it.

---

## 7. Regenerating

```bash
# from the repository root
cd regime_indicator

python -m src.ingest        # market data -> point-in-time panel
python -m src.features      # features + validation gate
python -m src.benchmark     # Arm A
python -m src.detector      # Arm B
python -m src.agents        # Arm C  (replays free from data/llm_cache; a cold run is ~2,910 LLM calls)
python -m src.contest       # score everything
python -m src.export_regime --to ../finagent_backtest/data/regime
```

A full rebuild from scratch takes hours and requires `ANTHROPIC_API_KEY` and
`FRED_API_KEY`. **You should not need to do this.** Consume the exported
artifact; regenerate only when the underlying study changes.

Every export writes a `manifest.json` alongside the parquet recording
`built_at`, `score_start`, `score_end`, `n_days` and a config hash. Record the
manifest with any backtest result so the regime vintage is identifiable.

---

## 8. What not to do

Ranked by how much damage it causes.

1. **Do not call this repo's pipeline live inside a backtest.** Arm C is
   LLM-driven and non-deterministic, so results stop being reproducible; each run
   costs ~2,910 API calls; and asking a live pipeline for "the regime on
   2008-10-15" is one careless line away from computing it with data through
   today. Read the frozen artifact.
2. **Do not read past `2023-12-29`.** Reindexing or forward-filling into 2024+
   produces a confident, stale, wrong answer that looks fine. The reader raises
   instead — let it.
3. **Do not use Arm A's `p_bull_quiet` / `p_bull_volatile`.** They are equal by
   construction (§2). Arm A is a two-state model; use `p_bear_raw` for it.
4. **Do not treat labels as fixed thresholds.** They are relative rankings that
   re-calibrate annually (§4).
5. **Do not use `confidence` as a direction or a risk level.** It is `max(p)` (§4).
6. **Do not downsample to monthly** (§6).
7. **Do not import this package into the portfolio repo to get a number.** It
   pulls PyTorch, transformers, hmmlearn, statsmodels, langgraph and the
   Anthropic SDK to deliver three floats.
8. **Do not claim Arm C outperformed the classical benchmark.** It did not (§2).
   The defensible claims are that it is the only arm with three genuine states,
   and that it is materially better calibrated than the detector it wraps.

---

## 9. Known limitations that affect interpretation

- **Three states was a taxonomy choice, not a fitted one.** BIC falls
  monotonically from k=2 to k=6 and never turns, so the data does not select
  three states.
- **The agent panel never convened during 2008.** The HMM sat confidently in
  `bear_stress` throughout, so nothing triggered its gate. Arm C is *identical to
  Arm B* across the entire financial crisis.
- **The news layer adds nothing measurable.** Tested twice — once with 7
  single-stock tickers, once after adding 89,380 market-wide articles. Arm C's
  Brier moved 0.8337 → 0.8336. Any regime signal here comes from price and
  volatility, not sentiment.
- **Arm C's advantage over Arm B is calibration only.** Agreement between them is
  κ = 0.999; it changed 2 labels out of 582 eligible days. Reliability improved
  (0.3015 → 0.1303) while resolution did not (0.0680 → 0.0639). The panel does
  not know more than the detector; it knows when the detector does not.
- **The contamination ablation has not been run.** Prompts carry no dates, but
  whether the panel reasons from evidence or recognises historical periods is
  argued, not yet measured.

---

## 10. Worked example

Reading the signal for 2020-09-03, the first day of the September 2020 tech
correction:

```
Arm A (use this):
  p_bear_raw    0.0108 -> 0.0518 -> 0.9999   over 09-01, 09-02, 09-03
  label         bear_stress

Arm B (detector):
  p_bear_stress 0.9935    <- flipped from 0.97 bull_volatile in one day
  confidence    0.9935

Arm C (panel):
  label         bull_volatile   <- panel overrode the detector
  confidence    0.5961          <- and damped confidence hard

Ground truth:  bear_stress      <- the panel was wrong; -3.5%, -2.8%, -1.8% followed
```

This is a fair illustration of the whole study: the statistical detectors caught
it, the agent panel talked itself out of it, and the damage was limited only
because the panel hedged its confidence rather than committing.

---

*Questions this file does not answer are covered in `regime_indicator/README.md`.*
