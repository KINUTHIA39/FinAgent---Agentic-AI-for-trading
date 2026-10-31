"""Daily FinGPT sentiment from the Alpaca SPY/QQQ scores, confidence-weighted.

Each article's FinGPT score is P(positive) - P(negative), renormalised over the
three answer words. `label_mass` is how much of the model's next-token
probability those three words held before renormalising: ~1.0 when FinGPT
clearly wanted to answer, low when it did not. 7.3% of articles sit below 0.5,
and 98% of those carry a full body (median 798 tokens against 67 overall) --
FinGPT was tuned on short financial sentences, and a long article leaves it
unsure how to answer at all.

So the daily mean and dispersion are weighted by `label_mass`. Weights are
normalised within each day (see sentiment._weighted_daily): an article the
model was unsure about counts for less than a cleanly labelled one on the
same day, but a day is never silenced because all its articles were long.

The Loughran-McDonald columns are computed on the same text FinGPT read, so the
lexicon track sees exactly the corpus the model did. Everything downstream
(decay on zero-news days, EWMA, trailing volume z-score) is the existing
sentiment.aggregate_daily, unchanged.

Output: results/fingpt_sentiment_daily.parquet. The FinBERT file
(results/sentiment_daily.parquet) is left untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .sentiment import aggregate_daily, lm_scores

START, END = "2015-01-01", "2025-12-31"
FG = config.ROOT / "data" / "alpaca" / "fingpt"


def _input_text(prompt: pd.Series) -> pd.Series:
    """The article text inside FinGPT's prompt template."""
    return (prompt.str.split("\nInput: ", n=1).str[1]
            .str.rsplit("\nAnswer: ", n=1).str[0])


def load_scored() -> pd.DataFrame:
    s = pd.read_parquet(FG / "fingpt_scores.parquet")
    if not (FG / "prompts.parquet").exists():
        # The Loughran-McDonald columns need the article text, which is
        # Benzinga content and is not redistributed with this repository.
        raise SystemExit(
            "data/alpaca/fingpt/prompts.parquet not found. Article text is not "
            "shipped; rebuild it with a free Alpaca key:\n"
            "  python -m src.alpaca_news\n  python -m src.fingpt_prep\n"
            "results/fingpt_sentiment_daily.parquet is already included.")
    p = pd.read_parquet(FG / "prompts.parquet", columns=["id", "prompt"])
    s = s.merge(p, on="id", how="left")
    assert s["score"].notna().all(), "unscored articles in fingpt_scores"
    lm = lm_scores(_input_text(s["prompt"]).tolist())
    return pd.concat([s.drop(columns="prompt").reset_index(drop=True),
                      lm.reset_index(drop=True)], axis=1)


def calendar() -> pd.DatetimeIndex:
    c = pd.read_parquet(config.ROOT / "data" / "raw"
                        / "yahoo__GSPC_Close__2025-12-31.parquet").index
    c = pd.DatetimeIndex(c).astype("datetime64[ns]")
    return c[(c >= START) & (c <= END)]


def build() -> pd.DataFrame:
    cfg = config.load()
    scored = load_scored()
    cal = calendar()
    daily = aggregate_daily(scored, cal, cfg, score_col="score",
                            name="fingpt", weight_col="label_mass")

    # The unweighted mean, for the record only: how much did weighting move it?
    raw = aggregate_daily(scored, cal, cfg, score_col="score", name="fingpt")
    have = daily["article_count"] > 0
    w, u = daily.loc[have, "mean_fingpt"], raw.loc[have, "mean_fingpt"]
    print(f"articles {len(scored):,}   trading days {len(cal):,}   "
          f"with news {have.mean():.1%}")
    print(f"weighted vs unweighted daily mean: corr {w.corr(u):.4f}   "
          f"mean |diff| {np.abs(w - u).mean():.4f}   "
          f"days moved > 0.10: {(np.abs(w - u) > 0.10).sum()}")
    print(f"daily confidence (mean label_mass): median "
          f"{daily.loc[have, 'fingpt_confidence'].median():.3f}   p05 "
          f"{daily.loc[have, 'fingpt_confidence'].quantile(.05):.3f}")
    yr = daily.loc[have].groupby(daily.loc[have].index.year)
    print(yr[["mean_fingpt", "sentiment_disp", "article_count"]].mean()
          .round(3).to_string())
    return daily


def main() -> None:
    daily = build()
    out = config.paths()["results"] / "fingpt_sentiment_daily.parquet"
    daily.to_parquet(out)
    print(f"written: {out}  ({len(daily):,} days, {daily.shape[1]} columns)")


if __name__ == "__main__":
    main()
