"""Headline sentiment: FinBERT and Loughran-McDonald, scored independently.

Two methods, deliberately. They fail differently, and disagreement between them
is informative rather than embarrassing:

  FinBERT   a 110M-parameter transformer fine-tuned on the Financial
            PhraseBank. Reads context and negation. Verified here: the label
            order is {0: positive, 1: negative, 2: neutral}, so the signed
            score is P(pos) - P(neg). On a hand-labelled set of 20 news-wire
            headlines it was directionally right 18 times; both misses were
            POSITIVE headlines scored negative ("Microsoft announces record
            quarterly profit" at -0.749). That negative lean on wire phrasing
            is the Financial-PhraseBank domain shift, and it is why the lexicon
            runs alongside rather than being replaced.

  Loughran-McDonald   a finance-specific word list, 86,486 entries. No context,
            no negation, but no domain shift either. It is here because
            general-purpose lexicons systematically mislabel financial
            vocabulary: LIABILITY, TAX, CAPITAL, CRUDE and DEPRECIATION are all
            NEUTRAL in this dictionary and negative in Harvard-IV. Loughran &
            McDonald (2011) found roughly three-quarters of Harvard-IV's
            negative words are not negative in a financial context.

HEADLINES ONLY. Bodies are present in FNSPID but out of scope -- they exceed
FinBERT's 512 tokens and are inconsistently populated across its five sources.

NONE OF THIS REACHES THE HMM. Sentiment feeds the agents only, so that C - B
measures the sentiment-and-agents layer and nothing else.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_WORD = re.compile(r"[A-Za-z][A-Za-z']+")
_LM = None
_FINBERT = None


# ------------------------------------------------------------------ dictionary
def load_lm_dictionary() -> dict[str, set[str]]:
    """Loughran-McDonald master dictionary, from the copy pysentiment2 ships.

    The Notre Dame page serves HTML rather than a stable CSV endpoint, so the
    bundled copy is used instead of scraping a link that moves.
    """
    global _LM
    if _LM is not None:
        return _LM
    import inspect

    import pysentiment2

    path = (Path(inspect.getfile(pysentiment2)).parent / "static" / "LM.csv")
    df = pd.read_csv(path)
    words = df["Word"].astype(str).str.upper()
    _LM = {}
    for cat in ("Negative", "Positive", "Uncertainty", "Litigious",
                "Constraining"):
        flag = pd.to_numeric(df[cat], errors="coerce").fillna(0) != 0
        _LM[cat.lower()] = set(words[flag])
    return _LM


def lm_scores(texts: list[str]) -> pd.DataFrame:
    """Per-headline Loughran-McDonald measures.

    `negative_pct` -- negative words over total words -- is the standard tone
    measure in the accounting literature, and is reported alongside net tone
    because the two behave differently on short text: a headline with one
    negative word and no positive one has net_tone of -1.0 regardless of length,
    whereas negative_pct scales with it.
    """
    lm = load_lm_dictionary()
    neg_d, pos_d, unc_d = lm["negative"], lm["positive"], lm["uncertainty"]
    rows = []
    for t in texts:
        w = _WORD.findall(str(t).upper())
        n = len(w)
        if not n:
            rows.append((0.0, 0.0, 0.0, 0))
            continue
        neg = sum(1 for x in w if x in neg_d)
        pos = sum(1 for x in w if x in pos_d)
        unc = sum(1 for x in w if x in unc_d)
        tone = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
        rows.append((tone, neg / n, unc / n, n))
    return pd.DataFrame(rows, columns=["lm_net_tone", "lm_negative_pct",
                                       "lm_uncertainty_pct", "n_words"])


# --------------------------------------------------------------------- FinBERT
def _finbert():
    """Load once. CPU is fine; a GPU only makes it comfortable."""
    global _FINBERT
    if _FINBERT is None:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        name = config.load()["sentiment"]["finbert_model"]
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        lab = {v.lower(): k for k, v in model.config.id2label.items()}
        # Verified rather than assumed. Getting this backwards inverts every
        # score and shows up only as a result nobody can explain.
        assert "positive" in lab and "negative" in lab, \
            f"unexpected FinBERT labels: {model.config.id2label}"
        _FINBERT = (tok, model, lab["positive"], lab["negative"], torch)
    return _FINBERT


def finbert_scores(texts: list[str], batch_size: int = 64,
                   progress_every: int = 20_000) -> np.ndarray:
    """Signed sentiment in [-1, 1]: P(positive) - P(negative)."""
    tok, model, pos, neg, torch = _finbert()
    out = np.empty(len(texts), dtype=float)
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = [str(x) for x in texts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=512)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)
        out[i:i + len(batch)] = (p[:, pos] - p[:, neg]).numpy()
        if progress_every and i and i % progress_every < batch_size:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"      {i:,}/{len(texts):,} scored  {rate:.0f}/s  "
                  f"eta {(len(texts) - i) / max(rate, 1e-9) / 60:.0f} min",
                  flush=True)
    return out


# ---------------------------------------------------------------- per-article
def score_articles(articles: pd.DataFrame, cfg: dict,
                   force: bool = False) -> pd.DataFrame:
    """Score every headline with both methods. Cached -- FinBERT is the slow part.

    The cache key is a hash of the headline text plus the model name, so adding
    articles rescores only the new ones and changing the model invalidates
    everything, which is the behaviour you want in both cases.
    """
    dest = config.ROOT / "data" / "fnspid" / "articles_scored.parquet"
    titles = articles["Article_title"].astype(str)
    model = cfg["sentiment"]["finbert_model"]
    keys = pd.Series([hashlib.sha256(f"{model}|{t}".encode()).hexdigest()[:20]
                      for t in titles], index=articles.index, name="key")

    cached = pd.DataFrame()
    if dest.exists() and not force:
        cached = pd.read_parquet(dest)
        print(f"   {len(cached):,} headlines already scored")

    known = set(cached["key"]) if len(cached) else set()
    todo_mask = ~keys.isin(known)
    todo = titles[todo_mask]
    if len(todo):
        print(f"   scoring {len(todo):,} new headlines with FinBERT "
              f"({cfg['sentiment']['finbert_model']}, CPU)")
        fb = finbert_scores(todo.tolist(),
                            int(cfg["sentiment"].get("batch_size", 64)))
        lm = lm_scores(todo.tolist())
        fresh = pd.DataFrame({"key": keys[todo_mask].values,
                              "finbert_score": fb})
        fresh = pd.concat([fresh.reset_index(drop=True),
                           lm.reset_index(drop=True)], axis=1)
        cached = pd.concat([cached, fresh], ignore_index=True) if len(cached) \
            else fresh
        cached = cached.drop_duplicates("key")
        cached.to_parquet(dest)
        print(f"   written: {dest}")

    out = articles.copy()
    out["key"] = keys
    return out.merge(cached, on="key", how="left")


# ------------------------------------------------------------ daily aggregation
def _weighted_daily(scored: pd.DataFrame, score_col: str,
                    weight_col: str) -> pd.DataFrame:
    """Per-day weighted mean and weighted SD of the model score.

    The weight only redistributes influence WITHIN a day: a day whose articles
    are all low-confidence still gets their (equally weighted) mean, because the
    weights are normalised per day. What changes is that on a mixed day a long
    article the model was unsure how to label counts for less than a headline
    it labelled cleanly.
    """
    w = scored[weight_col].clip(lower=1e-6)
    x = scored[score_col]
    t = pd.DataFrame({"d": scored["trading_day"], "w": w, "wx": w * x,
                      "wxx": w * x * x})
    g = t.groupby("d")
    sw = g["w"].sum()
    mean = g["wx"].sum() / sw
    var = (g["wxx"].sum() / sw - mean ** 2).clip(lower=0)
    # Weighted SD is 0 for a one-article day, matching the fillna(0) the
    # unweighted path applies to a single-article day's NaN.
    return pd.DataFrame({"mean": mean, "disp": np.sqrt(var),
                         "confidence": g["w"].mean()})


def aggregate_daily(scored: pd.DataFrame, calendar: pd.DatetimeIndex,
                    cfg: dict, score_col: str = "finbert_score",
                    name: str = "finbert",
                    weight_col: str | None = None) -> pd.DataFrame:
    """Collapse articles to one row per trading day, then impute and smooth.

    `article_count` is a feature in its own right, not bookkeeping. News volume
    spikes around regime transitions whatever the tone -- it is an attention
    proxy.

    `sentiment_disp` is the disagreement measure: high dispersion means the
    narrative is contested, which tends to coincide with transitional regimes.
    A pure-price model cannot see it at all.
    """
    n = cfg["news"]
    g = scored.groupby("trading_day")
    mcol = f"mean_{name}"
    daily = pd.DataFrame({
        mcol: g[score_col].mean(),
        "mean_lm_tone": g["lm_net_tone"].mean(),
        "lm_negative_pct": g["lm_negative_pct"].mean(),
        "lm_uncertainty": g["lm_uncertainty_pct"].mean(),
        "article_count": g.size(),
        "sentiment_disp": g[score_col].std(),
    })
    if weight_col is not None:
        wd = _weighted_daily(scored, score_col, weight_col)
        daily[mcol] = wd["mean"]
        daily["sentiment_disp"] = wd["disp"]
        daily[f"{name}_confidence"] = wd["confidence"]
    daily = daily.reindex(pd.DatetimeIndex(calendar))
    daily["article_count"] = daily["article_count"].fillna(0)
    daily["sentiment_disp"] = daily["sentiment_disp"].fillna(0)

    # Zero-news days: decay the last observation toward neutral rather than
    # holding it flat. News impact persists but fades; an indefinite forward
    # fill would keep a stale headline live for months.
    hl = float(n.get("decay_half_life_days", 3))
    tone_cols = [mcol, "mean_lm_tone", "lm_negative_pct",
                 "lm_uncertainty"]
    for c in tone_cols:
        s = daily[c]
        last = s.ffill()
        gap = s.notna().cumsum()
        since = s.isna().groupby(gap).cumsum()
        daily[c] = (last * np.power(0.5, since / hl)).fillna(0.0)

    ew = int(n.get("smoothing_ewma_days", 5))
    for c in tone_cols + ["sentiment_disp"]:
        daily[f"{c}_ewma"] = daily[c].ewm(span=ew, adjust=False).mean()
    daily["article_count_ewma"] = daily["article_count"].ewm(
        span=ew, adjust=False).mean()

    # DETRENDED VOLUME. Raw article_count cannot be shown to an agent as a
    # stress signal, because almost all of its variation is the corpus growing
    # rather than the market moving: 44.0 articles/day through the 2022 bear
    # market against 2.4/day in calm 2013, and 26.1 in COVID against 13.3 in
    # calm 2021. An agent handed the raw count reads "2022 = crisis, 2013 =
    # calm" straight off an artefact of how FNSPID was assembled.
    #
    # The z-score is against a TRAILING window, so it asks the only question
    # that survives the trend: is coverage unusually heavy relative to what
    # this corpus was recently producing? Trailing and shifted by one day, so
    # no part of it is knowable only in hindsight.
    win = int(n.get("volume_baseline_days", 252))
    cnt = daily["article_count"]
    base = cnt.shift(1).rolling(win, min_periods=60)
    daily["article_count_z"] = ((cnt - base.mean()) / base.std()
                                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily["article_count_z_ewma"] = daily["article_count_z"].ewm(
        span=ew, adjust=False).mean()
    return daily


# ------------------------------------------------------------------ checkpoint
def report(daily: pd.DataFrame, cfg: dict, close: pd.Series | None = None) -> dict:
    """Build-step-7 checkpoint."""
    res = {}
    print("=" * 92)
    print("BUILD STEP 7 CHECKPOINT - sentiment")
    print("=" * 92)

    have = daily["article_count"] > 0
    print(f"\ndaily series {daily.index[0].date()} -> {daily.index[-1].date()}  "
          f"n={len(daily)}")
    print(f"days with at least one article: {have.mean() * 100:.1f}%")
    print(f"articles per day: median {daily.loc[have, 'article_count'].median():.0f}  "
          f"mean {daily.loc[have, 'article_count'].mean():.1f}  "
          f"max {daily['article_count'].max():.0f}")
    res["day_coverage"] = float(have.mean())

    print("\n-- distributions on days with news --")
    for c in ("mean_finbert", "mean_lm_tone", "lm_negative_pct",
              "lm_uncertainty", "sentiment_disp"):
        s = daily.loc[have, c]
        print(f"   {c:18s} mean {s.mean():+.4f}  sd {s.std():.4f}  "
              f"p10 {s.quantile(.1):+.4f}  p90 {s.quantile(.9):+.4f}")

    print("\n-- do the two methods agree? --")
    sub = daily.loc[have, ["mean_finbert", "mean_lm_tone"]].dropna()
    rho = float(sub.corr().iloc[0, 1]) if len(sub) > 100 else float("nan")
    res["finbert_lm_corr"] = rho
    print(f"   corr(FinBERT, LM net tone) = {rho:+.3f}")
    print("   Low-to-moderate is expected and is the point: a transformer that")
    print("   reads negation and a context-free word list disagree by design.")

    print("\n-- sanity: sentiment during known stress vs calm --")
    for name, a, b in (("2008 GFC", "2008-09-01", "2008-12-31"),
                       ("2020 COVID", "2020-02-19", "2020-03-23"),
                       ("calm 2017", "2017-01-01", "2017-12-31"),
                       ("calm 2021", "2021-01-01", "2021-12-31")):
        seg = daily.loc[a:b]
        if not len(seg):
            continue
        print(f"   {name:12s} finbert {seg['mean_finbert'].mean():+.3f}   "
              f"LM tone {seg['mean_lm_tone'].mean():+.3f}   "
              f"uncertainty {seg['lm_uncertainty'].mean():.4f}   "
              f"articles/day {seg['article_count'].mean():5.1f}")
    print("   (stress should show lower tone, higher uncertainty, and more"
          " articles)")

    if close is not None:
        r = np.log(close).diff().reindex(daily.index)
        fwd = r.shift(-1)
        print("\n-- does sentiment relate to returns at all? --")
        for c in ("mean_finbert_ewma", "mean_lm_tone_ewma",
                  "lm_uncertainty_ewma", "article_count_ewma"):
            same = daily[c].corr(r)
            nxt = daily[c].corr(fwd)
            print(f"   {c:22s} same-day {same:+.3f}   next-day {nxt:+.3f}")
        print("   Same-day correlation is contemporaneous, not predictive.")
        print("   Next-day is the one that would matter, and near-zero is the")
        print("   honest expectation -- headline tone is not a return forecast.")

    out = config.paths()["results"] / "sentiment_daily.parquet"
    daily.to_parquet(out)
    print(f"\nwritten: {out}")
    return res


def export_article_detail(cfg: dict) -> Path:
    """One row per headline, with its scores attached. For inspection only.

    The pipeline deliberately keeps text and scores apart: articles_scored is
    keyed by a hash of the headline so it can be a pure cache, and the daily
    file is an aggregate with no text in it at all. Neither is readable on its
    own, which makes "why did this day score -0.4?" impossible to answer in a
    parquet viewer. This rejoins them.

    Nothing downstream reads this file. It exists so a human can sort by score,
    look at what FinBERT actually did, and disagree with it.
    """
    from . import ingest, news

    blocks = ingest.build_panel(cfg=cfg)
    cal = blocks["index"].index
    art = news.align(news.load_filtered(cfg), cal, cfg)

    model = cfg["sentiment"]["finbert_model"]
    art = art.copy()
    art["key"] = [hashlib.sha256(f"{model}|{t}".encode()).hexdigest()[:20]
                  for t in art["Article_title"].astype(str)]

    dest = config.ROOT / "data" / "fnspid" / "articles_scored.parquet"
    scored = pd.read_parquet(dest) if dest.exists() else pd.DataFrame()
    out = art.merge(scored, on="key", how="left")

    cols = ["trading_day", "Date", "Stock_symbol", "Publisher", "Article_title",
            "finbert_score", "lm_net_tone", "lm_negative_pct",
            "lm_uncertainty_pct", "n_words", "key"]
    out = out[[c for c in cols if c in out.columns]]
    out = out.sort_values(["trading_day", "Stock_symbol"]).reset_index(drop=True)

    p = config.paths()["results"] / "articles_detail.parquet"
    out.to_parquet(p)

    n_scored = int(out["finbert_score"].notna().sum()) if "finbert_score" in out else 0
    print(f"written: {p}")
    print(f"   {len(out):,} headlines   {n_scored:,} with scores "
          f"({n_scored / max(len(out), 1) * 100:.1f}%)")
    if n_scored < len(out):
        print(f"   {len(out) - n_scored:,} unscored -- run `python -m src.sentiment` "
              f"to score the newly added tickers")
    return p


def score_cli(texts: list[str]) -> None:
    """Score headlines from the command line with both methods side by side.

    Useful for sanity-checking the thing that produced the study's news
    features: FinBERT's negative lean on wire phrasing is easy to assert and
    much more convincing to watch happen on a headline you chose yourself.
    """
    fb = finbert_scores(texts)
    lm = lm_scores(texts)
    print()
    print(f"{'FinBERT':>9}  {'LM tone':>9}  {'LM neg%':>8}  {'words':>5}  headline")
    print("-" * 96)
    for i, t in enumerate(texts):
        f = float(fb[i])
        mark = "+" if f > 0.15 else ("-" if f < -0.15 else " ")
        short = t if len(t) <= 56 else t[:53] + "..."
        print(f"{f:+9.3f}  {lm['lm_net_tone'].iloc[i]:+9.3f}  "
              f"{lm['lm_negative_pct'].iloc[i] * 100:7.1f}%  "
              f"{int(lm['n_words'].iloc[i]):5d}  {mark} {short}")
    print()
    print("FinBERT is P(positive) - P(negative) from a transformer that reads")
    print("negation and context. LM is a finance-specific word count with no")
    print("context at all. They correlate +0.42 over the study corpus, which is")
    print("low on purpose -- they fail differently, and that is the point.")


# ------------------------------------------------------- market-wide track
MARKET_SOURCES = ("_MARKET", "SPY", "QQQ")


def load_market_articles(cfg: dict, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """The three market-wide corpora, stacked, with a `source` column.

    Kept SEPARATE from the single-stock tone. Seven mega-cap tickers and
    market-wide wire copy measure different things, and letting the news agent
    see both -- including where they disagree -- carries more information than
    averaging them into one number.
    """
    from . import news as N

    root = config.ROOT / "data" / "fnspid" / "by_ticker"
    frames = []
    for src in MARKET_SOURCES:
        p = root / f"{src}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d["source"] = src
        frames.append(d)
    if not frames:
        raise FileNotFoundError("no market-wide corpora found")

    out = pd.concat(frames, ignore_index=True)
    out["trading_day"] = N.assign_trading_day(
        out["Date"], calendar, cfg["news"]["market_close_et"])
    out = out.dropna(subset=["trading_day", "Article_title"])
    out = out[out["Article_title"].astype(str).str.len() > 10]
    return out.sort_values(["trading_day", "source"]).reset_index(drop=True)


def aggregate_market_daily(scored: pd.DataFrame, calendar: pd.DatetimeIndex,
                           cfg: dict) -> pd.DataFrame:
    """Daily market-wide tone, Z-SCORED WITHIN SOURCE before pooling.

    A paired test on days where two sources both published found every pair
    significantly different at the same market: _MARKET vs SPY +0.297 (t 4.73),
    _MARKET vs QQQ -0.087 (t -3.01), SPY vs QQQ -0.131 (t -5.56). SPY reads
    most negative and QQQ most positive -- the opposite of the expectation that
    ETF copy would be cheerier than newswire.

    In aggregate the sources largely cancel, so pooling raw would shift the
    series by only 0.04 sd when the mix flips in 2017. But that cancellation is
    an accident of the current SPY/QQQ balance, and the mix is not stable:
    100/0/0 in 2007-2010, 26/40/33 in 2017, 0/57/43 in 2021-2023. Standardizing
    within source removes the dependence rather than relying on it, so a day
    covered only by SPY is comparable to a day covered only by wire copy.

    Expanding, not full-sample: a source's z-score uses only its own history.
    """
    n = cfg["news"]
    min_obs = int(cfg["features"].get("min_standardize_obs", 250))
    cal = pd.DatetimeIndex(calendar)

    per = (scored.groupby(["trading_day", "source"])
           .agg(finbert=("finbert_score", "mean"),
                lm=("lm_net_tone", "mean"),
                n=("finbert_score", "size")).reset_index())

    # Within-source expanding z-score, on that source's own observed days.
    z = []
    for src, g in per.groupby("source"):
        g = g.sort_values("trading_day").copy()
        for col in ("finbert", "lm"):
            mu = g[col].expanding(min_periods=min(min_obs, 60)).mean()
            sd = g[col].expanding(min_periods=min(min_obs, 60)).std()
            g[f"z_{col}"] = ((g[col] - mu) / sd).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
        z.append(g)
    per = pd.concat(z, ignore_index=True)

    daily = per.groupby("trading_day").agg(
        market_finbert=("z_finbert", "mean"),
        market_lm=("z_lm", "mean"),
        market_articles=("n", "sum"),
        market_n_sources=("source", "nunique"))
    # Which corpora contributed, so the mix is visible rather than implicit.
    mix = (per.groupby("trading_day")["source"]
           .apply(lambda s: "+".join(sorted(set(s)))))
    daily["market_source_mix"] = mix

    daily = daily.reindex(cal)
    daily["market_articles"] = daily["market_articles"].fillna(0)
    daily["market_n_sources"] = daily["market_n_sources"].fillna(0)
    daily["market_source_mix"] = daily["market_source_mix"].fillna("none")

    hl = float(n.get("decay_half_life_days", 3))
    for c in ("market_finbert", "market_lm"):
        s = daily[c]
        last = s.ffill()
        gap = s.notna().cumsum()
        since = s.isna().groupby(gap).cumsum()
        daily[c] = (last * np.power(0.5, since / hl)).fillna(0.0)

    ew = int(n.get("smoothing_ewma_days", 5))
    for c in ("market_finbert", "market_lm"):
        daily[f"{c}_ewma"] = daily[c].ewm(span=ew, adjust=False).mean()
    return daily


if __name__ == "__main__":
    import sys

    if "--detail" in sys.argv:
        export_article_detail(config.load())
        raise SystemExit

    if "--score" in sys.argv:
        args = sys.argv[sys.argv.index("--score") + 1:]
        if args:
            score_cli(args)
        else:                                   # read headlines from stdin
            lines = [l.strip() for l in sys.stdin if l.strip()]
            if not lines:
                print("usage: python -m src.sentiment --score \"a headline\" ...")
            else:
                score_cli(lines)
        raise SystemExit

    from . import ingest, news

    cfg = config.load()
    blocks = ingest.build_panel(cfg=cfg)
    cal = blocks["index"].index

    # Single-stock track. articles_filtered also carries the SPY/QQQ blocks,
    # which belong to the market-wide track below, so they are kept out here.
    articles = news.load_filtered(cfg)
    articles = articles[articles["Stock_symbol"].isin(
        cfg["universe"]["news_tickers"])]
    articles = news.align(articles, cal, cfg)
    print(f"aligned articles: {len(articles):,}")
    scored = score_articles(articles, cfg)
    daily = aggregate_daily(scored, cal, cfg)
    report(daily, cfg, close=blocks["index"]["close"])

    # Market-wide track: _MARKET wire copy plus the SPY and QQQ blocks.
    market = aggregate_market_daily(
        score_articles(load_market_articles(cfg, cal), cfg), cal, cfg)
    out = config.paths()["results"] / "market_sentiment_daily.parquet"
    market.to_parquet(out)
    have = int((market["market_articles"] > 0).sum())
    print(f"written: {out}  ({have:,} days with market-wide articles)")
