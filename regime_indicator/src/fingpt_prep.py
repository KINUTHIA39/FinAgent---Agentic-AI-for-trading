"""Turn the Alpaca SPY/QQQ pull into ready-to-score FinGPT prompts.

Everything that needs judgement happens here, on the PC, so the GPU box does
nothing but forward passes and its bill is spent on the GPU alone.

  window      2015-01-01 -> 2025-12-31 by TRADING DAY, the earliest Alpaca goes.
  scope       articles tagged with at most MAX_SYMBOLS tickers. A roundup tagged
              with 40 names is not about the index; SPY/QQQ is incidental to it.
  reposts     the same headline and body re-sent within REPOST_WINDOW is one
              article, not two. Recurring wire headlines ("S&P 500 Index Futures
              Trading Higher") on different days are separate items and stay.
  timing      the existing 16:00 ET rule (src.news.assign_trading_day): after
              the close counts toward the next session. Alpaca stamps real times,
              so the midnight-unknown branch never fires (checked: 0.0%).
  text        headline, then the summary when the body does not already open
              with it, then the body with Benzinga's trailing boilerplate cut.
  length      tokenised with the base model's own tokenizer and cut to fit the
              4,096-token context, so the box never truncates silently.

The prompt is FinGPT's own sentiment template, character for character,
including the trailing space after "Answer:". The full prompt is stored rather
than rebuilt on the VM so the text that was measured is the text that is scored.

Output: data/alpaca/fingpt/prompts.parquet (all) and pilot.parquet (2,000).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .news import assign_trading_day

START, END = "2015-01-01", "2025-12-31"
MAX_SYMBOLS = 4
REPOST_WINDOW = pd.Timedelta("1h")
TOKENIZER = "NousResearch/Llama-2-13b-hf"   # base of the FinGPT LoRA
CONTEXT = 4096
ANSWER_ROOM = 4                              # label tokens after "Answer: "
PILOT_N = 2000
PILOT_LONGEST = 50                           # worst-case lengths, to test memory

TEMPLATE = ("Instruction: What is the sentiment of this news? Please choose an "
            "answer from {{negative/neutral/positive}}\nInput: {text}\nAnswer: ")

SRC = config.ROOT / "data" / "alpaca" / "news_spy_qqq.parquet"
OUT = config.ROOT / "data" / "alpaca" / "fingpt"

# Benzinga closes many pieces with a link-out or a photo credit. Cut from the
# first such marker, but only in the last third of the body, so a body that
# merely mentions "Read Next" early on keeps its substance.
_TAIL = re.compile(
    r"\s*(?:Read Next|Read Also|Also Read|Now Read|See Also|Read More|"
    r"Photo(?:s)? (?:via|courtesy|by|credit|:)|Image (?:via|courtesy|by|sourced|credit)|_{5,}|"
    r"Featured (?:photo|image)|This (?:article|story) was generated)\b.*$",
    re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"\s+")


def _clean_body(body: str) -> str:
    body = _WS.sub(" ", body or "").strip()
    m = _TAIL.search(body, pos=int(len(body) * 2 / 3))
    return body[:m.start()].rstrip() if m else body


def _compose(headline: str, summary: str, body: str) -> str:
    headline = _WS.sub(" ", headline or "").strip()
    summary = _WS.sub(" ", summary or "").strip()
    parts = [headline]
    if len(summary) > 20 and summary[:120] not in body[:600] \
            and summary != headline:
        parts.append(summary)
    if body:
        parts.append(body)
    return "\n\n".join(parts)


def _drop_reposts(d: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    d = d.sort_values("created_at")
    key = d["headline"].str.strip() + "\x00" + d["content_text"].fillna("")
    gap = d.groupby(key)["created_at"].diff()
    repost = gap.notna() & (gap < REPOST_WINDOW)
    return d[~repost], int(repost.sum())


def _calendar() -> pd.DatetimeIndex:
    s = pd.read_parquet(config.ROOT / "data" / "raw"
                        / "yahoo__GSPC_Close__2025-12-31.parquet")
    return pd.DatetimeIndex(s.index).astype("datetime64[ns]")


def _fit_to_context(texts: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Cut each input so BOS + template + input + answer room <= CONTEXT.

    Cuts on a token boundary using the tokenizer's own character offsets, so no
    decode/re-encode round trip can shift the count.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    overhead = len(tok(TEMPLATE.format(text=""))["input_ids"])   # incl. BOS
    budget = CONTEXT - overhead - ANSWER_ROOM

    out, n_tok, cut = [], np.empty(len(texts), int), np.zeros(len(texts), bool)
    for i0 in range(0, len(texts), 2000):
        enc = tok(texts[i0:i0 + 2000], add_special_tokens=False,
                  return_offsets_mapping=True)
        for j, (ids, offs) in enumerate(zip(enc["input_ids"],
                                            enc["offset_mapping"])):
            t = texts[i0 + j]
            if len(ids) > budget:
                t = t[:offs[budget][0]].rstrip()
                cut[i0 + j] = True
            out.append(t)
            n_tok[i0 + j] = overhead + min(len(ids), budget)
    return out, n_tok, cut


def build() -> pd.DataFrame:
    d = pd.read_parquet(SRC)
    n0 = len(d)
    print(f"Alpaca pull: {n0:,} articles")

    d = d[d["n_symbols"] <= MAX_SYMBOLS]
    print(f"  <= {MAX_SYMBOLS} tickers:       {len(d):,}  "
          f"(dropped {n0 - len(d):,} multi-ticker roundups)")

    d, n_rep = _drop_reposts(d)
    print(f"  reposts removed:     {n_rep:,}  -> {len(d):,}")

    d = d.copy()
    d["trading_day"] = assign_trading_day(d["created_at"], _calendar(), "16:00")
    d = d[(d["trading_day"] >= START) & (d["trading_day"] <= END)]
    print(f"  in window {START}..{END} by trading day: {len(d):,}")

    body = [_clean_body(b) if h else "" for b, h
            in zip(d["content_text"].fillna(""), d["has_body"])]
    text = [_compose(h, s, b) for h, s, b
            in zip(d["headline"], d["summary"].fillna(""), body)]

    print(f"  tokenising with {TOKENIZER} ...", flush=True)
    text, n_tok, cut = _fit_to_context(text)

    out = pd.DataFrame({
        "id": d["id"].to_numpy(),
        "created_at": d["created_at"].to_numpy(),
        "trading_day": d["trading_day"].to_numpy(),
        "symbols": d["symbols"].to_numpy(),
        "n_symbols": d["n_symbols"].to_numpy(),
        "has_body": d["has_body"].to_numpy(),
        "headline": d["headline"].to_numpy(),
        "prompt": [TEMPLATE.format(text=t) for t in text],
        "n_tokens": n_tok,
        "truncated": cut,
    }).sort_values("created_at").reset_index(drop=True)
    return out


def pilot(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """2,000 articles: the longest PILOT_LONGEST, plus a year x has_body
    stratified sample so the pilot sees every era and both text types."""
    longest = df.nlargest(PILOT_LONGEST, "n_tokens")
    rest = df.drop(longest.index)
    strata = rest.groupby([rest["trading_day"].dt.year, "has_body"])
    k = PILOT_N - PILOT_LONGEST
    take = (strata.size() / len(rest) * k).round().astype(int).clip(lower=5)
    parts = [g.sample(min(len(g), take[name]), random_state=seed)
             for name, g in strata]
    p = pd.concat([longest] + parts)
    return p.sample(frac=1, random_state=seed).reset_index(drop=True)


def report(df: pd.DataFrame, tok_per_s: float = 2200.0) -> None:
    yr = df["trading_day"].dt.year
    days = df.groupby(yr)["trading_day"].nunique()
    t = pd.DataFrame({
        "articles": df.groupby(yr).size(),
        "with_body_%": (df.groupby(yr)["has_body"].mean() * 100).round(0),
        "days_with_news": days,
        "per_news_day": (df.groupby(yr).size() / days).round(1),
        "Mtokens": (df.groupby(yr)["n_tokens"].sum() / 1e6).round(2),
    })
    print()
    print(t.to_string())
    total = df["n_tokens"].sum()
    print(f"\n  total {len(df):,} prompts   {total / 1e6:.1f}M tokens   "
          f"median {int(df['n_tokens'].median())}  p99 "
          f"{int(df['n_tokens'].quantile(.99))}  max {int(df['n_tokens'].max())}")
    print(f"  cut to fit {CONTEXT} tokens: {int(df['truncated'].sum())}")
    print(f"  rough GPU time at {tok_per_s:,.0f} tokens/s (A10 x2, to be "
          f"measured in the pilot): {total / tok_per_s / 3600:.1f} h")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build()
    report(df)
    df.to_parquet(OUT / "prompts.parquet")
    p = pilot(df)
    p.to_parquet(OUT / "pilot.parquet")
    print(f"\nwritten: {OUT / 'prompts.parquet'}  ({len(df):,})")
    print(f"written: {OUT / 'pilot.parquet'}  ({len(p):,}, "
          f"{p['n_tokens'].sum() / 1e6:.2f}M tokens)")


if __name__ == "__main__":
    sys.exit(main())
