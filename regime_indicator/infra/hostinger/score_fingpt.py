"""FinGPT sentiment scoring: one forward pass per article, no text generation.

Score = P(positive) - P(negative), the same definition as the FinBERT score it
replaces. The probabilities are read from the model's next-token distribution
after FinGPT's own prompt, at the three answer tokens it was trained to emit.
Checked against the tokenizer: the prompt ends "Answer:" + a lone "▁" token,
and each answer is a single token -- exactly the sequence FinGPT's training
code builds (prompt and target tokenised separately, then concatenated).

Fast on a single big GPU because:
  - the LoRA is merged into the base weights once, so each pass is a plain
    Llama forward with no adapter overhead;
  - prompts are sorted by length and packed by a token budget, so padding is
    a few percent instead of most of the batch;
  - only the LAST position goes through the vocabulary head (32,000 wide),
    not every position of every prompt;
  - left padding puts every prompt's last token in the same column.

Resumable: results are written in parts; ids already in --out are skipped.

  python score_fingpt.py --input pilot.parquet --out out/pilot --validate 256
  python score_fingpt.py --input prompts.parquet --out out/full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = "NousResearch/Llama-2-13b-hf"
ADAPTER = "FinGPT/fingpt-sentiment_llama2-13b_lora"
LABELS = {"negative": 8178, "neutral": 21104, "positive": 6374}   # ▁negative etc.


def load():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    for w, i in LABELS.items():
        assert tok.encode(w, add_special_tokens=False) == [i], f"label token drift: {w}"
    tok.pad_token = tok.unk_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
    print(f"model ready in {time.time() - t0:.0f}s  "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return tok, model


def batches(lengths: np.ndarray, budget: int, max_rows: int):
    """Indices in descending-length order, packed so rows x longest <= budget.
    Longest first: if anything is going to run out of memory it does so in the
    first few seconds, not after an hour."""
    order = np.argsort(-lengths, kind="stable")
    i = 0
    while i < len(order):
        n = max(1, min(max_rows, budget // int(lengths[order[i]])))
        yield order[i:i + n]
        i += n


@torch.inference_mode()
def score_batch(model, ids: list[list[int]], pad: int) -> np.ndarray:
    L = max(map(len, ids))
    x = torch.full((len(ids), L), pad, dtype=torch.long)
    m = torch.zeros((len(ids), L), dtype=torch.long)
    for r, s in enumerate(ids):
        x[r, L - len(s):] = torch.tensor(s)
        m[r, L - len(s):] = 1
    x, m = x.cuda(non_blocking=True), m.cuda(non_blocking=True)
    pos = (m.cumsum(-1) - 1).clamp_min(0)
    h = model.model(input_ids=x, attention_mask=m, position_ids=pos,
                    use_cache=False).last_hidden_state[:, -1]
    logp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
    lab = logp[:, list(LABELS.values())]                   # neg, neu, pos
    out = torch.cat([torch.softmax(lab, -1), lab.exp().sum(-1, keepdim=True)], -1)
    return out.cpu().numpy()


def score_safe(model, ids: list[list[int]], pad: int) -> np.ndarray:
    """Halve and retry on out-of-memory rather than lose the run."""
    try:
        return score_batch(model, ids, pad)
    except torch.cuda.OutOfMemoryError:
        if len(ids) == 1:
            raise
        torch.cuda.empty_cache()
        print(f"  OOM at {len(ids)} rows x {max(map(len, ids))} tokens; halving", flush=True)
        h = len(ids) // 2
        return np.concatenate([score_safe(model, ids[:h], pad),
                               score_safe(model, ids[h:], pad)])


@torch.inference_mode()
def validate(tok, model, df: pd.DataFrame, n: int) -> dict:
    """Does reading three token probabilities agree with what FinGPT actually
    writes when asked to answer? Greedy generation on a sample, compared with
    the argmax of the three-way probabilities."""
    s = df.sample(min(n, len(df)), random_state=0)
    agree, unparsed = 0, 0
    for p, want in zip(s["prompt"], s["label"]):
        enc = tok(p, return_tensors="pt").to("cuda")
        g = model.generate(**enc, max_new_tokens=4, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        txt = tok.decode(g[0, enc["input_ids"].shape[1]:]).strip().lower()
        got = next((w for w in LABELS if txt.startswith(w)), None)
        unparsed += got is None
        agree += got == want
    return {"validate_n": len(s), "agree": agree / len(s),
            "unparsed": unparsed / len(s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-per-batch", type=int, default=262_144)
    ap.add_argument("--max-rows", type=int, default=1024)
    ap.add_argument("--part-every", type=int, default=40, help="batches per saved part")
    ap.add_argument("--validate", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.input, columns=["id", "prompt"])
    done = set()
    for f in out.glob("part-*.parquet"):
        done |= set(pd.read_parquet(f, columns=["id"])["id"])
    df = df[~df["id"].isin(done)].reset_index(drop=True)
    print(f"{len(df):,} to score ({len(done):,} already done)", flush=True)

    tok, model = load()
    if len(df):
        t0 = time.time()
        ids = tok(df["prompt"].tolist())["input_ids"]
        lengths = np.fromiter(map(len, ids), int, len(ids))
        print(f"tokenised in {time.time() - t0:.0f}s: {lengths.sum() / 1e6:.2f}M tokens, "
              f"max {lengths.max()}", flush=True)

        torch.cuda.reset_peak_memory_stats()
        part = len(list(out.glob("part-*.parquet")))
        buf, n_tok, t0 = [], 0, time.time()
        for b, idx in enumerate(batches(lengths, a.tokens_per_batch, a.max_rows), 1):
            r = score_safe(model, [ids[i] for i in idx], tok.pad_token_id)
            buf.append(pd.DataFrame({"id": df["id"].to_numpy()[idx],
                                     "p_neg": r[:, 0], "p_neu": r[:, 1],
                                     "p_pos": r[:, 2], "label_mass": r[:, 3],
                                     "n_tokens": lengths[idx]}))
            n_tok += int(lengths[idx].sum())
            if b % a.part_every == 0:
                part += 1
                pd.concat(buf).to_parquet(out / f"part-{part:04d}.parquet")
                buf = []
                el = time.time() - t0
                print(f"  {n_tok / 1e6:6.2f}M tokens  {n_tok / el:,.0f} tok/s  "
                      f"eta {(lengths.sum() - n_tok) / (n_tok / el) / 60:.1f} min  "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB",
                      flush=True)
        if buf:
            part += 1
            pd.concat(buf).to_parquet(out / f"part-{part:04d}.parquet")
        el = time.time() - t0
        stats = {"articles": len(df), "tokens": int(lengths.sum()),
                 "seconds": round(el, 1), "tok_per_s": round(n_tok / el),
                 "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(stats), flush=True)

    res = pd.concat([pd.read_parquet(f) for f in sorted(out.glob("part-*.parquet"))])
    res["score"] = res["p_pos"] - res["p_neg"]
    res["label"] = np.array(list(LABELS))[res[["p_neg", "p_neu", "p_pos"]].to_numpy().argmax(1)]
    summary = {"scored": len(res),
               "label_share": res["label"].value_counts(normalize=True).round(3).to_dict(),
               "score_mean": round(float(res["score"].mean()), 3),
               "label_mass_median": round(float(res["label_mass"].median()), 3),
               "label_mass_p05": round(float(res["label_mass"].quantile(.05)), 3)}
    if a.validate:
        full = pd.read_parquet(a.input, columns=["id", "prompt"]).merge(res[["id", "label"]], on="id")
        summary.update(validate(tok, model, full, a.validate))
    print(json.dumps(summary), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
