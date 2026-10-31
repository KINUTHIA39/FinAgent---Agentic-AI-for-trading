"""Arm C: LangGraph panel over the HMM output.

    hmm_detect -> agent_panel -> critic -> emit

With a single detector there is no ensemble to arbitrate. The layer's three
jobs are narrower, and the write-up must say so: the claim is "LLM agents add
contextual judgement to a statistical detector", not "LLM agents arbitrate an
ensemble".

  1. Label interpretation -- the panel is shown fit-window statistics for each
     state, so it reasons about "the high-vol state" rather than "state 2"
  2. Confidence adjustment -- up or down on evidence the HMM never sees
  3. Veto -- override the label only when the panel is UNANIMOUS against it and
     the critic concurs, always with a logged rationale

Specialists see STRICTLY NON-OVERLAPPING inputs. Shared inputs produce
correlated opinions and fake consensus, which would make the panel look
decisive while carrying one opinion's worth of information.

COST GATE: the panel runs only on candidate days from detector.candidate_days.
Off-gate, Arm C IS Arm B by construction, which is why scoring reports every
metric restricted to the gated subset as well as over the full sample.

CONTAMINATION: prompts carry no calendar dates and only standardized, unitless
values. Residual risk remains -- a -8 sigma vol print identifies March 2020 with
or without a date stamp -- which is what the ablation is for.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from . import config
from .detector import STATES
from .features import expanding_standardize

load_dotenv(config.ROOT / ".env")

# Strictly non-overlapping. A feature may appear in exactly one row.
#
# Raw article_count is deliberately NOT here. Almost all of its variation is
# FNSPID growing over time rather than the market moving -- 44 articles/day
# through the 2022 bear market against 2.4/day in calm 2013 -- so an agent
# shown the raw count reads the dataset's construction as a crisis signal. The
# trailing z-score (article_count_z) is what the panel sees instead.
AGENT_INPUTS = {
    "volatility": ["realized_vol_", "vol_of_vol", "vix", "variance_risk_premium",
                   "vvix", "skew"],
    "market_sentiment": ["put_call_", "breadth_", "trend_efficiency_",
                         "momentum_"],
    # Two tone tracks, deliberately both. The single-stock series (mean_*) is
    # seven mega-caps; the market-wide series (market_*) is Reuters/Bloomberg
    # wire copy plus SPY and QQQ. They measure different things and behave
    # differently: across the study's episodes the single-stock series ordered
    # China 2015 as MORE positive than any calm year, while the market-wide
    # series orders GFC (-0.106) below calm 2013 (+0.151) below calm 2017
    # (+0.425), which is the ordering theory predicts. Showing the specialist
    # both -- and letting it see them disagree -- carries more information than
    # averaging them into one number.
    "news_sentiment": ["mean_finbert", "mean_lm_tone", "lm_negative_pct",
                       "lm_uncertainty", "sentiment_disp", "article_count_z",
                       "market_finbert", "market_lm",
                       # FinGPT run (news_track: fingpt). Absent from the
                       # FinBERT pool, so the FinBERT slice is unchanged.
                       "mean_fingpt"],
    "credit_conditions": ["credit_", "term_spread", "nfci"],
}

# The FinGPT run has ONE news track: Alpaca/Benzinga articles tagged SPY or QQQ,
# scored by FinGPT and confidence-weighted. The FinBERT question and note talk
# about two tracks, so the FinGPT run swaps in its own wording. Everything else
# the specialist sees -- system prompt, window, z-scoring -- is identical.
FINGPT_NEWS_QUESTION = ("Is the news narrative deteriorating, improving, or "
                        "simply absent? You have one track: market-level news "
                        "on the S&P 500 and Nasdaq-100 (SPY and QQQ).")
FINGPT_NEWS_NOTE = ("mean_fingpt is FinGPT sentiment of market-level news "
                    "(articles tagged SPY or QQQ), weighted by how confidently "
                    "FinGPT answered; mean_lm_tone, lm_negative_pct and "
                    "lm_uncertainty are a finance word list on the same text; "
                    "sentiment_disp is how much the day's articles disagree; "
                    "article_count_z is coverage against the trailing year. "
                    "Where articles_today is false, the values are a decayed "
                    "echo of older news rather than today's; report "
                    "informative=false unless a decayed level is itself "
                    "extreme.")


def _fingpt(cfg: dict) -> bool:
    return cfg["agents"].get("news_track", "finbert") == "fingpt"

AGENT_QUESTION = {
    "volatility": "Is the volatility regime shifting, and in which direction?",
    "market_sentiment": "Is positioning and market breadth confirming the price "
                        "action, or diverging from it?",
    "news_sentiment": "Is the news narrative deteriorating, improving, or "
                      "simply absent? You have two tracks: single-stock "
                      "headlines (mean_*) and market-wide wire copy "
                      "(market_*). Say so if they disagree.",
    "credit_conditions": "Are credit and financial conditions tightening or "
                         "easing?",
}

SYSTEM = """You are the {role} specialist on a US equity regime panel.

The market is in one of three regimes:
  bull_quiet     - low realized volatility, near the highs, orderly trading
  bull_volatile  - still broadly rising, but volatility is elevated and the
                   tape is choppy
  bear_stress    - high volatility and a meaningful drawdown from the peak

You see ONLY your own slice of evidence. Other specialists see different
slices; you must not speculate about theirs. Answer from what you are given.

Your question: {question}

All values are z-scores against an expanding history, so +2.0 means two
standard deviations above everything seen up to that point. Periods are
labelled t, t-1, t-2 relative to the decision day. There are no dates: do not
guess a date or a historical episode, and do not reason from one. If your slice
is uninformative, say so and report low confidence -- a candid abstention is
more useful to the panel than a confident guess."""

CRITIC_SYSTEM = """You are the critic on a US equity regime panel.

You see the statistical detector's proposal, every specialist's opinion, and
the recent history. The specialists each saw one slice; you see all of them.

Your job is to find what is wrong with the proposed call:
  - Is this recency bias? Would the same evidence have looked different a
    month ago?
  - Are the specialists agreeing because the evidence agrees, or because the
    proposal anchored them?
  - What single observation would falsify this call?

Override the detector ONLY when the specialists are unanimous against it AND
you judge their reasoning sound. The detector is a fitted statistical model
with a real track record; the bar for overriding it is high. Prefer adjusting
confidence over changing the label.

There are no dates. Do not guess a date or a historical episode."""


class Opinion(BaseModel):
    """One specialist's read of its own slice."""

    state: Literal["bull_quiet", "bull_volatile", "bear_stress"] = Field(
        description="The regime your slice alone points to.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="How strongly your slice supports that call.")
    confidence_adjustment: Literal["raise", "hold", "lower"] = Field(
        description="Whether the detector should be more or less certain.")
    informative: bool = Field(
        description="False if your slice says nothing useful today.")
    reasoning: str = Field(description="Two sentences maximum.")


class CriticVerdict(BaseModel):
    """The critic's ruling on the panel plus the detector's proposal."""

    verdict: Literal["accept", "adjust_confidence", "override"] = Field(
        description="accept keeps the label and confidence; adjust_confidence "
                    "keeps the label; override replaces the label.")
    final_state: Literal["bull_quiet", "bull_volatile", "bear_stress"]
    confidence_multiplier: float = Field(
        ge=0.25, le=1.5,
        description="Scales the detector's confidence. Below 1.0 expresses doubt.")
    recency_bias_risk: Literal["low", "medium", "high"]
    objection: str = Field(description="The strongest case against the call.")
    falsifier: str = Field(description="One observation that would disprove it.")


# ------------------------------------------------------------------------ views
def build_agent_views(feats: dict, cfg: dict,
                      sentiment: pd.DataFrame | None = None,
                      market: pd.DataFrame | None = None
                      ) -> dict[str, pd.DataFrame]:
    """Per-role standardized feature slices, strictly non-overlapping.

    Standardized expanding, so every value the model sees is unitless and
    comparable -- which is both the anonymisation requirement and what makes a
    z-score interpretable to the panel.

    Only the EWMA sentiment columns are pooled. The raw daily series is far
    noisier and, on the ~43% of days with no news at all, is a decayed echo of
    the last headline rather than an observation; smoothing is what makes the
    decay behave like evidence fading instead of a series of cliffs.
    """
    min_obs = int(cfg["features"]["min_standardize_obs"])
    frames = [feats["volatility"], feats["market_sentiment"],
              feats["trend"], feats["credit"]]
    if sentiment is not None and len(sentiment):
        frames.append(sentiment[[c for c in sentiment.columns
                                 if c.endswith("_ewma")]])
    if market is not None and len(market):
        # The market series is already z-scored WITHIN SOURCE before pooling,
        # because a paired test on shared days found every source pair
        # significantly different at the same market. Expanding-standardizing
        # it again here is harmless and keeps it on the same footing as every
        # other feature the panel sees.
        frames.append(market[[c for c in market.columns
                              if c.endswith("_ewma")]])
    pool = pd.concat(frames, axis=1)
    pool = pool.loc[:, ~pool.columns.duplicated()]

    views, claimed = {}, set()
    for role, prefixes in AGENT_INPUTS.items():
        cols = [c for c in pool.columns
                if any(c.startswith(p) for p in prefixes) and c not in claimed]
        claimed.update(cols)
        views[role] = pd.DataFrame(
            {c: expanding_standardize(pool[c], min_obs) for c in cols})
    return views


def _window(view: pd.DataFrame, day: pd.Timestamp, n: int = 5,
            anonymise: bool = True) -> dict:
    """The last n periods of one slice.

    Anonymised, periods are keyed by offset (t, t-1, ...) and carry only
    z-scores. With `anonymise=False` the keys become ISO dates -- used ONLY by
    the contamination ablation, to measure how much the model gains from being
    told when it is.
    """
    idx = view.index
    pos = idx.get_indexer([day])[0]
    if pos < 0:
        return {}
    lo = max(0, pos - n + 1)
    seg = view.iloc[lo:pos + 1]
    out = {}
    for i in range(lo, pos + 1):
        offset = f"t-{pos - i}" if pos - i else "t"
        key = offset if anonymise else str(idx[i].date())
        out[key] = {c: (None if pd.isna(v) else round(float(v), 2))
                    for c, v in seg.iloc[i - lo].items()}
    return out


# ------------------------------------------------------------------------ cache
def _cache_dir() -> Path:
    d = config.ROOT / "data" / "llm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(model: str, role: str, payload: dict) -> str:
    blob = json.dumps({"m": model, "r": role, "p": payload}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load(key: str):
    p = _cache_dir() / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save(key: str, value: dict) -> None:
    (_cache_dir() / f"{key}.json").write_text(json.dumps(value))


# -------------------------------------------------------------------- LLM calls
_CLIENT = None


class MissingAPIKey(RuntimeError):
    """A cache miss with no key. Fatal, never cached as a failed call."""


def _client():
    global _CLIENT
    if _CLIENT is None:
        from anthropic import AsyncAnthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise MissingAPIKey("ANTHROPIC_API_KEY not found in .env")
        _CLIENT = AsyncAnthropic(max_retries=4)
    return _CLIENT


def _strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema, tightened for structured outputs.

    Structured outputs require `additionalProperties: false` on every object,
    which pydantic does not emit by default. Field descriptions are kept -- they
    are what tells the model to keep `reasoning` to two sentences, and dropping
    them measurably lengthens the output.

    Numeric bounds must be stripped: the API rejects `minimum`/`maximum` on
    number types ("For 'number' type, properties maximum, minimum are not
    supported"). They are folded into the field description so the model still
    sees the range, and pydantic still enforces them on the way back in -- the
    constraint moves from generation-time to validation-time, not away.
    """
    schema = model.model_json_schema()
    bounds = {"minimum": ">=", "maximum": "<=",
              "exclusiveMinimum": ">", "exclusiveMaximum": "<"}

    def tighten(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node.setdefault("required", list(node["properties"]))
            found = [f"{op} {node.pop(k)}" for k, op in bounds.items() if k in node]
            if found:
                desc = node.get("description", "").rstrip(".")
                node["description"] = f"{desc}. Must be {' and '.join(found)}.".lstrip(". ")
            node.pop("title", None)
            for v in node.values():
                tighten(v)
        elif isinstance(node, list):
            for v in node:
                tighten(v)

    tighten(schema)
    return schema


def _fallback(schema: type[BaseModel]) -> dict:
    """Neutral, explicitly-marked result used when a call cannot be parsed.

    Recorded as `_failed` rather than silently substituted: a failure that looks
    like an abstention would quietly bias the panel toward the detector, and the
    failure rate is reported alongside the results.
    """
    if schema is Opinion:
        out = {"state": "bull_quiet", "confidence": 0.0,
               "confidence_adjustment": "hold", "informative": False,
               "reasoning": "call failed; excluded from the panel vote"}
    else:
        out = {"verdict": "accept", "final_state": "bull_quiet",
               "confidence_multiplier": 1.0, "recency_bias_risk": "low",
               "objection": "call failed", "falsifier": "n/a"}
    out["_failed"] = True
    return out


async def _ask(system: str, payload: dict, schema: type[BaseModel],
               role: str, cfg: dict, sem: asyncio.Semaphore,
               attempts: int = 3) -> dict:
    """One structured call, disk-cached by content hash.

    NOTE ON output_config. `messages.parse` sets `output_config.format` itself
    to constrain generation to the schema, so passing our own `output_config`
    alongside it REPLACED that field and left generation unconstrained -- which
    produced degenerate repetition ("}} 0}}}}}}") that failed JSON parsing.

    Dropping `output_config` entirely fixed the parse failures but also dropped
    `effort`, so the run reverted to the default and burned ~710 output tokens
    per call on what is a small judgement over five periods of one feature
    slice. Both settings are therefore passed together on `messages.create`,
    with the schema supplied explicitly, and the JSON validated here.

    Validation errors arrive on HTTP 200, so the SDK's own retry never sees
    them. One unparseable response killed a 3,665-call run; failures are now
    retried and then isolated so a single bad day cannot destroy the batch.
    """
    model = cfg["agents"]["model"]
    effort = cfg["agents"].get("effort", "low")
    # Effort is part of the key: mixing effort levels across the run would make
    # the panel reason harder on some days than others, and the low-effort
    # switch happened partway through -- chronologically, right at the GFC.
    key = _key(model, role, {"s": system, "p": payload, "e": effort})
    if cfg["agents"].get("cache", True):
        hit = _load(key)
        if hit is not None:
            return hit

    fmt = {"type": "json_schema", "schema": _strict_schema(schema)}
    last = None
    for attempt in range(attempts):
        try:
            async with sem:
                resp = await _client().messages.create(
                    model=model,
                    max_tokens=3000,
                    output_config={"effort": effort, "format": fmt},
                    system=[{"type": "text", "text": system,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user",
                               "content": json.dumps(payload, indent=1)}],
                )
            text = next(b.text for b in resp.content if b.type == "text")
            out = schema.model_validate_json(text).model_dump()
            out["_usage"] = {"in": resp.usage.input_tokens,
                             "out": resp.usage.output_tokens,
                             "cache_read": getattr(
                                 resp.usage, "cache_read_input_tokens", 0)}
            # Store the prompt alongside the answer. The key is a one-way hash,
            # so without this the cache records what was said and not what was
            # asked -- and src/export.py has to regenerate every payload to pair
            # them back up. Cheap insurance against that drifting out of sync.
            out["_prompt"] = {"role": role, "system": system, "payload": payload}
            if cfg["agents"].get("cache", True):
                _save(key, out)
            return out
        except MissingAPIKey:
            # A keyless replay that strays off the cache must stop rather than
            # write placeholder answers a later run would read back as real.
            raise SystemExit(
                f"LLM cache miss for the {role} agent and no ANTHROPIC_API_KEY "
                "is set. The shipped data/llm_cache covers the published runs "
                "only; set a key in .env to make new calls.")
        except Exception as exc:                      # noqa: BLE001
            last = exc
            await asyncio.sleep(1.5 * (attempt + 1))

    out = _fallback(schema)
    out["_error"] = f"{type(last).__name__}: {str(last)[:120]}"
    if cfg["agents"].get("cache", True):
        _save(key, out)
    return out


# ------------------------------------------------------------------- adjudication
def apply_panel_verdict(proposal: dict, opinions: dict, critic: dict) -> dict:
    """Combine the detector's proposal with the panel and the critic.

    A veto requires BOTH unanimity among informative specialists AND the critic
    ruling override. Either alone leaves the label untouched and can only move
    confidence -- the detector is a fitted model with a real track record, and
    one dissenting slice is not grounds to discard it.

    Blocked overrides are recorded as `override_blocked_not_unanimous` rather
    than silently downgraded, so how often the critic wanted to override and was
    stopped is visible in the provenance. That count is itself a result.

    Failed calls arrive marked `informative: False` and are excluded from the
    vote, so an API failure cannot manufacture unanimity.
    """
    informative = [o for o in opinions.values() if o.get("informative")]
    votes = {o["state"] for o in informative}
    unanimous_against = (len(informative) >= 2 and len(votes) == 1
                         and votes != {proposal["label"]})

    label, action = proposal["label"], "passthrough"
    if critic["verdict"] == "override" and unanimous_against:
        label, action = critic["final_state"], "veto"
    elif critic["verdict"] == "override":
        action = "override_blocked_not_unanimous"
    elif critic["verdict"] == "adjust_confidence":
        action = "confidence_only"

    conf = float(np.clip(proposal["confidence"] * critic["confidence_multiplier"],
                         1e-3, 1.0))
    return {
        "label": label, "confidence": conf, "action": action,
        "unanimous_against": unanimous_against,
        "panel_votes": {r: o["state"] for r, o in opinions.items()},
        "panel_informative": {r: o.get("informative") for r, o in opinions.items()},
        "n_failed_calls": sum(1 for o in opinions.values() if o.get("_failed")),
        "critic_verdict": critic["verdict"],
        "recency_bias_risk": critic["recency_bias_risk"],
        "objection": critic["objection"], "falsifier": critic["falsifier"],
    }


def rederive_probabilities(base: np.ndarray, label: str,
                           confidence: float) -> np.ndarray:
    """Rebuild the probability vector so it agrees with the emitted label.

    A damped confidence must move the vector TOWARD UNIFORM, not toward zero.
    The first version put `confidence` on the label and split the remainder by
    the detector's relative weights -- which, once the panel damped confidence
    below 0.5, left another class holding more mass than the emitted label. 64
    of 5,261 days came out with a label their own probabilities contradicted,
    so accuracy and Brier were scoring different things.

    Shrinking toward uniform is the right operation: it expresses "less certain"
    while preserving the ordering, and it cannot invert the argmax because
    uniform is the fixed point. Confidence is therefore floored at 1/K.

    On a veto the target label's mass is swapped with the current maximum first,
    so the new label leads before shrinking.
    """
    k = len(STATES)
    unif = 1.0 / k
    p = np.clip(np.asarray(base, dtype=float), 1e-9, None)
    p = p / p.sum()

    i = STATES.index(label)
    j = int(p.argmax())
    if i != j:                                   # veto: make the new label lead
        p[i], p[j] = p[j], p[i]

    target = float(np.clip(confidence, unif + 1e-6, 1.0))
    spread = p.max() - unif
    s = 1.0 if spread <= 1e-9 else (target - unif) / spread
    out = unif + (p - unif) * np.clip(s, 0.0, 1.0 / max(spread, 1e-9))
    out = np.clip(out, 1e-9, None)
    return out / out.sum()


# -------------------------------------------------------------------- the graph
def build_graph(cfg: dict):
    """Assemble the LangGraph state graph: panel -> critic -> emit."""
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict, total=False):
        payloads: dict
        proposal: dict
        memory: dict
        sem: object
        opinions: dict
        critic: dict
        final: dict

    async def agent_panel(state: State) -> dict:
        roles = list(AGENT_INPUTS)
        q = dict(AGENT_QUESTION)
        if _fingpt(cfg):
            q["news_sentiment"] = FINGPT_NEWS_QUESTION
        results = await asyncio.gather(*[
            _ask(SYSTEM.format(role=r, question=q[r]),
                 {"your_evidence": state["payloads"][r],
                  "recent_history": state["memory"]},
                 Opinion, r, cfg, state["sem"])
            for r in roles])
        return {"opinions": dict(zip(roles, results))}

    async def critic(state: State) -> dict:
        v = await _ask(
            CRITIC_SYSTEM,
            {"detector_proposal": state["proposal"],
             "specialist_opinions": {k: {kk: vv for kk, vv in o.items()
                                         if not kk.startswith("_")}
                                     for k, o in state["opinions"].items()},
             "recent_history": state["memory"]},
            CriticVerdict, "critic", cfg, state["sem"])
        return {"critic": v}

    def emit(state: State) -> dict:
        return {"final": apply_panel_verdict(
            state["proposal"], state["opinions"], state["critic"])}

    g = StateGraph(State)
    g.add_node("agent_panel", agent_panel)
    g.add_node("critic", critic)
    g.add_node("emit", emit)
    g.add_edge(START, "agent_panel")
    g.add_edge("agent_panel", "critic")
    g.add_edge("critic", "emit")
    g.add_edge("emit", END)
    return g.compile()


# ------------------------------------------------------------------ walk forward
async def _run_days(days, graph, views, arm_b, history, cfg, sem,
                    news_days: pd.Series | None = None,
                    market_days: pd.Series | None = None):
    async def one(day):
        row = arm_b.loc[day]
        proposal = {
            "label": str(row["label"]),
            "confidence": round(float(row["confidence"]), 4),
            "probabilities": {s: round(float(row[f"p_{s}"]), 4) for s in STATES},
            "state_meaning": {
                "bear_stress": "highest mean drawdown from the running peak in "
                               "the fit window",
                "bull_quiet": "of the two remaining states, the lower mean "
                              "realized volatility",
                "bull_volatile": "the remaining state"},
        }
        anon = bool(cfg["agents"].get("anonymise_dates", True))
        payloads = {r: _window(v, day, anonymise=anon) for r, v in views.items()}

        # The news slice needs to say whether it is looking at today's news or
        # at a decayed echo of the last headline. Without this the agent reads
        # a fading tone value as a fresh observation and reports it as
        # informative, which is exactly the false confidence the panel design
        # is meant to avoid. No count is passed, only the fact -- a raw count
        # would reintroduce the corpus-growth trend the z-score removes.
        if "news_sentiment" in payloads and news_days is not None \
                and _fingpt(cfg):
            payloads["news_sentiment"] = {
                "z_scores": payloads["news_sentiment"],
                "articles_today": bool(news_days.get(day, False)),
                "note": FINGPT_NEWS_NOTE,
            }
        elif "news_sentiment" in payloads and news_days is not None:
            tk = bool(news_days.get(day, False))
            mk = bool(market_days.get(day, False)) if market_days is not None \
                else False
            payloads["news_sentiment"] = {
                "z_scores": payloads["news_sentiment"],
                # Two flags, because the tracks have different coverage: the
                # single-stock corpus starts in 2011, the market-wide one in
                # 2007. Reporting one flag for both would tell the specialist a
                # day had no news when in fact only one track was dark.
                "single_stock_articles_today": tk,
                "market_wide_articles_today": mk,
                "note": "mean_* are single-stock headlines (7 mega-caps); "
                        "market_* are market-wide wire copy plus SPY and QQQ. "
                        "Where a track's flag is false, its values are a "
                        "decayed echo of older headlines rather than today's "
                        "news; weight that track accordingly. If BOTH flags "
                        "are false, report informative=false unless a decayed "
                        "level is itself extreme.",
            }
        out = await graph.ainvoke({
            "payloads": payloads,
            "proposal": proposal, "memory": history.get(day, {}), "sem": sem})
        return day, out

    return await asyncio.gather(*[one(d) for d in days])


def walk_forward(arm_b: pd.DataFrame, feats: dict, cfg: dict,
                 gate_mask: pd.Series, limit: int | None = None,
                 concurrency: int = 8,
                 sentiment: pd.DataFrame | None = None,
                 market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Arm C outputs, same contract as Arms A and B.

    Off-gate days pass the HMM through untouched, which is the design: Arm C
    differs from Arm B only where the panel was allowed to act.
    """
    from .memory import build_history

    views = build_agent_views(feats, cfg, sentiment, market)
    history = build_history(arm_b["label"],
                            arm_b[[f"p_{s}" for s in STATES]],
                            feats["hmm"], cfg)

    news_days = None
    if sentiment is not None and "article_count" in sentiment:
        news_days = sentiment["article_count"].fillna(0) > 0
    market_days = None
    if market is not None and "market_articles" in market:
        market_days = market["market_articles"].fillna(0) > 0

    days = list(arm_b.index[gate_mask.reindex(arm_b.index, fill_value=False)])
    if limit:
        days = days[:limit]

    graph = build_graph(cfg)
    sem = asyncio.Semaphore(concurrency)
    results = asyncio.run(_run_days(days, graph, views, arm_b, history, cfg,
                                    sem, news_days, market_days))

    out = arm_b.copy()
    out["agent_action"] = "off_gate"
    out["critic_verdict"] = None
    out["objection"] = None
    for day, r in results:
        f = r["final"]
        out.loc[day, "label"] = f["label"]
        out.loc[day, "confidence"] = f["confidence"]
        out.loc[day, "agent_action"] = f["action"]
        out.loc[day, "critic_verdict"] = f["critic_verdict"]
        out.loc[day, "objection"] = f["objection"]

    for day, r in results:
        f = r["final"]
        base = np.array([float(arm_b.loc[day, f"p_{s}"]) for s in STATES])
        new = rederive_probabilities(base, f["label"], f["confidence"])
        for s, v in zip(STATES, new):
            out.loc[day, f"p_{s}"] = v

    out.attrs["provenance"] = {str(d.date()): r["final"] for d, r in results}
    out.attrs["n_gated"] = len(days)
    return out


# -------------------------------------------------------------------- checkpoint
def report(arm_c: pd.DataFrame, arm_b: pd.DataFrame, cfg: dict,
           provenance: dict | None = None) -> dict:
    """Build-step-6 checkpoint: what the agentic layer actually did."""
    from .detector import candidate_days

    prov = provenance if provenance is not None else arm_c.attrs.get("provenance", {})
    gate = candidate_days(arm_b, cfg)
    res = {}

    print("=" * 92)
    print("BUILD STEP 6 CHECKPOINT - Arm C (agentic layer)")
    print("=" * 92)
    print(f"\nmodel: {cfg['agents']['model']}   gated days: {int(gate.sum())}   "
          f"panel calls: {int(gate.sum()) * 5}")

    if not prov:
        print("no provenance available")
        return res

    acts = pd.Series([v["action"] for v in prov.values()]).value_counts()
    verd = pd.Series([v["critic_verdict"] for v in prov.values()]).value_counts()
    recency = pd.Series([v.get("recency_bias_risk") for v in prov.values()]).value_counts()
    failed = sum(v.get("n_failed_calls", 0) for v in prov.values())

    print("\n-- what the panel did --")
    for k, v in acts.items():
        print(f"   {k:34s} {v:5d}  ({v / len(prov) * 100:5.1f}%)")
    res["actions"] = acts.to_dict()

    print("\n-- critic verdicts --")
    for k, v in verd.items():
        print(f"   {k:34s} {v:5d}  ({v / len(prov) * 100:5.1f}%)")
    res["critic_verdicts"] = verd.to_dict()

    blocked = int(acts.get("override_blocked_not_unanimous", 0))
    vetoed = int(acts.get("veto", 0))
    print(f"\n   critic wanted to override {blocked + vetoed} times; "
          f"{vetoed} passed the unanimity gate, {blocked} were blocked")
    print(f"   failed calls (isolated, counted as abstentions): {failed}")
    res["vetoes"], res["blocked"], res["failed_calls"] = vetoed, blocked, failed

    print("\n-- critic's recency-bias assessment --")
    for k, v in recency.items():
        print(f"   {k:10s} {v:5d}  ({v / len(prov) * 100:5.1f}%)")

    print("\n-- effect on labels and confidence (gated days only) --")
    idx = arm_c.index[gate]
    changed = int((arm_c.loc[idx, "label"] != arm_b.loc[idx, "label"]).sum())
    cb, cc = arm_b.loc[idx, "confidence"], arm_c.loc[idx, "confidence"]
    print(f"   labels changed:   {changed} of {len(idx)} ({changed / len(idx) * 100:.1f}%)")
    print(f"   mean confidence:  B {cb.mean():.4f}  ->  C {cc.mean():.4f}   "
          f"({(cc.mean() - cb.mean()) / cb.mean() * 100:+.1f}%)")
    print(f"   min confidence:   B {cb.min():.4f}  ->  C {cc.min():.4f}")
    print(f"   share above 0.99: B {(cb > .99).mean() * 100:.1f}%  ->  "
          f"C {(cc > .99).mean() * 100:.1f}%")
    print("\n   The detector's saturation was its biggest scoring problem, so a")
    print("   large downward move here is the mechanism by which Arm C could")
    print("   improve Brier without changing a single label.")
    res["labels_changed"] = changed
    res["conf_b"], res["conf_c"] = float(cb.mean()), float(cc.mean())

    print("\n-- sample critic objections on vetoed days --")
    vet = [(d, v) for d, v in prov.items() if v["action"] == "veto"][:5]
    if not vet:
        print("   none: the panel never cleared the unanimity bar")
    for d, v in vet:
        print(f"   {d}  {arm_b.loc[pd.Timestamp(d), 'label']} -> {v['label']}")
        print(f"      votes: {v['panel_votes']}")
        print(f"      {v['objection'][:160]}")

    return res


if __name__ == "__main__":
    import sys

    from . import config, ingest, features as F, sentiment as SENT
    from .detector import candidate_days

    cfg = config.load()
    res_dir = config.paths()["results"]

    blocks = ingest.build_panel(cfg=cfg)
    feats = F.build_features(blocks, cfg)

    arm_b = pd.read_parquet(res_dir / "arm_b_hmm.parquet")
    gate = candidate_days(arm_b, cfg)

    # Sentiment is optional: without it the news_sentiment specialist simply
    # has no columns and abstains, and the run still produces a valid Arm C.
    sent = None
    p = res_dir / "sentiment_daily.parquet"
    if p.exists():
        sent = pd.read_parquet(p)
    else:
        print("no sentiment_daily.parquet -- single-stock tone unavailable")

    mkt = None
    q = res_dir / "market_sentiment_daily.parquet"
    if q.exists():
        mkt = pd.read_parquet(q)
        have = int((mkt["market_articles"] > 0).sum())
        print(f"market-wide tone: {have:,} days with articles")
    else:
        print("no market_sentiment_daily.parquet -- market tone unavailable")

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
        print(f"SMOKE TEST: first {limit} gated days only")

    arm_c = walk_forward(arm_b, feats, cfg, gate, limit=limit,
                         concurrency=int(cfg["agents"].get("concurrency", 8)),
                         sentiment=sent, market=mkt)

    if limit is None:
        out = res_dir / "arm_c_agentic.parquet"
        arm_c.to_parquet(out)
        with open(res_dir / "arm_c_provenance.json", "w") as fh:
            json.dump(arm_c.attrs.get("provenance", {}), fh, indent=1,
                      default=str)
        print(f"written: {out}")

    report(arm_c, arm_b, cfg, arm_c.attrs.get("provenance"))
