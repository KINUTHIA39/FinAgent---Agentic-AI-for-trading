"""Rehydrate the full LLM record: what each agent was shown, and what it said.

The response cache is content-addressed -- `sha256(model, role, system, payload,
effort)` -- so the filename is one-way and carries neither the prompt nor the
day it belongs to. That is fine for deduplication and useless for inspection.

Because the key is deterministic and the payload is a pure function of the
feature matrix plus config, every (day, role) pair can be regenerated offline,
hashed, and matched back to its cached response WITHOUT a single API call. This
module does that and emits a flat record table a dashboard can read directly.

Regeneration must mirror agents.walk_forward exactly. If a payload is built even
slightly differently the hash misses and the record shows as uncached, so the
`matched` count in the summary is the check that this stayed in sync.

    python -m src.export
"""
from __future__ import annotations

import json

import pandas as pd

from . import agents, config
from .detector import STATES, candidate_days
from .memory import build_history

SPECIALISTS = list(agents.AGENT_INPUTS)


def _proposal(row: pd.Series) -> dict:
    """The detector proposal exactly as walk_forward builds it."""
    return {
        "label": str(row["label"]),
        "confidence": round(float(row["confidence"]), 4),
        "probabilities": {s: round(float(row[f"p_{s}"]), 4) for s in STATES},
        "state_meaning": {
            "risk_on": "highest carry-to-vol of the two low-volatility states",
            "risk_off": "highest mean realized volatility in the fit window",
            "range": "the remaining low-volatility state"},
    }


def build_records(arm_b: pd.DataFrame, feats: dict, cfg: dict,
                  gate: pd.Series | None = None) -> pd.DataFrame:
    """One row per (day, agent) with prompt, response, tokens and cache status."""
    gate = candidate_days(arm_b, cfg) if gate is None else gate
    views = agents.build_agent_views(feats, cfg)
    history = build_history(arm_b["label"],
                            arm_b[[f"p_{s}" for s in STATES]],
                            feats["hmm"], cfg)

    model = cfg["agents"]["model"]
    effort = cfg["agents"].get("effort", "low")
    anon = bool(cfg["agents"].get("anonymise_dates", True))
    days = list(arm_b.index[gate.reindex(arm_b.index, fill_value=False)])

    rows = []
    for day in days:
        proposal = _proposal(arm_b.loc[day])
        mem = history.get(day, {})
        opinions = {}

        for role in SPECIALISTS:
            system = agents.SYSTEM.format(role=role,
                                          question=agents.AGENT_QUESTION[role])
            payload = {"your_evidence": agents._window(views[role], day,
                                                       anonymise=anon),
                       "recent_history": mem}
            key = agents._key(model, role, {"s": system, "p": payload, "e": effort})
            resp = agents._load(key)
            if resp is not None:
                opinions[role] = resp
            rows.append(_row(day, role, system, payload, resp, key, model, effort))

        # The critic's prompt depends on the specialists' answers, so it can only
        # be rebuilt after they are loaded -- same ordering as the live graph.
        if len(opinions) == len(SPECIALISTS):
            payload = {
                "detector_proposal": proposal,
                "specialist_opinions": {
                    k: {kk: vv for kk, vv in o.items() if not kk.startswith("_")}
                    for k, o in opinions.items()},
                "recent_history": mem}
            key = agents._key(model, "critic",
                              {"s": agents.CRITIC_SYSTEM, "p": payload, "e": effort})
            rows.append(_row(day, "critic", agents.CRITIC_SYSTEM, payload,
                             agents._load(key), key, model, effort))

    return pd.DataFrame(rows)


def _row(day, role, system, payload, resp, key, model, effort) -> dict:
    u = (resp or {}).get("_usage", {})
    return {
        "date": day, "agent": role, "model": model, "effort": effort,
        "cache_key": key, "matched": resp is not None,
        "system_prompt": system,
        "prompt_payload": json.dumps(payload, indent=1),
        "prompt_chars": len(json.dumps(payload)),
        # Specialist fields (null for the critic) ...
        "state": (resp or {}).get("state"),
        "confidence": (resp or {}).get("confidence"),
        "informative": (resp or {}).get("informative"),
        "confidence_adjustment": (resp or {}).get("confidence_adjustment"),
        "reasoning": (resp or {}).get("reasoning"),
        # ... and critic fields (null for specialists)
        "verdict": (resp or {}).get("verdict"),
        "final_state": (resp or {}).get("final_state"),
        "confidence_multiplier": (resp or {}).get("confidence_multiplier"),
        "recency_bias_risk": (resp or {}).get("recency_bias_risk"),
        "objection": (resp or {}).get("objection"),
        "falsifier": (resp or {}).get("falsifier"),
        "failed": bool((resp or {}).get("_failed", False)),
        "tokens_in": u.get("in"), "tokens_out": u.get("out"),
        "raw_response": json.dumps(resp) if resp else None,
    }


def build_day_table(arm_b: pd.DataFrame, arm_c: pd.DataFrame, cfg: dict,
                    provenance: dict) -> pd.DataFrame:
    """One row per gated day: detector in, panel out, and what changed."""
    rows = []
    for d, v in provenance.items():
        day = pd.Timestamp(d)
        if day not in arm_b.index:
            continue
        b, c = arm_b.loc[day], arm_c.loc[day]
        rows.append({
            "date": day,
            "hmm_label": b["label"], "hmm_confidence": float(b["confidence"]),
            **{f"hmm_p_{s}": float(b[f"p_{s}"]) for s in STATES},
            "final_label": v["label"], "final_confidence": float(v["confidence"]),
            **{f"final_p_{s}": float(c[f"p_{s}"]) for s in STATES},
            "action": v["action"], "critic_verdict": v["critic_verdict"],
            "unanimous_against": v.get("unanimous_against"),
            "recency_bias_risk": v.get("recency_bias_risk"),
            "label_changed": v["label"] != b["label"],
            "confidence_delta": float(v["confidence"]) - float(b["confidence"]),
            "panel_votes": json.dumps(v.get("panel_votes", {})),
            "n_failed_calls": v.get("n_failed_calls", 0),
            "objection": v.get("objection"), "falsifier": v.get("falsifier"),
        })
    return pd.DataFrame(rows).sort_values("date")


def main() -> None:
    cfg = config.load()
    from . import features as F, ingest

    res = config.paths()["results"]
    blocks = ingest.build_panel(cfg=cfg)
    feats = F.build_features(blocks, cfg)
    arm_b = pd.read_parquet(res / "arm_b_hmm.parquet")

    print("rehydrating LLM records from the content-addressed cache ...")
    recs = build_records(arm_b, feats, cfg)
    matched = int(recs["matched"].sum())
    print(f"   {len(recs)} (day, agent) pairs   matched to a cached response: "
          f"{matched} ({matched / max(len(recs), 1) * 100:.1f}%)")
    if matched < len(recs):
        print("   unmatched rows mean the run is incomplete, or payload")
        print("   regeneration has drifted out of sync with agents.walk_forward")
    recs.to_parquet(res / "arm_c_records.parquet")
    print(f"written: {res / 'arm_c_records.parquet'}")

    cpath, ppath = res / "arm_c_agentic.parquet", res / "arm_c_provenance.json"
    if cpath.exists() and ppath.exists():
        arm_c = pd.read_parquet(cpath)
        prov = json.load(open(ppath))
        days = build_day_table(arm_b, arm_c, cfg, prov)
        days.to_parquet(res / "arm_c_days.parquet")
        print(f"written: {res / 'arm_c_days.parquet'}  ({len(days)} gated days)")

    ok = recs[recs["matched"]]
    if len(ok):
        print(f"\ntokens: in {ok['tokens_in'].sum():,.0f}  "
              f"out {ok['tokens_out'].sum():,.0f}")
        print(f"failed calls: {int(ok['failed'].sum())}")
        print("\nsample record (what one agent saw and said):")
        vol = ok[ok["agent"] == "volatility"]
        r = (vol if len(vol) else ok).iloc[-1]
        print(f"   date={r['date'].date()}  agent={r['agent']}  "
              f"state={r['state']}  informative={r['informative']}")
        print(f"   prompt ({r['prompt_chars']} chars): "
              f"{r['prompt_payload'][:180]}...")
        print(f"   reasoning: {str(r['reasoning'])[:200]}")


if __name__ == "__main__":
    main()
