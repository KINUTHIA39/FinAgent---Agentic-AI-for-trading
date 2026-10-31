"""DAG feature extraction + the defensiveness score.

Turns a planned ``DAG`` into a flat feature dict for ``runs.parquet``:
topology (node/stage counts, type histogram) and the risk-posture parameters the
planner chose. The defensiveness score is a transparent, documented composite of
those posture indicators — the central metric for the adaptation analysis (RQ1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Risk-posture parameters, with the synonym keys the LLM tends to use.
_POSTURE = {
    "max_position_pct": ["max_position_pct", "max_position_size_pct", "max_pos_pct",
                         "position_size_pct", "max_alloc_per_position_pct"],
    "sizing_mode": ["position_sizing_mode", "sizing_method", "sizing_mode"],
    "reduce_only": ["reduce_only"],
    "halt_new_longs": ["halt_new_longs"],
    "circuit_breaker": ["circuit_breaker_action", "daily_loss_circuit_breaker_pct",
                        "circuit_breaker"],
    "slippage_guard_bps": ["slippage_guard_bps", "slippage_tolerance_bps", "slippage_limit_bps"],
    "flagged_symbols": ["known_losing_symbols", "avoid_symbols", "suppress_symbols"],
    "suppress_low_conf": ["suppress_low_confidence"],
}

_STAGE_ORDER = ["data_fetch", "feature", "signal", "risk", "portfolio", "execution", "audit"]

# Reference values for normalising the defensiveness composite.
_AGGRESSIVE_POSITION_PCT = 0.15   # a "max" position cap; smaller => more defensive
_SLIPPAGE_REF_BPS = 500.0         # a loose guard; tighter => more defensive

# Posture params that must land in the parquet as a single number. The planner
# sometimes emits these per-symbol (e.g. slippage_guard_bps={"AAPL":9.2,...}),
# which breaks the columnar write; we summarise such values by their mean.
_NUMERIC_POSTURE = {"max_position_pct", "slippage_guard_bps"}


def _to_scalar(val: Any) -> Any:
    """Reduce a per-symbol dict/list of numbers to a representative float (mean).

    Leaves plain numbers untouched and returns None for empty/non-numeric so the
    parquet column stays a clean nullable double.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, dict):
        nums = [v for v in val.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return sum(nums) / len(nums) if nums else None
    if isinstance(val, list):
        nums = [v for v in val if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return sum(nums) / len(nums) if nums else None
    return val


def _find(value: Any, keys: List[str]) -> Any:
    """Recursively search a params value for any of ``keys``."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in keys:
                return v
        for v in value.values():
            found = _find(v, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for v in value:
            found = _find(v, keys)
            if found is not None:
                return found
    return None


def _posture(dag: Any) -> Dict[str, Any]:
    """Pull the first occurrence of each posture param across all node params."""
    out: Dict[str, Any] = {}
    for label, keys in _POSTURE.items():
        val = None
        for n in dag.nodes:
            val = _find(dict(getattr(n, "params", {}) or {}), keys)
            if val is not None:
                break
        if label in _NUMERIC_POSTURE:
            val = _to_scalar(val)
        out[label] = val
    return out


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def defensiveness_score(posture: Dict[str, Any], n_candidates: int) -> float:
    """Mean of available [0,1] defensiveness indicators (1 = most defensive).

    Transparent composite — document in the methodology and run a weight
    sensitivity check. Only indicators the planner actually set contribute.
    """
    ind: List[float] = []

    mp = posture.get("max_position_pct")
    if isinstance(mp, (int, float)):
        ind.append(1.0 - _clamp(float(mp) / _AGGRESSIVE_POSITION_PCT))

    ind.append(1.0 if posture.get("reduce_only") is True else 0.0)
    ind.append(1.0 if posture.get("circuit_breaker") not in (None, False) else 0.0)
    ind.append(1.0 if posture.get("halt_new_longs") is True else 0.0)

    sm = posture.get("sizing_mode")
    if sm is not None:
        ind.append(1.0 if str(sm).lower() in ("conservative", "defensive", "reduce_only")
                   else 0.0)

    sg = posture.get("slippage_guard_bps")
    if isinstance(sg, (int, float)):
        ind.append(1.0 - _clamp(float(sg) / _SLIPPAGE_REF_BPS))

    flagged = posture.get("flagged_symbols")
    if isinstance(flagged, list) and n_candidates > 0:
        ind.append(_clamp(len(flagged) / n_candidates))

    ind.append(1.0 if posture.get("suppress_low_conf") is True else 0.0)

    return round(sum(ind) / len(ind), 4) if ind else 0.0


def dag_features(dag: Any, n_candidates: int) -> Dict[str, Any]:
    """Flat feature dict for one planned DAG."""
    nodes = list(getattr(dag, "nodes", []))
    types = []
    for n in nodes:
        t = getattr(n, "type", None)
        types.append(getattr(t, "value", str(t)))
    type_counts = {t: types.count(t) for t in set(types)}

    try:
        n_stages = len(dag.to_execution_plan())
    except Exception:  # noqa: BLE001
        n_stages = None

    p = _posture(dag)
    flagged = p.get("flagged_symbols")
    return {
        "source": getattr(dag, "source", "?"),
        "n_nodes": len(nodes),
        "n_edges": len(getattr(dag, "edges", []) or []),
        "n_stages": n_stages,
        "n_risk_nodes": type_counts.get("risk", 0),
        "max_position_pct": p.get("max_position_pct"),
        "reduce_only": bool(p.get("reduce_only") is True),
        "circuit_breaker": bool(p.get("circuit_breaker") not in (None, False)),
        "halt_new_longs": bool(p.get("halt_new_longs") is True),
        "slippage_guard_bps": p.get("slippage_guard_bps"),
        "sizing_mode": (str(p.get("sizing_mode")) if p.get("sizing_mode") is not None else None),
        "n_flagged_symbols": len(flagged) if isinstance(flagged, list) else 0,
        "defensiveness_score": defensiveness_score(p, n_candidates),
    }


__all__ = ["dag_features", "defensiveness_score"]
