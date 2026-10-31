"""Phase 0: the parameter contract between the agent and the RS optimiser.

The main study let the planner return free-form ``params: {}``. Across 248
months it emitted 627 distinct keys of which 12 were ever read, and unrecognised
values fell through to hard-coded defaults without anything being logged. The
agentic claim was correspondingly hard to defend.

This module exists so that cannot happen again. The surface is three typed,
bounded parameters, enforced twice:

  * server-side, by handing ``JSON_SCHEMA`` to the API's structured-output mode,
    so the model physically cannot emit a differently-named key; and
  * client-side, by ``validate()``, which returns an explicit rejection reason
    rather than silently substituting a default.

A rejection is recorded in the run output as a rejection. The defaults are only
ever applied by the caller, deliberately, and are exactly the behaviour of the
unmodified RS model, so a fully-rejected run reproduces the classical baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# Bounds. Deliberately wide enough to permit genuine judgement, narrow enough
# that any value inside them is economically meaningful.
GAMMA_RANGE = (3.0, 15.0)        # risk aversion; HIGHER = more risk averse
CAP_RANGE = (0.10, 1.00)         # per-name ceiling; 1.00 == no cap == plain RS
TREATMENTS = ("hard_label", "probability_blend")


@dataclass(frozen=True)
class RSPosture:
    """What the agent is allowed to decide each month."""

    gamma: float
    regime_treatment: str
    max_position_pct: float
    rationale: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# Applied only when the agent is absent or its answer is rejected. These values
# reproduce the unmodified RS model exactly: gamma 6, the hard 0.5 regime
# threshold, and no concentration cap.
DEFAULT = RSPosture(gamma=6.0, regime_treatment="hard_label",
                    max_position_pct=1.00, rationale="default (plain RS)")


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "gamma": {
            "type": "number",
            "description": (
                "Risk aversion in the mean-variance objective "
                "max w'mu - (gamma/2) w'Sigma w. HIGHER means MORE risk averse "
                f"and a smaller, safer allocation. Range {GAMMA_RANGE[0]} to "
                f"{GAMMA_RANGE[1]}; the classical model fixes this at 6."
            ),
        },
        "regime_treatment": {
            "type": "string",
            "enum": list(TREATMENTS),
            "description": (
                "hard_label: use the regime the filter reports, as the classical "
                "model does. probability_blend: weight the two states' moments by "
                "P(bull) instead of committing to one, which is the safer choice "
                "when the filter is uncertain or unstable."
            ),
        },
        "max_position_pct": {
            "type": "number",
            "description": (
                "Ceiling on any single name's weight, as a FRACTION not a "
                f"percentage (0.20 means twenty percent). Range {CAP_RANGE[0]} to "
                f"{CAP_RANGE[1]}; 1.00 means no cap, which is what the classical "
                "model does."
            ),
        },
        "rationale": {
            "type": "string",
            "description": "One or two sentences on why, referencing the diagnostics.",
        },
    },
    "required": ["gamma", "regime_treatment", "max_position_pct", "rationale"],
    "additionalProperties": False,
}


def validate(payload: dict | None) -> tuple[RSPosture | None, str]:
    """Check one agent response.

    Returns ``(posture, "")`` when the answer is usable, or ``(None, reason)``
    when it is not. Never returns a partially-defaulted posture: the caller
    decides what to do with a rejection and must record it.
    """
    if not isinstance(payload, dict):
        return None, f"not_an_object: {type(payload).__name__}"

    missing = [k for k in ("gamma", "regime_treatment", "max_position_pct")
               if k not in payload]
    if missing:
        return None, f"missing_keys: {','.join(missing)}"

    g = payload["gamma"]
    if not isinstance(g, (int, float)) or isinstance(g, bool):
        return None, f"gamma_not_numeric: {g!r}"
    if not GAMMA_RANGE[0] <= float(g) <= GAMMA_RANGE[1]:
        return None, f"gamma_out_of_range: {g}"

    t = payload["regime_treatment"]
    if t not in TREATMENTS:
        return None, f"unknown_regime_treatment: {t!r}"

    c = payload["max_position_pct"]
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        return None, f"cap_not_numeric: {c!r}"
    c = float(c)
    # The main study saw the model emit a cap as 15 meaning 15 percent, which the
    # weight mapper read as effectively uncapped. Reject rather than guess.
    if c > CAP_RANGE[1]:
        return None, f"cap_looks_like_percent_not_fraction: {c}"
    if c < CAP_RANGE[0]:
        return None, f"cap_below_floor: {c}"

    return RSPosture(gamma=float(g), regime_treatment=t, max_position_pct=c,
                     rationale=str(payload.get("rationale", ""))[:400]), ""


__all__ = ["RSPosture", "DEFAULT", "JSON_SCHEMA", "validate",
           "GAMMA_RANGE", "CAP_RANGE", "TREATMENTS"]
