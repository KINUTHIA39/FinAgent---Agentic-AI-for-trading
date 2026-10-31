"""Swappable monthly regime signal for the agentic backtest path.

The backtest previously read ``regime_hamilton.parquet`` directly at four call
sites. This module puts one interface in front of that so the regime signal can
be exchanged without touching anything downstream, which is what makes a
comparison between signals attributable to the signal rather than the plumbing.

Two providers:

    HamiltonProvider   the existing monthly two-state Hamilton filter
    ArmCProvider       the frozen agentic detector in data/regime, collapsed
                       to two states

Scope: the **agentic path only** (``backtest``, ``regen_sessions``,
``capture_plans``, ``dump_prompt``). ``rs_optimal`` and ``rs_agentic`` keep
Hamilton and are deliberately untouched, because Markose & Yang Eqs. 15-17 need
a fitted two-state transition matrix that the frozen artifact does not carry.

Selection: ``--regime {hamilton,arm_c}`` on the scripts, or the environment
variable ``FINAGENT_REGIME``. Default is ``hamilton`` so existing behaviour is
unchanged unless asked for.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
REGIME_DIR = REPO / "data" / "regime"

# Collapse of the artifact's three states onto the two the pipeline speaks.
# Measured over the 223 usable months, the argmax collapse and a 0.5 threshold
# on (p_bull_quiet + p_bull_volatile) agree on every single month, so this
# mapping introduces no ambiguity of its own.
_BULL_STATES = ("bull_quiet", "bull_volatile")


@dataclass(frozen=True)
class RegimeMonth:
    """One month's regime as the backtest consumes it."""

    date: pd.Timestamp
    label: str          # "bull" | "bear"
    p_bull: float
    source: str

    # Within-month context. Populated by ArmCProvider, None for Hamilton, which
    # is a month-end series and has nothing else to give.
    regime_changed: bool | None = None
    worst_label: str | None = None
    max_p_bear: float | None = None
    days_in_state: int | None = None
    as_of: str | None = None

    def briefing_ctx(self) -> dict:
        """The dict ``SessionWriter`` expects.

        Within-month keys are omitted entirely when the provider has nothing to
        say, rather than sent as None, because the writer renders any key that
        is present. Omitting them is what keeps a month-end signal's briefing
        byte-identical to the archived ones.
        """
        ctx: dict = {
            "tag": self.label,
            "prob_bull": round(self.p_bull, 2),
            "prob_bear": round(1 - self.p_bull, 2),
            "prob_sideways": 0.0,
        }
        if self.worst_label is not None:
            ctx["worst_regime"] = self.worst_label
        if self.regime_changed is not None:
            ctx["regime_changed"] = self.regime_changed
        if self.max_p_bear is not None:
            ctx["max_prob_bear"] = round(self.max_p_bear, 2)
        if self.days_in_state is not None:
            ctx["days_in_state"] = self.days_in_state
        return ctx


class HamiltonProvider:
    """The existing monthly Hamilton filter, unchanged."""

    name = "hamilton"

    def __init__(self, reg: pd.DataFrame):
        self._reg = reg

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._reg.index

    def month(self, d) -> RegimeMonth:
        row = self._reg.loc[d]
        return RegimeMonth(date=pd.Timestamp(d),
                           label=str(row["regime"]),
                           p_bull=float(row["p_bull_filtered"]),
                           source="hamilton")


class ArmCProvider:
    """The frozen agentic detector, collapsed to two states.

    Reads through ``RegimeReader.month_summary`` rather than sampling one day,
    because the contract is explicit that a point read at month-end loses a
    third of the signal. Of the 223 usable months, 29% contain a regime change
    and 15 contain a ``bear_stress`` episode that a month-end sample misses
    entirely. Those facts arrive on the record as ``regime_changed``,
    ``worst_label`` and ``max_p_bear`` even though the headline label and
    probability are the month-end values.
    """

    name = "arm_c"

    def __init__(self, candidate_months: pd.DatetimeIndex,
                 directory: Path | str = REGIME_DIR, arm: str = "C_agentic"):
        # The reader ships beside the artifact and is not importable as a
        # package, so add its directory rather than copying the file.
        d = Path(directory)
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        from regime_reader import RegimeReader  # noqa: E402

        self._reader = RegimeReader(d, arm=arm)
        self.manifest = self._reader.manifest
        self._end = self._reader.end
        self._index = pd.DatetimeIndex(
            [m for m in candidate_months if self._covers(m)])

    def _covers(self, d) -> bool:
        """A month is usable if the artifact reaches into that same month.

        The artifact's last session is 2023-12-29 while the December month-end
        is the 31st. Clamping to the last session is correct point-in-time
        behaviour, but only when it stays inside the rebalance month. Reaching
        back into an earlier month would silently answer a question about
        December with November's data, so that case is refused.
        """
        ts = pd.Timestamp(d)
        if ts < self._reader.start:
            return False
        if ts <= self._end:
            return True
        return (self._end.year, self._end.month) == (ts.year, ts.month)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    def month(self, d) -> RegimeMonth:
        ts = pd.Timestamp(d)
        if not self._covers(ts):
            raise KeyError(f"{ts.date()} is not covered by the frozen artifact "
                           f"(ends {self._end.date()})")
        asof = min(ts, self._end)
        s = self._reader.month_summary(asof)
        p_bull = float(s["p_bull_quiet"] + s["p_bull_volatile"])
        return RegimeMonth(
            date=ts,
            label="bull" if s["label"] in _BULL_STATES else "bear",
            p_bull=p_bull,
            source=f"arm_c:{self.manifest.get('config_hash', '?')}",
            regime_changed=bool(s["regime_changed"]),
            worst_label="bull" if s["worst_regime"] in _BULL_STATES else "bear",
            max_p_bear=float(s["max_p_bear_stress"]),
            days_in_state=int(s["days_in_state"]),
            as_of=str(s["as_of"]),
        )


def get_provider(name: str | None, reg: pd.DataFrame,
                 candidate_months: pd.DatetimeIndex):
    """Build the requested provider. ``None`` falls back to the env var."""
    name = (name or os.getenv("FINAGENT_REGIME") or "hamilton").lower()
    if name in ("hamilton", "ham"):
        return HamiltonProvider(reg)
    if name in ("arm_c", "armc", "arm-c"):
        return ArmCProvider(candidate_months)
    raise ValueError(f"unknown regime provider {name!r}; "
                     f"expected 'hamilton' or 'arm_c'")


__all__ = ["RegimeMonth", "HamiltonProvider", "ArmCProvider", "get_provider"]
