"""Short-term rolling buffer. A deliberate scope decision, not an oversight.

No graph database, no episode retrieval. Episode-level precedent retrieval
(2008 credit crisis, 2018 volmageddon, 2020 COVID crash, 2022 rate shock) is
the natural extension and belongs on the future-work slide.

CONTAMINATION: nothing here may carry an absolute date. The buffer speaks in
offsets -- "changed 14 days ago" -- because a calendar date is the single
cheapest way for the model to recognise a period it has memorised rather than
inferring anything from the data.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ShortTermMemory:
    """Rolling context passed into every LLM call. Windows come from cfg['memory']."""

    max_labels: int = 60
    max_probs: int = 20
    max_features: int = 20

    recent_labels: deque = field(default_factory=deque)
    recent_probabilities: deque = field(default_factory=deque)
    recent_features: deque = field(default_factory=deque)
    days_in_current_state: int = 0
    last_transition: dict | None = None
    _n_seen: int = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "ShortTermMemory":
        m = cfg["memory"]
        return cls(max_labels=int(m["recent_labels"]),
                   max_probs=int(m["recent_probabilities"]),
                   max_features=int(m["recent_features"]))

    def update(self, label: str, probs: dict, features: dict) -> None:
        """Push one day. Offsets are relative -- never a calendar date."""
        prev = self.recent_labels[-1] if self.recent_labels else None
        if prev is not None and label != prev:
            self.last_transition = {"days_ago": 0, "from": prev, "to": label}
            self.days_in_current_state = 1
        else:
            self.days_in_current_state += 1
            if self.last_transition:
                self.last_transition = dict(self.last_transition)
                self.last_transition["days_ago"] += 1

        self.recent_labels.append(label)
        self.recent_probabilities.append(probs)
        self.recent_features.append(features)
        while len(self.recent_labels) > self.max_labels:
            self.recent_labels.popleft()
        while len(self.recent_probabilities) > self.max_probs:
            self.recent_probabilities.popleft()
        while len(self.recent_features) > self.max_features:
            self.recent_features.popleft()
        self._n_seen += 1

    def to_prompt_context(self) -> dict:
        """Serialize for the prompt. Must contain no absolute dates."""
        labels = list(self.recent_labels)
        counts = {s: labels.count(s) for s in set(labels)} if labels else {}
        probs = list(self.recent_probabilities)[-5:]
        return {
            "days_in_current_state": self.days_in_current_state,
            "last_transition": self.last_transition,
            "label_counts_last_%d_days" % len(labels): counts,
            "label_sequence_recent": labels[-15:],
            "prob_vector_last_5_days": [
                {k: round(float(v), 3) for k, v in p.items()} for p in probs],
        }


def build_history(labels: pd.Series, probs: pd.DataFrame, features: pd.DataFrame,
                  cfg: dict) -> dict[pd.Timestamp, dict]:
    """Replay the whole series once and snapshot memory at each day.

    Snapshotting up front means every gated day gets the memory state it would
    have had in a live run, without replaying the buffer per call.
    """
    mem = ShortTermMemory.from_config(cfg)
    out: dict[pd.Timestamp, dict] = {}
    for day in labels.index:
        # Snapshot BEFORE absorbing today: the panel sees history, not itself.
        out[day] = mem.to_prompt_context()
        mem.update(str(labels.loc[day]),
                   probs.loc[day].to_dict() if day in probs.index else {},
                   features.loc[day].round(3).to_dict() if day in features.index else {})
    return out
