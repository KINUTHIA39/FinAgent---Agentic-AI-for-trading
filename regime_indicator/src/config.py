"""Config loading. Every module reads config.yaml through here; nothing hardcodes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load(path: Path | str | None = None) -> dict[str, Any]:
    """Load and return the experiment config."""
    with open(Path(path) if path else CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def paths() -> dict[str, Path]:
    """Canonical directories, created on demand."""
    d = {
        "root": ROOT,
        "raw": ROOT / "data" / "raw",
        "pit": ROOT / "data" / "pit",
        "results": ROOT / "results",
    }
    for key, p in d.items():
        if key != "root":
            p.mkdir(parents=True, exist_ok=True)
    return d
