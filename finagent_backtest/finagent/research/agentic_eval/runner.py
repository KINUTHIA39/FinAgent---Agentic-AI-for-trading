"""CLI runner: sweep a scenario set through the planner -> runs.parquet.

Usage
-----
    python -m finagent.research.agentic_eval.runner --set pilot --k 8
    python -m finagent.research.agentic_eval.runner --set grid  --k 25 --out data/agentic_eval/runs.parquet

Requires ANTHROPIC_API_KEY (.env). Read-only w.r.t. dissertation assets; writes
only the output parquet.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_env() -> None:
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["pilot", "surface", "grid"], default="pilot")
    ap.add_argument("--k", type=int, default=8, help="repetitions per scenario")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "agentic_eval" / "runs.parquet"))
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: load --out, skip cells already done")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("agentic-eval")

    _load_env()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (.env)"); return 1

    import pandas as pd
    from finagent.core.claude_client import ClaudeClient
    from finagent.core.dag_planner import LLMDAGPlanner
    from .scenarios import build_pilot, build_surface, build_grid
    from .harness import run_scenario

    scenarios = {"pilot": build_pilot, "surface": build_surface, "grid": build_grid}[args.set]()
    planner = LLMDAGPlanner(client=ClaudeClient())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint/resume: the parquet is rewritten after EVERY cell, so a crash
    # loses at most the in-flight cell. --resume reloads it and skips done cells.
    rows: list = []
    done: set = set()
    if out.exists() and args.resume:
        prev = pd.read_parquet(out)
        rows = prev.to_dict("records")
        done = set(prev["scenario"].unique())
        log.info("Resuming: %d cells / %d rows already present", len(done), len(rows))
    elif out.exists():
        log.info("Overwriting %s (pass --resume to continue an interrupted run)", out)

    log.info("Running %d scenarios x k=%d = %d plans", len(scenarios), args.k,
             len(scenarios) * args.k)
    t0 = time.perf_counter()
    for n, sc in enumerate(scenarios, 1):
        if sc.label in done:
            log.info("[%d/%d] skip (done): %s", n, len(scenarios), sc.label)
            continue
        log.info("[%d/%d] %s", n, len(scenarios), sc.label)
        rows.extend(run_scenario(sc, planner, args.k))
        pd.DataFrame(rows).to_parquet(out, index=False)   # checkpoint after each cell
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)

    # Quick console summary
    llm_rate = df["llm_success"].mean() if len(df) else 0.0
    print("\n" + "=" * 66)
    print(f"  wrote {len(df)} rows -> {out}  ({elapsed:.0f}s)")
    print(f"  LLM-plan success rate: {llm_rate:.0%}")
    if len(df):
        print("\n  mean defensiveness by regime x exposure:")
        piv = df.groupby(["regime", "exposure"])["defensiveness_score"].mean().round(3)
        print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
