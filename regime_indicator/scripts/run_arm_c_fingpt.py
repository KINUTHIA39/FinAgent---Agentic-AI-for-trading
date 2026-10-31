"""Arm C with FinGPT news, 2015-2025. Everything but the news slice is the
FinBERT run's: same detector (extended to 2025, identical through 2023), same
gate, same specialists, critic, model, effort and unanimity rule.

Because the LLM cache is keyed on the exact prompt, the volatility, market-
sentiment and credit specialists hit the FinBERT run's cache on 2015-2023 and
cost nothing; only the news specialist and the critic (which reads the news
opinion) are new calls there. 2024-2025 is new for every role.

    python scripts/run_arm_c_fingpt.py --dry      # count new calls, no API
    python scripts/run_arm_c_fingpt.py --limit 3  # smoke test
    python scripts/run_arm_c_fingpt.py            # full run (resumable)

Outputs in results/fingpt/: arm_c_fingpt.parquet, arm_c_fingpt_provenance.json
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import pandas as pd

from src import agents, config, detector, features as F, ingest

START, END = "2015-01-01", "2025-12-31"
RES = config.paths()["results"] / "fingpt"

cfg = config.load()
cfg["agents"]["news_track"] = "fingpt"
dry = "--dry" in sys.argv
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

if dry:
    calls = {"n": 0}

    def _blocked():
        calls["n"] += 1
        raise RuntimeError("dry run")
    agents._client = _blocked
    agents._save = lambda k, v: None

feats = F.build_features(ingest.build_panel(cfg=cfg), cfg)
b_path = RES / "arm_b_hmm_2025.parquet"
if not b_path.exists():
    # Same walk-forward as results/arm_b_hmm.parquet, scored on through 2025.
    print("building the detector through 2025 ...", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    detector.walk_forward(feats["hmm"], cfg, score_end=END).to_parquet(b_path)
arm_b = pd.read_parquet(b_path)
sent = pd.read_parquet(config.paths()["results"] / "fingpt_sentiment_daily.parquet")

gate = detector.candidate_days(arm_b, cfg)
in_window = (gate.index >= START) & (gate.index <= END)
gate = gate & in_window
n = int(gate.sum())
print(f"model={cfg['agents']['model']}  effort={cfg['agents']['effort']}  "
      f"gated days {START}..{END}: {n}  "
      f"(2015-2023: {int(gate.loc[:'2023'].sum())}, 2024-2025: "
      f"{int(gate.loc['2024':].sum())})", flush=True)

t0 = time.time()
out = agents.walk_forward(arm_b, feats, cfg, gate, limit=limit,
                          concurrency=int(cfg["agents"].get("concurrency", 8)),
                          sentiment=sent, market=None)

if dry:
    print(f"new API calls needed: {calls['n'] // 3} "
          f"(of {min(n, limit or n) * 5} panel+critic calls)")
    sys.exit()

print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
prov = out.attrs["provenance"]
if limit is None:
    out = out.loc[START:END]
    out.drop(columns=["objection"]).to_parquet(RES / "arm_c_fingpt.parquet")
    json.dump(prov, open(RES / "arm_c_fingpt_provenance.json", "w"),
              indent=1, default=str)
    print(f"written: {RES / 'arm_c_fingpt.parquet'}")

acts = pd.Series([v["action"] for v in prov.values()]).value_counts()
print("\nagent actions:\n" + acts.to_string())
failed = sum(v.get("n_failed_calls", 0) for v in prov.values())
changed = int((out.loc[START:END, "label"] != arm_b.loc[START:END, "label"]).sum())
print(f"failed calls: {failed}   labels changed vs detector: {changed}")
