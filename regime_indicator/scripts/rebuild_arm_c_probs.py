"""Regenerate Arm C probabilities from saved provenance. No API calls."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src import agents, config, detector
from src.detector import STATES

res = config.paths()["results"]
arm_b = pd.read_parquet(res / "arm_b_hmm.parquet")
arm_c = pd.read_parquet(res / "arm_c_agentic.parquet")
prov = json.load(open(res / "arm_c_provenance.json"))

fixed = 0
for d, f in prov.items():
    day = pd.Timestamp(d)
    base = np.array([float(arm_b.loc[day, f"p_{s}"]) for s in STATES])
    new = agents.rederive_probabilities(base, f["label"], float(f["confidence"]))
    for s, v in zip(STATES, new):
        arm_c.loc[day, f"p_{s}"] = v
    arm_c.loc[day, "label"] = f["label"]
    arm_c.loc[day, "confidence"] = float(f["confidence"])
    fixed += 1

probs = arm_c[[f"p_{s}" for s in STATES]]
argmax = pd.Series([STATES[i] for i in probs.to_numpy().argmax(axis=1)], index=arm_c.index)
bad = int((argmax != arm_c["label"]).sum())
print(f"rebuilt {fixed} gated days")
print(f"rows where label != argmax: {bad} (was 64)")
print(f"probabilities sum to 1: {np.allclose(probs.sum(axis=1), 1.0)}")
print(f"mean confidence: B {arm_b['confidence'].mean():.4f}  C {arm_c['confidence'].mean():.4f}")
arm_c.to_parquet(res / "arm_c_agentic.parquet")
print(f"written: {res / 'arm_c_agentic.parquet'}")
