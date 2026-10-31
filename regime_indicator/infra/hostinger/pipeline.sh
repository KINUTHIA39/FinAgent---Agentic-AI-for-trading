#!/bin/bash
# Runs on the box, detached (nohup), so a dropped SSH session cannot stop it.
# setup -> pilot (with quality gate) -> full run. Leaves DONE or FAILED in /opt/fingpt.
set -uo pipefail
W=/opt/fingpt
export HF_HOME=$W/hf HF_HUB_OFFLINE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { stamp "FAILED: $*"; touch $W/FAILED; exit 1; }
rm -f $W/DONE $W/FAILED

[ -f $W/READY ] || bash $W/setup.sh || fail "setup"
export HF_HUB_OFFLINE=1          # weights are local now; never re-check the Hub
PY=$W/venv/bin/python

stamp "waiting for data upload"
for _ in $(seq 120); do [ -f $W/data.ok ] && break; sleep 5; done
[ -f $W/data.ok ] || fail "data never arrived"

stamp "pilot: 2,002 articles incl. the 50 longest, + 256 generation cross-checks"
$PY $W/score_fingpt.py --input $W/pilot.parquet --out $W/out/pilot --validate 256 \
    || fail "pilot crashed"

# Gate: the probability read-out must agree with what FinGPT writes when it
# answers, and the three labels must hold most of the next-token mass.
$PY - <<'PY' || fail "pilot quality gate"
import json, sys
s = json.load(open("/opt/fingpt/out/pilot/summary.json"))
ok = s["agree"] >= 0.90 and s["unparsed"] <= 0.10 and s["label_mass_median"] >= 0.5
print("gate", "PASS" if ok else "FAIL", s)
sys.exit(0 if ok else 1)
PY

stamp "full run: 67,624 articles"
$PY $W/score_fingpt.py --input $W/prompts.parquet --out $W/out/full || fail "full run"
touch $W/DONE
stamp "ALL DONE"
