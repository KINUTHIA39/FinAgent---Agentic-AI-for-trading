#!/bin/bash
# Run from Git Bash on the PC:   bash infra/hostinger/run.sh ubuntu@<IP> [port] [key]
#
# 1. sends the scripts and starts the pipeline on the box (weights begin
#    downloading at once), 2. uploads the 23 MB of prompts while that runs,
# 3. streams progress, 4. copies the scores down and builds one parquet.
# Safe to re-run: setup is skipped once done, scoring resumes from its parts.
set -euo pipefail
HOST=${1:?usage: run.sh user@ip [port] [key]}
PORT=${2:-22}
KEY=${3:-$HOME/.ssh/id_ed25519}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DATA=$ROOT/data/alpaca/fingpt
PYLOCAL=${PYTHON:-python}    # local interpreter with pandas + pyarrow
OPTS="-i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
SSH="ssh -p $PORT $OPTS"
SCP="scp -q -P $PORT $OPTS"
W=/opt/fingpt

echo "== 1/4 scripts + launch"
$SSH "$HOST" "sudo mkdir -p $W/out && sudo chown -R \$(id -u):\$(id -g) $W && rm -f $W/data.ok"
$SCP "$HERE/setup.sh" "$HERE/pipeline.sh" \
    "$HERE/score_fingpt.py" "$HOST:$W/"
$SSH "$HOST" "nohup bash $W/pipeline.sh >> $W/run.log 2>&1 < /dev/null & echo launched"

echo "== 2/4 prompts upload (pipeline is already installing in parallel)"
$SCP "$DATA/prompts.parquet" "$DATA/pilot.parquet" "$HOST:$W/"
$SSH "$HOST" "touch $W/data.ok"

echo "== 3/4 progress (Ctrl+C only stops watching; the box keeps going)"
$SSH "$HOST" "tail -n +1 -f $W/run.log & T=\$!; \
    while [ ! -f $W/DONE ] && [ ! -f $W/FAILED ]; do sleep 5; done; sleep 2; kill \$T"
$SSH "$HOST" "test -f $W/DONE" || { echo "pipeline FAILED - see log above"; exit 1; }

echo "== 4/4 results down"
mkdir -p "$DATA/scores"
$SCP -r "$HOST:$W/out/pilot" "$HOST:$W/out/full" "$HOST:$W/run.log" "$DATA/scores/"
"$PYLOCAL" - "$DATA" <<'PY'
import sys, pandas as pd
from pathlib import Path
d = Path(sys.argv[1])
s = pd.concat(pd.read_parquet(f) for f in sorted((d / "scores/full").glob("part-*.parquet")))
s = s.drop_duplicates("id")
s["score"] = s["p_pos"] - s["p_neg"]
meta = pd.read_parquet(d / "prompts.parquet",
                       columns=["id", "created_at", "trading_day", "symbols", "has_body", "headline"])
out = meta.merge(s.drop(columns="n_tokens"), on="id", how="left")
missing = int(out["score"].isna().sum())
out.to_parquet(d / "fingpt_scores.parquet")
print(f"fingpt_scores.parquet: {len(out):,} articles, {missing} unscored")
PY
echo "DONE. Delete the Hostinger instance now - it bills by the hour."
