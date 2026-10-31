#!/bin/bash
# One-time setup on the Hostinger B200 box (Ubuntu 24.04, run as root).
# Ordered for wall-clock time: the 26 GB weight download starts first and runs
# in the background while PyTorch installs, so the two overlap.
set -euo pipefail
W=/opt/fingpt
BASE_MODEL=NousResearch/Llama-2-13b-hf
ADAPTER=FinGPT/fingpt-sentiment_llama2-13b_lora
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_TELEMETRY=1
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p $W/out

stamp "GPU check"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
    echo "FATAL: no NVIDIA driver on this image. Ask Hostinger for their GPU image" \
         "(B200 needs driver >= 570, open kernel modules)."; exit 1
fi
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$DRV" -ge 570 ] || { echo "FATAL: driver $DRV < 570 cannot run Blackwell with CUDA 12.8"; exit 1; }

stamp "uv + venv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=$HOME/.local/bin:$PATH
uv venv -q --python 3.12 $W/venv
PY=$W/venv/bin/python
uv pip install -q --python $PY huggingface_hub hf_transfer

stamp "weights: background download"
$PY - > $W/download.log 2>&1 <<PY &
from huggingface_hub import list_repo_files, snapshot_download
base = "$BASE_MODEL"
files = list_repo_files(base)
w = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
snapshot_download(base, allow_patterns=["*.json", "tokenizer.model", w], max_workers=16)
snapshot_download("$ADAPTER", max_workers=16)
print("download ok")
PY
DL=$!

stamp "PyTorch (CUDA 12.8 build: first with Blackwell sm_100 kernels) + libs"
uv pip install -q --python $PY torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python $PY "transformers>=4.45" "peft>=0.13" accelerate \
    sentencepiece protobuf pandas pyarrow

stamp "waiting for weights"
wait $DL || { cat $W/download.log; exit 1; }
tail -1 $W/download.log
du -sh $HF_HOME

$PY - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  {p.name}  {p.total_memory/2**30:.0f} GiB  sm_{p.major}{p.minor}")
assert f"sm_{p.major}{p.minor}" in torch.cuda.get_arch_list(), \
    f"this torch build has no kernels for sm_{p.major}{p.minor}: {torch.cuda.get_arch_list()}"
PY
touch $W/READY
stamp "setup done"
