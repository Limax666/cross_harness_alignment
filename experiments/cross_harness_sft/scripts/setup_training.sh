#!/usr/bin/env bash
# Create the Linux CUDA training environment used by train_qwen35_4b.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
CONDA_BIN="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-cross-harness-sft}"

test -x "$CONDA_BIN" || { echo "Install Miniforge first, or set CONDA_BIN." >&2; exit 1; }
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  "$CONDA_BIN" create -y -n "$CONDA_ENV" python=3.11 pip
fi
"$CONDA_BIN" run -n "$CONDA_ENV" python -m pip install --upgrade pip
"$CONDA_BIN" run -n "$CONDA_ENV" python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.6.0 torchvision==0.21.0
"$CONDA_BIN" run -n "$CONDA_ENV" python -m pip install \
  'transformers>=5.17.0' 'peft>=0.15.0' 'accelerate>=1.6.0' \
  'bitsandbytes>=0.45.5' 'safetensors>=0.5.3' 'sentencepiece>=0.2.0' 'matplotlib>=3.8'
"$CONDA_BIN" run -n "$CONDA_ENV" python -m pip install causal-conv1d flash-linear-attention
"$CONDA_BIN" run -n "$CONDA_ENV" python -m pip install -e "$EXP"
"$CONDA_BIN" run -n "$CONDA_ENV" python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable: repair the NVIDIA driver before starting SFT"
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
PY
