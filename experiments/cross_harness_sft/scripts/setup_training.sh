#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
: "${TRANSFORMERS_VERSION:?Set exact TRANSFORMERS_VERSION}"
: "${PEFT_VERSION:?Set exact PEFT_VERSION}"
: "${ACCELERATE_VERSION:?Set exact ACCELERATE_VERSION}"
: "${BITSANDBYTES_VERSION:?Set exact BITSANDBYTES_VERSION}"
source "$EXP/.venv/bin/activate"
python -m pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio
python -m pip install "transformers==$TRANSFORMERS_VERSION" "peft==$PEFT_VERSION" \
  "accelerate==$ACCELERATE_VERSION" "bitsandbytes==$BITSANDBYTES_VERSION"
python -m cross_harness_sft.preflight --config "$EXP/configs/formal_native.yaml" --mode train \
  --dataset-dir "$EXP/data/sft/formal"
