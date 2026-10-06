#!/usr/bin/env bash
# Two-GPU launcher for harness-balanced, harness-conditioned TRL QLoRA SFT.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
CONDA_BIN="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-cross-harness-sft}"
GPUS="${GPUS:-0,1}"
DATA_DIR="${DATA_DIR:-$EXP/data/sft/multi_harness_trl_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXP/checkpoints/qwen35-9b-base-multiharness-trl-sft}"
MAX_LENGTH="${MAX_LENGTH:-8192}"

test -x "$CONDA_BIN" || { echo "Conda not found: $CONDA_BIN" >&2; exit 1; }
test -s "$DATA_DIR/train.jsonl" && test -s "$DATA_DIR/validation.jsonl" || {
  echo "Missing filtered multi-harness data in $DATA_DIR" >&2; exit 1;
}
GPU_COUNT="$(awk -F, '{print NF}' <<<"$GPUS")"
test "$GPU_COUNT" -le 2 || { echo "Only one or two GPUs are allowed; got GPUS=$GPUS" >&2; exit 1; }

exec "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  env CUDA_VISIBLE_DEVICES="$GPUS" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  torchrun --standalone --nproc_per_node="$GPU_COUNT" "$EXP/scripts/train_multiharness_trl_sft.py" \
    --train-data "$DATA_DIR/train.jsonl" \
    --validation-data "$DATA_DIR/validation.jsonl" \
    --base-model "Qwen/Qwen3.5-9B-Base" \
    --output-dir "$OUTPUT_DIR" \
    --max-length "$MAX_LENGTH" \
    --epochs "${EPOCHS:-1}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
    --gradient-accumulation "${GRADIENT_ACCUMULATION:-4}" \
    --eval-steps "${EVAL_STEPS:-32}" \
    --max-steps "${MAX_STEPS:-400}" \
    --lora-rank "${LORA_RANK:-32}" \
    --lora-alpha "${LORA_ALPHA:-64}"
