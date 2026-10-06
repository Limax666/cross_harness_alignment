#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../" && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
CONDA_BIN="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-cross-harness-sft}"
DATA_DIR="${DATA_DIR:-$EXP/data/sft/agentdojo_codex_claude_gpt56_sol_full}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXP/checkpoints/qwen35-9b-base-agentdojo-codex-claude-sft}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
GPUS="${GPUS:-0,1}"

test -x "$CONDA_BIN" || { echo "Conda not found: $CONDA_BIN" >&2; exit 1; }
test -s "$DATA_DIR/train.jsonl" && test -s "$DATA_DIR/validation.jsonl" || {
  echo "SFT data missing; run scripts/build_cross_harness_sft.py first." >&2; exit 1;
}
exec "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="$EXP/src${PYTHONPATH:+:$PYTHONPATH}" \
  torchrun --standalone --nproc_per_node="$(awk -F, '{print NF}' <<<"$GPUS")" -m cross_harness_sft.train_student \
    --train-data "$DATA_DIR/train.jsonl" --validation-data "$DATA_DIR/validation.jsonl" \
    --base-model Qwen/Qwen3.5-9B-Base --output-dir "$OUTPUT_DIR" --max-length "$MAX_LENGTH" \
    --epochs 2 --learning-rate 2e-5 --batch-size 1 --gradient-accumulation 8 \
    --lora-rank 32 --lora-alpha 64 --load-in-4bit
