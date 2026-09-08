#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../" && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
python -m cross_harness_sft.train_student \
  --train-data "$EXP/data/sft/formal/train.jsonl" \
  --validation-data "$EXP/data/sft/formal/validation.jsonl" \
  --base-model Qwen/Qwen3.5-4B --output-dir "$EXP/checkpoints/qwen35-4b-cross-harness-rft" \
  --max-length 8192 --epochs 2 --learning-rate 2e-5 --batch-size 1 \
  --gradient-accumulation 16 --lora-rank 32 --lora-alpha 64 --load-in-4bit
