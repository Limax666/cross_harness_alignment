#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_ROLE:?Set EVAL_ROLE to teacher, base, or sft}"
: "${EVAL_MODEL_ID:?Set EVAL_MODEL_ID}"
: "${EVAL_OPENAI_BASE_URL:?Set EVAL_OPENAI_BASE_URL}"
: "${EVAL_ANTHROPIC_BASE_URL:?Set an Anthropic-compatible endpoint (LiteLLM is acceptable)}"
: "${EVAL_API_KEY:?Set endpoint key}"
ROOT="$(cd "$(dirname "$0")/../../../" && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
export EVAL_OUTPUT_JSONL="experiments/cross_harness_sft/outputs/eval/${EVAL_ROLE}.jsonl"
python -m cross_harness_sft.preflight --config "$EXP/configs/eval_native.yaml" --mode collect
python -m cross_harness_sft.collect_native --config "$EXP/configs/eval_native.yaml"
