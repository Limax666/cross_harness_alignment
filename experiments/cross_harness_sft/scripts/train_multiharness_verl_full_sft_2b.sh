#!/usr/bin/env bash
# Full-parameter multi-turn SFT for the selected 2B student on exactly two
# RTX 4090s. It deliberately refuses a 3rd/4th GPU so evaluation and
# collection capacity remains available.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
GPUS="${GPUS:-1,2}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ "${#GPU_LIST[@]}" != "2" ]]; then
  echo "Full SFT for Qwen3.5-2B is registered for exactly two GPUs; got GPUS=$GPUS." >&2
  exit 2
fi

export MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
if [[ "$MODEL" != "Qwen/Qwen3.5-2B" ]]; then
  echo "This full-SFT resource profile is only for Qwen/Qwen3.5-2B; got MODEL=$MODEL" >&2
  exit 2
fi
export OUT_DIR="${OUT_DIR:-$EXP/checkpoints/qwen35-2b-base-multiharness-verl-full-sft-v1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35-2b-multiharness-verl-full-sft}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export LR="${LR:-2e-5}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export MIN_QUALIFIED_PER_HARNESS="${MIN_QUALIFIED_PER_HARNESS:-32}"
export MIN_QUALIFIED_VALIDATION="${MIN_QUALIFIED_VALIDATION:-8}"
export TEST_FREQ="${TEST_FREQ:-8}"
export LORA_RANK=0
export LORA_ALPHA=0
export CHSFT_REQUIRE_FULL_TRAINABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Launching full-parameter SFT: model=$MODEL, GPUS=$GPUS, max_length=$MAX_LENGTH"
echo "LoRA is disabled (LORA_RANK=0); do not point MODEL at an adapter checkpoint."

# Refresh the AgentHarm source-specific selection using the approved historical
# GPT-5.6-Sol and current Qwen judges, then join it with the audited real
# AgentDojo/ActBench rows. Legacy HarnessRisk text-only and synthetic records
# remain classified in the audit but are excluded from primary full-SFT.
python "$EXP/scripts/audit_agentharm_collection_coverage.py" --require-complete
python "$EXP/scripts/select_agentharm_native_sft.py" \
  --allowed-judges codex_cli,openai_compatible \
  --judge-models gpt-5.6-sol,ali/qwen3.8-flash \
  --legacy-judge-model-map codex_cli=gpt-5.6-sol,openai_compatible=ali/qwen3.8-flash \
  --min-harnesses 1 --min-benign-tools-per-harness 1 --min-harmful-per-harness 1 \
  --output-dir "$EXP/data/sft/agentharm_native_verified_v1"
JOINT_DIR="$EXP/data/sft/multidataset_joint_sft_v1"
python "$EXP/scripts/build_joint_multidataset_sft.py" \
  --legacy-dir "$EXP/data/sft/multi_harness_combined_v1" \
  --agentharm-dir "$EXP/data/sft/agentharm_native_verified_v1" \
  --output-dir "$JOINT_DIR" --replace-output
read -r HARNESSES < <(python - "$JOINT_DIR/train.jsonl" "$MIN_QUALIFIED_PER_HARNESS" <<'PY'
import json,sys
from collections import Counter
counts=Counter(json.loads(line)['metadata']['harness_name'] for line in open(sys.argv[1]) if line.strip())
names=sorted(name for name,count in counts.items() if count >= int(sys.argv[2]))
if len(names) < 3:
    raise SystemExit(f"Joint SFT has only {len(names)} harnesses with >= {sys.argv[2]} rows; need at least 3")
print(','.join(names))
PY
)
export TRAIN_JSONL="$JOINT_DIR/train.jsonl"
export VAL_JSONL="$JOINT_DIR/validation.jsonl"
export DATA_HARNESSES="$HARNESSES"
export DATA_DIR="${DATA_DIR:-$EXP/data/sft/multidataset_joint_verl_2b_v1}"
if [[ "${TOTAL_TRAINING_STEPS:-null}" != null ]]; then
  # VeRL stops at total_training_steps only while inside its epoch loop.
  # Provide enough epochs for a 100-step preflight even on a small dataset.
  export TOTAL_EPOCHS=100
  export OUT_DIR="${OUT_DIR}-preflight-${TOTAL_TRAINING_STEPS}steps"
fi

exec env GPUS="$GPUS" \
  bash "$EXP/scripts/train_multiharness_verl_sft.sh" "$@"
