#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?pass the completed RL run directory}"
EXPECTED_STEPS="${2:-6}"
ACTOR="$RUN/checkpoints/global_step_${EXPECTED_STEPS}/actor"
MERGED="$RUN/hf_merged_step_${EXPECTED_STEPS}"
EVAL_NAME="${EVAL_NAME:-heldout_agentdojo_validation}"
[[ "$EVAL_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid evaluation directory name' >&2; exit 2; }
EVAL="$RUN/$EVAL_NAME"
RL_PYTHON="$ROOT/.venv-rl-pilot/bin/python"
export HERMES_AGENT_ROOT="${HERMES_AGENT_ROOT:-$ROOT/vendor/hermes-agent}"
export HERMES_AGENT_PYTHON="${HERMES_AGENT_PYTHON:-$RL_PYTHON}"
export PYTHONPATH="$ROOT/vendor/verl:$ROOT/scripts:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

[[ -f "$RUN/checkpoints/latest_checkpointed_iteration.txt" ]] || { echo 'missing final checkpoint marker' >&2; exit 2; }
[[ "$(cat "$RUN/checkpoints/latest_checkpointed_iteration.txt")" == "$EXPECTED_STEPS" ]] || { echo 'incomplete RL run' >&2; exit 2; }
[[ -f "$ACTOR/fsdp_config.json" ]] || { echo 'missing FSDP checkpoint' >&2; exit 2; }
[[ ! -e "$EVAL" ]] || { echo 'refusing to overwrite validation results' >&2; exit 2; }

if [[ -e "$MERGED" ]]; then
  [[ "${REUSE_MERGED:-0}" == 1 && -s "$MERGED/model.safetensors" ]] || {
    echo 'refusing to reuse or overwrite an existing merged model' >&2; exit 2;
  }
else
  "$RL_PYTHON" -m verl.model_merger merge --backend fsdp --use_cpu_initialization \
    --local_dir "$ACTOR" --target_dir "$MERGED"
fi
mkdir -p "$EVAL"
"$RL_PYTHON" "$ROOT/scripts/validate_agentdojo_2b_rl_entry.py" \
  --role sft --families 3 17 29 31 --seeds 0 1 2 --max-tokens 512 \
  --output "$EVAL/sft.jsonl"
"$RL_PYTHON" "$ROOT/scripts/validate_agentdojo_2b_rl_entry.py" \
  --role rl --checkpoint "$MERGED" --families 3 17 29 31 --seeds 0 1 2 --max-tokens 512 \
  --output "$EVAL/rl.jsonl"
"$RL_PYTHON" "$ROOT/scripts/compare_agentdojo_rl_validation.py" \
  --sft "$EVAL/sft.jsonl" --rl "$EVAL/rl.jsonl" --output "$EVAL/comparison.json"
