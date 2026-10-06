#!/usr/bin/env bash
# Run the full-parameter, on-policy 2048-token ablation on the emptiest card
# without preempting other users. The launcher's occupancy check is repeated.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${RL_RUN_DIR:-$ROOT/outputs/rl/hermes_agentdojo_grpo_r17_budget2048_16cells_20261005}"
PYTHON="/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python"
PLOT="$ROOT/scripts/plot_rl_training_metrics.py"
mkdir -p "$RUN"
[[ ! -e "$RUN/metrics.jsonl" ]] || { echo "Existing metrics journal: $RUN" >&2; exit 2; }

gpu_available() {
  local used
  used="$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" || return 1
  [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 ))
}

echo "$(date -Is) queued: waiting for GPU1 to fall below the 1-GiB sharing threshold; no process will be stopped" >&2
until gpu_available 1; do
  sleep 60
done
echo "$(date -Is) GPU1 available; starting single-GPU 16-cell online GRPO, response budget 2048" >&2

refresh_plot() {
  while :; do
    if [[ -s "$RUN/metrics.jsonl" ]]; then
      "$PYTHON" "$PLOT" "$RUN/metrics.jsonl" --output "$RUN/rl_training_dashboard.png" >/dev/null 2>&1 || true
    fi
    sleep 120
  done
}
refresh_plot &
plot_pid=$!
trap 'kill "$plot_pid" 2>/dev/null || true' EXIT

set +e
CUDA_VISIBLE_DEVICES=1 \
RL_N_GPUS=1 \
RL_GPU_MAX_PREEXISTING_MIB=1024 \
RL_RUN_DIR="$RUN" \
RL_STEPS=12 \
RL_SAVE_FREQ=3 \
RL_RESPONSE_LENGTH=2048 \
RL_GPU_UTIL=0.20 \
RL_MAX_NUM_SEQS=2 \
RL_DATA="$ROOT/outputs/rl/native_pilot_train_20261005_hermes_mcp_16cells.parquet" \
RL_TRAIN_BATCH_SIZE=16 \
bash "$ROOT/scripts/train_rl_native_pilot_verl.sh"
train_status=$?
set -e
kill "$plot_pid" 2>/dev/null || true
wait "$plot_pid" 2>/dev/null || true
trap - EXIT
if [[ -s "$RUN/metrics.jsonl" ]]; then
  "$PYTHON" "$PLOT" "$RUN/metrics.jsonl" --output "$RUN/rl_training_dashboard.png" || true
fi
printf 'training_exit_code=%s\nfinished_at=%s\n' "$train_status" "$(date -Is)" > "$RUN/queue_status.txt"

# A zero-variance gate can end training after a saved online update. Evaluate
# only an actually saved checkpoint, on the same frozen held-out task cells.
if [[ -s "$RUN/checkpoints/latest_checkpointed_iteration.txt" ]]; then
  checkpoint_step="$(cat "$RUN/checkpoints/latest_checkpointed_iteration.txt")"
  if [[ "$checkpoint_step" =~ ^[1-9][0-9]*$ ]]; then
    echo "$(date -Is) paired held-out evaluation of checkpoint $checkpoint_step" >&2
    set +e
    EVAL_GPU=1 EVAL_NAME=heldout_agentdojo_validation_budget2048 \
      bash "$ROOT/scripts/evaluate_rl_sft_pair.sh" "$RUN" "$checkpoint_step"
    eval_status=$?
    set -e
    printf 'validation_exit_code=%s\n' "$eval_status" >> "$RUN/queue_status.txt"
  fi
fi
exit "$train_status"
