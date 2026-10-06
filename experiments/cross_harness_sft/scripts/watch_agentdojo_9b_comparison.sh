#!/usr/bin/env bash
# Keep the paired figures and provenance manifest current during evaluation.
set -euo pipefail

EXP="/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft"
OUT="$EXP/outputs/eval_agentdojo_9b"
MODEL_PY="/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python"

while true; do
  "$MODEL_PY" "$EXP/scripts/plot_agentdojo_9b_comparison.py" --output-dir "$OUT"
  if ! tmux has-session -t agentdojo-eval-base 2>/dev/null && \
     ! tmux has-session -t agentdojo-eval-sft 2>/dev/null; then
    break
  fi
  sleep 30
done
