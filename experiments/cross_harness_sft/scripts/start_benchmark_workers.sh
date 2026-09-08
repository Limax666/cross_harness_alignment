#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../" && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
mkdir -p "$EXP/runs/workers"
"${AGENTDOJO_PYTHON:-$EXP/.venv/bin/python}" -m cross_harness_sft.benchmark_server --driver cross_harness_sft.backends.agentdojo:create_driver \
  --config "$EXP/configs/agentdojo_worker.yaml" --port 8111 >"$EXP/runs/workers/agentdojo.log" 2>&1 &
echo $! >"$EXP/runs/workers/agentdojo.pid"
"${AGENTHARM_PYTHON:-$EXP/.venv/bin/python}" -m cross_harness_sft.benchmark_server --driver cross_harness_sft.backends.agentharm:create_driver \
  --config "$EXP/configs/agentharm_worker.yaml" --port 8112 >"$EXP/runs/workers/agentharm.log" 2>&1 &
echo $! >"$EXP/runs/workers/agentharm.pid"
echo "Workers started. Inspect logs and require both /health endpoints before collection."
