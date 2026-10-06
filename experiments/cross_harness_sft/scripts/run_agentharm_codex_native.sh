#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
PYTHON=${AGENTHARM_PYTHON:-/data/home/liumingxiao/.venvs/cross-harness-agentharm/bin/python}
CONFIG=${AGENTHARM_CONFIG:-$EXP/configs/agentharm_codex_gpt56_sol_val.yaml}
WORKER_CONFIG=${AGENTHARM_WORKER_CONFIG:-$EXP/configs/agentharm_worker_val.yaml}
PORT=${AGENTHARM_PORT:-18112}
export AGENTHARM_BASE_URL="http://127.0.0.1:$PORT"

if [[ ! -x "$PYTHON" ]]; then
  echo "AgentHarm runtime is unavailable: $PYTHON" >&2
  exit 2
fi

mkdir -p "$EXP/runs/workers" "$EXP/outputs/agentharm_codex_gpt56_sol_native"

if ! curl -fsS "$AGENTHARM_BASE_URL/health" >/dev/null 2>&1; then
  PYTHONPATH="$EXP/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m cross_harness_sft.benchmark_server \
      --driver cross_harness_sft.backends.agentharm:create_driver \
      --config "$WORKER_CONFIG" --port "$PORT" \
      >"$EXP/runs/workers/agentharm-val.log" 2>&1 &
  worker_pid=$!
  echo "$worker_pid" >"$EXP/runs/workers/agentharm-val.pid"
  for _ in $(seq 1 60); do
    curl -fsS "$AGENTHARM_BASE_URL/health" >/dev/null 2>&1 && break
    sleep 1
  done
fi

curl -fsS "$AGENTHARM_BASE_URL/health" >/dev/null
health_json=$(curl -fsS "$AGENTHARM_BASE_URL/health")
if [[ "$health_json" != *"agentharm/"* ]]; then
  echo "Unexpected worker on $AGENTHARM_BASE_URL: $health_json" >&2
  exit 3
fi
cd "$ROOT"
exec "$PYTHON" -m cross_harness_sft.collect_native --config "$CONFIG" "$@"
