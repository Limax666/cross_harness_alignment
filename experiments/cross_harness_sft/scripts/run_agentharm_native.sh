#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
PYTHON=${AGENTHARM_PYTHON:-/data/home/liumingxiao/.venvs/cross-harness-agentharm/bin/python}
CONFIG=${AGENTHARM_CONFIG:?set AGENTHARM_CONFIG to a native collector YAML}
WORKER_CONFIG=${AGENTHARM_WORKER_CONFIG:?set AGENTHARM_WORKER_CONFIG to an AgentHarm worker YAML}
PORT=${AGENTHARM_PORT:-18113}
EXPECTED_SPLIT=${AGENTHARM_EXPECTED_SPLIT:-test_public}
export AGENTHARM_BASE_URL="http://127.0.0.1:$PORT"

if [[ ! -x "$PYTHON" ]]; then
  echo "AgentHarm runtime is unavailable: $PYTHON" >&2
  exit 2
fi

mkdir -p "$EXP/runs/workers"
health_json=$(curl -fsS "$AGENTHARM_BASE_URL/health" 2>/dev/null || true)
if [[ -z "$health_json" ]]; then
  PYTHONPATH="$EXP/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m cross_harness_sft.benchmark_server \
      --driver cross_harness_sft.backends.agentharm:create_driver \
      --config "$WORKER_CONFIG" --port "$PORT" \
      >"$EXP/runs/workers/agentharm-${EXPECTED_SPLIT}-${PORT}.log" 2>&1 &
  worker_pid=$!
  echo "$worker_pid" >"$EXP/runs/workers/agentharm-${EXPECTED_SPLIT}-${PORT}.pid"
  for _ in $(seq 1 90); do
    health_json=$(curl -fsS "$AGENTHARM_BASE_URL/health" 2>/dev/null || true)
    [[ -n "$health_json" ]] && break
    sleep 1
  done
fi

if [[ "$health_json" != *"agentharm/${EXPECTED_SPLIT}"* ]]; then
  echo "Unexpected or unavailable worker on $AGENTHARM_BASE_URL: $health_json" >&2
  exit 3
fi

cd "$ROOT"
exec "$PYTHON" -m cross_harness_sft.collect_native --config "$CONFIG" "$@"
