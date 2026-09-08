#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
CONFIG="${CONFIG:-$EXP/configs/codex_agentdojo_native_full.yaml}"
PYTHON="$EXP/.venv/bin/python"
WORKER_LOG="$EXP/runs/workers/agentdojo.log"
WORKER_PID="$EXP/runs/workers/agentdojo.pid"

[[ -x "$PYTHON" ]] || { echo "Missing project environment: $PYTHON" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Missing collection config: $CONFIG" >&2; exit 1; }
[[ -f /root/.codex/auth.json ]] || { echo "Codex login missing; run: codex login" >&2; exit 1; }

codex login status
mkdir -p "$EXP/runs/workers"
started_worker=0
cleanup() {
  if [[ "$started_worker" == 1 ]] && [[ -f "$WORKER_PID" ]]; then
    kill "$(<"$WORKER_PID")" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! curl --fail --silent --show-error http://127.0.0.1:8111/health >/dev/null 2>&1; then
  "$PYTHON" -m cross_harness_sft.benchmark_server \
    --driver cross_harness_sft.backends.agentdojo:create_driver \
    --config "$EXP/configs/agentdojo_worker.yaml" --port 8111 \
    >"$WORKER_LOG" 2>&1 &
  echo $! >"$WORKER_PID"
  started_worker=1
  for _ in $(seq 1 60); do
    curl --fail --silent http://127.0.0.1:8111/health >/dev/null 2>&1 && break
    if ! kill -0 "$(<"$WORKER_PID")" 2>/dev/null; then
      echo "AgentDojo worker exited; see $WORKER_LOG" >&2
      exit 1
    fi
    sleep 1
  done
fi

curl --fail --silent --show-error http://127.0.0.1:8111/health
echo
"$PYTHON" -m cross_harness_sft.preflight --config "$CONFIG" --mode collect
# Collection only: no verification/rejection filtering is performed here.
"$PYTHON" -m cross_harness_sft.collect_native --config "$CONFIG" "$@"
