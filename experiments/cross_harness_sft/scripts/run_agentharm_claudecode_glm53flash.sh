#!/usr/bin/env bash
# Full AgentHarm test_public collection through Claude Code and a local facade.
set -euo pipefail
ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
set -a
source "$EXP/.agentharm_token_plan.env"
set +a
export FACADE_LOCAL_TOKEN="${FACADE_LOCAL_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8411/api/hello || true)" = "000" ]; then
  python3 "$EXP/scripts/shengsuanyun_anthropic_facade.py" --port 8411 \
    >"$EXP/outputs/agentharm_claudecode_glm53flash_native/facade.log" 2>&1 &
  facade_pid=$!
  trap 'kill "$facade_pid" 2>/dev/null || true' EXIT
  for _ in $(seq 1 30); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8411/api/hello || true)" != "000" ] && break
    sleep 1
  done
fi
export AGENTHARM_CONFIG="$EXP/configs/agentharm_claudecode_glm53flash_test_public.yaml"
export AGENTHARM_WORKER_CONFIG="$EXP/configs/agentharm_worker_test_public.yaml"
export AGENTHARM_PORT="${AGENTHARM_PORT:-18122}"
export AGENTHARM_EXPECTED_SPLIT=test_public
export AGENTHARM_PYTHON="${AGENTHARM_PYTHON:-/data/home/liumingxiao/.venvs/cross-harness-agentharm/bin/python}"
cd "$ROOT"
bash "$EXP/scripts/run_agentharm_native.sh" "$@"
