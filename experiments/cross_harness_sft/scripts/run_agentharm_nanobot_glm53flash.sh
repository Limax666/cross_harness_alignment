#!/usr/bin/env bash
# Full AgentHarm test_public collection through the real NanoBot CLI.
set -euo pipefail

ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
ENV_FILE="$EXP/.agentharm_token_plan.env"
[[ -f "$ENV_FILE" ]] || { echo "Missing credential env file: $ENV_FILE" >&2; exit 2; }

set -a
source "$ENV_FILE"
set +a
[[ -n "${SHENGSUANYUN_API_KEY:-}" ]] || { echo "SHENGSUANYUN_API_KEY is unset" >&2; exit 2; }

export AGENTHARM_CONFIG="$EXP/configs/agentharm_nanobot_glm53flash_test_public.yaml"
export AGENTHARM_WORKER_CONFIG="$EXP/configs/agentharm_worker_test_public.yaml"
export AGENTHARM_PORT="${AGENTHARM_PORT:-18114}"
export AGENTHARM_EXPECTED_SPLIT=test_public
export AGENTHARM_PYTHON="${AGENTHARM_PYTHON:-/data/home/liumingxiao/.venvs/cross-harness-agentharm/bin/python}"

cd "$ROOT"
exec bash "$EXP/scripts/run_agentharm_native.sh" "$@"
