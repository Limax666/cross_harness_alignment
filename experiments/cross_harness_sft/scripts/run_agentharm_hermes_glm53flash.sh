#!/usr/bin/env bash
# Full AgentHarm test_public collection through the real Hermes CLI.
set -euo pipefail
ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
set -a
source "$EXP/.agentharm_token_plan.env"
set +a
export AGENTHARM_CONFIG="$EXP/configs/agentharm_hermes_qwen38max_test_public.yaml"
export AGENTHARM_WORKER_CONFIG="$EXP/configs/agentharm_worker_test_public.yaml"
export AGENTHARM_PORT="${AGENTHARM_PORT:-18120}"
export AGENTHARM_EXPECTED_SPLIT=test_public
export AGENTHARM_PYTHON="${AGENTHARM_PYTHON:-/data/home/liumingxiao/.venvs/cross-harness-agentharm/bin/python}"
cd "$ROOT"
exec bash "$EXP/scripts/run_agentharm_native.sh" "$@"
