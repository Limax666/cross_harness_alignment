#!/usr/bin/env bash
# Collect the full HarnessRisk benchmark with the official Hermes adapter.
set -euo pipefail

PROJECT_ROOT="/data/home/liumingxiao/cross_harness_alignment"
BENCHMARK_ROOT="$PROJECT_ROOT/HarnessRisk"
EXPERIMENT_ROOT="$PROJECT_ROOT/experiments/cross_harness_sft"
ENV_FILE="$EXPERIMENT_ROOT/.env"
OUTPUT_ROOT="$EXPERIMENT_ROOT/outputs/harnessrisk_hermes_gpt56_sol"
STATE_DIR="$OUTPUT_ROOT/state"
RUN_ROOT="$OUTPUT_ROOT/runs"
HERMES_CMD="/data/home/liumingxiao/miniforge3/envs/harnessaudit/bin/hermes"

[[ -f "$ENV_FILE" ]] || { echo "Missing OpenRouter env file: $ENV_FILE" >&2; exit 2; }
[[ -x "$HERMES_CMD" ]] || {
  echo "Official Hermes CLI is unavailable: $HERMES_CMD" >&2
  echo "Install Hermes before starting collection." >&2
  exit 2
}

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is empty" >&2; exit 2; }

# OpenRouter access on this host requires the user-provided proxy.
export HTTPS_PROXY="http://ustc-course.com:48527"
export HTTP_PROXY="$HTTPS_PROXY"
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

mkdir -p "$OUTPUT_ROOT"

"$BENCHMARK_ROOT/harness_adapter/scripts/setup_scripts/setup_agent.sh" \
  --harness hermes \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --cmd "$HERMES_CMD" \
  --state-dir "$STATE_DIR" \
  --run-root "$RUN_ROOT" \
  --strict

batch_id="gpt56-sol-hermes-$(date -u +%Y%m%dT%H%M%SZ)"
"$BENCHMARK_ROOT/harness_adapter/scripts/run_case_scripts/run_batch.sh" \
  --harness hermes \
  --data-dir "$BENCHMARK_ROOT/data/HarnessRisk" \
  --run-root "$RUN_ROOT" \
  --batch-id "$batch_id" \
  -- \
  --state-dir "$STATE_DIR" \
  --cmd "$HERMES_CMD" \
  --model openai/gpt-5.6-sol \
  --provider custom:openrouter \
  --api-key-env OPENROUTER_API_KEY
