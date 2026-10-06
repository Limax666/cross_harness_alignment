#!/usr/bin/env bash
# Collect the full HarnessRisk benchmark with the official OpenClaw adapter.
set -euo pipefail

PROJECT_ROOT="/data/home/liumingxiao/cross_harness_alignment"
BENCHMARK_ROOT="$PROJECT_ROOT/HarnessRisk"
EXPERIMENT_ROOT="$PROJECT_ROOT/experiments/cross_harness_sft"
ENV_FILE="$EXPERIMENT_ROOT/.env"
OPENCLAW_ROOT="$PROJECT_ROOT/openclaw-runtime/node_modules/openclaw"
OUTPUT_ROOT="$EXPERIMENT_ROOT/outputs/harnessrisk_openclaw_gpt56_sol"
STATE_DIR="$OUTPUT_ROOT/state"
RUN_ROOT="$OUTPUT_ROOT/runs"

[[ -f "$ENV_FILE" ]] || { echo "Missing OpenRouter env file: $ENV_FILE" >&2; exit 2; }
[[ -d "$OPENCLAW_ROOT" && -f "$OPENCLAW_ROOT/scripts/run-node.mjs" ]] || {
  echo "OpenClaw source is unavailable at $OPENCLAW_ROOT." >&2
  echo "Install the official OpenClaw source there before starting collection." >&2
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
export OPENCLAW_REPO_ROOT="$OPENCLAW_ROOT"

mkdir -p "$OUTPUT_ROOT"

"$BENCHMARK_ROOT/harness_adapter/scripts/setup_scripts/setup_agent.sh" \
  --harness openclaw \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --compatibility openai \
  --state-dir "$STATE_DIR" \
  --non-interactive \
  --strict

batch_id="gpt56-sol-openclaw-$(date -u +%Y%m%dT%H%M%SZ)"
"$BENCHMARK_ROOT/harness_adapter/scripts/run_case_scripts/run_batch.sh" \
  --harness openclaw \
  --data-dir "$BENCHMARK_ROOT/data/HarnessRisk" \
  --run-root "$RUN_ROOT" \
  --batch-id "$batch_id" \
  -- \
  --state-dir "$STATE_DIR" \
  --model openrouter/openai/gpt-5.6-sol \
  --api-key-env OPENROUTER_API_KEY
