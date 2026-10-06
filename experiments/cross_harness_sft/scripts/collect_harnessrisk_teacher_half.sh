#!/usr/bin/env bash
# Collect exactly 64 real GPT-5.6 Sol HarnessRisk trajectories with an official harness adapter.
set -euo pipefail

PROJECT_ROOT="/data/home/liumingxiao/cross_harness_alignment"
BENCHMARK_ROOT="$PROJECT_ROOT/HarnessRisk"
EXPERIMENT_ROOT="$PROJECT_ROOT/experiments/cross_harness_sft"
ENV_FILE="$EXPERIMENT_ROOT/.env"
HARNESS="${HARNESS:?Set HARNESS to hermes or nanobot}"
[[ "$HARNESS" == hermes || "$HARNESS" == nanobot ]] || { echo "HARNESS must be hermes or nanobot" >&2; exit 2; }
OUTPUT_ROOT="$EXPERIMENT_ROOT/outputs/harnessrisk_${HARNESS}_gpt56_sol_half"
STATE_DIR="$OUTPUT_ROOT/state"
RUN_ROOT="$OUTPUT_ROOT/runs"
SELECTION="$OUTPUT_ROOT/real_64_cases.json"
HERMES_CMD="${HERMES_CMD:-/data/home/liumingxiao/.hermes/hermes-agent/venv/bin/hermes}"
NANOBOT_CMD="${NANOBOT_CMD:-$EXPERIMENT_ROOT/.venv-nanobot/bin/nanobot}"

[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE" >&2; exit 2; }
[[ "$HARNESS" != hermes || -x "$HERMES_CMD" ]] || { echo "Hermes CLI unavailable: $HERMES_CMD" >&2; exit 2; }
[[ "$HARNESS" != nanobot || -x "$NANOBOT_CMD" ]] || { echo "Nanobot CLI unavailable: $NANOBOT_CMD" >&2; exit 2; }

set -a
source "$ENV_FILE"
set +a
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is empty" >&2; exit 2; }
if [[ "${DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy ALL_PROXY all_proxy
else
  export HTTPS_PROXY="${HTTPS_PROXY:-http://ustc-course.com:48527}"
  export HTTP_PROXY="${HTTP_PROXY:-$HTTPS_PROXY}"
  export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="$NO_PROXY"
fi

mkdir -p "$OUTPUT_ROOT"
python3 "$EXPERIMENT_ROOT/scripts/select_harnessrisk_cases.py" \
  --data-dir "$BENCHMARK_ROOT/data/HarnessRisk" --count 64 \
  --output "$SELECTION"
CASES="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["case_ids"]))' "$SELECTION")"

CMD="$HERMES_CMD"
[[ "$HARNESS" == nanobot ]] && CMD="$NANOBOT_CMD"
"$BENCHMARK_ROOT/harness_adapter/scripts/setup_scripts/setup_agent.sh" \
  --harness "$HARNESS" --provider openrouter --model openai/gpt-5.6-sol \
  --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
  --cmd "$CMD" --state-dir "$STATE_DIR" --run-root "$RUN_ROOT" --strict

BATCH_ID="gpt56-sol-${HARNESS}-real64-$(date -u +%Y%m%dT%H%M%SZ)"
"$BENCHMARK_ROOT/harness_adapter/scripts/run_case_scripts/run_batch.sh" \
  --harness "$HARNESS" --multiturn --data-dir "$BENCHMARK_ROOT/data/HarnessRisk" \
  --run-root "$RUN_ROOT" --batch-id "$BATCH_ID" --cases "$CASES" -- \
  --state-dir "$STATE_DIR" --cmd "$CMD" --model openai/gpt-5.6-sol \
  --provider openrouter --api-key-env OPENROUTER_API_KEY
