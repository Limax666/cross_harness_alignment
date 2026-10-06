#!/usr/bin/env bash
# Run after teacher collection finishes. Re-score old saved traces with the
# current official rubric and Qwen judge, without invoking any teacher model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
ENV_FILE="$EXP/.agentharm_token_plan.env"
[[ -f "$ENV_FILE" ]] || { echo "Missing judge credential env file" >&2; exit 2; }
set -a
source "$ENV_FILE"
set +a
[[ -n "${AGENTHARM_JUDGE_API_KEY:-}" ]] || { echo "AGENTHARM_JUDGE_API_KEY is unset" >&2; exit 2; }
PYTHON="${AGENTHARM_PYTHON:-/data/home/liumingxiao/.venvs/cross-harness-agentharm/bin/python}"
for source in "$EXP"/outputs/agentharm*/test_public_v1.jsonl; do
  [[ -f "$source" ]] || continue
  output="${source%.jsonl}_rejudged.jsonl"
  "$PYTHON" "$EXP/scripts/rejudge_agentharm_native_sft.py" --input "$source" --output "$output"
done
"$PYTHON" "$EXP/scripts/select_agentharm_native_sft.py"
