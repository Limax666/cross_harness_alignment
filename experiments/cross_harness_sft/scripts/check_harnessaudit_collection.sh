#!/usr/bin/env bash
# Lightweight, read-only progress monitor for the two HarnessAudit collectors.
set -euo pipefail

BASE="/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs"
MONITOR_DIR="$BASE/harnessaudit_collection_monitor"
mkdir -p "$MONITOR_DIR"

count_files() {
  local dir="$1" pattern="$2"
  if [[ -d "$dir" ]]; then
    find "$dir" -type f -name "$pattern" | wc -l | tr -d ' '
  else
    echo 0
  fi
}

recent_errors() {
  local log_dir="$1"
  local count
  if [[ ! -d "$log_dir" ]]; then
    echo 0
    return
  fi
  count="$({ find "$log_dir" -type f -mmin -6 -print0 | xargs -0r grep -Ehi \
    'traceback|runner error|api_error|authentication_failed|failed to spawn|error:|no (codex )?rollout matched|actions: 0' | wc -l | tr -d ' '; } || true)"
  echo "${count:-0}"
}

runner_active() {
  local marker="$1"
  if ps -eo args= | awk -v marker="$marker" \
    'index($0, marker) && index($0, "python -m multi_agent run") { found = 1 } END { exit !found }'; then
    echo true
  else
    echo false
  fi
}

collector_json() {
  local name="$1" root="$2" raw_dir="$3" raw_pattern="$4"
  local raw traces reports errors active
  raw="$(count_files "$root/$raw_dir" "$raw_pattern")"
  traces="$(count_files "$root/traces" '*.jsonl')"
  reports="$(count_files "$root/results" '*.json')"
  errors="$(recent_errors "$root/run_logs")"
  active="$(runner_active "$root")"
  printf '    "%s": {"runner_active": %s, "raw_sessions": %s, "normalized_traces": %s, "reports": %s, "recent_log_errors": %s}' \
    "$name" "$active" "$raw" "$traces" "$reports" "$errors"
}

timestamp="$(date --iso-8601=seconds)"
tmp="$MONITOR_DIR/collection_status.json.tmp"
{
  printf '{\n  "checked_at": "%s",\n' "$timestamp"
  collector_json \
    codex \
    "$BASE/harnessaudit_codex_gpt56_sol_full" \
    codex_isolation \
    'rollout-*.jsonl'
  printf ',\n'
  collector_json \
    claude \
    "$BASE/harnessaudit_claude_openrouter_gpt56_sol_full" \
    claude_isolation \
    '*.jsonl'
  printf '\n}\n'
} > "$tmp"
mv "$tmp" "$MONITOR_DIR/collection_status.json"
cat "$MONITOR_DIR/collection_status.json" >> "$MONITOR_DIR/collection_status.log"
