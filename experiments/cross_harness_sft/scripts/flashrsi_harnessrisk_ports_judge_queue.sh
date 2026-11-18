#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/flashrsi_harnessrisk_4b_20261113
for harness in opencode openclaw hermes pi; do
 while [[ ! -f "$root/${harness}_initial_full/completed.json" ]]; do
  if ! tmux has-session -t =flashrsi-harnessrisk-ports 2>/dev/null; then echo "Port queue ended before $harness initial baseline"; exit 1; fi
  sleep 15
 done
 case_dir="$root/${harness}_initial_full/cases"
 if [[ "$harness" == opencode ]]; then
  while [[ ! -f "$root/opencode_initial_full/retry_finished.json" ]]; do
   if ! tmux has-session -t =flashrsi-harnessrisk-opencode 2>/dev/null; then echo 'OpenCode retry did not finish'; exit 1; fi
   sleep 15
  done
  case_dir="$root/opencode_initial_full/paired_cases"
 fi
 .venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$case_dir" --deployment openai/gpt-5.4-nano > "$root/${harness}_initial_judge.log" 2>&1 || echo "$harness initial judge incomplete; rule results retained"
done
