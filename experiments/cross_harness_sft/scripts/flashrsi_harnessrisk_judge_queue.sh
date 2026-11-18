#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/flashrsi_harnessrisk_4b_20261113
while tmux has-session -t =flashrsi-harnessrisk-miniagent 2>/dev/null; do sleep 15; done
[[ -f "$root/miniagent_initial_full_v2/completed.json" ]] || { echo 'initial baseline incomplete; judge not started'; exit 1; }
.venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/miniagent_initial_full_v2/cases" --deployment openai/gpt-5.4-nano > "$root/miniagent_initial_judge.log" 2>&1
