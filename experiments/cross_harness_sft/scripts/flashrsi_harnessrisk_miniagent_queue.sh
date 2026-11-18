#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/flashrsi_harnessrisk_4b_20261113
while tmux has-session -t =flashrsi-harnessrisk-miniagent 2>/dev/null; do sleep 15; done
[[ -f "$root/miniagent_initial_full_v2/completed.json" ]] || { echo 'Initial baseline incomplete; RSI queue stopped.'; exit 1; }
export PYTHONPATH=src:scripts:vendor/FlashRSI
for algorithm in opro gepa evoprompt ace skillopt evo_harness; do
 .venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm "$algorithm" > "$root/miniagent_${algorithm}_full.log" 2>&1
 .venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/miniagent_${algorithm}_full/selected_cases" --deployment openai/gpt-5.4-nano > "$root/miniagent_${algorithm}_judge.log" 2>&1 || echo "$algorithm judge incomplete; rule results retained"
 done
