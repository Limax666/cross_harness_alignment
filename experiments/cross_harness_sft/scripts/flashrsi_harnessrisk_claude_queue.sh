#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/flashrsi_harnessrisk_4b_20261113
while tmux has-session -t =flashrsi-harnessrisk-ports 2>/dev/null; do sleep 15; done
for harness in opencode openclaw hermes pi; do
 for algorithm in opro gepa evoprompt ace; do
  [[ -f "$root/${harness}_${algorithm}_full/result.json" ]] || { echo "${harness} ${algorithm} incomplete; Claude Code queue stopped"; exit 1; }
 done
done
export PYTHONPATH=src:scripts:vendor/FlashRSI HR_HARNESS=claude_code HR_RUN_NAME=claude_code_initial_full
.venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_miniagent_initial.py > "$root/claude_code_initial_full.log" 2>&1
.venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/claude_code_initial_full/cases" --deployment openai/gpt-5.4-nano > "$root/claude_code_initial_judge.log" 2>&1 || echo 'Claude Code initial judge incomplete; rule results retained'
for algorithm in opro gepa evoprompt ace skillopt evo_harness; do
 .venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm "$algorithm" > "$root/claude_code_${algorithm}_full.log" 2>&1
 .venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/claude_code_${algorithm}_full/selected_cases" --deployment openai/gpt-5.4-nano > "$root/claude_code_${algorithm}_judge.log" 2>&1 || echo "Claude Code $algorithm judge incomplete; rule results retained"
done
