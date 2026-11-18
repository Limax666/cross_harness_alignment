#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/flashrsi_harnessrisk_4b_20261113
while tmux has-session -t =flashrsi-harnessrisk-claude 2>/dev/null; do sleep 15; done
for algorithm in opro gepa evoprompt ace skillopt evo_harness; do
 [[ -f "$root/claude_code_${algorithm}_full/result.json" ]] || { echo "Claude Code ${algorithm} incomplete; Codex queue stopped"; exit 1; }
done
export PYTHONPATH=src:scripts:vendor/FlashRSI HR_HARNESS=codex HR_RUN_NAME=codex_initial_full
.venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_miniagent_initial.py > "$root/codex_initial_full.log" 2>&1
.venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/codex_initial_full/cases" --deployment openai/gpt-5.4-nano > "$root/codex_initial_judge.log" 2>&1 || echo 'Codex initial judge incomplete; rule results retained'
for algorithm in opro gepa evoprompt ace skillopt evo_harness; do
 .venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm "$algorithm" > "$root/codex_${algorithm}_full.log" 2>&1
 .venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/codex_${algorithm}_full/selected_cases" --deployment openai/gpt-5.4-nano > "$root/codex_${algorithm}_judge.log" 2>&1 || echo "Codex $algorithm judge incomplete; rule results retained"
done
