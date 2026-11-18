#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/flashrsi_harnessrisk_4b_20261113
while tmux has-session -t =flashrsi-harnessrisk-rsi 2>/dev/null; do sleep 15; done
for algorithm in opro gepa evoprompt ace skillopt evo_harness; do
 [[ -f "$root/miniagent_${algorithm}_full/result.json" ]] || { echo "MiniAgent $algorithm incomplete; port queue stopped"; exit 1; }
done
while tmux has-session -t =flashrsi-harnessrisk-opencode 2>/dev/null; do sleep 15; done
for algorithm in opro gepa evoprompt ace; do
 [[ -f "$root/opencode_${algorithm}_full/result.json" ]] || { echo "OpenCode $algorithm incomplete; port queue stopped"; exit 1; }
done
export PYTHONPATH=src:scripts:vendor/FlashRSI
for harness in openclaw hermes pi; do
 export HR_HARNESS="$harness" HR_RUN_NAME="${harness}_initial_full"
 .venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_miniagent_initial.py > "$root/${harness}_initial_full.log" 2>&1
 for algorithm in opro gepa evoprompt ace; do
  .venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm "$algorithm" > "$root/${harness}_${algorithm}_full.log" 2>&1
  .venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/${harness}_${algorithm}_full/selected_cases" --deployment openai/gpt-5.4-nano > "$root/${harness}_${algorithm}_judge.log" 2>&1 || echo "$harness $algorithm judge incomplete; rule results retained"
 done
 if [[ "$harness" != opencode ]]; then
  for algorithm in skillopt evo_harness; do
   .venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm "$algorithm" > "$root/${harness}_${algorithm}_full.log" 2>&1
   .venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/${harness}_${algorithm}_full/selected_cases" --deployment openai/gpt-5.4-nano > "$root/${harness}_${algorithm}_judge.log" 2>&1 || echo "$harness $algorithm judge incomplete; rule results retained"
  done
 else
  printf '%s\n' 'SkillOpt and Evo-Harness unavailable: upstream OpenCode Snapshot declares no skill component.' > "$root/${harness}_skill_algorithms_unavailable.txt"
 fi
done
