#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")/.."
: "${HR_OUTPUT_ROOT:?set a fresh output root}"
: "${HARNESSRISK_MODEL_PORT:?set model port}"
if [[ $# -eq 0 ]]; then echo 'expected harness names' >&2; exit 2; fi
export PYTHONPATH=src:scripts:vendor/FlashRSI
root="$HR_OUTPUT_ROOT"
python=.venv-rl-pilot/bin/python
judge="$python scripts/judge_harnessrisk_shengsuanyun.py"
for harness in "$@"; do
 export HR_HARNESS="$harness"
 if [[ "$harness" == miniagent ]]; then export HR_RUN_NAME=miniagent_initial_full_v2; else export HR_RUN_NAME="${harness}_initial_full"; fi
 echo "$(date -Is) START $harness initial"
 if ! "$python" -u scripts/flashrsi_harnessrisk_miniagent_initial.py > "$root/${harness}_initial_full.log" 2>&1; then
  echo "$(date -Is) FAILED $harness initial; skip algorithms"; continue
 fi
 if ! "$python" -c 'import json,sys; x=json.load(open(sys.argv[1])); sys.exit(0 if x.get("completed_scored",0)>0 else 1)' "$root/$HR_RUN_NAME/completed.json"; then
  echo "$(date -Is) FAILED $harness initial has zero scored cases; skip algorithms"; continue
 fi
 if ! "$python" scripts/flashrsi_harnessrisk_retry_infrastructure.py "$root/$HR_RUN_NAME" >> "$root/${harness}_initial_full.log" 2>&1; then
  echo "$(date -Is) WARNING $harness infrastructure retry incomplete"
 fi
 initial_cases="$root/$HR_RUN_NAME/cases"
 if [[ -d "$root/$HR_RUN_NAME/paired_cases" ]]; then initial_cases="$root/$HR_RUN_NAME/paired_cases"; fi
 $judge "$initial_cases" --deployment openai/gpt-5.4-nano > "$root/${harness}_initial_judge.log" 2>&1 &
 judge_pid=$!
 algorithms=(opro gepa evoprompt ace skillopt evo_harness)
 if [[ "$harness" == opencode ]]; then algorithms=(opro gepa evoprompt ace); fi
 for algorithm in "${algorithms[@]}"; do
  echo "$(date -Is) START $harness $algorithm"
  if "$python" -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm "$algorithm" > "$root/${harness}_${algorithm}_full.log" 2>&1; then
   $judge "$root/${harness}_${algorithm}_full/selected_cases" --deployment openai/gpt-5.4-nano > "$root/${harness}_${algorithm}_judge.log" 2>&1 || echo "$(date -Is) WARNING $harness $algorithm judge incomplete"
  else
   echo "$(date -Is) FAILED $harness $algorithm; continuing to next method"
  fi
 done
 wait "$judge_pid" || echo "$(date -Is) WARNING $harness initial judge incomplete"
 if [[ "$harness" == opencode ]]; then printf '%s\n' 'SkillOpt and Evo-Harness unavailable: OpenCode Snapshot declares no skill component.' > "$root/opencode_skill_algorithms_unavailable.txt"; fi
 echo "$(date -Is) FINISH $harness"
done
