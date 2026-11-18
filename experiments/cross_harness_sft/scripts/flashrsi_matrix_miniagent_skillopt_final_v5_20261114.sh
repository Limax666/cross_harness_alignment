#!/usr/bin/env bash
set -euo pipefail
cd /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft
export PYTHONPATH=src:scripts:vendor/FlashRSI HR_HARNESS=miniagent HARNESSRISK_MODEL_PORT=18762
export HR_STRICT_EVOLUTION=1 HR_EVOLUTION_CASE_ATTEMPTS=1 HR_WORKER_MAX_TOKENS=4096
export HR_HOST_SCRIPT=flashrsi_harnessrisk_host_final_v5.py
export HR_OUTPUT_ROOT=/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_final_v5_20261114
exec 9>"$HR_OUTPUT_ROOT/skillopt_launch.lock"
flock -n 9 || exit 75
trap '.venv-rl-pilot/bin/python scripts/flashrsi_matrix_repair_loop.py --scan > "$HR_OUTPUT_ROOT/skillopt_scan_after.json"' EXIT
.venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm skillopt > "$HR_OUTPUT_ROOT/miniagent_skillopt_full.log" 2>&1
.venv-rl-pilot/bin/python - <<'GATE'
import json,os
from pathlib import Path
from flashrsi_matrix_repair_loop import inspect_run
base=Path(os.environ['HR_OUTPUT_ROOT']);run=base/'miniagent_skillopt_full'
expected={x['case_id'] for x in json.loads((base/'case_manifest.json').read_text())['cases']}
row=inspect_run(run,expected,None)
(run/'prejudge_gate.json').write_text(json.dumps(row,indent=2)+'\n')
if row['status'] not in ('needs_judge','complete'):raise SystemExit('Full128 scoring/identity/privacy gate failed; no judge invoked')
GATE
.venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$HR_OUTPUT_ROOT/miniagent_skillopt_full/selected_cases" --deployment openai/gpt-5.4-nano > "$HR_OUTPUT_ROOT/miniagent_skillopt_judge.log" 2>&1
