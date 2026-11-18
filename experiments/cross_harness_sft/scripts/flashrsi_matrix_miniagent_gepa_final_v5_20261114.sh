#!/usr/bin/env bash
set -euo pipefail
cd /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft
export PYTHONPATH=src:scripts:vendor/FlashRSI HR_HARNESS=miniagent HARNESSRISK_MODEL_PORT=18761
export HR_STRICT_EVOLUTION=1 HR_EVOLUTION_CASE_ATTEMPTS=1 HR_WORKER_MAX_TOKENS=4096
export HR_HOST_SCRIPT=flashrsi_harnessrisk_host_final_v5.py
export HR_OUTPUT_ROOT=/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_final_v5_20261114
exec 9>"$HR_OUTPUT_ROOT/launch.lock"
flock -n 9 || exit 75
.venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm gepa > "$HR_OUTPUT_ROOT/miniagent_gepa_full.log" 2>&1
.venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$HR_OUTPUT_ROOT/miniagent_gepa_full/selected_cases" --deployment openai/gpt-5.4-nano > "$HR_OUTPUT_ROOT/miniagent_gepa_judge.log" 2>&1
.venv-rl-pilot/bin/python scripts/flashrsi_matrix_repair_loop.py --scan > "$HR_OUTPUT_ROOT/scan_after.json"
