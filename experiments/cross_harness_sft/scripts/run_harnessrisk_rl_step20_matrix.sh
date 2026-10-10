#!/usr/bin/env bash
# Evaluate the R11 step-20 merged RL model with the v7 SFT HarnessRisk protocol.
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROLE=rl
export RL_MODEL="${RL_MODEL:-$EXP/multiharness_grpo_pool107_20261007_r11-300steps/hf_merged_step_20}"
export OUT="${OUT:-$EXP/outputs/harnessrisk_qwen35_2b_rl_step20}"
export RUN_TAG="${RUN_TAG:-rl20p1}"
export REPETITIONS="${REPETITIONS:-3}"
export HARNESS_LIST="${HARNESS_LIST:-hermes nanobot openclaw}"
export JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.4-nano}"

exec bash "$EXP/scripts/run_harnessrisk_qwen35_2b_v7_matrix.sh"
