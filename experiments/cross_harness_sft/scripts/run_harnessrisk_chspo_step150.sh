#!/usr/bin/env bash
# One independent harness job; repeat seeds and judging follow the frozen v7 protocol.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${HARNESS:?Set HARNESS=hermes, nanobot, or openclaw}"
case "$HARNESS" in hermes|nanobot|openclaw) ;; *) exit 2 ;; esac
export ROLE=rl
export RL_MODEL="${RL_MODEL:-$EXP/outputs/rl/multiharness_chspo_r20g_v4_from130_20261012/hf_merged_step_150}"
export OUT="${OUT:-$EXP/outputs/harnessrisk_qwen35_2b_chspo_v4_step150}"
export RUN_TAG="${RUN_TAG:-chspo150v4p1}"
export HARNESS_LIST="$HARNESS"
export REPETITIONS="${REPETITIONS:-3}"
export JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.4-nano}"
# This evaluation always covers all 128 cases. A CASES/BATCH_CASES value
# inherited from a prior subset run silently narrowed the Hermes batch to 38.
unset CASES BATCH_CASES BATCH_CASE_GLOB
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec nice -n 10 ionice -c 3 bash "$EXP/scripts/run_harnessrisk_qwen35_2b_v7_matrix.sh"
