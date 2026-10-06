#!/usr/bin/env bash
# Resume Hermes Base or full-SFT in a dedicated process and model server.
# For two-GPU concurrent Hermes/OpenClaw evaluation, set ROLE, GPU, PORT,
# and RUN_TAG (v7p5 for Base; v7p6 for full-SFT).
set -euo pipefail
: "${ROLE:?Set ROLE=base or sft}"
: "${GPU:?Set GPU to the assigned GPU}"
: "${PORT:?Set the Hermes-only model server port}"
export HARNESS_LIST=hermes
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_harnessrisk_qwen35_2b_v7_matrix.sh"
