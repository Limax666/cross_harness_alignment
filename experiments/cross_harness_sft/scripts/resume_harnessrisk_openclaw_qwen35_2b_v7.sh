#!/usr/bin/env bash
# Resume OpenClaw Base or full-SFT in a dedicated process and model server.
# Run independently of the Hermes-only launcher; use a distinct PORT and
# server session even if both launchers use the same GPU and model role.
set -euo pipefail
: "${ROLE:?Set ROLE=base or sft}"
: "${GPU:?Set GPU to the assigned GPU}"
: "${PORT:?Set the OpenClaw-only model server port}"
export HARNESS_LIST=openclaw
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_harnessrisk_qwen35_2b_v7_matrix.sh"
