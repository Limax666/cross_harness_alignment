#!/usr/bin/env bash
# 2B full-parameter CHS-PO with frozen cross-harness K=2 assignments.
set -euo pipefail
export RL_MODEL_SIZE=2b
export RL_SEPARATE_REFERENCE_GPU=false
export RL_STEPS="${RL_STEPS:-300}"
export RL_RESPONSE_LENGTH="${RL_RESPONSE_LENGTH:-4096}"
export RL_MAX_MODEL_LEN="${RL_MAX_MODEL_LEN:-8192}"
exec bash "$(dirname "$0")/train_rl_9b_whitebox.sh" "$@"
