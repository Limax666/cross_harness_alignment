#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/cross_harness_sft"
: "${AGENTDOJO_REF:?Set exact AGENTDOJO_REF}"
: "${INSPECT_EVALS_REF:?Set exact INSPECT_EVALS_REF}"
mkdir -p "$EXP/vendor"
[[ -d "$EXP/vendor/agentdojo/.git" ]] || git clone https://github.com/ethz-spylab/agentdojo.git "$EXP/vendor/agentdojo"
[[ -d "$EXP/vendor/inspect_evals/.git" ]] || git clone https://github.com/UKGovernmentBEIS/inspect_evals.git "$EXP/vendor/inspect_evals"
git -C "$EXP/vendor/agentdojo" fetch --tags origin
git -C "$EXP/vendor/agentdojo" checkout --detach "$AGENTDOJO_REF"
git -C "$EXP/vendor/inspect_evals" fetch --tags origin
git -C "$EXP/vendor/inspect_evals" checkout --detach "$INSPECT_EVALS_REF"
source "$EXP/.venv/bin/activate"
python -m pip install -e "$EXP/vendor/agentdojo"
python -m pip install -e "$EXP/vendor/inspect_evals"
git -C "$EXP/vendor/agentdojo" rev-parse HEAD
git -C "$EXP/vendor/inspect_evals" rev-parse HEAD
