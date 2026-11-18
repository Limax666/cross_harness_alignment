#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/flashrsi_harnessrisk_4b_20261113
while [[ ! -f "$root/opencode_initial_full/completed.json" ]]; do
 if ! tmux has-session -t =flashrsi-harnessrisk-opencode 2>/dev/null; then echo 'OpenCode initial run stopped before all cases'; exit 1; fi
 sleep 15
done
export PYTHONPATH=src:scripts:vendor/FlashRSI HR_HARNESS=opencode HARNESSRISK_MODEL_PORT=18762
.venv-rl-pilot/bin/python scripts/flashrsi_harnessrisk_retry_infrastructure.py "$root/opencode_initial_full" > "$root/opencode_retry.log" 2>&1
