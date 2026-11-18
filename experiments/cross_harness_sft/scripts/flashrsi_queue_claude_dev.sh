#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while tmux has-session -t flashrsi-port-sequential 2>/dev/null; do sleep 15; done
.venv-rl-pilot/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('outputs/flashrsi_local_20261112/uniform4096_serial')
for harness in ('opencode','openclaw','hermes','pi'):
    for algorithm in ('opro','gepa','evoprompt','ace'):
        value=json.loads((root/f'{harness}_{algorithm}_dev/result.json').read_text())
        assert value['status']=='finished', (harness,algorithm)
PY
export PYTHONPATH=src:scripts:vendor/FlashRSI FLASHRSI_COMPLETION_BUDGET=4096
root=outputs/flashrsi_local_20261112/uniform4096_serial
.venv-rl-pilot/bin/python -u scripts/flashrsi_native_initial_dev.py --harness claude_code > "$root/claude_code_initial.log" 2>&1
for algorithm in opro gepa evoprompt ace; do
    .venv-rl-pilot/bin/python -u scripts/flashrsi_port_algorithm_dev.py --harness claude_code --algorithm "$algorithm" > "$root/claude_code_${algorithm}.log" 2>&1
done
