#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/flashrsi_harnessrisk_4b_20261113
export PYTHONPATH=src:scripts:vendor/FlashRSI HR_HARNESS=opencode HR_RUN_NAME=opencode_initial_full HARNESSRISK_MODEL_PORT=18762
ready=0
for ((i=0;i<180;i++)); do
 if curl --noproxy '*' -fsS http://127.0.0.1:18762/v1/models | .venv-rl-pilot/bin/python -c 'import json,sys;x=json.load(sys.stdin);assert any(m.get("id")=="qwen35-4b-preflight" for m in x.get("data",[]))' 2>/dev/null; then ready=1;break; fi
 if ! tmux has-session -t =flashrsi-4b-gpu3 2>/dev/null; then echo 'GPU 3 model server stopped'; exit 1; fi
 sleep 5
done
[[ "$ready" == 1 ]] || { echo 'GPU 3 model server did not become ready'; exit 1; }
.venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_miniagent_initial.py > "$root/opencode_initial_full.log" 2>&1
.venv-rl-pilot/bin/python scripts/flashrsi_harnessrisk_retry_infrastructure.py "$root/opencode_initial_full" >> "$root/opencode_initial_full.log" 2>&1
.venv-rl-pilot/bin/python - <<'PY'
import json
from pathlib import Path
x=json.loads(Path('outputs/flashrsi_harnessrisk_4b_20261113/opencode_initial_full/completed.json').read_text())
if x['attempts']!=128 or x['completed_scored']==0:raise SystemExit('OpenCode initial baseline has no usable scored case')
PY
for algorithm in opro gepa evoprompt ace; do
 .venv-rl-pilot/bin/python -u scripts/flashrsi_harnessrisk_algorithm.py --algorithm "$algorithm" > "$root/opencode_${algorithm}_full.log" 2>&1
 .venv-rl-pilot/bin/python scripts/judge_harnessrisk_shengsuanyun.py "$root/opencode_${algorithm}_full/selected_cases" --deployment openai/gpt-5.4-nano > "$root/opencode_${algorithm}_judge.log" 2>&1 || echo "$algorithm judge incomplete; rule results retained"
done
printf '%s\n' 'SkillOpt and Evo-Harness unavailable: upstream OpenCode Snapshot declares no skill component.' > "$root/opencode_skill_algorithms_unavailable.txt"
