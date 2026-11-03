#!/usr/bin/env python3
"""Start the frozen 2B run only after GPU 0/3 are truly free."""
import json
import os
import subprocess
import time
from pathlib import Path
from rl_resource_guard import select_idle_pair

root = Path(__file__).resolve().parents[1]
run = Path(os.environ['RL_RUN_DIR'])
candidate_text=os.environ.get('RL_GPU_CANDIDATES')
if candidate_text:
    candidates=tuple(int(x) for x in candidate_text.split(','))
    if len(candidates)<2 or len(set(candidates))!=len(candidates):
        raise ValueError('RL_GPU_CANDIDATES must name at least two distinct GPUs')
else:
    candidates=tuple(int(x) for x in os.environ.get('RL_GPU_IDS', '0,3').split(','))
    if len(candidates) != 2 or len(set(candidates)) != 2:
        raise ValueError('RL_GPU_IDS must name two different GPUs')
run.mkdir(parents=True, exist_ok=True)
stable = 0
while True:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    used = {int(a): int(b) for a, b in (line.split(',') for line in gpu.splitlines())}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(a): b.strip() for a, b in (line.split(',', 1) for line in uuid.splitlines())}
    active = {line.split(',', 1)[0].strip() for line in apps.splitlines()}
    mem = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    io = float(dict(item.split('=') for item in next(line for line in Path('/proc/pressure/io').read_text().splitlines() if line.startswith('full ')).split()[1:])['avg60'])
    selected=select_idle_pair(candidates,used,uuids,active)
    ready = selected is not None and mem > 80 * 1024 * 1024 and io < 15
    stable = stable + 1 if ready else 0
    (run/'queue_status.json').write_text(json.dumps({'state':'waiting' if stable < 3 else 'starting', 'gpu_candidates':list(candidates), 'selected_gpu_pair':list(selected) if selected else None, 'gpu_used_mib':{str(i):used[i] for i in candidates}, 'available_ram_gib':round(mem/1024**2,1), 'io_full_avg60':io, 'stable_checks':stable})+'\n')
    if stable >= 3:
        env=dict(os.environ, RL_RUN_DIR=str(run), CUDA_VISIBLE_DEVICES=','.join(map(str,selected)))
        with (run/'launch.log').open('w') as log:
            code=subprocess.run(['bash', str(root/'scripts/train_rl_2b_whitebox.sh'), '+trainer.whitebox_pilot_health_guard=true'], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (run/'exit_code').write_text(f'{code}\n')
        raise SystemExit(code)
    time.sleep(20)
