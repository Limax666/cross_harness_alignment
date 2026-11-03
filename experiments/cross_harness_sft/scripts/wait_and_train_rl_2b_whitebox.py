#!/usr/bin/env python3
"""Start the frozen 2B run only after GPU 0/3 are truly free."""
import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = Path(os.environ['RL_RUN_DIR'])
devices = tuple(int(x) for x in os.environ.get('RL_GPU_IDS', '0,3').split(','))
if len(devices) != 2 or len(set(devices)) != 2:
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
    ready = all(used[i] < 256 and uuids[i] not in active for i in devices) and mem > 80 * 1024 * 1024 and io < 15
    stable = stable + 1 if ready else 0
    (run/'queue_status.json').write_text(json.dumps({'state':'waiting' if stable < 3 else 'starting', 'gpu_used_mib':{str(i):used[i] for i in devices}, 'available_ram_gib':round(mem/1024**2,1), 'io_full_avg60':io, 'stable_checks':stable})+'\n')
    if stable >= 3:
        env=dict(os.environ, RL_RUN_DIR=str(run), CUDA_VISIBLE_DEVICES=','.join(map(str,devices)))
        with (run/'launch.log').open('w') as log:
            code=subprocess.run(['bash', str(root/'scripts/train_rl_2b_whitebox.sh')], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (run/'exit_code').write_text(f'{code}\n')
        raise SystemExit(code)
    time.sleep(20)
