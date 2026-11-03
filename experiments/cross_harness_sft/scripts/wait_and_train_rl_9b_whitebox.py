#!/usr/bin/env python3
"""Wait for genuinely idle requested GPUs; never preempt another user's work."""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
root=Path(__file__).resolve().parents[1];run=args.run;run.mkdir(exist_ok=True)
stable=0;previous=None
for check in range(8640):
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    usage={int(a):int(b) for a,b in (line.split(',') for line in gpu.stdout.splitlines())}
    mem={s[0].rstrip(':'):int(s[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if len(s:=l.split())==3}
    requested=[int(x) for x in os.environ.get('RL_GPU_IDS','').split(',') if x.strip()]
    if requested and (len(requested)!=2 or len(set(requested))!=2 or any(i not in usage for i in requested)):
        raise ValueError('RL_GPU_IDS must name two distinct existing GPUs')
    candidates=requested or sorted(usage)
    idle=[i for i in candidates if usage[i]<256]
    selected=idle[:2]
    io_full=next(line for line in Path('/proc/pressure/io').read_text().splitlines() if line.startswith('full '))
    io_avg60=float(dict(part.split('=') for part in io_full.split()[1:])['avg60'])
    validation=root/'outputs/rl/9b_preparation_20261027/config_validation_r3.log'
    config_ok=validation.exists() and 'VALIDATED' in validation.read_text()
    ready=len(selected)==2 and mem['MemAvailable']>190*2**30 and io_avg60<15 and config_ok
    stable=stable+1 if ready else 0
    status=dict(state='waiting_for_two_idle_gpus_and_190GiB_ram',time_utc=datetime.now(timezone.utc).isoformat(),
                requested_gpus=requested or 'automatic',config_validated=config_ok,io_full_stall_avg60=io_avg60,gpu_used_mib=usage,available_ram_gib=round(mem['MemAvailable']/2**30,1),stable_checks=stable)
    (run/'queue_status.json').write_text(json.dumps(status,indent=2)+'\n')
    signature=(tuple(usage.values()),ready,config_ok)
    if signature!=previous:
        print(json.dumps(status),flush=True);previous=signature
    if stable>=3:
        import hashlib,shutil
        snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=True)
        sources=list((root/'scripts').glob('rl_whitebox_*.py'))+list((root/'scripts').glob('train_rl_9b_whitebox.*'))+[(root/'scripts/rl_verl_multiharness_loop.py')]
        for source in sources:shutil.copy2(source,snapshot/source.name)
        (run/'source_hashes.json').write_text(json.dumps({source.name:hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},indent=2)+'\n')
        env=dict(os.environ,RL_RUN_DIR=str(run),CUDA_VISIBLE_DEVICES=','.join(map(str,selected)),RL_N_GPUS='2')
        with (run/'launch.log').open('w') as log:
            result=subprocess.run(['bash',str(root/'scripts/train_rl_9b_whitebox.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
        if (run/'launch.started').exists():
            (run/'exit_code').write_text(str(result.returncode)+'\n')
            (run/'queue_status.json').write_text(json.dumps({'state':'training_process_exited','exit_code':result.returncode})+'\n')
            raise SystemExit(result.returncode)
        print('Resource guard lost race; waiting again.',flush=True);stable=0
    time.sleep(10)
(run/'queue_status.json').write_text(json.dumps({'state':'queue_expired_after_24h'})+'\n')
