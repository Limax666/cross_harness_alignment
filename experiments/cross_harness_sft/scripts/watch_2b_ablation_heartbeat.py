#!/usr/bin/env python3
"""Read-only ten-minute experiment heartbeat. Does not launch or alter jobs."""
import json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
base=Path(os.environ.get('RL_HEARTBEAT_BASE', Path(__file__).resolve().parents[1]/'outputs/rl/2b_verified_progress_ablation_remaining_20261030'))
base.mkdir(parents=True,exist_ok=True)
session=os.environ.get('RL_HEARTBEAT_TMUX','rl2b-v4-curriculum-wait')
while True:
    live=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
    memory={p[0][:-1]:int(p[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(p:=line.split())==3}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    arms=[]
    for run in sorted(p for p in base.iterdir() if p.is_dir()):
        metrics=run/'metrics.jsonl'; lines=metrics.read_text().splitlines() if metrics.exists() else []
        data=json.loads(lines[-1]) if lines else None
        arms.append({'name':run.name,'latest_step':data.get('step') if data else None,'zero_variance_group_rate':data.get('zero_variance_group_rate') if data else None,'mean_reward':data.get('optimizer',{}).get('mean_reward') if data else None,'health_stop':json.loads((run/'health_stop.json').read_text()) if (run/'health_stop.json').exists() else None,'exit_code':(run/'exit_code').read_text().strip() if (run/'exit_code').exists() else None})
    queue=Path(os.environ.get('RL_HEARTBEAT_QUEUE', base/'CHSPO_progress_protocol'/'queue_status.json'))
    status={'utc':datetime.now(timezone.utc).isoformat(),'queue_live':live,'queue_session':session,
            'queue_status':json.loads(queue.read_text()) if queue.exists() else None,
            'available_ram_gib':round(memory['MemAvailable']/2**30,2),'gpus':gpu.stdout.strip().splitlines(),'arms':arms}
    with (base/'heartbeat.jsonl').open('a') as f:f.write(json.dumps(status,ensure_ascii=False)+'\n')
    (base/'heartbeat_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    if not live:break
    time.sleep(600)
