#!/usr/bin/env python3
"""Low-write process/RAM telemetry and plots when an admitted round lands."""
import argparse,json,time,subprocess
from datetime import datetime,timezone
from pathlib import Path
from plot_rl_9b_whitebox import plot
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();run=args.run
last_size=-1
while True:
    memory={parts[0].rstrip(':'):int(parts[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if len(parts:=line.split())==3}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    status={'time_utc':datetime.now(timezone.utc).isoformat(),'memory_available_gib':memory['MemAvailable']/2**30,
            'memory_total_gib':memory['MemTotal']/2**30,'gpus':gpu.stdout.strip().splitlines()}
    try:
        pressure=Path('/proc/pressure/io').read_text().splitlines()
        status['io_pressure']={line.split()[0]:{k:float(v) for token in line.split()[1:]
                                                for k,v in [token.split('=',1)]} for line in pressure}
    except (OSError,ValueError):
        status['io_pressure']=None
    status['exit_code']=(run/'exit_code').read_text().strip() if (run/'exit_code').exists() else None
    with (run/'resource_history.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
    (run/'resource_status.json').write_text(json.dumps(status,indent=2)+'\n')
    metrics=run/'metrics.jsonl';size=metrics.stat().st_size if metrics.exists() else 0
    if size and size!=last_size:
        plot(run);last_size=size
    if status['exit_code'] is not None:break
    time.sleep(60)
