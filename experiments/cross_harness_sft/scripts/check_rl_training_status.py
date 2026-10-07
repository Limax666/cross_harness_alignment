#!/usr/bin/env python3
"""Cron-friendly status check for one VeRL RL run; emits durable alerts."""
from __future__ import annotations
import argparse
import fcntl
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc)


def parse_time(value):
    return datetime.fromisoformat(value) if value else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--expected-steps',type=int,default=300)
    p.add_argument('--stall-minutes',type=int,default=15)
    a=p.parse_args()
    run=a.run_dir.resolve(); log=a.log.resolve(); root=run.parent
    state_path=root/f'{a.run_id}.watchdog_state.json'
    latest_path=root/f'{a.run_id}.watchdog_status.json'
    history_path=root/f'{a.run_id}.watchdog_history.jsonl'
    alert_path=root/f'{a.run_id}.watchdog_alerts.log'
    lock_path=root/f'{a.run_id}.watchdog.lock'
    root.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        now=utcnow(); now_s=now.isoformat()
        old=json.loads(state_path.read_text()) if state_path.exists() else {}
        metrics=run/'metrics.jsonl'; rows=[]
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                if line.strip(): rows.append(json.loads(line))
        updates=[r for r in rows if r.get('record_type')=='metrics' and r.get('phase')=='train']
        step=max((int(r['step']) for r in updates),default=0)
        checkpoints=run/'checkpoints'; marker=checkpoints/'latest_checkpointed_iteration.txt'
        checkpoint=int(marker.read_text().strip()) if marker.exists() else 0
        ps=subprocess.run(['ps','-eo','pid=,args='],text=True,capture_output=True,check=True).stdout
        matches=[]
        script_suffix='/scripts/train_rl_multiharness_verl.py'
        for line in ps.splitlines():
            fields=line.strip().split(None,1)
            if len(fields)!=2 or a.run_id not in fields[1] or script_suffix not in fields[1]:
                continue
            executable=fields[1].split(None,1)[0]
            # Count only the actual Python trainer, not wrapper/watchdog shells
            # that may mention its command in their own arguments.
            if Path(executable).name not in {'python','python3','python3.12'}:
                continue
            try: matches.append(int(fields[0]))
            except ValueError: pass
        # Exclude this monitor's own command line if its args happen to mention the run.
        matches=sorted(set(pid for pid in matches if pid != __import__('os').getpid()))
        alive=bool(matches)
        errors=[]
        if log.exists():
            for line in log.read_text(errors='replace').splitlines()[-200:]:
                if 'Error executing job with overrides' in line or 'Traceback (most recent call last)' in line:
                    errors.append(line.strip()[:500])
        last_step_at=old.get('last_step_at_utc')
        if step > int(old.get('last_step',0)):
            last_step_at=now_s
        if not last_step_at and alive:
            last_step_at=old.get('started_at_utc',now_s)
        started=old.get('started_at_utc',now_s)
        if errors:
            status='failed'; reason='training log contains traceback/error'
        elif not alive and step >= a.expected_steps and checkpoint >= a.expected_steps:
            status='completed'; reason='expected final step and checkpoint recorded'
        elif not alive:
            status='failed'; reason='training process exited before expected completion'
        elif now-parse_time(last_step_at) > __import__('datetime').timedelta(minutes=a.stall_minutes):
            status='stalled'; reason=f'no optimizer-step progress for over {a.stall_minutes} minutes'
        else:
            status='running'; reason='training process alive; step progress within threshold'
        result={'checked_at_utc':now_s,'run_id':a.run_id,'status':status,'reason':reason,
                'optimizer_step':step,'expected_steps':a.expected_steps,'latest_checkpoint':checkpoint,
                'process_pids':matches,'process_alive':alive,'last_step_at_utc':last_step_at,
                'metrics_rows':len(updates),'curve_png_exists':(run/'rl_training_dashboard.png').exists(),
                'recent_error_lines':errors[-3:]}
        latest_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        with history_path.open('a') as f:f.write(json.dumps(result,sort_keys=True)+'\n')
        alert_state=old.get('alert_state')
        new_alert=status in ('failed','stalled')
        if new_alert and alert_state != status:
            with alert_path.open('a') as f:f.write(json.dumps(result,sort_keys=True)+'\n')
        state={'last_step':step,'last_step_at_utc':last_step_at,'started_at_utc':started,
               'alert_state':status if new_alert else None}
        state_path.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
        print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
