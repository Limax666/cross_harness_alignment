#!/usr/bin/env python3
"""Poll the RL status and start one bounded Codex repair turn per incident.

Run from cron once a minute. Codex is not invoked while training is healthy.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z'):
            return False
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--stall-minutes', type=int, default=30)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[2]
    root = args.run_dir.resolve().parent
    prefix = root / args.run_id
    lock_path = Path(str(prefix) + '.codex_wake.lock')
    state_path = Path(str(prefix) + '.codex_wake_state.json')
    events_path = Path(str(prefix) + '.codex_wake_events.jsonl')
    root.mkdir(parents=True, exist_ok=True)

    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        check = subprocess.run([
            sys.executable, str(scripts / 'check_rl_training_status.py'),
            '--run-id', args.run_id, '--run-dir', str(args.run_dir),
            '--log', str(args.log), '--stall-minutes', str(args.stall_minutes),
        ], capture_output=True, text=True, check=True)
        status = json.loads(check.stdout)
        old = json.loads(state_path.read_text()) if state_path.exists() else {}
        state = status['status']
        if state not in {'failed', 'stalled'}:
            if old.get('incident_status'):
                events_path.open('a').write(json.dumps({'at': now_utc().isoformat(), 'event': 'cleared', 'status': state}) + '\n')
            state_path.write_text(json.dumps({'incident_status': None}, indent=2) + '\n')
            print(json.dumps({'status': state, 'wake': 'none', 'step': status['optimizer_step']}))
            return

        same = old.get('incident_status') == state
        attempts = int(old.get('attempts', 0)) if same else 0
        last = datetime.fromisoformat(old['last_wake_at']) if same and old.get('last_wake_at') else None
        active = process_alive(old.get('agent_pid')) if same else False
        if active or attempts >= 3 or (last and now_utc() - last < timedelta(minutes=15)):
            reason = 'agent_active' if active else ('attempt_limit' if attempts >= 3 else 'cooldown')
            print(json.dumps({'status': state, 'wake': reason, 'step': status['optimizer_step']}))
            return

        prompt = (
            'The cross-harness online RL run has a monitoring incident. Diagnose and repair it now. '
            f'Run ID: {args.run_id}. Status: {state}. Reason: {status["reason"]}. '
            f'Optimizer step: {status["optimizer_step"]}/300; checkpoint: {status["latest_checkpoint"]}. '
            f'Run dir: {args.run_dir.resolve()}. Log: {args.log.resolve()}. '
            'Read AGENTS.md, AGENT.md, the experiment README, the latest metrics, and the log. '
            'Check whether the trainer process is truly alive before touching it. '
            'For a stall, inspect rollout/judge activity and GPU use before deciding it is hung. '
            'Fix a confirmed infrastructure or code fault, run focused validation, and resume safely '
            'from a valid checkpoint when possible. Keep a single training process. '
            'Do not bypass safety/admission gates or use validation/test families for training. '
            'Do not print secrets or API keys. Record the diagnosis, changes, and outcome in '
            f'{prefix}.codex_repair_report.md. If the run cannot be safely resumed, report the blocker.'
        )
        event = {'at': now_utc().isoformat(), 'event': 'wake', 'status': state,
                 'step': status['optimizer_step'], 'attempt': attempts + 1}
        if args.dry_run:
            print(json.dumps({**event, 'dry_run': True}))
            return
        codex_log = Path(str(prefix) + f'.codex_repair_{attempts + 1}.log')
        env = {key: value for key, value in os.environ.items()
               if key in {'HOME', 'PATH', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TERM', 'CODEX_HOME', 'XDG_CONFIG_HOME'}}
        with codex_log.open('w') as stream:
            child = subprocess.Popen([
                'codex', 'exec', '-C', str(repo), '--sandbox', 'danger-full-access',
                '-c', 'approval_policy="never"', prompt,
            ], cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        event['agent_pid'] = child.pid
        event['agent_log'] = str(codex_log)
        with events_path.open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        state_path.write_text(json.dumps({
            'incident_status': state, 'attempts': attempts + 1,
            'last_wake_at': event['at'], 'agent_pid': child.pid,
        }, indent=2) + '\n')
        print(json.dumps(event))


if __name__ == '__main__':
    main()
