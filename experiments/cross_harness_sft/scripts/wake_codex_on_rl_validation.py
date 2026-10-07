#!/usr/bin/env python3
"""Start one Codex decision turn after the step-20 paired validation is complete.

Cron polls cheaply; no model call occurs before a complete, stable comparison.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timedelta, timezone


def candidate_reports(run_dir: Path):
    yield from sorted(run_dir.glob('heldout_agentdojo_validation*/comparison.json'))


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat')
        return not (stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z'))
    except ProcessLookupError:
        return False


def complete_report(path: Path, *, min_age_seconds: int) -> tuple[str, dict] | None:
    if time.time() - path.stat().st_mtime < min_age_seconds:
        return None
    parent = path.parent
    for name in ('sft.jsonl', 'rl.jsonl'):
        source = parent / name
        if not source.is_file() or time.time() - source.stat().st_mtime < min_age_seconds:
            return None
        try:
            rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        except (json.JSONDecodeError, OSError):
            return None
        if len(rows) != 24:
            return None
    try:
        raw = path.read_bytes()
        report = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return None
    if report.get('paired_cases') != 24 or len(report.get('held_out_families', [])) != 4:
        return None
    for arm in ('sft', 'rl'):
        summary = report.get(arm, {})
        if (summary.get('episodes'), summary.get('clean_episodes'),
                summary.get('injection_episodes')) != (24, 12, 12):
            return None
    return hashlib.sha256(raw).hexdigest(), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--min-age-seconds', type=int, default=30)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    marker = run_dir / 'checkpoints/latest_checkpointed_iteration.txt'
    if not marker.is_file() or marker.read_text().strip() != '20':
        print(json.dumps({'status': 'waiting_checkpoint'}))
        return
    actor = run_dir / 'checkpoints/global_step_20/actor'
    if not (actor / 'fsdp_config.json').is_file() or len(list(actor.glob('model_world_size_*_rank_*.pt'))) != 2:
        print(json.dumps({'status': 'waiting_checkpoint_files'}))
        return
    for path in candidate_reports(run_dir):
        result = complete_report(path, min_age_seconds=args.min_age_seconds)
        if result:
            digest, report = result
            break
    else:
        print(json.dumps({'status': 'waiting_validation'}))
        return

    state_path = run_dir / 'step20_validation_codex_wake.json'
    lock_path = run_dir / 'step20_validation_codex_wake.lock'
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        old = json.loads(state_path.read_text()) if state_path.exists() else {}
        same = old.get('comparison_sha256') == digest
        attempts = int(old.get('attempts', 0)) if same else 0
        decision = run_dir / 'step20_codex_decision.md'
        if same and decision.is_file() and decision.stat().st_size > 0:
            print(json.dumps({'status': 'decision_recorded', 'report': str(decision)}))
            return
        if same and process_alive(old.get('agent_pid')):
            print(json.dumps({'status': 'agent_active', 'agent_pid': old['agent_pid']}))
            return
        last = datetime.fromisoformat(old['woken_at_utc']) if same and old.get('woken_at_utc') else None
        if same and (attempts >= 3 or (last and datetime.now(timezone.utc) - last < timedelta(minutes=10))):
            print(json.dumps({'status': 'attempt_limit' if attempts >= 3 else 'cooldown',
                              'attempts': attempts}))
            return
        event = {'status': 'validation_complete', 'report': str(path),
                 'comparison_sha256': digest, 'paired_cases': report['paired_cases']}
        if args.dry_run:
            print(json.dumps({**event, 'dry_run': True}))
            return
        prompt = (
            'The user asked you to decide and act after Claude Code completes R11 step-20 '
            'family-disjoint paired validation. Read AGENTS.md, AGENT.md, the Phase 3/4 '
            'experiment README, and the validation artifacts. '
            f'Run directory: {run_dir}. Comparison: {path}. '
            'Check step-20 checkpoint integrity; inspect sft.jsonl and rl.jsonl and verify '
            'paired cases, official outcomes, and family separation. Check that Claude has '
            'stopped training and finished evaluation; do not interfere with an active job. '
            'Compare clean/injection utility, ASR, invalid calls, incomplete rate, and '
            'per-family changes, acknowledging the small sample. Inspect training zero-variance '
            'and response-length trends. Decide whether evidence warrants continuing online RL '
            'from the step-20 checkpoint with a verified VeRL optimizer-state resume, or '
            'first changing reward/sampling/update strategy and launching a scientifically '
            'traceable revised run. Never silently restart from SFT while claiming step-20 '
            'continuation, never train on validation/test families, never disable admission '
            'or safety gates, and never run overlapping GPU jobs. If neither route is '
            'defensible, preserve artifacts and explain the blocker. Do the authorized work '
            f'and write a reviewable decision report to {run_dir}/step20_codex_decision.md. '
            'Do not print secrets.'
        )
        log = run_dir / f'step20_codex_decision_attempt_{attempts + 1}.log'
        env = {key: value for key, value in os.environ.items()
               if key in {'HOME', 'PATH', 'USER', 'LOGNAME', 'LANG', 'LC_ALL',
                          'TERM', 'CODEX_HOME', 'XDG_CONFIG_HOME'}}
        with log.open('w') as stream:
            child = subprocess.Popen([
                'codex', 'exec', '-C', str(repo), '--sandbox', 'danger-full-access',
                '-c', 'approval_policy="never"', prompt,
            ], cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        event.update({'agent_pid': child.pid, 'agent_log': str(log),
                      'attempts': attempts + 1,
                      'woken_at_utc': datetime.now(timezone.utc).isoformat()})
        state_path.write_text(json.dumps(event, indent=2) + '\n')
        print(json.dumps(event))


if __name__ == '__main__':
    main()
