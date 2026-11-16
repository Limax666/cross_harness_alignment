"""Full real ClawGym environment/checker readiness, without solving tasks.

Each check uses actual input files, genuine tool execution, the unchanged private
source grader and an independent tmpfs workspace reset. It is not an agent run
or a task-solvability claim; source solutions are not synthesized for admission.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .claw_backend import ClawBackend
from .io import atomic_json, append_jsonl, digest, file_hash
from .isolation import IsolationError


def validate_record(record, directory):
    directory = Path(directory)
    terminal, identity = directory / 'quality.json', digest(record)
    if terminal.exists():
        previous = json.loads(terminal.read_text())
        if previous['record_sha256'] != identity:
            raise ValueError('resumed_source_record_changed')
        if previous.get('failure_class') != 'isolation_or_runtime_infrastructure':
            return previous
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / ('attempt_' + str(time.time_ns()))
    attempt.mkdir(mode=0o700)
    proof = {'source': 'clawgym', 'id': record['id'], 'family': record['family'],
             'condition': 'benign', 'record_sha256': identity, 'ready': False,
             'reasons': [], 'attempt_directory': str(attempt), 'source_execution_performed': False,
             'task_solvability_proven': False, 'model_calls': 0, 'judge_calls': 0}
    atomic_json(attempt / 'started.json', proof)
    try:
        with ClawBackend(record, attempt / 'actual_episode') as first:
            initial = first.initial_entries
            # Check the real supplied task data through the advertised API.
            # No generated task answer or reference trajectory enters the pool.
            if initial:
                name = sorted(initial)[0]
                response = json.loads(first.call('run_python', {'code':
                    'from pathlib import Path\nimport hashlib,json\np=Path(' + repr(name) + ')\n'
                    'print(json.dumps({"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"private_grader_visible":Path("/grader").exists()}))'}))
            else:
                response = json.loads(first.call('run_python', {'code':
                    'from pathlib import Path\nimport json\nprint(json.dumps({"entries":sorted(p.name for p in Path(".").iterdir()),"private_grader_visible":Path("/grader").exists()}))'}))
            if response.get('returncode') != 0:
                raise ValueError('actual_task_workspace_tool_failed')
            observed = json.loads(response['stdout'])
            if observed['private_grader_visible'] or (initial and observed['sha256'] != initial[name]['sha256']):
                raise ValueError('source_asset_or_private_grader_separation_failed')
            first_grade = first.finish()
            proof['source_execution_performed'] = True
            if not first_grade['valid'] or first.protocol_errors:
                raise ValueError('actual_source_checker_or_tools_not_valid')
        with ClawBackend(record, attempt / 'reset_episode') as reset:
            if reset.initial_entries != initial:
                raise ValueError('fresh_process_reset_source_assets_mismatch')
            reset_grade = reset.finish()
            if not reset_grade['valid'] or reset_grade['utility'] != first_grade['utility'] or reset_grade['metrics'] != first_grade['metrics']:
                raise ValueError('untouched_source_checker_new_process_replay_mismatch')
        proof.update(ready=True, untouched_workspace_grade=first_grade['utility'],
                     reset={'initial_assets_sha256': digest(initial), 'fresh_os_process_and_tmpfs': True,
                            'same_checker_scores': True},
                     verifier_profile=record['reward_profile'],
                     original_full_rubric_utility_claimed=False)
    except Exception as exc:
        proof['reasons'].append(type(exc).__name__ + ':' + str(exc))
        proof['failure_class'] = 'isolation_or_runtime_infrastructure' if isinstance(exc, IsolationError) else 'source_quality_or_control_pending'
        (attempt / 'error.traceback.txt').write_text(traceback.format_exc())
    atomic_json(terminal, proof)
    return proof


def bounded_admission(records, out, workers, *, validate=validate_record):
    """Keep at most workers in flight; drain and stop on infrastructure failure."""
    records, results, blocked = iter(records), [], False
    out = Path(out)
    def submit(executor, pending):
        record = next(records, None)
        if record is not None:
            pending.add(executor.submit(validate, record, out / 'cases' / hashlib.sha256(record['id'].encode()).hexdigest()))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(workers):
            submit(executor, pending)
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                proof = future.result()
                results.append(proof)
                append_jsonl(out / 'events.jsonl', {'id': proof['id'], 'ready': proof['ready'], 'reasons': proof['reasons']})
                blocked |= proof.get('failure_class') == 'isolation_or_runtime_infrastructure'
            atomic_json(out / 'progress.json', {'status': 'blocked_shared_runtime' if blocked else 'running',
                        'completed_this_dispatch': len(results), 'ready_this_dispatch': sum(r['ready'] for r in results)})
            if len(results) % 10 == 0 or blocked:
                print(json.dumps({'completed_this_dispatch': len(results), 'blocked': blocked}), flush=True)
            if not blocked:
                for _ in completed:
                    submit(executor, pending)
    return results, blocked


def run(source_dir, out, workers=2):
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError('workers_must_be_1_or_2')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    # An advisory lock survives stale PID files and prevents duplicate resumes.
    with (out / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IsolationError('quality_run_already_active') from exc
        return _run_locked(source_dir, out, workers)


def _run_locked(source_dir, out, workers):
    source_dir, out = Path(source_dir), Path(out)
    manifest = json.loads((source_dir / 'source_manifest.json').read_text())
    if file_hash(source_dir / 'accepted.jsonl') != manifest['file_sha256']['accepted.jsonl']:
        raise ValueError('prepared_source_manifest_changed')
    records = [row for row in map(json.loads, (source_dir / 'accepted.jsonl').open()) if row['split'] == 'train']
    records.sort(key=lambda row: digest([20261116, row['family'], row['id']]))
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('duplicate_train_record_id')
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out, 0o700)
    protocol = {'schema': 'hsrl-v5-claw-real-quality/v2', 'source_manifest_sha256': file_hash(source_dir / 'source_manifest.json'),
                'records': len(records), 'workers': workers,
                'source_dir': str(source_dir.resolve()), 'train_record_id_sha256': digest([r['id'] for r in records]),
                'code_sha256': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                'controls': 'actual_input_read_and_private_original_checker_plus_fresh_os_process_tmpfs_reset',
                'task_solvability_proven': False, 'model_calls': 0, 'judge_calls': 0}
    if (out / 'protocol.json').exists() and json.loads((out / 'protocol.json').read_text()) != protocol:
        raise ValueError('quality_protocol_changed_requires_independent_run')
    atomic_json(out / 'protocol.json', protocol)
    atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'running', 'planned': len(records)})
    results = []
    if records:
        first = validate_record(records[0], out / 'cases' / hashlib.sha256(records[0]['id'].encode()).hexdigest())
        results.append(first)
        append_jsonl(out / 'events.jsonl', {'id': first['id'], 'ready': first['ready'], 'reasons': first['reasons']})
        if first.get('failure_class') == 'isolation_or_runtime_infrastructure':
            atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'blocked_shared_runtime', 'first_failure': first})
            raise IsolationError('full_admission_shared_runtime_requires_repair_see_first_case')
    remainder, blocked = bounded_admission(records[1:], out, workers)
    results.extend(remainder)
    if blocked:
        atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'blocked_shared_runtime',
                    'planned': len(records), 'completed': len(results), 'pending': len(records) - sum(r['ready'] for r in results)})
        raise IsolationError('full_admission_shared_runtime_requires_repair_see_cases')
    summary = {'status': 'complete_runtime_validation_pass' if all(r['ready'] for r in results) else 'complete_with_pending_cases', 'planned': len(records), 'completed': len(results),
               'ready': sum(row['ready'] for row in results), 'reasons': dict(Counter(x for row in results for x in row['reasons'])),
               'protocol_sha256': file_hash(out / 'protocol.json')}
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'complete'})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.out, args.workers)))
