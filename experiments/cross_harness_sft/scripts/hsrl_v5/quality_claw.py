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
from .grader_pool import LaneGraderPools
from .io import atomic_json, append_jsonl, digest, file_hash
from .isolation import IsolationError
from .workspace import checker_boundary_contract, CHECKER_BOUNDARY_VERSION


def prefilter_checker_boundaries(records, out, *, cache_dir=None):
    """Retain every static decision before dispatch; pending sources never run."""
    out = Path(out)
    cached = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        old_protocol = json.loads((cache_dir / 'protocol.json').read_text())
        summary = json.loads((cache_dir / 'private_grader_boundary_summary.json').read_text())
        audit = cache_dir / 'private_grader_boundary_audit.json'
        if (old_protocol['private_grader_boundary_gate'] != CHECKER_BOUNDARY_VERSION
                or old_protocol['code_sha256']['workspace.py'] != file_hash(Path(__file__).with_name('workspace.py'))
                or summary['audit_sha256'] != file_hash(audit)):
            raise ValueError('boundary_cache_policy_or_audit_identity_changed')
        rows = json.loads(audit.read_text())
        cached = {row['id']: row for row in rows}
        if len(cached) != len(rows) or set(cached) != {row['id'] for row in records}:
            raise ValueError('boundary_cache_record_inventory_changed')
    runnable, pending, audits = [], [], []
    for index, record in enumerate(records):
        if cached is None:
            boundary = checker_boundary_contract(record['private_checker_path'])
        else:
            previous = cached[record['id']]
            if previous['record_sha256'] != digest(record):
                raise ValueError('boundary_cache_source_record_changed')
            boundary = previous['private_grader_boundary']
            if boundary['checker_sha256'] != file_hash(record['private_checker_path']):
                raise ValueError('boundary_cache_original_checker_bytes_changed')
        expected = record['original_asset_sha256']['reward/test.py']
        if boundary['checker_sha256'] != expected:
            raise ValueError('private_checker_changed_before_boundary_admission:' + record['id'])
        audits.append({'id': record['id'], 'family': record['family'], 'split': record['split'],
                       'record_sha256': digest(record), 'private_grader_boundary': boundary})
        if boundary['status'] == 'readonly_data_checker_static_boundary_pass':
            runnable.append(record)
        else:
            proof = {'source': 'clawgym', 'id': record['id'], 'family': record['family'],
                     'condition': 'benign', 'record_sha256': digest(record), 'ready': False,
                     'reasons': ['private_grader_boundary_pending', *boundary['reasons']],
                     'failure_class': 'source_private_grader_boundary_pending',
                     'private_grader_boundary': boundary, 'source_execution_performed': False,
                     'task_solvability_proven': False, 'model_calls': 0, 'judge_calls': 0,
                     'utility': None, 'c_source': None, 'attempt_directory': None}
            directory = out / 'cases' / hashlib.sha256(record['id'].encode()).hexdigest()
            atomic_json(directory / 'quality.json', proof)
            pending.append(proof)
        if (index + 1) % 100 == 0:
            progress = {'status': 'static_boundary_prefilter', 'scanned': index + 1,
                        'total': len(records), 'pending_unsupported': len(pending)}
            atomic_json(out / 'prefilter_progress.json', progress)
            print(json.dumps(progress), flush=True)
    atomic_json(out / 'private_grader_boundary_audit.json', audits)
    summary = {'version': CHECKER_BOUNDARY_VERSION, 'records': len(records),
               'eligible_for_runtime_qc': len(runnable), 'pending_unsupported': len(pending),
               'readonly_compatibility_only_pending': sum(row['private_grader_boundary']['reasons'] == ['readonly_snapshot_write_capability_pending'] for row in pending),
               'pending_reasons': dict(Counter(reason for row in pending for reason in row['private_grader_boundary']['reasons'])),
               'source_execution_performed': False, 'all_original_checker_hashes_matched': True,
               'audit_sha256': file_hash(out / 'private_grader_boundary_audit.json')}
    if cache_dir is not None:
        summary['verified_static_cache'] = {'path': str(cache_dir.resolve()),
            'original_audit_sha256': file_hash(cache_dir / 'private_grader_boundary_audit.json'),
            'original_protocol_sha256': file_hash(cache_dir / 'protocol.json'),
            'workspace_policy_sha256': file_hash(Path(__file__).with_name('workspace.py')),
            'all_records_and_actual_checker_bytes_rechecked': True}
    atomic_json(out / 'private_grader_boundary_summary.json', summary)
    return runnable, pending, summary


def validate_record(record, directory, *, grader_pool=None):
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
    backend_options = {'grader_pool': grader_pool} if grader_pool is not None else {}
    try:
        with ClawBackend(record, attempt / 'actual_episode', **backend_options) as first:
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
        with ClawBackend(record, attempt / 'reset_episode', **backend_options) as reset:
            if reset.initial_entries != initial:
                raise ValueError('fresh_process_reset_source_assets_mismatch')
            reset_grade = reset.finish()
            if not reset_grade['valid'] or reset_grade['utility'] != first_grade['utility'] or reset_grade['metrics'] != first_grade['metrics']:
                raise ValueError('untouched_source_checker_new_process_replay_mismatch')
        if grader_pool is not None:
            identities = [grade['evidence']['private_grader'] for grade in (first_grade, reset_grade)]
            if (identities[0]['container'] != identities[1]['container']
                    or identities[0]['request_nonce'] == identities[1]['request_nonce']
                    or identities[0]['child_pid'] == identities[1]['child_pid']):
                grader_pool.poison('actual_reset_grader_identity_contract_failed')
                raise IsolationError('actual_reset_grader_identity_contract_failed')
            proof['private_grader_pool'] = {'same_private_container': True, 'fresh_children': True,
                'requests': [{key: identity[key] for key in ('container', 'request_nonce', 'child_pid',
                    'snapshot_manifest_sha256', 'source_manifest_sha256')} for identity in identities]}
        proof.update(ready=True, untouched_workspace_grade=first_grade['utility'],
                     reset={'initial_assets_sha256': digest(initial), 'fresh_os_process_and_tmpfs': True,
                            'same_checker_scores': True},
                     verifier_profile=record['reward_profile'],
                     original_full_rubric_utility_claimed=False)
        if first_grade['utility'] == 1.0:
            proof.update(ready=False, failure_class='source_already_satisfied_pending')
            proof['reasons'].append('untouched_source_verifier_already_satisfied_pending')
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


def run(source_dir, out, workers=4, *, boundary_cache=None, priority_case=None):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('workers_must_be_1_to_16')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    # An advisory lock survives stale PID files and prevents duplicate resumes.
    with (out / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IsolationError('quality_run_already_active') from exc
        # A fresh dispatch cannot replace containers left uncertain by a prior
        # interruption. Preserve their evidence for exact-container recovery.
        for previous in (out / 'private_grader_pools').glob('dispatch_*'):
            try:
                cleanup = json.loads((previous / 'cleanup.json').read_text())
            except (OSError, ValueError) as exc:
                raise IsolationError('previous_private_grader_pool_cleanup_unconfirmed:' + str(previous)) from exc
            if not isinstance(cleanup, dict) or cleanup.get('all_owned_containers_removal_confirmed') is not True:
                raise IsolationError('previous_private_grader_pool_cleanup_unconfirmed:' + str(previous))
        return _run_locked(source_dir, out, workers, boundary_cache=boundary_cache, priority_case=priority_case)


def _run_locked(source_dir, out, workers, *, boundary_cache=None, priority_case=None):
    source_dir, out = Path(source_dir), Path(out)
    manifest = json.loads((source_dir / 'source_manifest.json').read_text())
    if file_hash(source_dir / 'accepted.jsonl') != manifest['file_sha256']['accepted.jsonl']:
        raise ValueError('prepared_source_manifest_changed')
    records = [row for row in map(json.loads, (source_dir / 'accepted.jsonl').open()) if row['split'] == 'train']
    records.sort(key=lambda row: digest([20261116, row['family'], row['id']]))
    if priority_case is not None:
        if priority_case not in {record['id'] for record in records}:
            raise ValueError('priority_case_not_in_full_train_inventory')
        records.sort(key=lambda row: row['id'] != priority_case)
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('duplicate_train_record_id')
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out, 0o700)
    protocol = {'schema': 'hsrl-v5-claw-real-quality/v5', 'source_manifest_sha256': file_hash(source_dir / 'source_manifest.json'),
                'records': len(records), 'workers': workers,
                'source_dir': str(source_dir.resolve()), 'train_record_id_sha256': digest([r['id'] for r in records]),
                'private_grader_boundary_gate': CHECKER_BOUNDARY_VERSION,
                'boundary_cache_protocol_sha256': file_hash(Path(boundary_cache) / 'protocol.json') if boundary_cache else None,
                'priority_case_in_full_job': priority_case,
                'private_grader_reuse': {'policy': 'serial_host_projection_readonly_data_only/v1',
                    'max_cases_per_container': 64, 'max_grade_children_per_container': 128,
                    'fresh_actor_containers_per_case': 2, 'fresh_python_child_per_grade': True,
                    'tmp_cleared_before_each_grade': True, 'case_local_source_and_isolation_evidence': True},
                'numeric_library_threads': 1, 'command_nice_increment': 10,
                'unsupported_checker_execution': 'pending_without_source_execution_or_utility',
                'code_sha256': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                'controls': 'actual_input_read_and_private_original_checker_plus_fresh_os_process_tmpfs_reset',
                'task_solvability_proven': False, 'model_calls': 0, 'judge_calls': 0}
    if (out / 'protocol.json').exists() and json.loads((out / 'protocol.json').read_text()) != protocol:
        raise ValueError('quality_protocol_changed_requires_independent_run')
    atomic_json(out / 'protocol.json', protocol)
    atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'running', 'planned': len(records)})
    planned = len(records)
    records, pending, boundary_summary = prefilter_checker_boundaries(records, out, cache_dir=boundary_cache)
    results = list(pending)
    atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'running', 'planned': planned,
                'eligible_for_runtime_qc': len(records), 'boundary_pending': len(pending)})
    pools = LaneGraderPools(out / 'private_grader_pools' / ('dispatch_' + str(time.time_ns())))
    try:
        if records:
            first = validate_record(records[0], out / 'cases' / hashlib.sha256(records[0]['id'].encode()).hexdigest(), grader_pool=pools.get())
            # The serial first-case lane retires before up to 16 executor lanes start.
            pools.get().close()
            results.append(first)
            append_jsonl(out / 'events.jsonl', {'id': first['id'], 'ready': first['ready'], 'reasons': first['reasons']})
            if first.get('failure_class') == 'isolation_or_runtime_infrastructure':
                atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'blocked_shared_runtime', 'first_failure': first})
                raise IsolationError('full_admission_shared_runtime_requires_repair_see_first_case')
        remainder, blocked = bounded_admission(records[1:], out, workers, validate=lambda record, directory:
            validate_record(record, directory, grader_pool=pools.get()))
        results.extend(remainder)
        if blocked:
            atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'blocked_shared_runtime',
                        'planned': planned, 'completed': len(results), 'pending': planned - sum(r['ready'] for r in results)})
            raise IsolationError('full_admission_shared_runtime_requires_repair_see_cases')
    finally:
        try:
            pools.close()
        except BaseException as exc:
            atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'blocked_pool_cleanup',
                                               'error': repr(exc)})
            raise
    summary = {'status': 'complete_runtime_validation_pass' if all(r['ready'] for r in results) else 'complete_with_pending_cases', 'planned': planned, 'completed': len(results),
               'boundary_prefilter': boundary_summary,
               'private_grader_pool_cleanup_confirmed': True,
               'private_grader_pool_cleanup': str(pools.directory / 'cleanup.json'),
               'ready': sum(row['ready'] for row in results), 'reasons': dict(Counter(x for row in results for x in row['reasons'])),
               'protocol_sha256': file_hash(out / 'protocol.json')}
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'complete'})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 17))
    parser.add_argument('--boundary-cache', type=Path)
    parser.add_argument('--priority-case')
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.out, args.workers, boundary_cache=args.boundary_cache, priority_case=args.priority_case)))
