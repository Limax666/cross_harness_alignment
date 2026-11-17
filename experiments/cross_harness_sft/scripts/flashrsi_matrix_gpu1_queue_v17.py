"""Finite GPU1-only strict matrix queue; retain incomplete stages for repair."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from flashrsi_matrix_repair_loop import inspect_run

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
DATA = ROOT.parents[1] / 'HarnessRisk/data/HarnessRisk'
REFERENCE = OUTPUTS / 'flashrsi_harnessrisk_4b_private_opencode_opro_suffix_v5_20261116/opencode_initial_full'
WORKER_LOCK = OUTPUTS / 'flashrsi_harnessrisk_4b_private_v1_20261113/monitor/matrix_loop/worker.lock'
LANE_LOCK = OUTPUTS / '.opencode_gpu1_deferred_repair.lock'
ALGORITHMS = ('opro', 'gepa', 'evoprompt', 'ace', 'skillopt', 'evo_harness')
HOSTS = {
    'opencode': 'flashrsi_harnessrisk_host_opencode_session_v11.py',
    'openclaw': 'flashrsi_harnessrisk_host_openclaw_caseenv_v3.py',
    'codex': 'flashrsi_harnessrisk_host_native_v3.py',
    'claude_code': 'flashrsi_harnessrisk_host_native_v8.py',
    'pi': 'flashrsi_harnessrisk_host_pi_interpreter_v4.py',
}


class QueueCleanupUncertain(RuntimeError):
    """No further dispatch before external exact descendant reconciliation."""


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin_path(name):
    return ROOT.parents[1] / 'HarnessRisk' / name[len('@harnessrisk/'):] if name.startswith('@harnessrisk/') else ROOT / name


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        if path.stat().st_uid != os.getuid():
            return None
        command = (path / 'cmdline').read_bytes().split(b'\0')
        raw = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        identity = {'pid': pid, 'start_ticks': raw[19], 'command': [v.decode(errors='replace') for v in command if v]}
        if '--algorithm' in identity['command']:
            env = dict(v.split(b'=', 1) for v in (path / 'environ').read_bytes().split(b'\0') if b'=' in v)
            harness = env.get(b'HR_HARNESS', b'').decode()
            algorithm = identity['command'][identity['command'].index('--algorithm') + 1]
            if harness in HOSTS and algorithm in ALGORITHMS and env.get(b'HARNESSRISK_MODEL_PORT') == b'18764' and env.get(b'HR_OUTPUT_ROOT'):
                identity['run_scope'] = str((Path(env[b'HR_OUTPUT_ROOT'].decode()) / f'{harness}_{algorithm}_full').resolve())
        return identity
    except (OSError, IndexError):
        return None


def lane_clients(exclude=()):
    result = []
    for path in Path('/proc').glob('[0-9]*'):
        pid = int(path.name)
        if pid in exclude:
            continue
        identity = process_identity(pid)
        if not identity or not any('flashrsi_harnessrisk_' in arg for arg in identity['command']):
            continue
        if not any(('algorithm' in arg or 'initial' in arg or 'host_' in arg) and 'flashrsi' in arg for arg in identity['command']):
            continue
        try:
            env = dict(v.split(b'=', 1) for v in (path / 'environ').read_bytes().split(b'\0') if b'=' in v)
            if env.get(b'HARNESSRISK_MODEL_PORT') == b'18764':
                result.append(identity)
        except OSError:
            continue
    return result


def expected_initial_contract(manifest_bytes, harness):
    manifest = json.loads(manifest_bytes)
    if not all(isinstance(manifest.get(key), str) and manifest[key]
               for key in ('model_revision', 'flashrsi_revision')):
        raise ValueError('trusted_manifest_model_runtime_revisions_missing')
    contract = {'case_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'model_revision': manifest['model_revision'], 'flashrsi_revision': manifest['flashrsi_revision'],
        'harness': harness, 'host_script': HOSTS[harness], 'model_port': 18764,
        'privacy_protocol': 'harnessrisk-private-case-v1', 'temperature': 0,
        'max_tokens_per_turn': 4096, 'case_count': 128,
        'max_turns_per_owner_turn': None if harness == 'codex' else 16 if harness == 'claude_code' else 12,
        'owner_turn_protocol': 'sequential native SQLite session; environment state preserved' if harness == 'openclaw'
            else 'sequential, previous owner/assistant text replay; environment state preserved'}
    # The retained v11 manifest predates these explicit native/sandbox fields;
    # its declared12-turn4096-token contract and exact host remain mandatory.
    if harness != 'opencode':
        contract.update(native_timeout_seconds_per_owner_turn=600 if harness == 'claude_code' else 360,
                        sandbox_wall_seconds_per_owner_turn=900 if harness == 'claude_code' else 360)
    return contract


def initial_binding(run):
    run = Path(run)
    paths = {'snapshot_sha256': run / 'initial_snapshot.json', 'run_manifest_sha256': run / 'run_manifest.json',
             'case_manifest_sha256': run.parent / 'case_manifest.json'}
    return {key: sha(path) if path.is_file() else None for key, path in paths.items()}


def initial_acceptance(run, expected, contract):
    """A file or attempted case is never sufficient for strict H0 acceptance."""
    try:
        run = Path(run)
        manifest = read(run / 'run_manifest.json', {})
        retained_manifest = run.parent / 'case_manifest.json'
        if (not retained_manifest.is_file() or sha(retained_manifest) != contract['case_manifest_sha256']
                or any(key not in manifest or manifest[key] != value for key, value in contract.items())):
            return False
        snapshot = read(run / 'initial_snapshot.json', {})
        sid = snapshot['id']
        completed = read(run / 'completed.json', {})
        attempts = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines() if line]
        if (snapshot.get('agent_type') != manifest.get('harness')
                or manifest.get('snapshot_id') != sid
                or manifest.get('privacy_protocol') != 'harnessrisk-private-case-v1'
                or completed.get('snapshot_id') != sid or completed.get('completed_scored') != 128
                or completed.get('attempts') != 128 or {r['case_id'] for r in attempts} != expected or len(attempts) != 128
                or any(r.get('exit_code') != 0 or r.get('rule_evaluation_present') is not True for r in attempts)):
            return False
        actual = {path.name for path in (run / 'cases').iterdir() if path.is_dir() and (path / 'harness_result.json').is_file()}
        if actual != expected:
            return False
        for cid in expected:
            case = run / 'cases' / cid
            result, audit = read(case / 'harness_result.json', {}), read(case / 'privacy_audit.json', {})
            rules = read(case / 'official_rule_evaluation.json', {}).get('runs', [])
            reset = read(case / 'adapter_metadata.json', {}).get('mock_start', {}).get('reset', {}).get('body', {})
            if (result.get('status') != 'completed' or result.get('process_exit_code') != 0 or result.get('snapshot_id') != sid
                    or audit.get('protocol') != 'harnessrisk-private-case-v1'
                    or reset.get('case_id') != cid or reset.get('ok') is not True
                    or len(rules) != 1 or rules[0].get('case_id') != cid or rules[0].get('status') != 'completed'):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def judge_acceptance(batch, expected):
    rows = read(Path(batch) / 'llm_judge_multi_harness/manifest.json', [])
    accepted = [row for row in rows if row.get('status') in ('judged', 'existing')]
    return len(rows) == 128 and len(accepted) == 128 and {row.get('case_id') for row in accepted} == expected


def failure_feedback(run):
    run = Path(run)
    counts, paths = Counter(), []
    attempts = run / 'attempts.jsonl'
    if attempts.exists():
        for line in attempts.read_text().splitlines():
            row = json.loads(line)
            result = read(run / 'cases' / row['case_id'] / 'harness_result.json', {})
            if row.get('exit_code') == 0 and result.get('status') == 'completed':
                continue
            reasons = [turn.get('stop_reason') for turn in result.get('turns', []) if turn.get('stop_reason') != 'completed']
            for reason in reasons or ['host_startup_or_missing_result']:
                reason = str(reason)
                classification = ('context_capacity' if 'context length' in reason or 'context cannot' in reason else
                    'budget_exhausted_invalid_under_current_strict_contract' if 'max_turn' in reason or 'max_iterations' in reason or 'maximum number of turns' in reason else
                    'dependency_missing' if 'No module named' in reason or 'not found' in reason else 'runtime_or_protocol_failure')
                counts[classification] += 1
            paths.append(str(run / 'host_observability' / row['case_id']))
    for path in run.glob('evidence/*/host_execution.json'):
        receipt = read(path, {})
        if receipt.get('status') == 'error':
            counts['host_or_protocol_failure'] += 1
            paths.append(str(path))
    error = read(run / 'error.json', {})
    fingerprint = hashlib.sha256(json.dumps({'counts': counts, 'error': error}, sort_keys=True).encode()).hexdigest()
    return {'classification_counts': dict(counts), 'evidence_paths': paths, 'error': error,
            'failure_fingerprint': fingerprint, 'repair_needed': True, 'unchanged_retry_allowed': False}


def freeze_sources(destination):
    paths = set((ROOT / 'scripts').glob('flashrsi*.py'))
    paths.update([ROOT / 'scripts/judge_harnessrisk_shengsuanyun.py', ROOT / 'scripts/flashrsi_harnessrisk_sandbox.py'])
    for path in (ROOT / 'vendor/FlashRSI/flashrsi').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and 'reference' not in path.parts and path.suffix != '.pyc':
            paths.add(path)
    named = {str(path.relative_to(ROOT)): path for path in paths}
    harness_root = ROOT.parents[1] / 'HarnessRisk'
    for path in (harness_root / 'harness_adapter').rglob('*.py'):
        named['@harnessrisk/' + str(path.relative_to(harness_root))] = path
    hashes = {}
    for key, path in sorted(named.items()):
        hashes[key] = sha(path)
        target = destination / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return hashes


def verify_frozen(hashes):
    drift = [name for name, expected in hashes.items() if not pin_path(name).is_file() or sha(pin_path(name)) != expected]
    if drift:
        raise RuntimeError('source_hash_drift_requires_new_protocol:' + ','.join(drift[:5]))


class Queue:
    def __init__(self, out):
        self.out, self.tasks, self.judges, self.judge = Path(out), {}, [], None
        self.model_pid = None

    def check_cleanup(self):
        # These receipts survive their host/stage subprocess. Descendants may
        # no longer match the lane scanner after that direct process exits.
        block_path = self.out / 'queue_cleanup_block.json'
        block = read(block_path)
        if block is None and block_path.exists():
            block = {'type': 'QueueCleanupUncertain', 'error': 'unreadable_existing_sticky_cleanup_block'}
        uncertain = []
        receipt_paths = []
        for pattern in ('*/*/host_observability/*/host_execution.json',
                        '*/*/evidence/*/host_execution.json', 'stage_receipts/*.json'):
            receipt_paths.extend(self.out.glob(pattern))
        for run in getattr(self, 'predecessor_scopes', []):
            receipt_paths.extend(run.glob('evidence/*/host_execution.json'))
        for path in receipt_paths:
            receipt = read(path)
            if (not isinstance(receipt, dict) or receipt.get('timed_out') is True
                    or receipt.get('cleanup_confirmed') is False
                    or receipt.get('owned_child_still_running') is True):
                uncertain.append({'path': str(path), 'sha256': sha(path)})
        if block is None and not uncertain:
            return
        if block is None:
            block = {'type': 'QueueCleanupUncertain', 'receipts': uncertain,
                     'identity_based_descendant_reconciliation_required': True,
                     'automatic_unblock_or_retry': False}
            save(block_path, block)
        for task in self.tasks.values():
            task['dispatch_blocked_by_cleanup'] = str(block_path)
            if task.get('status') in ('queued', 'running'):
                task['status'] = 'blocked_cleanup_uncertain'
        ready = read(self.out / 'ready.json', {})
        ready.update(ready_for_authorized_finite_queue=False, dispatch_blocked_cleanup=str(block_path))
        save(self.out / 'ready.json', ready)
        self.event('queue', 'blocked_cleanup_uncertain', blocker=str(block_path))
        raise QueueCleanupUncertain('queue_wide_descendant_cleanup_unconfirmed:' + str(block_path))

    def board(self):
        save(self.out / 'pidboard.json', {'launcher': process_identity(os.getpid()), 'model_pid': self.model_pid,
             'judge_pid': self.judge[0].pid if self.judge is not None else None, 'port': 18764})

    def event(self, stage, status, **data):
        row = {'at': datetime.datetime.now().astimezone().isoformat(), 'launcher_pid': os.getpid(),
               'stage': stage, 'status': status, **data}
        with (self.out / 'queue_events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        save(self.out / 'launch_state.json', row)
        save(self.out / 'remaining_tasks.json', {'tasks': self.tasks})
        self.board()
        print(json.dumps({'stage': stage, 'status': status}), flush=True)

    def poll_judge(self):
        if self.judge is None:
            return
        child, lock, log, key, batch = self.judge
        code = child.poll()
        if code is None:
            return
        log.close(); lock.close()
        valid = judge_acceptance(batch, self.expected)
        self.tasks[key]['judge_status'] = 'complete' if code == 0 and valid else 'needs_judge_or_billing_repair'
        if code == 0 and valid:
            self.tasks[key]['status'] = 'complete'
        self.tasks[key]['judge_returncode'] = code
        self.judge = None
        self.event(key, self.tasks[key]['judge_status'])

    def schedule_judge(self):
        self.poll_judge()
        if self.judge is not None or not self.judges:
            return
        self.check_cleanup()
        verify_frozen(getattr(self, 'hashes', {}))
        guard = read(Path.home() / '.cache/harnessrisk-judge/billing-blocked.json', {})
        if guard.get('retry_after', 0) > time.time():
            return
        lock = WORKER_LOCK.open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close(); return
        key, batch = self.judges.pop(0)
        if judge_acceptance(batch, self.expected):
            lock.close()
            self.tasks[key].update(status='complete', judge_status='complete_cached')
            self.event(key, 'complete_cached')
            return
        log = (self.out / (key.replace('/', '_') + '_judge.log')).open('ab')
        command = [sys.executable, 'scripts/judge_harnessrisk_shengsuanyun.py', str(batch), '--deployment', 'openai/gpt-5.4-nano']
        try:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, pass_fds=(lock.fileno(),))
        except BaseException:
            lock.close(); log.close(); raise
        self.judge = child, lock, log, key, batch
        self.tasks[key]['judge_status'] = 'running'
        self.event(key, 'judge_running', pid=child.pid, successful_existing_not_overwritten=True)

    def execute(self, key, command, env, hashes, case_hash):
        self.check_cleanup()
        verify_frozen(hashes)
        contract = self.contracts[env['HR_HARNESS']]
        if '--algorithm' in command and not initial_acceptance(Path(env['HR_INITIAL_REFERENCE']), self.expected, contract):
            raise RuntimeError('initial_reference_contract_drift_before_dispatch')
        for row in self.manifest['cases']:
            if sha(DATA / (row['case_id'] + '.json')) != row['sha256']:
                raise RuntimeError('source_case_hash_drift:' + row['case_id'])
        if sha(Path(env['HR_OUTPUT_ROOT']) / 'case_manifest.json') != case_hash:
            raise RuntimeError('case_manifest_drift')
        clients = lane_clients(exclude=(os.getpid(),))
        while clients:
            self.event(key, 'waiting_endpoint_capacity', clients=clients)
            self.schedule_judge(); time.sleep(15)
            clients = lane_clients(exclude=(os.getpid(),))
        log = (self.out / (key.replace('/', '_') + '.log')).open('xb')
        receipt = {'command': [sys.executable, '-u', *command], 'port': 18764,
                   'host': env['HR_HOST_SCRIPT'], 'case_manifest_sha256': case_hash,
                   'frozen_protocol_sha256': sha(self.out / 'repair_protocol.json'),
                   'initial_reference': env['HR_INITIAL_REFERENCE'], 'started_time_ns': time.time_ns(),
                   'expected_initial_contract': contract,
                   'initial_binding_before': initial_binding(env['HR_INITIAL_REFERENCE'])}
        initial = Path(env['HR_INITIAL_REFERENCE']) / 'initial_snapshot.json'
        receipt['initial_snapshot_sha256_before'] = sha(initial) if initial.exists() else None
        receipt_path = self.out / 'stage_receipts' / (key.replace('/', '_') + '.json')
        save(receipt_path, receipt)
        child = None
        try:
            child = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, env=env, stdout=log, stderr=log)
            self.model_pid = child.pid
            receipt['pid'] = child.pid
            self.tasks[key] = {'status': 'running', 'pid': child.pid, 'command': command}
            self.event(key, 'running', pid=child.pid)
            while child.poll() is None:
                self.schedule_judge(); time.sleep(5)
            verify_frozen(hashes)
            if '--algorithm' in command and initial_binding(env['HR_INITIAL_REFERENCE']) != receipt['initial_binding_before']:
                raise RuntimeError('initial_reference_contract_drift_during_stage')
            if receipt['initial_snapshot_sha256_before'] is not None and sha(initial) != receipt['initial_snapshot_sha256_before']:
                raise RuntimeError('initial_snapshot_drift_during_stage')
            receipt['returncode'] = child.returncode
            return child.returncode
        finally:
            log.close()
            receipt['finished_time_ns'] = time.time_ns()
            receipt['initial_snapshot_sha256_after'] = sha(initial) if initial.exists() else None
            receipt['initial_binding_after'] = initial_binding(env['HR_INITIAL_REFERENCE'])
            save(receipt_path, receipt)
            still_running = child is not None and child.poll() is None
            receipt['owned_child_still_running'] = still_running
            save(receipt_path, receipt)
            if not still_running:
                self.model_pid = None
            self.board()

    def run(self, manifest, predecessors, *, manifest_bytes=None):
        self.out.mkdir(mode=0o700, exist_ok=False)
        self.manifest = manifest
        self.predecessor_scopes = [Path(row['run_scope']) for row in predecessors if row.get('run_scope')]
        if manifest_bytes is None:
            manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
        if json.loads(manifest_bytes) != manifest:
            raise ValueError('trusted_manifest_bytes_object_mismatch')
        self.contracts = {harness: expected_initial_contract(manifest_bytes, harness)
                          for harness in ('opencode', 'openclaw', 'codex')}
        (self.out / 'case_manifest.json').write_bytes(manifest_bytes)
        self.tasks = {'opencode/' + a: {'status': 'queued'} for a in ('evoprompt', 'gepa')}
        self.tasks.update({h + '/' + a: {'status': 'queued'} for h in ('openclaw', 'codex', 'claude_code', 'pi') for a in ('initial', *ALGORITHMS)})
        for harness, samples in {'claude_code': ['action_041', 'action_042'], 'pi': ['action_041', 'action_043']}.items():
            for algorithm in ('initial', *ALGORITHMS):
                self.tasks[harness + '/' + algorithm] = {'status': 'blocked_protocol_repair',
                    'reason': 'native_partial_history_and_settled_calls_not_yet_preserved_for_bounded_owner_protocol',
                    'failed_sample_cases': samples, 'unchanged_retry_allowed': False,
                    'review': 'docs/research/flashrsi_bounded_owner_profile_review_20261117.md'}
        coverage = {'model_harnesses_this_queue': ['opencode', 'openclaw', 'codex'],
                    'deferred_harnesses': ['claude_code', 'pi'], 'full_five_harness_objective_complete': False,
                    'external_predecessor_outcome_not_counted_here': True}
        protocol = {'protocol': 'gpu1-finite-strict-matrix/v17', 'port': 18764,
            'queue_launcher': 'scripts/flashrsi_matrix_gpu1_queue_v17.py', 'job_ids': list(self.tasks),
            'predecessors': predecessors, 'coverage': coverage,
            'trusted_case_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'initial_contracts': self.contracts, 'opencode_reference_binding': initial_binding(REFERENCE),
            'backend_contract': {'context_window_tokens': 32768, 'max_reply_tokens': 4096,
                                 'existing_service_only': True, 'service_or_token_budget_changed': False}}
        save(self.out / 'repair_protocol.json', {**protocol, 'sources_frozen': False})
        save(self.out / 'ready.json', {'ready_for_authorized_finite_queue': True, 'coverage': coverage,
                                     'full_objective_complete': False})
        save(self.out / 'predecessors.json', predecessors)
        # No source snapshot churn while the existing real predecessor is active.
        while any(process_identity(p['pid']) == p for p in predecessors) or lane_clients(exclude=(os.getpid(),)):
            self.event('queue', 'waiting_endpoint_capacity', clients=lane_clients(exclude=(os.getpid(),)))
            time.sleep(15)
        self.lane_lock = LANE_LOCK.open('a')
        while True:
            try:
                fcntl.flock(self.lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                self.event('queue', 'waiting_endpoint_capacity', reason='existing_gpu1_lane_lease')
                time.sleep(15)
        self.check_cleanup()
        hashes = freeze_sources(self.out / 'source_snapshot')
        self.hashes = hashes
        expected = {row['case_id'] for row in manifest['cases']}
        self.expected = expected
        if len(expected) != 128 or len(manifest['cases']) != 128:
            raise RuntimeError('full_manifest_must_have_128_unique_cases')
        for row in manifest['cases']:
            if sha(DATA / (row['case_id'] + '.json')) != row['sha256']:
                raise RuntimeError('source_case_hash_changed')
        save(self.out / 'repair_protocol.json', {**protocol, 'sources_frozen': True, 'sources_sha256': hashes,
            'no_new_model_service': True, 'invalid_or_budget_error_never_zero': True,
            'declared_codex_parser_dependencies': {'tree-sitter': '0.25.2', 'tree-sitter-bash': '0.25.1'}})
        for harness in ('opencode', 'openclaw', 'codex'):
            base = self.out / harness
            base.mkdir()
            (base / 'case_manifest.json').write_bytes(manifest_bytes)
            case_hash = sha(base / 'case_manifest.json')
            env = {**os.environ, 'PYTHONPATH': 'scripts:src:vendor/FlashRSI', 'HR_HARNESS': harness,
                'CUDA_VISIBLE_DEVICES': '',
                'HARNESSRISK_MODEL_PORT': '18764', 'HR_OUTPUT_ROOT': str(base), 'HR_HOST_SCRIPT': HOSTS[harness],
                'HR_TARGET_PROTOCOL': HOSTS[harness], 'HR_RUN_NAME': harness + '_initial_full',
                'HR_STRICT_EVOLUTION': '1', 'HR_EVOLUTION_CASE_ATTEMPTS': '1', 'HR_WORKER_MAX_TOKENS': '1024'}
            env.pop('HR_RESUME', None)
            reference = REFERENCE if harness == 'opencode' else base / (harness + '_initial_full')
            env['HR_INITIAL_REFERENCE'] = str(reference)
            if harness != 'opencode':
                key = harness + '/initial'
                if harness == 'codex':
                    try:
                        versions = {name: importlib.metadata.version(name) for name in ('tree-sitter', 'tree-sitter-bash')}
                    except importlib.metadata.PackageNotFoundError:
                        versions = {}
                    if versions != {'tree-sitter': '0.25.2', 'tree-sitter-bash': '0.25.1'}:
                        self.tasks[key] = {'status': 'blocked_declared_dependency_drift', 'observed': versions}
                        for algorithm in ALGORITHMS:
                            self.tasks[harness + '/' + algorithm] = {'status': 'blocked_same_protocol_initial', 'initial': str(reference)}
                        self.event(key, 'blocked_declared_dependency_drift')
                        continue
                code = self.execute(key, ['scripts/flashrsi_harnessrisk_initial_v17.py'], env, hashes, case_hash)
                self.check_cleanup()
                good = code == 0 and initial_acceptance(reference, expected, self.contracts[harness])
                self.tasks[key] = {'status': 'source_valid_needs_judge' if good else 'blocked_initial_requires_repair',
                                   'returncode': code, 'path': str(reference)}
                if good:
                    self.judges.append((key, reference / 'cases'))
                else:
                    self.tasks[key]['feedback'] = failure_feedback(reference)
                    for algorithm in ALGORITHMS:
                        self.tasks[harness + '/' + algorithm] = {'status': 'blocked_same_protocol_initial', 'initial': str(reference)}
                    self.event(key, 'blocked_initial_requires_repair')
                    continue
            elif not (initial_acceptance(reference, expected, self.contracts[harness])
                      and judge_acceptance(reference / 'cases', expected)):
                raise RuntimeError('opencode_v11_reference_not_strictly_valid')
            reference_hash = sha(reference / 'initial_snapshot.json')
            algorithms = ('evoprompt', 'gepa') if harness == 'opencode' else ALGORITHMS
            for algorithm in algorithms:
                key = harness + '/' + algorithm
                if sha(reference / 'initial_snapshot.json') != reference_hash:
                    raise RuntimeError('initial_snapshot_drift')
                entry = 'flashrsi_harnessrisk_algorithm_reference_v18.py' if harness == 'opencode' and algorithm == 'gepa' else 'flashrsi_harnessrisk_algorithm_reference_v19.py'
                env['HR_GEPA_REFLECTION_PROTOCOL'] = 'opencode-v6-contextfit' if harness == 'opencode' else ''
                code = self.execute(key, ['scripts/' + entry, '--algorithm', algorithm], env, hashes, case_hash)
                self.check_cleanup()
                run = base / (harness + '_' + algorithm + '_full')
                accepted = inspect_run(run, expected, None)
                good = code == 0 and accepted['status'] in ('needs_judge', 'complete')
                self.tasks[key] = {'status': 'source_valid_needs_judge' if good else 'requires_repair', 'returncode': code,
                                   'path': str(run), 'acceptance': accepted}
                if good:
                    self.judges.append((key, run / 'selected_cases'))
                else:
                    self.tasks[key]['feedback'] = failure_feedback(run)
                self.event(key, self.tasks[key]['status'])
        while self.judge is not None or self.judges:
            before = len(self.judges)
            self.schedule_judge()
            if self.judge is not None:
                time.sleep(5)
            elif len(self.judges) == before:
                break  # Lease/billing blocked: explicit pending, finite exit.
        for key, batch in self.judges:
            self.tasks[key]['judge_status'] = 'needs_judge_lease_or_billing'
        save(self.out / 'remaining_tasks.json', {'tasks': self.tasks})
        self.event('queue', 'finite_plan_ended_not_necessarily_complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case-manifest', type=Path, default=REFERENCE.parent / 'case_manifest.json')
    parser.add_argument('--predecessor-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('queue_output_exists_preserved_requires_explicit_recovery:' + str(args.out))
    lock = (OUTPUTS / '.flashrsi_gpu1_finite_queue_v17.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    predecessors = [p for pid in args.predecessor_pid if (p := process_identity(pid)) is not None]
    queue = Queue(args.out)
    try:
        manifest_bytes = args.case_manifest.read_bytes()
        queue.run(json.loads(manifest_bytes), predecessors, manifest_bytes=manifest_bytes)
    except BaseException as exc:
        for task in queue.tasks.values():
            if task.get('status') in ('running', 'queued'):
                task.update(status='blocked_queue_exception', error_type=type(exc).__name__, error=str(exc))
        queue.event('queue', 'blocked_cleanup_uncertain' if isinstance(exc, QueueCleanupUncertain) else 'blocked_requires_repair',
                    error_type=type(exc).__name__, error=str(exc))
        raise
    return 0 if all(task.get('status') == 'complete' for task in queue.tasks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
