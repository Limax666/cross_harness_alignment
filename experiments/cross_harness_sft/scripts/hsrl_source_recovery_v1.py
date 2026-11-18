"""Audited continuation after the known source-namespace failure.

No source/model execution occurs here. Parent evidence remains immutable. A
uniform repaired step starts from the accepted checkpoint and known final RNG;
all previous attempts, including quarantined partial work, remain spent.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import textwrap

from hsrl_v5.io import file_hash, digest
from hsrl_v3.policy import validate_checkpoint_files

SCHEMA = 'hsrl-v9-source-repair-continuation-grpo/v1'
PROOF_SCHEMA = 'hsrl-known-source-failure-continuation/v1'


def known_source_failure(value, depth=0):
    """Decode bounded framework repr wrappers; a matching phrase alone is not a cause."""
    if depth > 6:
        return False
    if isinstance(value, dict):
        if (value.get('ok') is False and value.get('error_type') == 'NameError'
                and value.get('error') == "name 'ValueError' is not defined"):
            return True
        return (value.get('type') in ('SourceExecutionError', 'RuntimeError')
                and known_source_failure(value.get('detail'), depth + 1))
    if not isinstance(value, str) or len(value) > 16000:
        return False
    if value.startswith('wave_aborted:'):
        return known_source_failure(value[len('wave_aborted:'):], depth + 1)
    for prefix in ('real_episode_invalid_stop_for_repair:', 'source_implementation_exception:'):
        if value.startswith(prefix):
            try:
                return known_source_failure(ast.literal_eval(value[len(prefix):]), depth + 1)
            except (SyntaxError, ValueError, RecursionError):
                return False
    return False


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or file_hash(path) != expected:
            raise ValueError('continuation_evidence_changed:' + name)


def max_accepted_updates(state, max_attempts):
    return min(300, state['updates'] + (max_attempts - state['attempts']) // 128)


def make_continuation_state(failed, checkpoint_state, reconciliation, *, manifest_sha256, checkpoint):
    if (reconciliation.get('all_inflight_accounted') is not True
            or reconciliation.get('source_cleanup_confirmed') is not True):
        raise ValueError('verified_inflight_reconciliation_required')
    if (reconciliation.get('parent_attempts') != failed['attempts']
            or reconciliation.get('parent_updates') != failed['updates']):
        raise ValueError('continuation_counters_not_bound')
    for key in ('step', 'updates', 'effective_updates', 'policy_version', 'adapter_hash'):
        if failed[key] != checkpoint_state[key]:
            raise ValueError('parent_checkpoint_policy_or_update_mismatch:' + key)
    if failed['checkpoint'] != checkpoint_state['checkpoint']:
        raise ValueError('parent_checkpoint_pointer_mismatch')
    result = copy.deepcopy(checkpoint_state)
    result.update(schema=SCHEMA, manifest_sha256=manifest_sha256,
        checkpoint=str(checkpoint), attempts=failed['attempts'], episodes=[],
        inflight=None, assignment=None, status='ready')
    result.pop('failure', None)
    result['continuation'] = {
        'parent_journal_sha256': reconciliation['parent_journal_sha256'],
        'quarantined_parent_attempts': reconciliation['quarantined_current_step_attempts'],
        'uniform_repaired_step_required': True,
        'known_postfailure_RNG_not_checkpoint_start_RNG': True}
    return result


def assert_source_repair(api_proof_path, dataset_root):
    """Verify narrow code changes and the actual isolated source regression."""
    from hsrl_v5 import agentdog_worker as old_worker
    from hsrl_v5.isolation import JSONWorker as OldJSONWorker
    from hsrl_v5.agentdog_backend import AgentDogBackend as OldBackend
    from hsrl_source_repair_v1 import agentdog_worker as worker
    from hsrl_source_repair_v1.adapter import JSONWorker, AgentDogBackend
    proof = json.loads(Path(api_proof_path).read_text())
    if (proof.get('schema') != 'hsrl-real-source-exception-repair-regression/v1'
            or proof.get('valid') is not True or proof.get('actual_call_returned_records') is not True
            or len(proof.get('cleanup', [])) != 2
            or any(x.get('removal_confirmed') is not True for x in proof['cleanup'])):
        raise ValueError('real_source_repair_regression_required')
    verify_pins(proof['source_code_sha256'])
    verify_pins({proof['official_grader_response_path']: proof['official_grader_response_sha256'],
                 proof['result_path']: proof['result_sha256']})
    root = Path(dataset_root)
    if (file_hash(root / 'READY.json') != proof['READY_sha256']
            or file_hash(root / 'manifest.json') != proof['dataset_manifest_sha256']):
        raise ValueError('repair_regression_dataset_identity_changed')
    if set(worker.SAFE_BUILTINS) != set(old_worker.SAFE_BUILTINS) | {'ValueError', 'TypeError', 'Exception'}:
        raise ValueError('source_repair_builtin_scope_changed')
    original = ast.parse(Path(old_worker.__file__).read_text())
    repaired = ast.parse(Path(worker.__file__).read_text())
    for left, right in zip(original.body, repaired.body):
        if isinstance(left, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SAFE_BUILTINS' for t in left.targets):
            right.value.generators[0].iter = left.value.generators[0].iter
    if ast.dump(original) != ast.dump(repaired):
        raise ValueError('source_worker_changed_beyond_standard_exceptions')
    for old, new in ((OldJSONWorker, JSONWorker), (OldBackend, AgentDogBackend)):
        if ast.dump(ast.parse(textwrap.dedent(inspect.getsource(old.__init__)))) != ast.dump(
                ast.parse(textwrap.dedent(inspect.getsource(new.__init__)))):
            raise ValueError('source_repair_constructor_gate_changed')
    if (AgentDogBackend.call is not OldBackend.call or AgentDogBackend.grade is not OldBackend.grade
            or AgentDogBackend.close is not OldBackend.close or JSONWorker.close is not OldJSONWorker.close
            or OldBackend.grade.__globals__['JSONWorker'] is not OldJSONWorker):
        raise ValueError('source_reward_grader_or_cleanup_contract_changed')
    return proof


def assess_parent(parent, actor_pids):
    """Account for every reservation of this one known, fully closed failure."""
    import torch
    parent = Path(parent).resolve(strict=True)
    manifest = json.loads((parent / 'run_manifest.json').read_text())
    saved = torch.load(parent / 'journal.pt', map_location='cpu', weights_only=False)
    state = saved['state']
    if digest(state) != digest(json.loads((parent / 'state.json').read_text())):
        raise ValueError('parent_json_and_authoritative_journal_differ')
    if (state.get('status') != 'blocked' or state['updates'] != state['step']
            or not known_source_failure(state.get('failure'))):
        raise ValueError('unrecognized_parent_failure_requires_new_reconciliation')
    if file_hash(parent / 'run_manifest.json') != state['manifest_sha256']:
        raise ValueError('parent_manifest_changed')
    root = Path(__file__).resolve().parent
    verify_pins({str(root / name): pin for name, pin in manifest['identity']['code_sha256'].items()})
    checkpoint = Path(state['checkpoint'])
    if checkpoint.parent != parent / 'checkpoints':
        raise ValueError('parent_checkpoint_outside_owned_run')
    validate_checkpoint_files(checkpoint)
    recovery = torch.load(checkpoint / 'resume.pt', map_location='cpu', weights_only=False)
    if recovery['version'] != state['policy_version'] or recovery['adapter_hash'] != state['adapter_hash']:
        raise ValueError('parent_checkpoint_identity_mismatch')
    if set(saved['rng']) != {'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng'}:
        raise ValueError('known_final_sampling_RNG_required')
    if any(saved['rng'][key].numel() <= 0 for key in ('torch_rng', 'cuda_rng')):
        raise ValueError('missing_final_sampling_RNG')
    if any(Path('/proc', str(pid)).exists() for pid in actor_pids):
        raise ValueError('parent_actor_or_controller_still_alive')
    inflight = state.get('inflight') or {}
    attempts = inflight.get('attempts', [])
    start = len(state['episodes'])
    if (inflight.get('schema') != 'hsrl-eight-worker-main-thread-broker/v2'
            or inflight.get('step') != state['step'] or len(attempts) != 8 or start % 8
            or state['attempts'] != 128 * state['updates'] + start + 8):
        raise ValueError('unknown_or_unaccounted_parent_inflight')
    wave = parent / 'waves' / f'step_{state["step"]:04d}_wave_{start:03d}.json'
    if json.loads(wave.read_text()).get('unjoined_workers') != []:
        raise ValueError('parent_source_threads_not_joined')
    pins = {str(parent / name): file_hash(parent / name) for name in
            ('journal.pt', 'state.json', 'run_manifest.json', 'training_metrics.jsonl')}
    pins[str(checkpoint / 'COMPLETE.json')] = file_hash(checkpoint / 'COMPLETE.json')
    pins[str(wave)] = file_hash(wave)
    quarantined = []
    for entry in state['episodes']:
        path = Path(entry['episode_path'])
        episode = json.loads(path.read_text())
        if file_hash(path) != entry['sha256'] or episode.get('valid') is not True:
            raise ValueError('parent_retained_episode_changed_or_invalid')
        if episode['policy_version'] != state['policy_version'] or episode['adapter_hash_at_collection'] != state['adapter_hash']:
            raise ValueError('parent_partial_episode_policy_mismatch')
        pins[str(path)] = entry['sha256']
        quarantined.append({'path': str(path), 'sha256': entry['sha256'], 'valid_but_not_updated': True})
    failed_slots = []
    for offset, attempt in enumerate(attempts):
        directory = parent / 'episodes' / f'step_{state["step"]:04d}' / f'rollout_{start+offset:03d}'
        if (attempt['slot'] != start + offset or Path(attempt['directory']) != directory
                or attempt['attempt'] != state['attempts'] - 7 + offset):
            raise ValueError('parent_reserved_slot_identity_unknown')
        path = directory / 'episode.json'
        episode = json.loads(path.read_text())
        if (episode.get('valid') is not False or not known_source_failure(episode.get('error'))
                or episode.get('training_reward') is not None):
            raise ValueError('failed_slot_not_explicitly_invalid_for_known_cause')
        receipts = list((directory / 'private/backend').glob('*/cleanup.json'))
        if not receipts:
            raise ValueError('source_cleanup_receipt_missing')
        for receipt in receipts:
            data = json.loads(receipt.read_text())
            if data.get('removal_confirmed') is not True or data.get('prior_removal_error'):
                raise ValueError('source_cleanup_uncertain')
            result = subprocess.run(['docker', '--host=unix:///run/user/1026/docker.sock',
                'container', 'inspect', data['container']], capture_output=True, text=True, timeout=20)
            if result.returncode == 0 or 'No such' not in result.stderr:
                raise ValueError('source_container_disappearance_not_confirmed')
            pins[str(receipt)] = file_hash(receipt)
        pins[str(path)] = file_hash(path)
        failed_slots.append({'slot': attempt['slot'], 'episode_path': str(path), 'valid': False,
                             'all_source_cleanup_confirmed': True})
    receipt = {'all_inflight_accounted': True, 'source_cleanup_confirmed': True,
        'parent_attempts': state['attempts'], 'parent_updates': state['updates'],
        'quarantined_current_step_attempts': start + 8,
        'parent_journal_sha256': file_hash(parent / 'journal.pt')}
    return {'parent_run': str(parent), 'parent_checkpoint': str(checkpoint),
        'parent_actor_pids': list(actor_pids), 'pins': pins, 'reconciliation': receipt,
        'quarantined_valid_unupdated': quarantined, 'failed_slots': failed_slots,
        'assignment': state['assignment'], 'parent_identity': manifest['identity'],
        'parent_initialized_policy': manifest['initialized_policy']}


def load_continuation_proof(path):
    import torch
    path = Path(path).resolve(strict=True)
    proof = json.loads(path.read_text())
    if proof.get('schema') != PROOF_SCHEMA:
        raise ValueError('verified_continuation_proof_required')
    verify_pins(proof['pins'])
    actual = assess_parent(proof['parent_run'], proof['parent_actor_pids'])
    for key in ('parent_checkpoint', 'reconciliation', 'quarantined_valid_unupdated',
                'failed_slots', 'assignment', 'parent_identity', 'parent_initialized_policy', 'pins'):
        if digest(actual[key]) != digest(proof[key]):
            raise ValueError('continuation_reconciliation_changed:' + key)
    verify_pins({proof['repair_api_proof_path']: proof['repair_api_proof_sha256']})
    assert_source_repair(proof['repair_api_proof_path'], proof['parent_identity']['dataset_root'])
    saved = torch.load(Path(proof['parent_run']) / 'journal.pt', map_location='cpu', weights_only=False)
    recovery = torch.load(Path(proof['parent_checkpoint']) / 'resume.pt', map_location='cpu', weights_only=False)
    return {'proof': proof, 'failed_state': saved['state'], 'rng': saved['rng'],
            'checkpoint_state': recovery['state'], 'proof_sha256': file_hash(path)}
