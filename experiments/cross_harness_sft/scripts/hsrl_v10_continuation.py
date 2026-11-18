"""Reconcile the closed v9 wave and bind accepted episodes to its checkpoint."""
from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path
import subprocess

from hsrl_slot_recovery_v10 import cleanup_receipts, entry_for, require_resumable, validate_rejection, REJECTION
from hsrl_source_recovery_v1 import SCHEMA as PARENT_SCHEMA, verify_pins
from hsrl_v3.policy import validate_checkpoint_files
from hsrl_v5.io import digest, file_hash
from hsrl_batched_rollout_v2 import VERSION as PARENT_WAVE_SCHEMA
from train_hsrl_v5 import config_name

SCHEMA = 'hsrl-v10-slot-recovery-continuation-grpo/v1'
PROOF_SCHEMA = 'hsrl-closed-v9-wave-reconciliation/v1'
ROOTLESS_SOCKET = '/run/user/1026/docker.sock'
CLAW_SOCKET = '/run/user/1026/hsrl-v7/docker.sock'


def assess_parent(parent, actor_pids):
    """Read the actual journal, scorer artifacts and Docker cleanup evidence."""
    parent = Path(parent).resolve(strict=True)
    if any(Path('/proc', str(pid)).exists() for pid in actor_pids):
        raise ValueError('parent_actor_or_controller_still_alive')
    with (parent / 'run.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _assess_closed_parent(parent)


def _assess_closed_parent(parent):
    import torch

    manifest_path = parent / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    identity = manifest['identity']
    journal_path = parent / 'journal.pt'
    saved = torch.load(journal_path, map_location='cpu', weights_only=False)
    state = saved['state']
    if (state['schema'] != PARENT_SCHEMA or state['status'] != 'blocked'
            or state['updates'] != state['step'] or state['policy_version'] != state['updates']
            or digest(state) != digest(json.loads((parent / 'state.json').read_text()))
            or file_hash(manifest_path) != state['manifest_sha256']
            or state['failure']['detail'] != "real_episode_invalid_stop_for_repair:{'type': 'ValueError', 'detail': 'source_measurement_invalid_or_incomplete'}"):
        raise ValueError('parent_is_not_a_closed_source_measurement_rejection')
    scripts = Path(__file__).resolve().parent
    verify_pins({str(scripts / name): sha for name, sha in identity['code_sha256'].items()})
    provenance = identity['source_execution_repair']
    verify_pins({provenance['real_API_regression_proof']: provenance['real_API_regression_sha256']})
    checkpoint = Path(state['checkpoint'])
    if checkpoint.parent != parent / 'checkpoints':
        raise ValueError('parent_checkpoint_outside_run')
    validate_checkpoint_files(checkpoint)
    recovery = torch.load(checkpoint / 'resume.pt', map_location='cpu', weights_only=False)
    checkpoint_state = recovery['state']
    for field in ('schema', 'step', 'updates', 'effective_updates', 'policy_version', 'adapter_hash',
                  'checkpoint', 'manifest_sha256'):
        if checkpoint_state[field] != state[field]:
            raise ValueError('checkpoint_and_failed_journal_identity_mismatch:' + field)
    if (recovery['version'] != state['policy_version'] or recovery['adapter_hash'] != state['adapter_hash']
            or set(saved['rng']) != {'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng'}
            or any(saved['rng'][key].numel() <= 0 for key in ('torch_rng', 'cuda_rng'))):
        raise ValueError('parent_policy_or_RNG_identity_mismatch')
    prefix = state['episodes']
    start = len(prefix)
    wave = state['inflight']
    reservations = wave['attempts']
    if (not 0 <= start < 128 or start % 8 or wave['schema'] != PARENT_WAVE_SCHEMA
            or wave['step'] != state['step'] or len(reservations) != 8
            or checkpoint_state['attempts'] + start + 8 != state['attempts']):
        raise ValueError('parent_inflight_cannot_be_accounted')
    wave_path = parent / 'waves' / f'step_{state["step"]:04d}_wave_{start:03d}.json'
    wave_metrics = json.loads(wave_path.read_text())
    if wave_metrics['unjoined_workers'] != [] or wave_metrics['slots'] != list(range(start, start+8)):
        raise ValueError('parent_workers_not_confirmed_joined')
    pins = {str(p): file_hash(p) for p in (journal_path, manifest_path, parent / 'state.json',
             parent / 'training_metrics.jsonl', checkpoint / 'COMPLETE.json', wave_path)}
    pending, rejected, failures = {}, [], {}
    containers = {}
    plan = state['assignment']
    for slot in range(start + 8):
        group = plan['groups'][slot // 8]
        directory = parent / 'episodes' / f'step_{state["step"]:04d}' / f'rollout_{slot:03d}'
        path = directory / 'episode.json'
        entry, record = entry_for(path, (group['source'], group['id']), state,
                                  config_name(plan['configuration']))
        if slot < start and (prefix[slot] != entry or record['valid'] is not True):
            raise ValueError('parent_committed_episode_evidence_changed')
        if slot >= start:
            attempt = reservations[slot-start]
            if (attempt['slot'] != slot or Path(attempt['directory']) != directory
                    or attempt['attempt'] != checkpoint_state['attempts'] + slot + 1):
                raise ValueError('parent_reserved_slot_identity_mismatch')
        pins[str(path)] = entry['sha256']
        if record['valid'] is True:
            receipts = cleanup_receipts(directory, group['source'])
            if slot >= start:
                pending[str(slot)] = entry
        elif slot >= start:
            receipts = validate_rejection(directory, record)
            failures[str(slot)] = 1
            rejected.append({**entry, 'step': state['step'], 'slot': slot,
                             'attempt': attempt['attempt'], 'reason': REJECTION})
        else:
            raise ValueError('parent_committed_episode_invalid')
        for receipt, container in receipts.items():
            pins[str(receipt)] = file_hash(receipt)
            socket = ROOTLESS_SOCKET
            if group['source'] == 'clawgym':
                isolation = receipt.with_name('isolation.json')
                boundary = json.loads(isolation.read_text())
                if boundary['container'] != container or boundary.get('docker_socket') != CLAW_SOCKET:
                    raise ValueError('parent_workspace_socket_or_container_identity_changed')
                pins[str(isolation)] = file_hash(isolation)
                socket = boundary['docker_socket']
            containers.setdefault(socket, set()).add(container)
        if record['valid'] is not True:
            result = directory / 'private/backend/result.json'
            pins[str(result)] = file_hash(result)
    if not rejected:
        raise ValueError('parent_rejected_attempt_not_found')
    for socket, names in containers.items():
        owned = subprocess.run(['docker', f'--host=unix://{socket}', 'ps', '-a', '--format', '{{.Names}}'],
                               check=True, capture_output=True, text=True, timeout=20)
        if names.intersection(owned.stdout.splitlines()):
            raise ValueError('parent_container_disappearance_not_confirmed')
    return {
        'schema': PROOF_SCHEMA, 'parent_run': str(parent), 'parent_checkpoint': str(checkpoint),
        'parent_identity': identity, 'parent_initialized_policy': manifest['initialized_policy'],
        'assignment': plan, 'pins': pins, 'parent_attempts': state['attempts'],
        'parent_updates': state['updates'], 'parent_episode_count': start,
        'pending_slots': pending, 'slot_failures': failures, 'rejected_attempts': rejected,
        'closed_source_container_count': sum(map(len, containers.values())),
        'checked_source_sockets': sorted(containers),
    }


def load_continuation_proof(path):
    import torch

    path = Path(path).resolve(strict=True)
    proof = json.loads(path.read_text())
    pids = proof.pop('parent_actor_pids')
    verify_pins(proof['pins'])
    if digest(proof) != digest(assess_parent(proof['parent_run'], pids)):
        raise ValueError('parent_wave_reconciliation_changed')
    saved = torch.load(Path(proof['parent_run']) / 'journal.pt', map_location='cpu', weights_only=False)
    return {'proof': proof, 'failed_state': saved['state'], 'rng': saved['rng'],
            'proof_sha256': file_hash(path)}


def make_continuation_state(continuation, *, manifest_sha256, checkpoint):
    proof = continuation['proof']
    state = copy.deepcopy(continuation['failed_state'])
    state.update(schema=SCHEMA, status='ready', manifest_sha256=manifest_sha256,
                 checkpoint=str(checkpoint), inflight=None,
                 pending_slots=copy.deepcopy(proof['pending_slots']),
                 slot_failures=copy.deepcopy(proof['slot_failures']),
                 rejected_attempts=copy.deepcopy(proof['rejected_attempts']))
    state.pop('failure')
    state['continuation'] = {'parent_proof_sha256': continuation['proof_sha256'],
                             'committed_parent_episodes': proof['parent_episode_count'],
                             'accepted_parent_wave_peers': len(proof['pending_slots']),
                             'known_postfailure_RNG_not_checkpoint_start_RNG': True}
    require_resumable(state)
    return state
