"""Single physical GPU0, real READY-v5 online LoRA GRPO. No synthetic fallback.

The durable journal reserves every real attempt before environment execution.
An interrupted in-flight episode requires explicit reconciliation; it is never
silently repeated. Accepted updates and partial collections restore from the
same atomic checkpoint plus per-episode RNG journal.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import stat
import subprocess
import time

from hsrl_v5.io import atomic_json, append_jsonl, digest, file_hash
import hsrl_checkpoint_retention_v1 as checkpoint_retention
from hsrl_v5.reward import VERSION as REWARD_VERSION
from hsrl_v5.runtime import LIMITS, VERSION as RUNTIME_VERSION, run_episode
from hsrl_v3.policy import LOGPROB_CONTRACT, VERSION as POLICY_VERSION
from hsrl_v3.components import config_name

SCHEMA = 'hsrl-v5-real-online-grpo/v1'
OFFICIAL_IDENTITY_PATH = Path(__file__).resolve().parents[1] / 'configs/hsrl_official4b_851bf6e.json'
CONFIG = {'optimizer': 'AdamW', 'lr': 1e-5, 'betas': [0.9, 0.999], 'eps': 1e-8,
    'weight_decay': 0., 'warmup_actual_updates': 2, 'grad_clip_norm': 1.,
    'ppo_clip': .2, 'kl_coefficient': .001, 'optimization_epochs': 1,
    'advantage_std': 'population', 'advantage_epsilon': 1e-6,
    'temperature': .7, 'top_p': .95, 'top_k': 0, 'rank': 16, 'alpha': 32, 'dropout': 0.,
    'G': 8, 'K': 2, 'groups_per_step': 16, 'safety_ratio': .5,
    'limits': LIMITS, 'logprob_contract': LOGPROB_CONTRACT,
    'distribution_note': 'raw-policy PPO surrogate; not transformed temperature/nucleus probabilities',
    'pg_reduction': 'sum_over_applicable_sampled_action_tokens/global_applicable_action_tokens',
    'kl_reduction': 'sum_over_all_admitted_sampled_action_tokens/global_admitted_action_tokens',
    'utility_NA_policy': 'PG masked, reference KL retained; no malicious utility imputation',
    'workflow_multicall': 'execute generated calls atomically within budgets; check at next model turn',
    'partial_group_policy': 'complete G8 required; infrastructure failure blocks run',
    'zero_variance_policy': 'retain zero PG advantage and costs; reference KL still applies',
    'max_attempts': 38400, 'max_steps': 300, 'min_mem_available_GiB': 32,
    'min_disk_free_GiB': 100, 'max_new_run_disk_GiB': 60,
    'checkpoint_retention': checkpoint_retention.POLICY}


def reserve_attempt(state, directory, *, max_attempts):
    if state.get('inflight') is not None:
        raise ValueError('inflight_attempt_requires_reconciliation_no_reexecution')
    if state['attempts'] >= max_attempts:
        raise ValueError('attempt_budget_exhausted')
    state['attempts'] += 1
    state['inflight'] = {'directory': str(directory), 'attempt': state['attempts']}


def complete_attempt(state, entry):
    if state.get('inflight') is None:
        raise ValueError('no_reserved_attempt')
    state['episodes'].append(entry)
    state['inflight'] = None


def seal_episode_evidence(out, episode):
    """Durable quiescent episode before its completed journal pointer exists."""
    out, episode = Path(out).absolute(), Path(episode).absolute()
    if out.resolve() != out or episode.resolve() != episode:
        raise ValueError('episode_evidence_symlink_path')
    if not episode.is_relative_to(out / 'episodes') or not episode.is_dir():
        raise ValueError('episode_evidence_outside_owned_run')
    paths = list(episode.rglob('*'))
    directories = [episode]
    for path in paths:
        if path.is_symlink():
            raise ValueError('episode_evidence_symlink')
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            directories.append(path)
        elif stat.S_ISREG(mode):
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise ValueError('episode_evidence_special_file')
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        descriptor = os.open(path, os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    parent = episode.parent
    while True:
        descriptor = os.open(parent, os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if parent == out.parent:
            break
        parent = parent.parent


def commit_completed_attempt(out, state, entry, policy):
    episode = Path(entry['episode_path']).parent
    if ((state.get('inflight') or {}).get('directory') != str(episode)
            or file_hash(entry['episode_path']) != entry['sha256']):
        raise ValueError('completed_episode_does_not_match_reserved_attempt')
    # Every barrier precedes mutation. A failure leaves the reserved attempt
    # and previous authoritative journal available for explicit reconciliation.
    seal_episode_evidence(out, episode)
    complete_attempt(state, entry)
    persist_journal(out, state, policy)


def prepare_update(groups, policy_version):
    """Validate complete real groups and precompute global action reductions."""
    pg_tokens, kl_tokens, prepared = 0, 0, []
    for group in groups:
        if len(group) != 8 or any(ep.get('valid') is not True for ep in group):
            raise ValueError('complete_G8_required_no_partial_group_update')
        flags = {ep['pg_applicable'] for ep in group}
        if len(flags) != 1:
            raise ValueError('mixed_objective_applicability_within_group')
        pg_enabled = flags.pop()
        if type(pg_enabled) is not bool:
            raise ValueError('missing_objective_mask')
        rewards = [ep['training_reward'] for ep in group]
        if pg_enabled:
            if any(type(r) not in (int, float) or not math.isfinite(r) for r in rewards):
                raise ValueError('invalid_applicable_reward')
            mean = sum(rewards) / 8
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / 8)
            advantages = [(r - mean) / (std + CONFIG['advantage_epsilon']) for r in rewards]
        else:
            if any(r is not None for r in rewards):
                raise ValueError('nonapplicable_utility_must_remain_None')
            advantages, std = [0.] * 8, None
        for ep in group:
            if ep.get('reward_contract') != REWARD_VERSION or not ep['segments']:
                raise ValueError('unfrozen_reward_or_empty_real_segments')
            for segment in ep['segments']:
                if segment['policy_version'] != policy_version:
                    raise ValueError('stale_rollout_policy')
                count = len(segment['response_ids'])
                if (not segment['prompt_ids'] or not count
                        or len(segment['old_logprobs']) != count
                        or len(segment['reference_logprobs']) != count
                        or any(not math.isfinite(lp) for lp in
                               segment['old_logprobs'] + segment['reference_logprobs'])):
                    raise ValueError('invalid_exact_segment_provenance')
                mask = segment.get('action_mask', [True] * count)
                if len(mask) != count or any(type(v) is not bool for v in mask):
                    raise ValueError('invalid_action_token_mask')
                effective = sum(mask)
                kl_tokens += effective
                pg_tokens += effective if pg_enabled else 0
        prepared.append({'episodes': group, 'advantages': advantages,
                         'pg_enabled': pg_enabled, 'reward_std': std})
    if not groups or kl_tokens <= 0:
        raise ValueError('no_admitted_action_tokens')
    return {'groups': prepared, 'pg_tokens': pg_tokens, 'kl_tokens': kl_tokens}


def _atomic_torch(path, value, torch):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def persist_journal(out, state, policy):
    # State and all RNG streams are in ONE atomic file. JSON is a human-readable
    # mirror only and never authoritative for continuation.
    _atomic_torch(out / 'journal.pt', {'state': state, 'rng': policy.rng_state()}, policy.torch)
    descriptor = os.open(Path(out).parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)  # Persist the run directory itself before action.
    finally:
        os.close(descriptor)
    atomic_json(out / 'state.json', state)


def verify_pinned_artifact(path, pin):
    """Compare actual bytes with a source object ID, returning measured SHA256."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('official_artifact_missing_or_symlink:' + path.name)
    sha256, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha256.update(block)
            blob.update(block)
    algorithm = pin.get('algorithm')
    if algorithm not in ('sha256', 'git_blob_sha1'):
        raise ValueError('unsupported_official_artifact_source_identity')
    observed = sha256.hexdigest() if algorithm == 'sha256' else blob.hexdigest()
    if observed != pin['digest']:
        raise ValueError('official_artifact_content_differs_from_trusted_source:' + path.name)
    return sha256.hexdigest()


def verify_official_checkpoint(model):
    """Authenticate initial input against the separately frozen source map.

    The approval map is code-owned, not supplied by the checkpoint or run. HF
    LFS ETags are SHA256; regular repository object IDs use Git blob SHA1.
    Download metadata is retained provenance, never runtime authorization.
    """
    approved = json.loads(OFFICIAL_IDENTITY_PATH.read_text())
    if (approved.get('schema') != 'hsrl-trusted-official-checkpoint/v1'
            or approved.get('source_repo') != 'Qwen/Qwen3.5-4B'
            or approved.get('source_revision') != '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'):
        raise ValueError('trusted_official_identity_missing_or_incompatible')
    model = Path(model)
    if model.resolve() != Path(approved['canonical_path']) or model.is_symlink():
        raise ValueError('model_differs_from_trusted_canonical_official_path')
    model = model.resolve(strict=True)
    names = {p.name for p in model.iterdir() if p.is_file() and p.suffix in (
        '.json', '.safetensors', '.jinja', '.txt', '.model', '.bin', '.pt', '.pth', '.py')}
    pins = approved['artifacts']
    if names != set(pins) or any(Path(name).name != name for name in pins):
        raise ValueError('official_artifact_load_relevant_file_map_changed')
    measured = {name: verify_pinned_artifact(model / name, pin) for name, pin in sorted(pins.items())}
    return {'approved_identity': approved, 'approved_identity_sha256': file_hash(OFFICIAL_IDENTITY_PATH),
            'verified_artifact_sha256': measured}


def commit_checkpoint(out, policy, optimizer, scheduler, next_state, *, current_state=None):
    """Only a completely installed durable recovery point may enter the journal."""
    policy.save(Path(next_state['checkpoint']), optimizer, next_state, scheduler)
    if current_state is not None:
        # After durable installation, even an error writing the readable mirror
        # must leave the exception handler pointing at the installed recovery
        # state. Before successful save the previous state remains untouched.
        current_state.clear()
        current_state.update(next_state)
        next_state = current_state
    persist_journal(out, next_state, policy)
    # Legacy runs lack ownership and cannot opt in through resume: run identity
    # pins both this policy and the helper bytes for every fresh owned run.
    if (Path(out) / checkpoint_retention.LEDGER).exists():
        anchors = list(next_state.get('retention_anchors', []))
        anchors += [next_state[k] for k in ('best_checkpoint', 'best_dev_checkpoint',
                    'terminal_checkpoint') if next_state.get(k)]
        checkpoint_retention.after_commit(out, next_state['checkpoint'], anchors=anchors,
            load_recovery=lambda path: policy.torch.load(path, map_location='cpu', weights_only=False))
    return next_state


def _code_identity():
    root = Path(__file__).resolve().parent
    files = [Path(__file__).resolve(), root / 'hsrl_v3/policy.py', root / 'hsrl_v3/components.py',
             root / 'hsrl_checkpoint_retention_v1.py']
    files += sorted((root / 'hsrl_v5').glob('*.py'))
    return {p.relative_to(root).as_posix(): file_hash(p) for p in files}


def validate_lease(path):
    lease = json.loads(Path(path).read_text())
    if (lease.get('schema') != 'hsrl-gpu0-training-handoff/v1'
            or lease.get('physical_gpu') != 0 or lease.get('exclusive') is not True
            or lease.get('gpu1_baselines_preserved') is not True
            or lease.get('forbidden_gpus') != [2, 3]
            or lease.get('boundary_status') not in ('accepted_complete', 'stages_ended_preserved_unresolved_work')):
        raise ValueError('explicit_GPU0_only_boundary_handoff_required')
    if any(Path('/proc', str(pid)).exists() for pid in lease['old_service_pids']):
        raise ValueError('old_GPU0_service_or_waiter_still_live')
    ledger = Path(lease['baseline_ledger_path'])
    if not ledger.is_absolute() or file_hash(ledger) != lease['baseline_ledger_sha256']:
        raise ValueError('baseline_migration_ledger_identity_changed')
    if not Path(lease['lock_path']).is_absolute():
        raise ValueError('absolute_GPU0_lease_lock_required')
    return lease


def resource_gate(out, *, initial=False):
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    free = shutil.disk_usage(out).free
    if available < 32 * 2**30 or free < 100 * 2**30:
        raise RuntimeError('resource_gate_memory_or_disk_below_registered_floor')
    # Include all retained adapters, evidence and bounded recovery states.
    # Retention runs only after durable commits, never to bypass this cap.
    run_bytes = sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())
    if run_bytes >= CONFIG['max_new_run_disk_GiB'] * 2**30:
        raise RuntimeError('registered_60GiB_run_growth_cap_reached')
    if initial:
        pids = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True).stdout.strip()
        if pids:
            raise RuntimeError('GPU0_has_compute_processes_before_exclusive_load:' + pids)
    return {'mem_available_bytes': available, 'disk_free_bytes': free, 'run_bytes': run_bytes,
            'host_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('CUDA_VISIBLE_DEVICES_must_be_exactly_physical_0')
    if not (256 <= args.max_attempts <= 38400 and args.max_attempts % 256 == 0):
        raise ValueError('budget_must_be_complete_K2_blocks_up_to_38400_attempts')
    from hsrl_v5.dataset import Dataset
    from hsrl_v3.policy import Policy
    out, model, lease_path = args.out.resolve(), args.model.resolve(strict=True), args.lease.resolve(strict=True)
    dataset = Dataset(args.dataset)
    official = verify_official_checkpoint(model)
    lease = validate_lease(lease_path)
    if args.resume:
        if not (out / 'journal.pt').is_file():
            raise ValueError('resume_requires_existing_durable_journal')
    else:
        out.mkdir(parents=True, exist_ok=False)
        descriptor = os.open(out.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    lock_path = Path(lease['lock_path'])
    with lock_path.open('a+') as lease_lock, (out / 'run.lock').open('a+') as run_lock:
        fcntl.flock(lease_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resources = resource_gate(out, initial=True)
        identity = {'schema': SCHEMA, 'dataset_root': str(dataset.root),
            'READY_sha256': file_hash(dataset.root / 'READY.json'),
            'dataset_manifest_sha256': file_hash(dataset.root / 'manifest.json'),
            'reward_contract': REWARD_VERSION, 'runtime_version': RUNTIME_VERSION,
            'policy_version': POLICY_VERSION, 'config': CONFIG, 'seed': args.seed,
            'objective': args.objective, 'max_attempts': args.max_attempts,
            'model_root': str(model), 'base_artifact_sha256': official['verified_artifact_sha256'],
            'approved_model_identity': official['approved_identity'],
            'approved_model_identity_sha256': official['approved_identity_sha256'],
            'model_revision': official['approved_identity']['source_revision'],
            'code_sha256': _code_identity(), 'lease_path': str(lease_path),
            'lease_sha256': file_hash(lease_path), 'resource_handoff': lease,
            'reference': 'same_immutable_official_base_with_LoRA_disabled',
            'frozen_signature_semantics': 'parameter version/name/shape/dtype mutation guard, not weight-content hash',
            'checkpoint_commit': 'journal checkpoint pointer authoritative; unpointed checkpoints retained as uncommitted evidence',
            'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'numpy')}}
        manifest_path = out / 'run_manifest.json'
        if args.resume:
            if json.loads(manifest_path.read_text())['identity'] != identity:
                raise ValueError('resume_data_model_code_config_or_lease_identity_changed')
        else:
            atomic_json(manifest_path, {'identity': identity, 'identity_sha256': digest(identity),
                'created_at': datetime.now(timezone.utc).isoformat(), 'initial_resources': resources})
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        policy = Policy(model, args.seed)
        torch = policy.torch
        actual_policy = {'language_LoRA_target_modules': policy.targets,
            'trainable_parameter_count': sum(p.numel() for p in policy.trainable),
            'trainable_parameter_dtypes': sorted({str(p.dtype) for p in policy.trainable}),
            'actual_tokenizer_chat_template_sha256': digest(policy.tokenizer.chat_template),
            'actual_protocol_eos_ids': policy.eos_ids, 'logprob_contract': LOGPROB_CONTRACT}
        manifest = json.loads(manifest_path.read_text())
        if args.resume:
            if manifest.get('initialized_policy') != actual_policy:
                raise ValueError('resumed_actual_policy_or_tokenizer_contract_changed')
        else:
            manifest['initialized_policy'] = actual_policy
            atomic_json(manifest_path, manifest)
            checkpoint_retention.initialize(out)
        if not (out / checkpoint_retention.LEDGER).is_file():
            raise ValueError('fresh_owned_checkpoint_retention_ledger_required')
        optimizer = torch.optim.AdamW(policy.trainable, lr=CONFIG['lr'], betas=tuple(CONFIG['betas']),
                                     eps=CONFIG['eps'], weight_decay=CONFIG['weight_decay'])
        # LambdaLR applies lambda(0) at construction: update1 uses 0.5*lr,
        # update2 and later use lr. Only accepted optimizer steps advance it.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: min((index + 1) / 2, 1.))
        if args.resume:
            saved = torch.load(out / 'journal.pt', map_location='cpu', weights_only=False)
            state = saved['state']
            if state.get('inflight') is not None:
                raise ValueError('inflight_attempt_requires_reconciliation_no_reexecution')
            policy.load(Path(state['checkpoint']), optimizer, scheduler)
            policy.restore_rng(saved['rng'])
        else:
            initial = out / 'checkpoints/initial'
            state = {'schema': SCHEMA, 'manifest_sha256': file_hash(manifest_path),
                'step': 0, 'updates': 0, 'attempts': 0, 'inflight': None, 'episodes': [],
                'checkpoint': str(initial), 'status': 'ready', 'policy_version': 0,
                'adapter_hash': policy.adapter_hash(), 'covered_task_conditions': [],
                'covered_families': [], 'next_K_member': 0, 'effective_updates': 0, 'assignment': None}
            commit_checkpoint(out, policy, optimizer, scheduler, state)
        if state['manifest_sha256'] != file_hash(manifest_path):
            raise ValueError('journal_manifest_identity_changed')
        if state['policy_version'] != policy.version or state['adapter_hash'] != policy.adapter_hash():
            raise ValueError('journal_checkpoint_policy_identity_mismatch')
        try:
            while state['step'] < CONFIG['max_steps']:
                if state['attempts'] + (128 - len(state['episodes'])) > args.max_attempts:
                    break
                if _code_identity() != identity['code_sha256']:
                    raise ValueError('training_code_changed_mid_run')
                if file_hash(dataset.root / 'manifest.json') != identity['dataset_manifest_sha256']:
                    raise ValueError('training_manifest_changed_mid_run')
                before_resources = resource_gate(out)
                plan = dataset.sampler.step(state['step'])
                if len(plan['groups']) != 16 or plan['G'] != 8:
                    raise ValueError('frozen_16_group_G8_contract_changed')
                row = plan['configuration']
                if state.get('assignment') is not None and state['assignment'] != plan:
                    raise ValueError('resumed_sampler_assignment_changed')
                state['assignment'] = plan
                # Reject invalid retained evidence before any new attempt.
                # A repair cannot silently replace its failed source outcome.
                for entry in state['episodes']:
                    retained = json.loads(Path(entry['episode_path']).read_text())
                    if file_hash(entry['episode_path']) != entry['sha256'] or retained['valid'] is not True:
                        raise ValueError('retained_episode_invalid_or_changed_requires_reconciliation')
                state['status'] = 'collecting'
                if state['adapter_hash'] != policy.adapter_hash():
                    raise ValueError('actor_weights_do_not_match_committed_policy')
                policy.collection_adapter_hash = state['adapter_hash']
                policy.model.eval()
                torch.cuda.reset_peak_memory_stats()
                batch_started = time.monotonic()
                for slot in range(len(state['episodes']), 128):
                    group = plan['groups'][slot // 8]
                    key = (group['source'], group['id'])
                    episode_dir = out / 'episodes' / f'step_{state["step"]:04d}' / f'rollout_{slot:03d}'
                    reserve_attempt(state, episode_dir, max_attempts=args.max_attempts)
                    persist_journal(out, state, policy)
                    episode = run_episode(dataset, key, row, policy, episode_dir, objective=args.objective)
                    entry = {'episode_path': str(episode_dir / 'episode.json'),
                             'sha256': file_hash(episode_dir / 'episode.json'), 'key': list(key)}
                    commit_completed_attempt(out, state, entry, policy)
                    if not episode['valid']:
                        raise RuntimeError('real_episode_invalid_stop_for_repair:' + str(episode['error']))
                episodes = []
                for entry in state['episodes']:
                    if file_hash(entry['episode_path']) != entry['sha256']:
                        raise ValueError('persisted_episode_evidence_changed')
                    episodes.append(json.loads(Path(entry['episode_path']).read_text()))
                if len(episodes) != 128:
                    raise ValueError('complete_128_rollout_step_required')
                for slot, ep in enumerate(episodes):
                    expected = plan['groups'][slot // 8]
                    if (ep['key'] != [expected['source'], expected['id']]
                            or ep['configuration'] != config_name(row)
                            or ep['policy_version'] != policy.version
                            or ep['adapter_hash_at_collection'] != state['adapter_hash']):
                        raise ValueError('episode_task_configuration_or_actor_identity_mismatch')
                batch = prepare_update([episodes[i:i+8] for i in range(0, 128, 8)], policy.version)
                state['status'] = 'updating'
                persist_journal(out, state, policy)
                optimizer.zero_grad(set_to_none=True)
                before_hash, frozen_before = policy.adapter_hash(), policy.frozen_versions()
                if frozen_before != policy.frozen_signature:
                    raise ValueError('frozen_base_mutation_guard_failed')
                loss_value = 0.
                for group in batch['groups']:
                    for episode, advantage in zip(group['episodes'], group['advantages']):
                        for segment in episode['segments']:
                            loss_value += policy.backward_segment(segment, advantage,
                                pg_denominator=batch['pg_tokens'], kl_denominator=batch['kl_tokens'],
                                pg_enabled=group['pg_enabled'])
                gradients = [p.grad for p in policy.trainable if p.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise ValueError('missing_or_nonfinite_adapter_gradients')
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.trainable, CONFIG['grad_clip_norm'],
                                                          error_if_nonfinite=True)
                lr_used = optimizer.param_groups[0]['lr']
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                after_hash = policy.adapter_hash()
                if policy.frozen_versions() != frozen_before:
                    raise ValueError('frozen_base_mutated_during_update')
                changed = before_hash != after_hash
                if not changed and float(grad_norm) > 0:
                    raise ValueError('nonzero_gradient_without_persisted_adapter_change')
                policy.version += 1
                next_state = copy.deepcopy(state)
                next_state.update(step=state['step'] + 1, updates=state['updates'] + 1,
                    episodes=[], inflight=None, status='ready', policy_version=policy.version,
                    adapter_hash=after_hash, next_K_member=(state['step'] + 1) % 2, assignment=None,
                    effective_updates=state['effective_updates'] + int(changed))
                next_state['covered_task_conditions'] = sorted({tuple(k) for k in
                    state['covered_task_conditions']} | {tuple(e['key']) for e in state['episodes']})
                next_state['covered_families'] = sorted(set(state['covered_families']) |
                    {e['key'][0] + ':' + str(dataset.index[tuple(e['key'])]['family']) for e in state['episodes']})
                checkpoint = out / 'checkpoints' / f'update_{policy.version:04d}_{time.time_ns()}'
                next_state['checkpoint'] = str(checkpoint)
                metrics = {'step': state['step'], 'K_block': plan['K_block'],
                    'configuration': row, 'policy_before': policy.version - 1,
                    'policy_after': policy.version, 'attempts': state['attempts'],
                    'adapter_before': before_hash, 'adapter_after': after_hash,
                    'adapter_changed': changed, 'frozen_mutation_guard_unchanged': True,
                    'effective_updates': next_state['effective_updates'],
                    'groups': plan['groups'],
                    'loss': loss_value, 'grad_norm_before_clip': float(grad_norm), 'lr_used': lr_used,
                    'pg_tokens': batch['pg_tokens'], 'kl_tokens': batch['kl_tokens'],
                    'zero_variance_groups': sum(g['reward_std'] == 0 for g in batch['groups']),
                    'PG_masked_groups': sum(not g['pg_enabled'] for g in batch['groups']),
                    'group_rewards': [[e['training_reward'] for e in g['episodes']] for g in batch['groups']],
                    'source_action_tokens': {source: sum(e['generated_action_tokens'] for e in episodes
                        if e['key'][0] == source) for source in ('clawgym', 'agentdog')},
                    'seconds': time.monotonic() - batch_started, 'resources_before': before_resources,
                    'cuda_max_allocated': torch.cuda.max_memory_allocated(),
                    'cuda_max_reserved': torch.cuda.max_memory_reserved(), 'checkpoint': str(checkpoint)}
                next_state['last_update'] = metrics
                state = commit_checkpoint(out, policy, optimizer, scheduler, next_state, current_state=state)
                append_jsonl(out / 'training_metrics.jsonl', metrics)
            state['status'] = 'budget_complete'
            persist_journal(out, state, policy)
            checkpoint_retention.after_commit(out, state['checkpoint'], anchors=[state['checkpoint']],
                load_recovery=lambda path: torch.load(path, map_location='cpu', weights_only=False))
        except BaseException as exc:
            # Never save possibly partly updated weights as a recovery point.
            # The journal still references the last committed adapter/optimizer;
            # persisted exact episodes can rebuild an interrupted update.
            state['status'] = 'blocked'
            state['failure'] = {'type': type(exc).__name__, 'detail': str(exc),
                                'at': datetime.now(timezone.utc).isoformat()}
            persist_journal(out, state, policy)
            atomic_json(out / f'failure_{time.time_ns()}.json', state['failure'])
            raise
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--max-attempts', type=int, default=38400)
    parser.add_argument('--seed', type=int, default=267)
    parser.add_argument('--objective', choices=('safety', 'utility'), default='safety')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == '__main__':
    main()
