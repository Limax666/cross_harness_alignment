"""Physical-GPU1 real two-worker batched LoRA GRPO; independent v7 journal.

Frozen source runtime/reward/sampler and reviewed v5 optimizer/checkpoint helpers
are reused. This entry never imports fake environments or changes source quotas.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

import train_hsrl_v5 as original
from train_hsrl_v5 import (CONFIG, LOGPROB_CONTRACT, POLICY_VERSION, REWARD_VERSION,
    RUNTIME_VERSION, atomic_json, append_jsonl, checkpoint_retention, commit_checkpoint,
    config_name, digest, file_hash, persist_journal, prepare_update, resource_gate,
    validate_lease, verify_official_checkpoint)
import hsrl_batched_rollout_v1 as rollout

SCHEMA = 'hsrl-v7-real-two-worker-batched-grpo/v1'


def validate_batch_contract(args):
    if (args.physical_gpu != 1 or args.episode_workers != 2 or args.model_batch_size != 2):
        raise ValueError('physical_GPU1_and_exactly_two_workers_and_two_model_rows_required')
    if type(args.max_batch_tokens) is not int or args.max_batch_tokens <= 0:
        raise ValueError('positive_max_batch_tokens_required')


def _code_identity():
    root = Path(__file__).resolve().parent
    files = [Path(__file__).resolve(), root / 'hsrl_batched_rollout_v1.py',
             root / 'hsrl_batched_policy_v1.py']
    # Include every frozen runtime package module as well as the original trainer
    # helpers and selected initializer. Legacy four package/version pins remain.
    files += [root / name for name in original._code_identity()]
    files += sorted((root / 'hsrl_v3').glob('*.py'))
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(set(files))}


def gpu_identity():
    fields = ['index', 'uuid', 'name', 'pci.bus_id', 'driver_version']
    result = subprocess.run(['nvidia-smi', '--id=1', '--query-gpu=' + ','.join(fields),
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    values = [v.strip() for v in result.stdout.strip().split(',')]
    if len(values) != len(fields) or values[0] != '1':
        raise ValueError('physical_GPU1_runtime_identity_unavailable')
    return dict(zip(fields, values))


def aggregate_wave_metrics(waves):
    return {'committed_waves': len(waves),
        'model_batch_calls': sum(w['model_batch_calls'] for w in waves),
        'batch_size_2_calls': sum(w['batch_size_2_calls'] for w in waves),
        'generated_action_tokens': sum(w['generated_action_tokens'] for w in waves),
        'model_seconds': sum(w['model_seconds'] for w in waves),
        'tokenizer_seconds': sum(w['tokenizer_seconds'] for w in waves),
        'peak_active_workers': max((w['peak_active_workers'] for w in waves), default=0),
        'source_stages_seconds_summed_over_workers': {key: sum(
            slot.get(key, 0.) for wave in waves for slot in wave['source_stages_by_slot'])
            for key in ('source_setup_seconds', 'source_call_seconds',
                        'source_finish_seconds', 'source_cleanup_seconds')}}


def run(args):
    validate_batch_contract(args)
    physical_gpu = 1
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical_gpu):
        raise ValueError(f'CUDA_VISIBLE_DEVICES_must_be_exactly_physical_{physical_gpu}')
    if not (256 <= args.max_attempts <= 38400 and args.max_attempts % 256 == 0):
        raise ValueError('budget_must_be_complete_K2_blocks_up_to_38400_attempts')
    from hsrl_v5.dataset import Dataset
    from hsrl_batched_policy_v1 import BatchedPolicy
    from hsrl_selected_gpu_policy_v1 import verify_initializer_contract
    verify_initializer_contract()
    Policy = BatchedPolicy
    out, model, lease_path = args.out.resolve(), args.model.resolve(strict=True), args.lease.resolve(strict=True)
    dataset = Dataset(args.dataset)
    official = verify_official_checkpoint(model)
    lease = validate_lease(lease_path, physical_gpu)
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
        if validate_lease(lease_path, physical_gpu) != lease:
            raise ValueError('lease_changed_before_exclusive_load')
        resources = resource_gate(out, initial=True, physical_gpu=physical_gpu)
        identity = {'schema': SCHEMA, 'physical_gpu': physical_gpu, 'dataset_root': str(dataset.root),
            'READY_sha256': file_hash(dataset.root / 'READY.json'),
            'dataset_manifest_sha256': file_hash(dataset.root / 'manifest.json'),
            'reward_contract': REWARD_VERSION, 'runtime_version': RUNTIME_VERSION,
            'policy_version': POLICY_VERSION, 'selected_policy': {'class': 'hsrl_batched_policy_v1.BatchedPolicy',
                'source_sha256': file_hash(Path(__file__).with_name('hsrl_batched_policy_v1.py'))}, 'config': CONFIG, 'seed': args.seed,
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
            'sampler': {'class': type(dataset.sampler).__module__ + '.' + type(dataset.sampler).__name__,
                'source_sha256': file_hash(Path(__file__).with_name('hsrl_v5') / 'sampler.py'),
                'seed': 20261116, 'step_assignment': 'deterministic_original_G8_K2_50percent'},
            'batching': {'episode_workers': 2, 'model_batch_size': 2,
                'max_batch_tokens': args.max_batch_tokens, 'wave_protocol': rollout.VERSION,
                'token_cost': 'batch_size*(max_prompt_length+max_new_tokens)'},
            'physical_gpu_identity': gpu_identity(),
            'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'numpy')}}
        manifest_path = out / 'run_manifest.json'
        if args.resume:
            if json.loads(manifest_path.read_text())['identity'] != identity:
                raise ValueError('resume_data_model_code_config_or_lease_identity_changed')
        else:
            atomic_json(manifest_path, {'identity': identity, 'identity_sha256': digest(identity),
                'created_at': datetime.now(timezone.utc).isoformat(), 'initial_resources': resources})
        if args.resume:
            import torch as cpu_torch
            saved = cpu_torch.load(out / 'journal.pt', map_location='cpu', weights_only=False)
            rollout.require_resumable(saved['state'])
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        policy = Policy(model, args.seed)
        torch = policy.torch
        actual_policy = {'language_LoRA_target_modules': policy.targets,
            'trainable_parameter_count': sum(p.numel() for p in policy.trainable),
            'trainable_parameter_dtypes': sorted({str(p.dtype) for p in policy.trainable}),
            'actual_tokenizer_chat_template_sha256': digest(policy.tokenizer.chat_template),
            'actual_protocol_eos_ids': policy.eos_ids, 'logprob_contract': LOGPROB_CONTRACT,
            'fast_runtime': policy.fast_runtime_identity(),
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_device_capability': list(torch.cuda.get_device_capability(0))}
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
            rollout.require_resumable(state)
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
                before_resources = resource_gate(out, physical_gpu=1)
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
                collection_started = time.monotonic()
                wave_metrics = []
                policy_event_start = len(policy.batch_events)
                while len(state['episodes']) < 128:
                    wave_metrics.append(rollout.collect_wave(dataset, plan, policy, out, state,
                        objective=args.objective, max_attempts=args.max_attempts,
                        max_batch_tokens=args.max_batch_tokens))
                collection_seconds = time.monotonic() - collection_started
                preparation_started = time.monotonic()
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
                preparation_seconds = time.monotonic() - preparation_started
                state['status'] = 'updating'
                persist_journal(out, state, policy)
                backward_started = time.monotonic()
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
                torch.cuda.synchronize()
                backward_seconds = time.monotonic() - backward_started
                optimizer_started = time.monotonic()
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
                torch.cuda.synchronize()
                optimizer_seconds = time.monotonic() - optimizer_started
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
                    'seconds': time.monotonic() - batch_started,
                    'stage_seconds': {'collection': collection_seconds,
                        'preparation': preparation_seconds, 'backward': backward_seconds,
                        'optimizer': optimizer_seconds},
                    'rollout_broker': aggregate_wave_metrics(wave_metrics),
                    'policy_batch_events': policy.batch_events[policy_event_start:],
                    'collection_episodes_this_session': 2 * len(wave_metrics),
                    'collection_episodes_per_second': 2 * len(wave_metrics) / collection_seconds,
                    'collection_action_tokens_per_second': sum(w['generated_action_tokens'] for w in wave_metrics) / collection_seconds,
                    'resources_before': before_resources,
                    'cuda_max_allocated': torch.cuda.max_memory_allocated(),
                    'cuda_max_reserved': torch.cuda.max_memory_reserved(), 'checkpoint': str(checkpoint)}
                next_state['last_update'] = metrics
                checkpoint_started = time.monotonic()
                state = commit_checkpoint(out, policy, optimizer, scheduler, next_state, current_state=state)
                # This duration is post-commit observability; the installed recovery
                # state's last_update contains all pre-commit measurements.
                metrics = copy.deepcopy(metrics)
                metrics['stage_seconds']['checkpoint'] = time.monotonic() - checkpoint_started
                metrics['seconds_including_checkpoint'] = time.monotonic() - batch_started
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



def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, choices=(1,), default=1)
    parser.add_argument('--episode-workers', type=int, default=2)
    parser.add_argument('--model-batch-size', type=int, default=2)
    parser.add_argument('--max-batch-tokens', type=int, default=32768)
    parser.add_argument('--max-attempts', type=int, default=38400)
    parser.add_argument('--seed', type=int, default=267)
    parser.add_argument('--objective', choices=('safety', 'utility'), default='safety')
    parser.add_argument('--resume', action='store_true')
    return parser


def main():
    print(json.dumps(run(argument_parser().parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
