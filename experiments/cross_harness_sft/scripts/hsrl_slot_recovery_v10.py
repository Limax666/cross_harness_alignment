"""Bounded replacement of isolated rejected episodes within a fixed G8 group."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import hsrl_batched_rollout_v2 as previous
import train_hsrl_v5 as original
from hsrl_v5.io import atomic_json, append_jsonl, file_hash
from hsrl_v5.runtime import run_episode

VERSION = 'hsrl-eight-worker-isolated-slot-retry/v3'
WORKERS = 8
MAX_REJECTIONS_PER_SLOT = 3  # original draw plus at most two replacements
REJECTION = 'immutable_grading_reference_modified'


class AttemptBudgetExhausted(ValueError):
    pass


def retryable_rejection(record):
    """Only a completed source rejection can be replaced automatically."""
    result = record.get('source_result') or {}
    return (record.get('valid') is False
            and record.get('key', [None])[0] == 'clawgym'
            and record.get('error') == {'type': 'ValueError',
                                        'detail': 'source_measurement_invalid_or_incomplete'}
            and record.get('training_reward') is None
            and record.get('reward_components') is None
            and result.get('done') is True and result.get('valid') is False
            and result.get('utility') is None and result.get('c_source') is None
            and result.get('reason') == REJECTION
            and bool(result.get('changed_inputs')))


def cleanup_receipts(directory, source, *, rejected=False):
    backend = Path(directory) / 'private/backend'
    paths = list(backend.glob('*/cleanup.json'))
    roles = {path.parent.name for path in paths}
    graded = ('private_grader' in roles if source == 'clawgym'
              else any(role.startswith('grader_') for role in roles))
    if 'tool_worker' not in roles or (not rejected and not graded):
        raise ValueError('required_source_cleanup_receipt_missing')
    receipts = {}
    for path in paths:
        if path.resolve() != path:
            raise ValueError('source_cleanup_receipt_path_not_owned')
        receipt = json.loads(path.read_text())
        if (receipt.get('removal_confirmed') is not True or receipt.get('prior_removal_error') or receipt.get('error')
                or not isinstance(receipt.get('container'), str) or not receipt['container']):
            raise ValueError('source_cleanup_requires_reconciliation')
        receipts[path] = receipt['container']
    return receipts


def validate_rejection(directory, record):
    backend = Path(directory) / 'private/backend/result.json'
    if (not retryable_rejection(record) or backend.resolve() != backend
            or json.loads(backend.read_text()) != record['source_result']):
        raise ValueError('source_rejection_evidence_mismatch')
    return cleanup_receipts(directory, 'clawgym', rejected=True)


def require_resumable(state):
    previous.require_resumable(state)
    start = len(state['episodes'])
    slots = {str(i) for i in range(start, min(start + WORKERS, 128))}
    pending = state['pending_slots']
    failures = state['slot_failures']
    if (start > 128 or not isinstance(pending, dict) or not isinstance(failures, dict)
            or not pending.keys() <= slots or not failures.keys() <= slots
            or len(pending) >= WORKERS or any(type(n) is not int or n < 1 for n in failures.values())):
        raise ValueError('partial_wave_state_requires_reconciliation')


def plan_reservations(out, state, *, max_attempts):
    require_resumable(state)
    first = len(state['episodes'])
    missing = [i for i in range(first, first + WORKERS) if str(i) not in state['pending_slots']]
    if first >= 128:
        raise ValueError('no_remaining_collection_slots')
    if any(state['slot_failures'].get(str(i), 0) >= MAX_REJECTIONS_PER_SLOT for i in missing):
        raise ValueError('episode_slot_retry_limit_reached')
    if state['attempts'] + 128 - first - len(state['pending_slots']) > max_attempts:
        raise AttemptBudgetExhausted('attempt_budget_cannot_complete_current_update')
    reservations = []
    for offset, slot in enumerate(missing, start=1):
        attempt = state['attempts'] + offset
        directory = (Path(out).absolute() / 'episodes' / f'step_{state["step"]:04d}'
                     / f'rollout_{slot:03d}_attempt_{attempt:06d}')
        if directory.exists() or directory.is_symlink():
            raise ValueError('attempt_directory_already_exists')
        reservations.append({'slot': slot, 'attempt': attempt, 'directory': str(directory)})
    return reservations


def publish(out, state, next_state, policy):
    original.persist_journal(out, next_state, policy)
    state.clear()
    state.update(next_state)


def entry_for(path, key, state, configuration):
    path = Path(path)
    record = json.loads(path.read_text())
    if (path.resolve() != path or record['key'] != list(key)
            or record['configuration'] != configuration
            or record['policy_version'] != state['policy_version']
            or record['adapter_hash_at_collection'] != state['adapter_hash']):
        raise ValueError('episode_slot_identity_mismatch')
    return {'episode_path': str(path), 'sha256': file_hash(path), 'key': list(key)}, record


def collect_wave(dataset, plan, policy, out, state, *, objective, max_attempts, max_batch_tokens):
    """Collect one complete group, durably retaining successful peers across retries."""
    start = len(state['episodes'])
    group = plan['groups'][start // WORKERS]
    key = (group['source'], group['id'])
    configuration = original.config_name(plan['configuration'])
    rounds, accepted, event_start = [], 0, len(policy.batch_events)

    while len(state['episodes']) == start:
        reservations = plan_reservations(out, state, max_attempts=max_attempts)
        next_state = copy.deepcopy(state)
        next_state['attempts'] += len(reservations)
        next_state['inflight'] = {'schema': VERSION, 'step': state['step'], 'attempts': reservations}
        publish(out, state, next_state, policy)

        broker = previous.WaveBroker(policy, max_batch_tokens=max_batch_tokens)
        source_metrics = [{} for _ in range(WORKERS)]
        tasks = [lambda facade: None for _ in range(WORKERS)]

        def collect(facade, reservation, metrics):
            directory = Path(reservation['directory'])
            timed = previous.TimedDataset(dataset, metrics, check_cancelled=broker.check_cancelled)
            result = run_episode(timed, key, plan['configuration'], facade, directory, objective=objective)
            if retryable_rejection(result):
                validate_rejection(directory, result)
                # Return this closed, known rejection through the v2 broker as
                # a completed worker. Unknown invalid returns still abort it.
                return {'closed_rejected_episode': True}
            return result

        for reservation in reservations:
            offset = reservation['slot'] - start
            tasks[offset] = lambda facade, r=reservation, i=offset: collect(facade, r, source_metrics[i])
        outcome = []
        try:
            finished = broker.run(tasks)
            next_state = copy.deepcopy(state)
            for reservation in reservations:
                slot = reservation['slot']
                path = Path(reservation['directory']) / 'episode.json'
                entry, record = entry_for(path, key, state, configuration)
                rejected = finished[slot - start].get('closed_rejected_episode') is True
                if rejected:
                    validate_rejection(path.parent, record)
                    next_state['slot_failures'][str(slot)] = next_state['slot_failures'].get(str(slot), 0) + 1
                    next_state['rejected_attempts'].append(
                        {**entry, 'step': state['step'], 'slot': slot,
                         'attempt': reservation['attempt'], 'reason': REJECTION})
                elif record['valid'] is True and finished[slot - start] == record:
                    if str(slot) in next_state['pending_slots']:
                        raise ValueError('attempt_would_replace_accepted_peer')
                    next_state['pending_slots'][str(slot)] = entry
                    accepted += 1
                else:
                    raise ValueError('invalid_or_changed_attempt_cannot_be_committed')
                original.seal_episode_evidence(out, path.parent)
                outcome.append({'slot': slot, 'attempt': reservation['attempt'], 'accepted': not rejected})
            next_state['inflight'] = None
            if len(next_state['pending_slots']) == WORKERS:
                next_state['episodes'].extend(next_state['pending_slots'][str(i)]
                                              for i in range(start, start + WORKERS))
                next_state['pending_slots'] = {}
                next_state['slot_failures'] = {}
            publish(out, state, next_state, policy)
        finally:
            measured = {**broker.metrics, 'source_stages_by_slot': source_metrics,
                        'step': state['step'], 'reservations': reservations, 'outcomes': outcome,
                        'all_outcomes_in_journal': state['inflight'] is None}
            rounds.append(measured)
            first_attempt = reservations[0]['attempt']
            atomic_json(Path(out) / 'waves' / f'step_{state["step"]:04d}_attempt_{first_attempt:06d}.json', measured)
            append_jsonl(Path(out) / 'rollout_attempt_metrics.jsonl', measured)

    totals = {name: sum(r[name] for r in rounds) for name in (
        'model_batch_calls', 'batch_size_2_calls', 'batch_size_8_calls',
        'batch_size_gt1_calls', 'generated_action_tokens', 'model_seconds', 'tokenizer_seconds')}
    totals.update(step=state['step'], slots=list(range(start, start + WORKERS)), accepted_this_session=accepted,
                  attempt_count=sum(len(r['reservations']) for r in rounds),
                  source_stages_by_slot=[slot for r in rounds for slot in r['source_stages_by_slot']],
                  peak_active_workers=max(r['peak_active_workers'] for r in rounds),
                  batch_size_histogram={str(i): sum(r['batch_size_histogram'][str(i)] for r in rounds)
                                        for i in range(1, WORKERS + 1)},
                  policy_batch_events=policy.batch_events[event_start:])
    append_jsonl(Path(out) / 'rollout_wave_metrics.jsonl', totals)
    return totals
