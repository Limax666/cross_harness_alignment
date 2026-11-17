"""Eight-worker G8 waves; fair main-thread batches of at most eight model rows.

Reuse the immutable v1 public facades, timing/cancellation and model dispatch
boundary. New schema/driver reject old two-slot journals and episode prefixes.
"""
from __future__ import annotations

from concurrent.futures import Future
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

import train_hsrl_v5 as original
import hsrl_batched_rollout_v1 as previous
from hsrl_v5.io import append_jsonl, atomic_json, file_hash
from hsrl_v5.runtime import run_episode

VERSION = 'hsrl-eight-worker-main-thread-broker/v2'
WORKERS = 8
MODEL_BATCH_SIZE = 8
TimedDataset = previous.TimedDataset


def require_resumable(state):
    if 'inflight' not in state or state['inflight'] is not None:
        raise ValueError('inflight_wave_requires_reconciliation_no_reexecution')
    if len(state.get('episodes', [])) % WORKERS:
        raise ValueError('partial_wave_requires_reconciliation_no_reexecution')


def reserve_wave(out, state, directories, policy, *, max_attempts,
                 persist=original.persist_journal):
    require_resumable(state)
    directories = [Path(p).absolute() for p in directories]
    slot = len(state['episodes'])
    expected = [Path(out).absolute() / 'episodes' / f'step_{state["step"]:04d}' /
                f'rollout_{index:03d}' for index in range(slot, slot + WORKERS)]
    if directories != expected or slot >= 128 or any(p.exists() for p in directories):
        raise ValueError('wave_requires_eight_fresh_deterministic_slots')
    if state['attempts'] + WORKERS > max_attempts:
        raise ValueError('attempt_budget_exhausted')
    # Mutate to reserved before persistence: a failed write cannot leave callers
    # with an apparently reusable in-memory wave. No source has run yet.
    start = state['attempts']
    state['attempts'] += WORKERS
    state['inflight'] = {'schema': VERSION, 'step': state['step'],
        'attempts': [{'directory': str(path), 'attempt': start + offset + 1,
                      'slot': slot + offset} for offset, path in enumerate(directories)]}
    persist(Path(out), state, policy)


def commit_wave(out, state, entries, policy, *, persist=original.persist_journal,
                seal=original.seal_episode_evidence):
    inflight = state.get('inflight') or {}
    reserved = inflight.get('attempts', [])
    if len(entries) != WORKERS or len(reserved) != WORKERS or inflight.get('schema') != VERSION:
        raise ValueError('complete_reserved_eight_slot_wave_required')
    for entry, attempt in zip(entries, reserved):
        episode = Path(entry['episode_path'])
        if (episode != Path(attempt['directory']) / 'episode.json'
                or file_hash(episode) != entry['sha256']):
            raise ValueError('episode_does_not_match_reserved_wave')
        if json.loads(episode.read_text()).get('valid') is not True:
            raise ValueError('invalid_episode_blocks_entire_wave')
    for attempt in reserved:
        seal(out, Path(attempt['directory']))
    next_state = copy.deepcopy(state)
    next_state['episodes'].extend(entries)
    next_state['inflight'] = None
    # Publish joint state only after all episode barriers. If persistence raises,
    # conservatively retain the original in-memory reservation too.
    persist(Path(out), next_state, policy)
    state.clear()
    state.update(next_state)


@dataclass
class _Request(previous._Request):
    generation_ordinal: int | None = None


class WaveBroker(previous.WaveBroker):
    def __init__(self, policy, *, max_batch_tokens=32768, wave_timeout=3600., cleanup_timeout=180.):
        super().__init__(policy, max_batch_tokens=max_batch_tokens,
                         wave_timeout=wave_timeout, cleanup_timeout=cleanup_timeout)
        self._next_generation_ordinal = [0] * WORKERS
        self.metrics.update(worker_capacity=WORKERS, model_batch_capacity=MODEL_BATCH_SIZE,
            batch_size_8_calls=0, batch_size_gt1_calls=0,
            batch_size_histogram={str(i): 0 for i in range(1, MODEL_BATCH_SIZE + 1)})

    def request(self, slot, operation, args, kwargs):
        future = Future()
        with self._condition:
            if self._abort is not None:
                raise RuntimeError('wave_aborted_no_more_policy_calls') from self._abort
            if type(slot) is not int or not 0 <= slot < WORKERS:
                raise ValueError('invalid_wave_slot')
            if slot in self._pending or slot in self._finished:
                raise RuntimeError('worker_has_more_than_one_outstanding_request')
            ordinal = None
            if operation == 'generate':
                ordinal = self._next_generation_ordinal[slot]
                self._next_generation_ordinal[slot] += 1
            self._pending[slot] = _Request(slot, operation, args, kwargs, future, ordinal)
            self._condition.notify_all()
        return future.result()

    def _select_batch(self, pending):
        # Oldest generation round wins even when low slots finish a previous
        # split and submit another turn before higher slots have been served.
        pending = sorted(pending, key=lambda request: (request.generation_ordinal, request.slot))
        first = pending[0]
        budget, width = first.args[1], len(first.args[0])
        selected = [first]
        for request in pending[1:]:
            next_width = max(width, len(request.args[0]))
            if (request.args[1] == budget and len(selected) < MODEL_BATCH_SIZE
                    and (len(selected) + 1) * (next_width + budget) <= self.max_batch_tokens):
                selected.append(request)
                width = next_width
        return selected

    def _generate(self, requests):
        if not 1 <= len(requests) <= MODEL_BATCH_SIZE:
            raise ValueError('one_to_eight_model_rows_required')
        start = len(self.metrics['batches'])
        try:
            return super()._generate(requests)
        finally:
            # Base dispatch records started/failed invocations too. Preserve its
            # exact boundary and add fairness/scale evidence for each real call.
            for event in self.metrics['batches'][start:]:
                count = event['batch_size']
                event['generation_ordinals'] = [r.generation_ordinal for r in requests]
                self.metrics['batch_size_histogram'][str(count)] += 1
                self.metrics['batch_size_8_calls'] += int(count == 8)
                self.metrics['batch_size_gt1_calls'] += int(count > 1)

    def run(self, tasks):
        if len(tasks) != WORKERS or self._threads:
            raise ValueError('exactly_eight_tasks_one_wave_per_broker_required')
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('broker_requires_main_thread')
        started = time.monotonic()
        error = None
        try:
            for slot, task in enumerate(tasks):
                thread = threading.Thread(target=self._worker, args=(slot, task),
                    name=f'hsrl-environment-{slot}', daemon=True)
                self._threads.append(thread)
                thread.start()
            while True:
                if time.monotonic() - started > self.wave_timeout:
                    raise TimeoutError('real_environment_wave_deadline_exceeded')
                with self._condition:
                    failures = [exc for _, exc in self._finished.values() if exc is not None]
                    if failures:
                        raise failures[0]
                    if len(self._finished) == WORKERS:
                        break
                    token_requests = [r for _, r in sorted(self._pending.items())
                                      if r.operation != 'generate']
                    generation_ready = (len(self._pending) + len(self._finished) == WORKERS
                        and all(r.operation == 'generate' for r in self._pending.values()))
                    pending = list(self._pending.values()) if generation_ready else []
                    if not token_requests and not pending:
                        self._condition.wait(timeout=.25)
                        continue
                for request in token_requests:
                    tick = time.monotonic()
                    value = getattr(self.policy.tokenizer, request.operation)(*request.args, **request.kwargs)
                    self.metrics['tokenizer_seconds'] += time.monotonic() - tick
                    with self._condition:
                        self._pending.pop(request.slot)
                        request.future.set_result(value)
                        self._condition.notify_all()
                if token_requests:
                    # Rendering is not a generation row. Recompute all active
                    # readiness and peer failure/completion after waking workers.
                    continue
                if pending:
                    self._generate(self._select_batch(pending))
        except BaseException as exc:
            error = exc
            self._fail(exc)
        finally:
            deadline = time.monotonic() + self.cleanup_timeout
            for thread in self._threads:
                if thread.ident is not None:
                    thread.join(timeout=max(0., deadline - time.monotonic()))
            self.metrics['wave_seconds'] = time.monotonic() - started
            self.metrics['unjoined_workers'] = self.live_threads()
        if self.live_threads():
            raise RuntimeError('worker_cleanup_deadline_exceeded:' + ','.join(self.live_threads())) from error
        if error is not None:
            raise error
        return [self._finished[slot][0] for slot in range(WORKERS)]


def collect_wave(dataset, plan, policy, out, state, *, objective, max_attempts, max_batch_tokens):
    """Production entry: durable G8-attempt wave -> real collectors -> joint commit."""
    start = len(state['episodes'])
    directories = [Path(out) / 'episodes' / f'step_{state["step"]:04d}' /
                   f'rollout_{slot:03d}' for slot in range(start, start + WORKERS)]
    reserve_wave(out, state, directories, policy, max_attempts=max_attempts)
    broker = WaveBroker(policy, max_batch_tokens=max_batch_tokens)
    tasks = []
    keys = []
    source_metrics = [{} for _ in range(WORKERS)]
    event_start = len(getattr(policy, 'batch_events', []))
    for offset, directory in enumerate(directories):
        group = plan['groups'][(start + offset) // 8]
        key = (group['source'], group['id'])
        keys.append(key)
        timed = TimedDataset(dataset, source_metrics[offset], check_cancelled=broker.check_cancelled)
        tasks.append(lambda facade, key=key, directory=directory, timed=timed:
            run_episode(timed, key, plan['configuration'], facade, directory, objective=objective))
    try:
        broker.run(tasks)
        entries = [{'episode_path': str(directory / 'episode.json'),
                    'sha256': file_hash(directory / 'episode.json'), 'key': list(key)}
                   for directory, key in zip(directories, keys)]
        commit_wave(out, state, entries, policy)
    finally:
        metrics = {'step': state['step'], 'slots': list(range(start, start + WORKERS)), **broker.metrics,
            'source_stages_by_slot': source_metrics,
            'policy_batch_events': getattr(policy, 'batch_events', [])[event_start:]}
        # Evidence also survives invalid episodes, model failures and failed fsync.
        atomic_json(Path(out) / 'waves' / f'step_{state["step"]:04d}_wave_{start:03d}.json', metrics)
        append_jsonl(Path(out) / 'rollout_wave_metrics.jsonl', metrics)
    return metrics
