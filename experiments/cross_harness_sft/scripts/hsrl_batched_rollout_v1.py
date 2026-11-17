"""Deterministic two-slot real-environment waves with one main-thread GPU broker.

No source work starts until reserve_wave has installed the durable reservation.
Workers own their source/context state; tokenizer and policy calls cross only this
broker. A failed wave is evidence for reconciliation, never a sampling retry.
"""
from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

import train_hsrl_v5 as original
from hsrl_v5.io import append_jsonl, atomic_json, file_hash
from hsrl_v5.runtime import run_episode

VERSION = 'hsrl-two-worker-main-thread-broker/v1'


def require_resumable(state):
    if 'inflight' not in state or state['inflight'] is not None:
        raise ValueError('inflight_wave_requires_reconciliation_no_reexecution')
    if len(state.get('episodes', [])) % 2:
        raise ValueError('partial_wave_requires_reconciliation_no_reexecution')


def reserve_wave(out, state, directories, policy, *, max_attempts,
                 persist=original.persist_journal):
    require_resumable(state)
    directories = [Path(p).absolute() for p in directories]
    slot = len(state['episodes'])
    expected = [Path(out).absolute() / 'episodes' / f'step_{state["step"]:04d}' /
                f'rollout_{index:03d}' for index in range(slot, slot + 2)]
    if directories != expected or slot >= 128 or any(p.exists() for p in directories):
        raise ValueError('wave_requires_two_fresh_deterministic_slots')
    if state['attempts'] + 2 > max_attempts:
        raise ValueError('attempt_budget_exhausted')
    # Mutate to reserved before persistence: a failed write cannot leave callers
    # with an apparently reusable in-memory wave. No source has run yet.
    start = state['attempts']
    state['attempts'] += 2
    state['inflight'] = {'schema': VERSION, 'step': state['step'],
        'attempts': [{'directory': str(path), 'attempt': start + offset + 1,
                      'slot': slot + offset} for offset, path in enumerate(directories)]}
    persist(Path(out), state, policy)


def commit_wave(out, state, entries, policy, *, persist=original.persist_journal,
                seal=original.seal_episode_evidence):
    inflight = state.get('inflight') or {}
    reserved = inflight.get('attempts', [])
    if len(entries) != 2 or len(reserved) != 2 or inflight.get('schema') != VERSION:
        raise ValueError('complete_reserved_two_slot_wave_required')
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


class _TimedBackend:
    def __init__(self, backend, metrics, check_cancelled):
        self._backend, self._metrics = backend, metrics
        self._check_cancelled = check_cancelled

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def _invoke(self, stage, operation, *args):
        self._check_cancelled()
        started = time.monotonic()
        try:
            return operation(*args)
        finally:
            key = 'source_' + stage + '_seconds'
            self._metrics[key] = self._metrics.get(key, 0.) + time.monotonic() - started

    def call(self, name, arguments):
        return self._invoke('call', self._backend.call, name, arguments)

    def finish(self, final_text):
        return self._invoke('finish', self._backend.finish, final_text)


class TimedDataset:
    """Observe the frozen Dataset/source API without changing its calls or results."""
    def __init__(self, dataset, metrics, *, check_cancelled=lambda: None):
        self._dataset, self._metrics = dataset, metrics
        self._check_cancelled = check_cancelled

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    @contextmanager
    def episode(self, key, evidence_dir):
        self._check_cancelled()
        context = self._dataset.episode(key, evidence_dir)
        started = time.monotonic()
        try:
            backend = context.__enter__()
        finally:
            self._metrics['source_setup_seconds'] = time.monotonic() - started
        # Preserve the context manager's exception/suppression semantics exactly.
        try:
            yield _TimedBackend(backend, self._metrics, self._check_cancelled)
        except BaseException:
            import sys
            started = time.monotonic()
            try:
                if not context.__exit__(*sys.exc_info()):
                    raise
            finally:
                self._metrics['source_cleanup_seconds'] = time.monotonic() - started
        else:
            started = time.monotonic()
            try:
                context.__exit__(None, None, None)
            finally:
                self._metrics['source_cleanup_seconds'] = time.monotonic() - started


@dataclass
class _Request:
    slot: int
    operation: str
    args: tuple
    kwargs: dict
    future: Future


class _TokenizerFacade:
    def __init__(self, broker, slot):
        self._broker, self._slot = broker, slot

    def apply_chat_template(self, *args, **kwargs):
        return self._broker.request(self._slot, 'apply_chat_template', args, kwargs)

    def encode(self, *args, **kwargs):
        return self._broker.request(self._slot, 'encode', args, kwargs)

    def decode(self, *args, **kwargs):
        return self._broker.request(self._slot, 'decode', args, kwargs)


class QueuedPolicy:
    """Per-episode public collector interface; no model, RNG or private backend."""
    def __init__(self, broker, slot):
        self._broker, self._slot = broker, slot
        self.version = broker.policy.version
        self.collection_adapter_hash = broker.policy.collection_adapter_hash
        self.tokenizer = _TokenizerFacade(broker, slot)

    def generate(self, prompt_ids, max_new_tokens):
        return self._broker.request(self._slot, 'generate',
                                    (list(prompt_ids), max_new_tokens), {})


class WaveBroker:
    def __init__(self, policy, *, max_batch_tokens=32768, wave_timeout=3600., cleanup_timeout=180.):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('broker_requires_main_thread')
        if type(max_batch_tokens) is not int or max_batch_tokens <= 0:
            raise ValueError('positive_max_batch_tokens_required')
        if (not math.isfinite(wave_timeout) or not math.isfinite(cleanup_timeout)
                or wave_timeout <= 0 or cleanup_timeout <= 0):
            raise ValueError('finite_positive_worker_deadlines_required')
        self.policy, self.max_batch_tokens = policy, max_batch_tokens
        self.wave_timeout, self.cleanup_timeout = wave_timeout, cleanup_timeout
        self._condition = threading.Condition()
        self._pending, self._finished = {}, {}
        self._abort = None
        self._active_workers = 0
        self._threads = []
        self.metrics = {'model_batch_calls': 0, 'batch_size_2_calls': 0,
            'generated_action_tokens': 0, 'model_seconds': 0., 'tokenizer_seconds': 0.,
            'peak_active_workers': 0, 'batches': []}

    def check_cancelled(self):
        with self._condition:
            if self._abort is not None:
                raise RuntimeError('wave_aborted_no_more_source_calls') from self._abort

    def live_threads(self):
        return [t.name for t in self._threads if t.is_alive()]

    def request(self, slot, operation, args, kwargs):
        future = Future()
        with self._condition:
            if self._abort is not None:
                raise RuntimeError('wave_aborted_no_more_policy_calls') from self._abort
            if slot in self._pending or slot in self._finished:
                raise RuntimeError('worker_has_more_than_one_outstanding_request')
            self._pending[slot] = _Request(slot, operation, args, kwargs, future)
            self._condition.notify_all()
        return future.result()

    def _worker(self, slot, task):
        with self._condition:
            self._active_workers += 1
            self.metrics['peak_active_workers'] = max(
                self.metrics['peak_active_workers'], self._active_workers)
        try:
            result = task(QueuedPolicy(self, slot))
            if isinstance(result, dict) and result.get('valid') is False:
                raise RuntimeError('real_episode_invalid_stop_for_repair:' + str(result.get('error')))
            finished = (result, None)
        except BaseException as exc:
            finished = (None, exc)
        with self._condition:
            self._active_workers -= 1
            self._finished[slot] = finished
            self._condition.notify_all()

    def _fail(self, exc):
        with self._condition:
            self._abort = exc
            requests = list(self._pending.values())
            self._pending.clear()
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(RuntimeError('wave_aborted:' + str(exc)))
            self._condition.notify_all()

    def _generate(self, requests):
        values = []
        for request in requests:
            ids, budget = request.args
            if (not ids or any(type(i) is not int or i < 0 for i in ids)
                    or type(budget) is not int or budget <= 0):
                raise ValueError('invalid_prompt_or_generation_budget')
            if len(ids) + budget > self.max_batch_tokens:
                raise ValueError('request_exceeds_max_batch_tokens_no_truncation')
            values.append({'prompt_ids': ids, 'max_new_tokens': budget})
        with self._condition:
            failures = [exc for _, exc in self._finished.values() if exc is not None]
            if failures:
                raise failures[0]
        event = {'slots': [r.slot for r in requests], 'batch_size': len(requests),
            'prompt_lengths': [len(v['prompt_ids']) for v in values],
            'max_new_tokens': values[0]['max_new_tokens'],
            'padded_token_budget': len(values) * (max(len(v['prompt_ids']) for v in values)
                                               + values[0]['max_new_tokens']),
            'status': 'started'}
        # These count actual generate_batch invocations. Actual GPU forward
        # counts and raw generation/reference timings come from policy events.
        self.metrics['model_batch_calls'] += 1
        self.metrics['batch_size_2_calls'] += int(len(values) == 2)
        self.metrics['batches'].append(event)
        started = time.monotonic()
        try:
            outputs = self.policy.generate_batch(values)
            if not isinstance(outputs, list) or len(outputs) != len(requests):
                raise ValueError('batch_generation_row_count_mismatch')
            tokens = sum(len(row['response_ids']) for row in outputs)
            event.update(status='complete', generated_tokens=tokens,
                         response_lengths=[len(row['response_ids']) for row in outputs])
            self.metrics['generated_action_tokens'] += tokens
        except BaseException as exc:
            event.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            event['elapsed_seconds'] = time.monotonic() - started
            self.metrics['model_seconds'] += event['elapsed_seconds']
        # Finish all GPU work for this batch before waking either environment.
        for request, result in zip(requests, outputs):
            with self._condition:
                self._pending.pop(request.slot)
                request.future.set_result(result)
                self._condition.notify_all()

    def run(self, tasks):
        if len(tasks) != 2 or self._threads:
            raise ValueError('exactly_two_tasks_one_wave_per_broker_required')
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('broker_requires_main_thread')
        started = time.monotonic()
        error = None
        try:
            # Daemon is only a final process-exit safeguard for a broken external
            # source boundary. Cleanup is bounded and any surviving worker fails
            # the wave explicitly; it can never lead to an optimizer update.
            for slot, task in enumerate(tasks):
                thread = threading.Thread(target=self._worker, args=(slot, task),
                    name=f'hsrl-environment-{slot}', daemon=True)
                self._threads.append(thread)
                thread.start()
            while True:
                if time.monotonic() - started > self.wave_timeout:
                    raise TimeoutError('real_environment_wave_deadline_exceeded')
                with self._condition:
                    failures = [error for _, error in self._finished.values() if error is not None]
                    if failures:
                        raise failures[0]
                    if len(self._finished) == 2:
                        break
                    token_requests = [r for _, r in sorted(self._pending.items())
                                      if r.operation != 'generate']
                    generation_ready = (len(self._pending) + len(self._finished) == 2
                        and all(r.operation == 'generate' for r in self._pending.values()))
                    generate = [r for _, r in sorted(self._pending.items())
                                if r.operation == 'generate'] if generation_ready else []
                    if not token_requests and not generate:
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
                    # Released renderers may now request more tokenization or
                    # generation. Recheck failures/completion and readiness from
                    # current state; a tokenizer request never fills a GPU row.
                    continue
                if generate:
                    first, *rest = generate
                    if rest:
                        second = rest[0]
                        same_budget = first.args[1] == second.args[1]
                        cost = 2 * (max(len(first.args[0]), len(second.args[0])) + first.args[1])
                        if same_budget and cost <= self.max_batch_tokens:
                            self._generate([first, second])
                            continue
                    # Snapshot order is stable even as a released worker advances.
                    for request in generate:
                        self._generate([request])
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
        return [self._finished[slot][0] for slot in range(2)]


def collect_wave(dataset, plan, policy, out, state, *, objective, max_attempts, max_batch_tokens):
    """Production entry: durable two-attempt wave -> real collectors -> joint commit."""
    start = len(state['episodes'])
    directories = [Path(out) / 'episodes' / f'step_{state["step"]:04d}' /
                   f'rollout_{slot:03d}' for slot in range(start, start + 2)]
    reserve_wave(out, state, directories, policy, max_attempts=max_attempts)
    broker = WaveBroker(policy, max_batch_tokens=max_batch_tokens)
    tasks = []
    keys = []
    source_metrics = [{}, {}]
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
        metrics = {'step': state['step'], 'slots': [start, start + 1], **broker.metrics,
            'source_stages_by_slot': source_metrics,
            'policy_batch_events': getattr(policy, 'batch_events', [])[event_start:]}
        # Evidence also survives invalid episodes, model failures and failed fsync.
        atomic_json(Path(out) / 'waves' / f'step_{state["step"]:04d}_wave_{start:03d}.json', metrics)
        append_jsonl(Path(out) / 'rollout_wave_metrics.jsonl', metrics)
    return metrics
