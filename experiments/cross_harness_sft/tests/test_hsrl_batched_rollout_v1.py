"""CPU broker and filesystem transaction checks; no task/source/model runs."""
import copy
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def rollout_module():
    try:
        import hsrl_batched_rollout_v1 as rollout
    except ImportError:
        pytest.fail('two-worker rollout implementation is missing')
    return rollout


class BoundaryPolicy:
    version = 4
    collection_adapter_hash = 'adapter'
    tokenizer = None

    def __init__(self):
        self.calls = []
        self.owner = threading.get_ident()
        self.tokenizer = self

    def encode(self, text, **kwargs):
        assert threading.get_ident() == self.owner
        return [ord(c) for c in text]

    def generate_batch(self, requests):
        assert threading.get_ident() == self.owner
        self.calls.append(copy.deepcopy(requests))
        return [{'response_ids': [req['prompt_ids'][0]], 'policy_version': self.version}
                for req in requests]


def test_two_independent_worker_requests_reach_one_actual_batch_in_slot_order():
    rollout = rollout_module()
    policy = BoundaryPolicy()
    workers = set()
    def episode(slot, facade):
        workers.add(threading.get_ident())
        ids = facade.tokenizer.encode(chr(65 + slot))
        return facade.generate(ids, 7)
    broker = rollout.WaveBroker(policy, max_batch_tokens=128)
    result = broker.run([lambda p: episode(0, p), lambda p: episode(1, p)])
    assert [r['response_ids'] for r in result] == [[65], [66]]
    assert len(workers) == 2 and policy.owner not in workers
    assert policy.calls == [[{'prompt_ids': [65], 'max_new_tokens': 7},
                             {'prompt_ids': [66], 'max_new_tokens': 7}]]
    assert broker.metrics['batch_size_2_calls'] == 1
    assert broker.metrics['peak_active_workers'] == 2
    assert not broker.live_threads()


@pytest.mark.parametrize('budgets,limit', [([7, 8], 128), ([7, 7], 15)])
def test_different_budgets_or_padding_cost_dispatch_all_requests_without_dropping(budgets, limit):
    rollout = rollout_module()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy, max_batch_tokens=limit)
    result = broker.run([lambda p: p.generate([65], budgets[0]),
                         lambda p: p.generate([66], budgets[1])])
    assert len(result) == 2 and [len(c) for c in policy.calls] == [1, 1]
    assert [c[0]['max_new_tokens'] for c in policy.calls] == budgets


def test_worker_failure_releases_peer_future_and_joins_workers():
    rollout = rollout_module()
    policy = BoundaryPolicy()
    ready = threading.Event()
    def broken(p):
        ready.wait(1)
        raise RuntimeError('source boundary failure')
    def waiting(p):
        ready.set()
        return p.generate([65], 7)
    broker = rollout.WaveBroker(policy, max_batch_tokens=128)
    with pytest.raises(RuntimeError, match='source boundary failure'):
        broker.run([broken, waiting])
    assert not policy.calls
    assert not broker.live_threads()


def test_model_failure_releases_both_waiters_and_never_retries():
    rollout = rollout_module()
    policy = BoundaryPolicy()
    def fail(requests):
        raise RuntimeError('allocation failure')
    policy.generate_batch = fail
    broker = rollout.WaveBroker(policy, max_batch_tokens=128)
    with pytest.raises(RuntimeError, match='allocation failure'):
        broker.run([lambda p: p.generate([65], 7), lambda p: p.generate([66], 7)])
    assert not broker.live_threads()
    assert broker.metrics['model_batch_calls'] == 1
    assert broker.metrics['batches'][0]['status'] == 'failed'


def wave_fixture(tmp_path):
    rollout = rollout_module()
    out = tmp_path / 'run'
    directories = [out / 'episodes/step_0000' / f'rollout_{slot:03}' for slot in range(2)]
    state = {'attempts': 0, 'inflight': None, 'episodes': [], 'step': 0}
    writes = []
    persist = lambda out, state, policy: writes.append(copy.deepcopy(state))
    rollout.reserve_wave(out, state, directories, None, max_attempts=256, persist=persist)
    entries = []
    from hsrl_v5.io import file_hash
    for slot, directory in enumerate(directories):
        directory.mkdir(parents=True)
        (directory / 'episode.json').write_text('{"valid": true}\n')
        entries.append({'episode_path': str(directory / 'episode.json'),
                        'sha256': file_hash(directory / 'episode.json'), 'key': ['source', str(slot)]})
    return rollout, out, directories, state, writes, persist, entries


def test_wave_is_reserved_before_sources_and_jointly_committed_after_both_barriers(tmp_path):
    rollout, out, directories, state, writes, persist, entries = wave_fixture(tmp_path)
    assert writes[0]['attempts'] == 2 and len(writes[0]['inflight']['attempts']) == 2
    sealed = []
    def seal(out, directory):
        assert not state['episodes'] and len(writes) == 1
        sealed.append(directory)
    rollout.commit_wave(out, state, entries, None, persist=persist, seal=seal)
    assert sealed == directories
    assert state['episodes'] == entries and state['inflight'] is None
    assert len(writes) == 2
    rollout.require_resumable(state)


def test_second_barrier_failure_preserves_all_inflight_and_rejects_resume(tmp_path):
    rollout, out, directories, state, writes, persist, entries = wave_fixture(tmp_path)
    before = copy.deepcopy(state)
    def seal(out, directory):
        if directory == directories[1]:
            raise OSError('fsync failed')
    with pytest.raises(OSError, match='fsync'):
        rollout.commit_wave(out, state, entries, None, persist=persist, seal=seal)
    assert state == before and len(writes) == 1
    with pytest.raises(ValueError, match='reconciliation'):
        rollout.require_resumable(state)
    with pytest.raises(ValueError, match='reconciliation'):
        rollout.reserve_wave(out, state, directories, None, max_attempts=256, persist=persist)


def test_invalid_episode_and_journal_failure_cannot_commit_partial_wave(tmp_path):
    rollout, out, directories, state, writes, persist, entries = wave_fixture(tmp_path)
    (directories[1] / 'episode.json').write_text('{"valid": false}\n')
    from hsrl_v5.io import file_hash
    entries[1]['sha256'] = file_hash(entries[1]['episode_path'])
    with pytest.raises(ValueError, match='invalid'):
        rollout.commit_wave(out, state, entries, None, persist=persist)
    assert not state['episodes'] and state['inflight'] and len(writes) == 1


def test_driver_exposes_fixed_gpu1_batch_contract_and_reuses_reviewed_helpers():
    try:
        import train_hsrl_v7 as driver
    except ImportError:
        pytest.fail('batched trainer implementation is missing')
    import train_hsrl_v5 as original
    args = driver.argument_parser().parse_args(['--dataset', '/d', '--model', '/m', '--out', '/o', '--lease', '/l'])
    assert (args.physical_gpu, args.episode_workers, args.model_batch_size, args.max_batch_tokens) == (1, 2, 2, 32768)
    assert driver.prepare_update is original.prepare_update
    assert driver.commit_checkpoint is original.commit_checkpoint
    assert driver.CONFIG == original.CONFIG
    driver.validate_batch_contract(args)
    args.episode_workers = 1
    with pytest.raises(ValueError, match='two'):
        driver.validate_batch_contract(args)


def test_successful_peer_completion_does_not_leave_remaining_single_waiting():
    rollout = rollout_module()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy)
    def later(p):
        first = p.generate([65], 7)
        second = p.generate([66], 7)
        return [first, second]
    result = broker.run([later, lambda p: p.generate([67], 7)])
    assert len(result[0]) == 2 and result[1]['response_ids'] == [67]
    assert [len(c) for c in policy.calls] == [2, 1]
    assert not broker.live_threads()


def test_invalid_episode_record_aborts_peer_without_admitting_a_wave():
    rollout = rollout_module()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy)
    with pytest.raises(RuntimeError, match='invalid'):
        broker.run([lambda p: {'valid': False, 'error': 'source scoring failed'},
                    lambda p: p.generate([66], 7)])
    assert not broker.live_threads() and not policy.calls


def test_joint_journal_write_failure_does_not_clear_in_memory_reservation(tmp_path):
    rollout, out, directories, state, writes, persist, entries = wave_fixture(tmp_path)
    before = copy.deepcopy(state)
    def fail(*args):
        raise OSError('journal write failed')
    with pytest.raises(OSError, match='journal'):
        rollout.commit_wave(out, state, entries, None, persist=fail)
    assert state == before
    assert all((p / 'episode.json').exists() for p in directories)


def test_budget_and_existing_directory_rejection_precede_any_reservation_write(tmp_path):
    rollout, out, directories, state, writes, persist, entries = wave_fixture(tmp_path)
    state.update(inflight=None, attempts=256, episodes=[])
    for path in directories:
        (path / 'episode.json').unlink()
        path.rmdir()
    with pytest.raises(ValueError, match='budget'):
        rollout.reserve_wave(out, state, directories, None, max_attempts=256, persist=persist)
    directories[0].mkdir()
    state['attempts'] = 0
    with pytest.raises(ValueError, match='fresh'):
        rollout.reserve_wave(out, state, directories, None, max_attempts=256, persist=persist)
    assert len(writes) == 1 and state['attempts'] == 0


def test_worker_deadline_is_finite_cancels_waiters_and_reports_unjoined_worker():
    rollout = rollout_module()
    policy = BoundaryPolicy()
    release = threading.Event()
    broker = rollout.WaveBroker(policy, wave_timeout=.03, cleanup_timeout=.03)
    try:
        with pytest.raises(RuntimeError, match='cleanup_deadline'):
            broker.run([lambda p: release.wait(1), lambda p: p.generate([65], 7)])
        assert not policy.calls
    finally:
        release.set()
        for thread in broker._threads:
            thread.join(1)
    assert not broker.live_threads()


def test_source_stage_timer_measures_each_boundary_and_preserves_backend_cleanup():
    rollout = rollout_module()
    from contextlib import contextmanager
    events = []
    class Backend:
        messages = ['public']
        tools = []
        def call(self, name, arguments):
            events.append(('call', name))
            return 'public observation'
        def finish(self, final):
            events.append(('finish', final))
            return {'valid': True}
    class Dataset:
        @contextmanager
        def episode(self, key, evidence):
            events.append('enter')
            try:
                yield Backend()
            finally:
                events.append('close')
    metrics = {}
    dataset = rollout.TimedDataset(Dataset(), metrics)
    with dataset.episode(('source', '1'), Path('/unused')) as backend:
        assert backend.messages == ['public']
        assert backend.call('tool', {}) == 'public observation'
        assert backend.finish('answer') == {'valid': True}
    assert events == ['enter', ('call', 'tool'), ('finish', 'answer'), 'close']
    assert set(metrics) == {'source_setup_seconds', 'source_call_seconds',
                            'source_finish_seconds', 'source_cleanup_seconds'}
    assert all(value >= 0 for value in metrics.values())


def test_cancellation_prevents_additional_source_calls_but_closes_owned_backend():
    rollout = rollout_module()
    from contextlib import contextmanager
    calls = []
    class Backend:
        def call(self, name, args):
            calls.append(name)
            return 'observation'
    class Dataset:
        @contextmanager
        def episode(self, key, evidence):
            try:
                yield Backend()
            finally:
                calls.append('closed')
    cancelled = False
    def check():
        if cancelled:
            raise RuntimeError('peer failed')
    dataset = rollout.TimedDataset(Dataset(), {}, check_cancelled=check)
    with pytest.raises(RuntimeError, match='peer failed'):
        with dataset.episode(('s', '1'), '/unused') as backend:
            backend.call('first', {})
            cancelled = True
            backend.call('second', {})
    assert calls == ['first', 'closed']


def test_real_cpu_journal_reserves_both_slots_then_atomically_commits_both(tmp_path):
    rollout = rollout_module()
    import torch
    from types import SimpleNamespace
    from hsrl_v5.io import file_hash
    out = tmp_path / 'run'
    out.mkdir()
    directories = [out / 'episodes/step_0000' / f'rollout_{slot:03}' for slot in range(2)]
    state = {'attempts': 0, 'inflight': None, 'episodes': [], 'step': 0}
    # Real torch serialization on CPU at the existing journal boundary. This
    # fixture does not instantiate a policy/model, call sources, or train.
    policy = SimpleNamespace(torch=torch, rng_state=lambda: {'unit_test_stream': [1, 2]})
    rollout.reserve_wave(out, state, directories, policy, max_attempts=256)
    reserved = torch.load(out / 'journal.pt', map_location='cpu', weights_only=False)
    assert reserved['state']['attempts'] == 2 and not reserved['state']['episodes']
    assert len(reserved['state']['inflight']['attempts']) == 2
    assert not any(path.exists() for path in directories)
    with pytest.raises(ValueError, match='reconciliation'):
        rollout.require_resumable(reserved['state'])
    entries = []
    for directory in directories:
        (directory / 'private').mkdir(parents=True)
        (directory / 'private/evidence.json').write_text('{}\n')
        (directory / 'episode.json').write_text('{"valid":true}\n')
        entries.append({'episode_path': str(directory / 'episode.json'),
                        'sha256': file_hash(directory / 'episode.json')})
    rollout.commit_wave(out, state, entries, policy)
    committed = torch.load(out / 'journal.pt', map_location='cpu', weights_only=False)
    assert committed['state'] == state and state['inflight'] is None
    assert state['episodes'] == entries and committed['rng'] == reserved['rng']
    rollout.require_resumable(committed['state'])
    import json
    assert json.loads((out / 'state.json').read_text()) == state
    assert all((path / 'private/evidence.json').exists() for path in directories)


@pytest.mark.parametrize('state', [
    {'inflight': {}}, {'inflight': []}, {'inflight': {'unknown': 'wave'}},
    {'inflight': None, 'episodes': [{}]},
])
def test_unknown_or_partial_inflight_never_resumes(state):
    rollout = rollout_module()
    with pytest.raises(ValueError, match='reconciliation'):
        rollout.require_resumable(state)


def test_missing_inflight_field_is_unknown_and_cannot_resume():
    rollout = rollout_module()
    with pytest.raises(ValueError, match='reconciliation'):
        rollout.require_resumable({'episodes': []})


def test_staggered_renderer_waits_for_peer_generation_before_batch_dispatch():
    rollout = rollout_module()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy)
    def rendered_first(facade):
        return facade.generate([65], 7)
    def still_rendering(facade):
        # Force the reviewed race: slot 0 is already at generation when slot 1
        # queues its tokenizer operation. Rendering cannot count as generation.
        with broker._condition:
            assert broker._condition.wait_for(
                lambda: 0 in broker._pending and broker._pending[0].operation == 'generate',
                timeout=1)
        prompt = facade.tokenizer.encode('B')
        assert not policy.calls, 'generation dispatched before the peer finished rendering'
        return facade.generate(prompt, 7)
    results = broker.run([rendered_first, still_rendering])
    assert [result['response_ids'] for result in results] == [[65], [66]]
    assert policy.calls == [[{'prompt_ids': [65], 'max_new_tokens': 7},
                             {'prompt_ids': [66], 'max_new_tokens': 7}]]
    assert broker.metrics['batch_size_2_calls'] == 1
    assert not broker.live_threads()
