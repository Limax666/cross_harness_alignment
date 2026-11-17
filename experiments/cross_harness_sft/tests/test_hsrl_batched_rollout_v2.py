"""Eight-worker CPU/filesystem contract tests; no task/model/GPU execution."""
import copy
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def implementation():
    try:
        import hsrl_batched_rollout_v2 as rollout
    except ImportError:
        pytest.fail('eight-worker rollout implementation missing')
    return rollout


class BoundaryPolicy:
    version = 2
    collection_adapter_hash = 'adapter'

    def __init__(self):
        self.owner = threading.get_ident()
        self.tokenizer = self
        self.calls = []

    def encode(self, text, **kwargs):
        assert threading.get_ident() == self.owner
        return [ord(c) for c in text]

    def generate_batch(self, requests):
        assert threading.get_ident() == self.owner
        self.calls.append(copy.deepcopy(requests))
        return [{'response_ids': [r['prompt_ids'][0]], 'policy_version': self.version}
                for r in requests]


def test_eight_real_worker_requests_dispatch_one_eight_row_model_boundary_call():
    rollout = implementation()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy)
    workers = set()
    def task(slot, facade):
        workers.add(threading.get_ident())
        return facade.generate([65 + slot], 7)
    result = broker.run([lambda p, slot=slot: task(slot, p) for slot in range(8)])
    assert [row['response_ids'] for row in result] == [[65 + slot] for slot in range(8)]
    assert len(workers) == 8 and policy.owner not in workers
    assert len(policy.calls) == 1 and len(policy.calls[0]) == 8
    assert broker.metrics['batch_size_8_calls'] == 1
    assert broker.metrics['batch_size_histogram']['8'] == 1
    assert broker.metrics['peak_active_workers'] == 8
    assert not broker.live_threads()


def test_staggered_tokenizer_is_never_mistaken_for_eighth_generation_request():
    rollout = implementation()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy)
    def renderer(p):
        with broker._condition:
            assert broker._condition.wait_for(lambda: len(broker._pending) == 7, timeout=1)
        prompt = p.tokenizer.encode('H')
        assert not policy.calls
        return p.generate(prompt, 7)
    result = broker.run([lambda p, slot=i: p.generate([65 + slot], 7) for i in range(7)] + [renderer])
    assert len(result) == 8 and [len(call) for call in policy.calls] == [8]
    assert not broker.live_threads()


def test_padding_limited_batches_serve_old_pending_ordinals_before_faster_new_turns():
    rollout = implementation()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy, max_batch_tokens=16)
    def task(slot, p):
        return [p.generate([65 + slot], 7), p.generate([65 + slot], 7)]
    results = broker.run([lambda p, slot=i: task(slot, p) for i in range(8)])
    assert all(len(rows) == 2 for rows in results)
    actual = [[r['prompt_ids'][0] for r in call] for call in policy.calls]
    assert actual == [[65, 66], [67, 68], [69, 70], [71, 72]] * 2
    assert [event['generation_ordinals'] for event in broker.metrics['batches']] == [[0, 0]] * 4 + [[1, 1]] * 4
    assert all(event['padded_token_budget'] <= 16 for event in broker.metrics['batches'])
    assert not broker.live_threads()


def test_mixed_budgets_and_unequal_prompts_split_without_changing_any_request():
    rollout = implementation()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy, max_batch_tokens=32)
    prompts = [[65 + i] * (i + 1) for i in range(8)]
    budgets = [3, 4] * 4
    broker.run([lambda p, i=i: p.generate(prompts[i], budgets[i]) for i in range(8)])
    flattened = [r for call in policy.calls for r in call]
    assert sorted((tuple(r['prompt_ids']), r['max_new_tokens']) for r in flattened) == sorted(
        (tuple(prompt), budget) for prompt, budget in zip(prompts, budgets))
    for call in policy.calls:
        assert len({r['max_new_tokens'] for r in call}) == 1
        assert len(call) * (max(len(r['prompt_ids']) for r in call) + call[0]['max_new_tokens']) <= 32


def test_one_source_boundary_failure_cancels_all_seven_waiters_and_joins():
    rollout = implementation()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy)
    def fail(p):
        with broker._condition:
            assert broker._condition.wait_for(lambda: len(broker._pending) == 7, timeout=1)
        raise RuntimeError('source boundary rejected')
    with pytest.raises(RuntimeError, match='source boundary rejected'):
        broker.run([lambda p, i=i: p.generate([65 + i], 7) for i in range(7)] + [fail])
    assert not policy.calls and not broker.live_threads()


def test_eight_row_model_failure_preserves_actual_invocation_and_releases_futures():
    rollout = implementation()
    policy = BoundaryPolicy()
    def fail(requests):
        raise RuntimeError('no allocation retry')
    policy.generate_batch = fail
    broker = rollout.WaveBroker(policy)
    with pytest.raises(RuntimeError, match='no allocation retry'):
        broker.run([lambda p: p.generate([65], 7) for _ in range(8)])
    assert broker.metrics['model_batch_calls'] == 1
    assert broker.metrics['batch_size_8_calls'] == 1
    assert broker.metrics['batches'][0]['status'] == 'failed'
    assert not broker.live_threads()


def test_completed_peers_allow_remaining_worker_to_finish_more_turns():
    rollout = implementation()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy)
    def twice(p):
        return [p.generate([65], 7), p.generate([65], 7)]
    result = broker.run([twice] + [lambda p: p.generate([66], 7) for _ in range(7)])
    assert len(result[0]) == 2 and [len(call) for call in policy.calls] == [8, 1]
    assert not broker.live_threads()


def test_real_cpu_journal_reserves_one_G8_group_and_commits_after_all_eight_fsyncs(tmp_path):
    rollout = implementation()
    import torch
    from types import SimpleNamespace
    from hsrl_v5.io import file_hash
    import train_hsrl_v5 as original
    out = tmp_path / 'run'
    out.mkdir()
    state = {'attempts': 0, 'inflight': None, 'episodes': [], 'step': 0}
    directories = [out / 'episodes/step_0000' / f'rollout_{i:03}' for i in range(8)]
    policy = SimpleNamespace(torch=torch, rng_state=lambda: {'cpu_fixture': 1})
    rollout.reserve_wave(out, state, directories, policy, max_attempts=256)
    saved = torch.load(out / 'journal.pt', map_location='cpu', weights_only=False)
    assert saved['state']['attempts'] == 8 and len(saved['state']['inflight']['attempts']) == 8
    assert not any(directory.exists() for directory in directories)
    with pytest.raises(ValueError, match='reconciliation'):
        rollout.require_resumable(saved['state'])
    entries = []
    for directory in directories:
        directory.mkdir(parents=True)
        (directory / 'episode.json').write_text('{"valid":true}\n')
        entries.append({'episode_path': str(directory / 'episode.json'),
                        'sha256': file_hash(directory / 'episode.json')})
    before = copy.deepcopy(state)
    def failed_barrier(out, directory):
        original.seal_episode_evidence(out, directory)
        if directory == directories[-1]:
            raise OSError('eighth barrier failed')
    with pytest.raises(OSError, match='eighth barrier'):
        rollout.commit_wave(out, state, entries, policy, seal=failed_barrier)
    assert state == before
    assert torch.load(out / 'journal.pt', map_location='cpu', weights_only=False)['state'] == before
    rollout.commit_wave(out, state, entries, policy)
    committed = torch.load(out / 'journal.pt', map_location='cpu', weights_only=False)['state']
    assert committed == state and state['episodes'] == entries and state['inflight'] is None
    rollout.require_resumable(committed)
    with pytest.raises(ValueError, match='fresh'):
        rollout.reserve_wave(out, {**state, 'episodes': []}, directories, policy, max_attempts=256)


@pytest.mark.parametrize('episodes,inflight', [(0, {}), (0, []), (2, None), (7, None)])
def test_unknown_inflight_and_non_G8_prefixes_cannot_resume(episodes, inflight):
    rollout = implementation()
    with pytest.raises(ValueError, match='reconciliation'):
        rollout.require_resumable({'episodes': [{}] * episodes, 'inflight': inflight})


def test_v8_cli_and_identity_are_new_while_reviewed_optimizer_helpers_stay_bound():
    try:
        import train_hsrl_v8 as driver
    except ImportError:
        pytest.fail('eight-worker trainer implementation missing')
    import train_hsrl_v5 as original
    args = driver.argument_parser().parse_args(['--dataset', '/d', '--model', '/m', '--out', '/o', '--lease', '/l'])
    assert (args.physical_gpu, args.episode_workers, args.model_batch_size, args.max_batch_tokens) == (1, 8, 8, 32768)
    assert driver.prepare_update is original.prepare_update and driver.commit_checkpoint is original.commit_checkpoint
    assert driver.CONFIG == original.CONFIG
    assert driver.SCHEMA != original.SCHEMA
    driver.validate_batch_contract(args)
    args.episode_workers = 2
    with pytest.raises(ValueError, match='eight'):
        driver.validate_batch_contract(args)


def test_invalid_eighth_episode_cannot_commit_or_clear_any_reserved_attempt(tmp_path):
    rollout = implementation()
    from hsrl_v5.io import file_hash
    out = tmp_path / 'run'
    directories = [out / 'episodes/step_0000' / f'rollout_{i:03}' for i in range(8)]
    state = {'attempts': 0, 'inflight': None, 'episodes': [], 'step': 0}
    writes = []
    persist = lambda out, state, policy: writes.append(copy.deepcopy(state))
    rollout.reserve_wave(out, state, directories, None, max_attempts=256, persist=persist)
    entries = []
    for i, directory in enumerate(directories):
        directory.mkdir(parents=True)
        (directory / 'episode.json').write_text('{"valid":true}\n' if i < 7 else '{"valid":false}\n')
        entries.append({'episode_path': str(directory / 'episode.json'),
                        'sha256': file_hash(directory / 'episode.json')})
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='invalid'):
        rollout.commit_wave(out, state, entries, None, persist=persist)
    assert state == before and len(writes) == 1
    (directories[-1] / 'episode.json').write_text('{"valid":true}\n')
    entries[-1]['sha256'] = file_hash(directories[-1] / 'episode.json')
    def fail(*args):
        raise OSError('journal boundary failed')
    with pytest.raises(OSError, match='journal boundary'):
        rollout.commit_wave(out, state, entries, None, persist=fail)
    assert state == before and all((directory / 'episode.json').exists() for directory in directories)


def test_attempt_budget_and_incomplete_wave_rejections_precede_worker_activity(tmp_path):
    rollout = implementation()
    out = tmp_path / 'run'
    directories = [out / 'episodes/step_0000' / f'rollout_{i:03}' for i in range(8)]
    state = {'attempts': 250, 'inflight': None, 'episodes': [], 'step': 0}
    writes = []
    with pytest.raises(ValueError, match='budget'):
        rollout.reserve_wave(out, state, directories, None, max_attempts=256,
                             persist=lambda *args: writes.append(True))
    assert not writes and state['attempts'] == 250
    for count in (2, 7, 9):
        broker = rollout.WaveBroker(BoundaryPolicy())
        with pytest.raises(ValueError, match='eight'):
            broker.run([lambda p: None] * count)
        assert not broker.live_threads()
    with pytest.raises(ValueError, match='reconciliation'):
        rollout.require_resumable({'episodes': []})


def test_eight_worker_deadline_cancels_seven_futures_and_bounds_cleanup():
    rollout = implementation()
    policy = BoundaryPolicy()
    broker = rollout.WaveBroker(policy, wave_timeout=.03, cleanup_timeout=.03)
    release = threading.Event()
    try:
        with pytest.raises(RuntimeError, match='cleanup_deadline'):
            broker.run([lambda p: release.wait(1)] + [lambda p: p.generate([65], 7) for _ in range(7)])
        assert not policy.calls
    finally:
        release.set()
        for thread in broker._threads:
            thread.join(1)
    assert not broker.live_threads()


def test_v8_code_identity_binds_both_policy_generations_and_reused_helpers():
    import train_hsrl_v8 as driver
    from hsrl_v5.io import file_hash
    identity = driver._code_identity()
    root = Path(driver.__file__).parent
    for name in ('train_hsrl_v5.py', 'train_hsrl_v7.py', 'train_hsrl_v8.py',
                 'hsrl_batched_policy_v1.py', 'hsrl_batched_policy_v2.py',
                 'hsrl_batched_rollout_v1.py', 'hsrl_batched_rollout_v2.py',
                 'hsrl_v5/runtime.py', 'hsrl_v5/sampler.py', 'hsrl_v3/policy.py'):
        assert identity[name] == file_hash(root / name)
