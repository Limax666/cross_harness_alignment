"""Fault recovery state tests. These do not run an RL experiment."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_only_isolated_episode_rejection_may_enter_slot_retry():
    from hsrl_slot_recovery_v10 import retryable_rejection

    record = {
        'key': ['clawgym', 'task'], 'valid': False,
        'error': {'type': 'ValueError', 'detail': 'source_measurement_invalid_or_incomplete'},
        'training_reward': None, 'reward_components': None,
        'source_result': {'done': True, 'valid': False, 'utility': None, 'c_source': None,
                          'reason': 'immutable_grading_reference_modified', 'changed_inputs': ['test-input']},
    }
    assert retryable_rejection(record)
    assert not retryable_rejection({**record, 'training_reward': 0})
    assert not retryable_rejection({**record, 'error': {'type': 'CleanupUncertain'}})
    assert not retryable_rejection({**record, 'key': ['agentdog', 'task']})


def test_only_rejected_slot_is_reserved_again_and_valid_peers_are_retained(tmp_path):
    from hsrl_slot_recovery_v10 import plan_reservations

    state = {
        'step': 9, 'attempts': 1304, 'episodes': [{}] * 24, 'inflight': None,
        'pending_slots': {str(i): {'episode_path': f'preserved-{i}'} for i in range(24, 31)},
        'slot_failures': {'31': 1},
    }
    reservations = plan_reservations(tmp_path, state, max_attempts=38400)
    assert [(a['slot'], a['attempt']) for a in reservations] == [(31, 1305)]
    assert 'attempt_001305' in reservations[0]['directory']
    assert state['attempts'] == 1304 and len(state['pending_slots']) == 7
    with pytest.raises(ValueError, match='budget'):
        plan_reservations(tmp_path, state, max_attempts=1304)
    state['inflight'] = {}
    with pytest.raises(ValueError, match='reconciliation'):
        plan_reservations(tmp_path, state, max_attempts=38400)
    state['inflight'] = None
    state['slot_failures']['31'] = 3
    with pytest.raises(ValueError, match='retry_limit'):
        plan_reservations(tmp_path, state, max_attempts=38400)


def test_collect_wave_durably_retries_only_one_rejected_worker(tmp_path, monkeypatch):
    import copy
    import json
    from types import SimpleNamespace

    import hsrl_slot_recovery_v10 as recovery

    out = tmp_path.resolve()
    state = {'step': 9, 'attempts': 1304, 'episodes': [{}] * 24, 'inflight': None,
             'pending_slots': {}, 'slot_failures': {}, 'rejected_attempts': [],
             'policy_version': 9, 'adapter_hash': 'adapter'}
    plan = {'groups': [{'source': 'clawgym', 'id': 'task'}] * 4, 'configuration': [0] * 5}
    policy = SimpleNamespace(batch_events=[], version=9, collection_adapter_hash="adapter")
    snapshots = []
    monkeypatch.setattr(recovery.original, 'persist_journal',
                        lambda _out, value, _policy: snapshots.append(copy.deepcopy(value)))

    def episode(_dataset, key, row, _policy, directory, *, objective):
        directory.mkdir(parents=True)
        rejected = directory.name == 'rollout_031_attempt_001312'
        result = {'done': True, 'valid': not rejected}
        error = None
        if rejected:
            result.update(reason=recovery.REJECTION, changed_inputs=['fixture'],
                          utility=None, c_source=None)
            error = {'type': 'ValueError', 'detail': 'source_measurement_invalid_or_incomplete'}
            worker = directory / 'private/backend/tool_worker'
            worker.mkdir(parents=True)
            (worker / 'cleanup.json').write_text(json.dumps(
                {'container': 'closed-fixture', 'removal_confirmed': True}))
            (worker.parent / 'result.json').write_text(json.dumps(result))
        record = {'key': list(key), 'configuration': recovery.original.config_name(row),
                  'policy_version': 9, 'adapter_hash_at_collection': 'adapter', 'valid': not rejected,
                  'reward_components': None, 'training_reward': None if rejected else 1.0,
                  'source_result': result, 'error': error}
        (directory / 'episode.json').write_text(json.dumps(record))
        return record

    monkeypatch.setattr(recovery, 'run_episode', episode)
    metrics = recovery.collect_wave(None, plan, policy, out, state, objective='safety',
                                    max_attempts=38400, max_batch_tokens=32768)
    assert [len(snapshot['inflight']['attempts']) if snapshot['inflight'] else 0
            for snapshot in snapshots] == [8, 0, 1, 0]
    assert len(snapshots[1]['pending_slots']) == 7
    assert len(state['episodes']) == 32 and state['pending_slots'] == {}
    assert state['attempts'] == 1313
    assert state['rejected_attempts'][0]['attempt'] == 1312
    assert '001313' in state['episodes'][-1]['episode_path']
    assert metrics['attempt_count'] == 9 and metrics['accepted_this_session'] == 8
