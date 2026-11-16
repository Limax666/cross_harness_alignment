"""Workspace/reward boundaries; full source admission supplies runtime proof."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def modules():
    assert importlib.util.find_spec('hsrl_v5.workspace'), 'bounded workspace runtime missing'
    from hsrl_v5.workspace import safe_path, numeric_reward, snapshot_entries
    return safe_path, numeric_reward, snapshot_entries


def test_workspace_refuses_absolute_parent_and_symlink_paths(tmp_path):
    scoped, _, _ = modules()
    (tmp_path / 'link').symlink_to('/tmp')
    for name in ('../../outside', '/etc/passwd', 'link/file'):
        with pytest.raises(ValueError):
            scoped(tmp_path, name)
    assert scoped(tmp_path, 'output/report.json') == tmp_path / 'output/report.json'


def test_empty_nan_and_out_of_range_grades_are_invalid_not_zero():
    _, grade, _ = modules()
    for value in ({}, [], None, float('nan'), 2.0, {'x': 'not-a-score'}):
        with pytest.raises(ValueError):
            grade(value)
    assert grade({'a': True, 'b': False, 'c': 0.5}) == .5


def test_snapshot_digest_covers_real_source_bytes_and_rejects_symlinks(tmp_path):
    _, _, entries = modules()
    (tmp_path / 'input.json').write_text('{"real":true}')
    original = entries(tmp_path)
    (tmp_path / 'input.json').write_text('{"real":false}')
    assert entries(tmp_path) != original
    (tmp_path / 'escape').symlink_to('/etc/passwd')
    with pytest.raises(ValueError):
        entries(tmp_path)


def test_component_grades_cannot_hide_out_of_range_values_in_average():
    _, grade, _ = modules()
    for value in ({'negative': -1, 'positive': 1}, {'too_high': 1.5, 'zero': 0}):
        with pytest.raises(ValueError, match='unit_interval'):
            grade(value)


def test_pending_infrastructure_record_is_retried_but_completed_record_is_reused(tmp_path, monkeypatch):
    from hsrl_v5 import quality_claw as qc
    from hsrl_v5.io import atomic_json, digest
    record = {'id': 'real-id', 'family': 'family', 'reward_profile': 'profile'}
    previous = {'record_sha256': digest(record), 'ready': False,
                'failure_class': 'isolation_or_runtime_infrastructure'}
    atomic_json(tmp_path / 'quality.json', previous)
    class BoundaryReached(Exception):
        pass
    def reach_real_backend(*args):
        raise BoundaryReached('backend-boundary-reached')
    monkeypatch.setattr(qc, 'ClawBackend', reach_real_backend)
    proof = qc.validate_record(record, tmp_path)
    assert proof['reasons'] == ['BoundaryReached:backend-boundary-reached']
    assert len(list(tmp_path.glob('attempt_*'))) == 1
    proof['ready'] = True
    atomic_json(tmp_path / 'quality.json', proof)
    assert qc.validate_record(record, tmp_path) == proof
    assert len(list(tmp_path.glob('attempt_*'))) == 1


def test_bounded_dispatch_stops_after_infrastructure_failure(tmp_path):
    from hsrl_v5.quality_claw import bounded_admission
    import threading
    import time
    seen, lock = [], threading.Lock()
    def external_boundary(record, directory):
        with lock:
            seen.append(record['id'])
        if record['id'] == '1':
            return {'id': '1', 'ready': False, 'reasons': ['engine unavailable'],
                    'failure_class': 'isolation_or_runtime_infrastructure'}
        time.sleep(.05)
        return {'id': record['id'], 'ready': True, 'reasons': []}
    records = [{'id': str(i)} for i in range(100)]
    proofs, blocked = bounded_admission(records, tmp_path, 2, validate=external_boundary)
    assert blocked is True
    assert len(seen) <= 2
    assert any(p['id'] == '1' for p in proofs)
    assert len(list((tmp_path / 'events.jsonl').open())) == len(proofs)


def test_oversized_worker_count_is_rejected_before_source_access(tmp_path):
    from hsrl_v5.quality_claw import run
    with pytest.raises(ValueError, match='workers'):
        run(tmp_path / 'missing', tmp_path / 'out', 4)


def test_original_checker_crash_and_malformed_scores_are_invalid_not_zero():
    from hsrl_v5.claw_backend import parse_checker_response
    for response in ({'ok': False, 'error': 'wall_timeout'},
                     {'ok': True, 'result': {'returncode': 1, 'stdout': '0', 'stderr': 'traceback'}},
                     {'ok': True, 'result': {'returncode': 0, 'stdout': '{}', 'stderr': ''}},
                     {'ok': True, 'result': {'returncode': 0, 'stdout': 'not-json', 'stderr': ''}}):
        with pytest.raises(ValueError):
            parse_checker_response(response)
    assert parse_checker_response({'ok': True, 'result': {'returncode': 0, 'stdout': '{"a":0,"b":1}', 'stderr': ''}}) == ({'a': 0, 'b': 1}, .5)


def test_real_child_output_limit_kills_command_and_preserves_workspace(tmp_path, monkeypatch):
    from hsrl_v5 import workspace_worker as worker
    monkeypatch.setattr(worker, 'PUBLIC', tmp_path)
    with pytest.raises(RuntimeError, match='output_limit'):
        worker.bounded_command([sys.executable, '-c', 'import sys;sys.stdout.write("x"*1100000)'])
    assert list(tmp_path.iterdir()) == []
