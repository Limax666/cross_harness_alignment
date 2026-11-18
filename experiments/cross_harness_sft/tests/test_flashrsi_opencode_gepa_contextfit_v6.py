"""Bound the reflection contract without changing target trajectories or scores."""
from copy import deepcopy
import json
import pytest


def test_compact_projection_retains_every_task_metric_and_trace_hash():
    from flashrsi_opencode_gepa_contextfit_v6 import compact_evidence, reflection_prompt
    rows = [{'task': 'task' * 500, 'output': 'answer' * 500,
             'metrics': {'utility_score': 0.75}, 'feedback': {'status': 'finished', 'ground_truth': None, 'error_message': None},
             'trajectory': [{'type': 'tool_end', 'result': 'long raw tool trace' * 3000},
                            {'type': 'agent_end', 'stop_reason': 'stop', 'iterations': 12}]} for _ in range(3)]
    before = deepcopy(rows)
    rendered = compact_evidence(rows)
    assert rows == before
    assert len(rendered) <= 6000
    projected = json.loads(rendered)
    assert len(projected) == 3
    assert all(r['metrics'] == rows[i]['metrics'] and len(r['trajectory_sha256']) == 64 for i, r in enumerate(projected))
    assert projected[0]['terminal']['iterations'] == 12
    assert 'long raw tool trace' not in rendered
    prompt = reflection_prompt('AGENTS.md', 'Original instruction', rendered)
    assert 'at most 300 words' in prompt
    assert prompt.endswith('```\nOriginal instruction\n```')


def test_oversize_required_feedback_is_rejected_not_silently_dropped():
    from flashrsi_opencode_gepa_contextfit_v6 import compact_evidence
    rows = [{'task': 't', 'output': 'o', 'metrics': {},
             'feedback': {'ground_truth': 'x' * 10000}, 'trajectory': []}]
    with pytest.raises(ValueError, match='required feedback'):
        compact_evidence(rows)


def test_selector_is_explicit_and_opencode_only(monkeypatch):
    from flashrsi_harnessrisk_algorithm_opencode_gepa_v6 import resolve_algorithm
    monkeypatch.setenv('HR_GEPA_REFLECTION_PROTOCOL', 'opencode-v6-contextfit')
    monkeypatch.setenv('HR_HARNESS', 'opencode')
    assert resolve_algorithm('gepa')[0].__name__ == 'OpenCodeContextfitGEPA'
    monkeypatch.setenv('HR_HARNESS', 'hermes')
    assert resolve_algorithm('gepa')[0].__name__ == 'GEPA'
