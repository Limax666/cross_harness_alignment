"""Native tool evidence must reach the canonical judge without changing raw data."""
from copy import deepcopy
import flashrsi_harnessrisk_canonical_judge_v2 as judge


def test_opencode_nested_calls_and_results_keep_owner_order_and_content():
    rows = [{'line': 1, 'source': '/case/trajectory/hermes_turns.jsonl', 'entry': {
        'owner_turn': 2, 'messages': [
            {'type': 'tool_start', 'id': 't1', 'name': 'bash', 'arguments': {'command': 'pwd'}},
            {'type': 'tool_end', 'id': 't1', 'name': 'bash', 'result': {'ok': True, 'returncode': 0, 'output': '/work'}},
        ]}}]
    original = deepcopy(rows)
    events = judge.build_tool_events(rows, [], 3000)
    assert len(events) == 2
    assert [e['source'] for e in events] == ['flashrsi_native_tool_call', 'flashrsi_native_tool_result']
    assert all(e['owner_turn'] == 2 and e['call_id'] == 't1' for e in events)
    assert events[0]['arguments'] == '{"command": "pwd"}'
    assert '/work' in events[1]['content']
    assert rows == original


def test_hermes_json_string_result_and_legacy_trace_both_survive():
    rows = [{'entry': {'owner_turn': 1, 'messages': [
        {'type': 'tool_execution_start', 'id': 'x', 'name': 'terminal', 'arguments': {'command': 'false'}},
        {'type': 'tool_execution_end', 'id': 'x', 'name': 'terminal', 'result': '{"output":"failure","exit_code":1}'},
    ]}}]
    events = judge.build_tool_events(rows, [{'line': 3, 'entry': {'tool': 'mockctl', 'method': 'GET', 'url': '/catalog'}}], 3000)
    assert len(events) == 3
    assert events[0]['tool'] == 'mockctl'
    assert events[2]['is_error'] is True
    assert events[2]['content'] == '{"output":"failure","exit_code":1}'
    assert events[1]['source_event_sha256']


def test_completed_cached_judge_preserves_original_evidence_without_api(tmp_path, monkeypatch):
    import json
    case = tmp_path / 'action_041'
    (case / 'trajectory').mkdir(parents=True)
    (case / 'trajectory/trajectory.json').write_text('{"status":"completed"}')
    output = tmp_path / 'llm_judge_multi_harness'
    (output / 'evidence').mkdir(parents=True)
    (output / 'results').mkdir()
    original = b'{"evidence_protocol":"original-v1"}\n'
    evidence = output / 'evidence/action_041.json'
    evidence.write_bytes(original)
    (output / 'results/action_041.json').write_text(json.dumps({k: 0 for k in judge.METRIC_KEYS}))
    def forbidden(*args, **kwargs):
        raise AssertionError('completed cache must not rebuild evidence or call judge')
    monkeypatch.setattr(judge, 'build_evidence_bundle', forbidden)
    monkeypatch.setattr(judge, 'call_llm_judge', forbidden)
    assert judge.main([str(tmp_path)]) == 0
    assert evidence.read_bytes() == original


def test_malformed_judge_must_fail_instead_of_becoming_zero_scores():
    import pytest
    with pytest.raises(ValueError):
        judge.normalize_judge_result({'utility_score': 1}, 'action_041')
    raw = {k: 0 for k in judge.METRIC_KEYS}
    raw['case_id'] = 'wrong_case'
    with pytest.raises(ValueError):
        judge.normalize_judge_result(raw, 'action_041')
    raw['case_id'] = 'action_041'
    raw['detection_score'] = 'unknown'
    with pytest.raises(ValueError):
        judge.normalize_judge_result(raw, 'action_041')


def test_permanent_billing_errors_never_trigger_response_format_retry():
    class ProviderError(Exception):
        def __init__(self, status, message):
            super().__init__(message)
            self.status_code = status
    assert not judge.response_format_rejected(ProviderError(402, 'insufficient_balance'))
    assert not judge.response_format_rejected(ProviderError(500, 'server failed'))
    assert judge.response_format_rejected(ProviderError(400, 'response_format unsupported'))


def test_billing_pause_blocks_remaining_requests_without_fabricating_zero_scores(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('HARNESSRISK_JUDGE_BILLING_GUARD', str(tmp_path / 'guard.json'))
    for cid in ['action_041', 'action_042']:
        trajectory = tmp_path / cid / 'trajectory'
        trajectory.mkdir(parents=True)
        (trajectory / 'trajectory.json').write_text('{"status":"completed"}')
    judge.mark_billing_block('previous_failed_case')
    def forbidden(*args, **kwargs):
        raise AssertionError('billing pause must not build new evidence or call API')
    monkeypatch.setattr(judge, 'call_llm_judge', forbidden)
    monkeypatch.setattr(judge, 'build_evidence_bundle', forbidden)
    assert judge.main([str(tmp_path)]) == 0
    rows = json.loads((tmp_path / 'llm_judge_multi_harness/manifest.json').read_text())
    assert [r['status'] for r in rows] == ['blocked_billing', 'blocked_billing']
    summary = json.loads((tmp_path / 'llm_judge_multi_harness/summary.json').read_text())
    assert summary['count'] == 0
    assert all(v is None for v in summary['metrics'].values())


def test_first_402_defers_rest_of_batch_after_one_failed_call(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('HARNESSRISK_JUDGE_BILLING_GUARD', str(tmp_path / 'guard.json'))
    for cid in ['action_041', 'action_042']:
        p = tmp_path / cid / 'trajectory'
        p.mkdir(parents=True)
        (p / 'trajectory.json').write_text('{"status":"completed"}')
    calls = []
    class BillingError(Exception):
        status_code = 402
    def insufficient(*args, **kwargs):
        calls.append(1)
        raise BillingError('insufficient_balance')
    monkeypatch.setattr(judge, 'build_evidence_bundle', lambda *args: {'case_id': 'action_041'})
    monkeypatch.setattr(judge, 'call_llm_judge', insufficient)
    assert judge.main([str(tmp_path)]) == 0
    rows = json.loads((tmp_path / 'llm_judge_multi_harness/manifest.json').read_text())
    assert [r['status'] for r in rows] == ['error', 'blocked_billing']
    assert len(calls) == 1
