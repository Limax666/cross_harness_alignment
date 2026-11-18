"""Regression for the real v7 compacted request losing every user role."""
from copy import deepcopy


def test_compacted_request_preserves_owner_as_user_without_mutating_native_archive():
    from flashrsi_harnessrisk_opencode_owner_anchor_v8 import anchor_owner_request
    raw = {'messages': [{'role': 'system', 'content': 'policy'},
                        {'role': 'assistant', 'content': '[OpenCode compacted prior context]\n[User]: old'}],
           'max_tokens': 4096, 'tools': []}
    before = deepcopy(raw)
    fixed = anchor_owner_request(raw, 'Original current owner task')
    assert raw == before
    assert [m['role'] for m in fixed['messages']] == ['system', 'user', 'assistant']
    assert fixed['messages'][1]['content'] == 'Original current owner task'
    assert fixed['messages'][2] == raw['messages'][1]
    assert fixed['max_tokens'] == 4096


def test_existing_user_and_tool_continuation_are_unchanged():
    from flashrsi_harnessrisk_opencode_owner_anchor_v8 import anchor_owner_request
    raw = {'messages': [{'role': 'user', 'content': 'task'},
                        {'role': 'assistant', 'tool_calls': [{'id': 'a'}]},
                        {'role': 'tool', 'tool_call_id': 'a', 'content': 'untrusted data'}]}
    assert anchor_owner_request(raw, 'task') == raw


def test_wire_audit_rejects_owner_rewrite_and_unrecorded_projection():
    from flashrsi_harnessrisk_opencode_owner_anchor_v8 import anchor_owner_request, audit_forwarded_requests
    import pytest
    raw = {'messages': [{'role': 'system', 'content': 'policy'}, {'role': 'assistant', 'content': 'summary'}]}
    wire = anchor_owner_request(raw, 'current task')
    record = audit_forwarded_requests([raw], [wire], 'current task')
    assert record['anchored_requests'] == 1
    wire['messages'][1]['content'] = 'different task'
    with pytest.raises(ValueError):
        audit_forwarded_requests([raw], [wire], 'current task')
