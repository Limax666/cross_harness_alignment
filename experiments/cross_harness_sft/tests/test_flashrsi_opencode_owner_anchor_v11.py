"""Keep the current task after compacted history, not before an assistant prefix."""
from copy import deepcopy
import pytest


def test_compacted_history_is_followed_by_exact_current_owner():
    from flashrsi_harnessrisk_opencode_owner_anchor_v11 import anchor_owner_request
    native = {'messages': [{'role': 'system', 'content': 'policy'},
              {'role': 'assistant', 'content': '[OpenCode compacted prior context]\n[Tool result]: incomplete historical text'}],
              'max_tokens': 4096, 'tools': []}
    before = deepcopy(native)
    wire = anchor_owner_request(native, 'Current public owner task')
    assert native == before
    assert wire['messages'][:-1] == native['messages']
    assert wire['messages'][-1] == {'role': 'user', 'content': 'Current public owner task'}
    assert wire['max_tokens'] == 4096


def test_existing_user_and_tool_continuation_are_unchanged():
    from flashrsi_harnessrisk_opencode_owner_anchor_v11 import anchor_owner_request
    native = {'messages': [{'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'tool_calls': [{'id': 'a'}]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'untrusted data'}]}
    assert anchor_owner_request(native, 'task') == native


def test_audit_rejects_old_order_and_owner_rewrite():
    from flashrsi_harnessrisk_opencode_owner_anchor_v11 import anchor_owner_request, audit_forwarded_requests
    raw = {'messages': [{'role': 'system', 'content': 'p'}, {'role': 'assistant', 'content': 'summary'}]}
    wire = anchor_owner_request(raw, 'current task')
    assert audit_forwarded_requests([raw], [wire], 'current task')['anchored_requests'] == 1
    with pytest.raises(ValueError):
        audit_forwarded_requests([raw], [wire], 'different task')
    old = deepcopy(wire)
    old['messages'].insert(1, old['messages'].pop())
    with pytest.raises(ValueError):
        audit_forwarded_requests([raw], [old], 'current task')
