"""Replay the exact current owner after compacted history in provider requests.

The native archive is immutable. Requests already containing a user are intact.
No instructions, token budgets, tool schemas or authority are added.
"""
import copy
from flashrsi_harnessrisk_opencode_session_audit_v4 import digest


def anchor_owner_request(request, owner):
    result = copy.deepcopy(request)
    messages = result.get('messages')
    if not isinstance(messages, list):
        raise ValueError('native request messages must be a list')
    if any(m.get('role') == 'user' for m in messages):
        return result
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError('missing original public owner task; refusing invented user')
    messages.append({'role': 'user', 'content': owner})
    return result


def audit_forwarded_requests(native, forwarded, owner):
    if len(native) != len(forwarded) or not native:
        raise ValueError('native/provider evidence count mismatch')
    anchored = 0
    for original, wire in zip(native, forwarded):
        if wire != anchor_owner_request(original, owner):
            raise ValueError('provider projection differs from exact current-owner suffix')
        anchored += not any(m.get('role') == 'user' for m in original['messages'])
    return {'protocol': 'opencode-provider-owner-anchor-v11-suffix',
            'requests': len(native), 'anchored_requests': anchored,
            'native_requests_sha256': digest(native), 'forwarded_requests_sha256': digest(forwarded),
            'owner_sha256': digest(owner), 'original_native_requests_retained': True}
