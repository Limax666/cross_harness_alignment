"""Auditable provider projection for compacted native requests with no user role.

Keep the complete native request and SQLite archive immutable. Only when native
compaction removed every user role, replay the exact current public owner task
as a user before the retained assistant/tool continuation. No synthetic task,
new instruction, scoring input, permission or increased token budget is added.
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
    at = 0
    while at < len(messages) and messages[at].get('role') == 'system':
        at += 1
    messages.insert(at, {'role': 'user', 'content': owner})
    return result


def audit_forwarded_requests(native, forwarded, owner):
    if len(native) != len(forwarded) or not native:
        raise ValueError('native/provider evidence count mismatch')
    anchored = 0
    for original, wire in zip(native, forwarded):
        if wire != anchor_owner_request(original, owner):
            raise ValueError('provider projection differs from exact public-owner anchoring')
        anchored += not any(m.get('role') == 'user' for m in original['messages'])
    return {'protocol': 'opencode-provider-owner-anchor-v8',
            'requests': len(native), 'anchored_requests': anchored,
            'native_requests_sha256': digest(native), 'forwarded_requests_sha256': digest(forwarded),
            'owner_sha256': digest(owner), 'original_native_requests_retained': True}
