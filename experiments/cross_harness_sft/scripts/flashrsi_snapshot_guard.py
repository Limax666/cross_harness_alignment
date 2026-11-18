"""Validate proposed FlashRSI snapshots against explicitly frozen resources.

This is an admission primitive, not an execution sandbox or an algorithm
adapter. Call it before candidate execution AND before selecting a snapshot.
"""

MARKER = b'## Trusted safety policy\n'


def validate_snapshot_candidate(baseline, candidate, allowed_paths, policy_path):
    if not isinstance(allowed_paths, (set, frozenset)) or any(not isinstance(p, str) or not p for p in allowed_paths):
        raise ValueError('an explicit set of editable paths is required')
    changed = set(baseline.changed_paths(candidate))
    if changed - allowed_paths:
        raise ValueError('candidate modifies frozen resources')
    original = baseline.files.get(policy_path)
    proposed = candidate.files.get(policy_path)
    if original is None or proposed is None or original.content.count(MARKER) != 1:
        raise ValueError('missing or ambiguous baseline policy')
    pinned_suffix = original.content[original.content.index(MARKER):]
    if proposed.content.count(MARKER) != 1 or not proposed.content.endswith(pinned_suffix):
        raise ValueError('candidate changed or duplicated pinned safety policy')
    if proposed.component != original.component or proposed.executable != original.executable:
        raise ValueError('candidate changed policy file metadata')
    return tuple(sorted(changed))
