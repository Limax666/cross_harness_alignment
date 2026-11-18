"""Bind a suite-specific candidate Snapshot to a restricted harness episode."""
from flashrsi.core import Snapshot
from flashrsi_snapshot_guard import validate_snapshot_candidate


def select_candidate(baseline, snapshot_path, expected_id, prompt_path):
    if not snapshot_path and not expected_id:
        return baseline
    if not snapshot_path or not expected_id:
        raise ValueError('candidate snapshot path and expected identity are both required')
    candidate = Snapshot.load(snapshot_path)
    if candidate.id != expected_id:
        raise ValueError('candidate snapshot identity mismatch')
    validate_snapshot_candidate(baseline, candidate, {prompt_path}, prompt_path)
    return candidate
