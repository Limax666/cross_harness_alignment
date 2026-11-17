"""Audit acceptance guards; no policy, source code, Docker or API execution."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_hsrl_v6_candidate as audit
from hsrl_v5.sampler import OA18


def measurements():
    return [{'configuration': list(c), 'initial_tokens': 1000} for c in OA18]


def test_audit_requires_exact_oa18_and_full_first_call_headroom():
    rows = measurements()
    audit.check_context_lengths(rows)
    for value in (True, 0, 14337):
        rows = measurements()
        rows[0]['initial_tokens'] = value
        with pytest.raises(ValueError):
            audit.check_context_lengths(rows)
    rows = measurements()
    rows[0]['configuration'] = [99] * 5
    with pytest.raises(ValueError, match='OA18'):
        audit.check_context_lengths(rows)


def test_official_tokenizer_rejects_untrusted_path_and_measures_real_pinned_bytes(tmp_path):
    with pytest.raises(ValueError, match='canonical'):
        audit.verify_tokenizer(tmp_path)
    authority = json.loads(audit.IDENTITY_PATH.read_text())
    proof = audit.verify_tokenizer(authority['canonical_path'])
    assert proof['source_revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    assert len(proof['verified_tokenizer_artifact_sha256']) == 9
    assert proof['verified_tokenizer_artifact_sha256']['tokenizer.json'] == authority['artifacts']['tokenizer.json']['digest']
    assert not proof['model_weights_verified_by_this_audit']


def test_no_ready_candidate_is_invented_for_absent_export(tmp_path):
    candidate = tmp_path / 'not_exported'
    candidate.mkdir()
    out = tmp_path / 'audit'
    with pytest.raises(ValueError, match='unpublished_real_candidate'):
        audit.run(candidate, out, tmp_path)
    assert not out.exists()
