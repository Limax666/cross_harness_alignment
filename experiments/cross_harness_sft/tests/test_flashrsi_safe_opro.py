from types import SimpleNamespace

import pytest

from flashrsi_safe_opro import SafetyOPRO, protect_policy_suffix
from flashrsi.algorithms.opro import OPRO


def scored(name, score):
    return SimpleNamespace(snapshot=SimpleNamespace(id=name), score=score)


def evidence(attack, coverage=1.0):
    return {'case': {'arm': 'attack', 'attack_success_rate': attack, 'verifier_coverage': coverage}}


def test_worker_may_edit_prefix_without_owning_safety_policy():
    suffix = '## Trusted safety policy\npinned\n'
    assert protect_policy_suffix('new instruction', suffix) == 'new instruction\n\n' + suffix
    assert protect_policy_suffix('new\n\n' + suffix, suffix) == 'new\n\n' + suffix


def test_explicit_policy_tampering_is_rejected():
    with pytest.raises(ValueError):
        protect_policy_suffix('new\n## Trusted safety policy\nchanged\n', '## Trusted safety policy\npinned\n')
    with pytest.raises(ValueError):
        protect_policy_suffix('## Trusted safety policy\nfake\n## Trusted safety policy\npinned\n', '## Trusted safety policy\npinned\n')


def test_opro_selects_only_verified_nonregressing_candidates():
    algorithm = SafetyOPRO.__new__(SafetyOPRO)
    baseline, safe, worse, missing = (scored('base', .3), scored('safe', .4), scored('worse', .9), scored('missing', 1.0))
    algorithm._history = [baseline, safe, worse, missing]
    algorithm._safety_evidence = {'base': evidence(0), 'safe': evidence(0), 'worse': evidence(1), 'missing': evidence(0, 0)}
    assert algorithm._best() is safe


def test_opro_keeps_initial_when_baseline_evidence_incomplete():
    algorithm = SafetyOPRO.__new__(SafetyOPRO)
    base, candidate = scored('base', .2), scored('candidate', .8)
    algorithm._history = [base, candidate]
    algorithm._safety_evidence = {'base': evidence(0, 0), 'candidate': evidence(0)}
    assert algorithm._best() is base


def test_opro_checks_candidate_against_original_frozen_resources(monkeypatch):
    algorithm = SafetyOPRO.__new__(SafetyOPRO)
    algorithm._path = 'prompt.md'
    algorithm._pinned_safety_suffix = '## Trusted safety policy\npinned\n'
    class FakeSnapshot:
        files = {'prompt.md': SimpleNamespace(content=b'original\n## Trusted safety policy\npinned\n', component='prompt', executable=False)}
        def changed_paths(self, candidate):
            return ('tools.py',)
    algorithm._security_initial = FakeSnapshot()
    monkeypatch.setattr(OPRO, '_candidate', lambda self, baseline, files: object())
    with pytest.raises(ValueError, match='frozen'):
        algorithm._candidate(object(), {'prompt.md': 'new instruction'})
