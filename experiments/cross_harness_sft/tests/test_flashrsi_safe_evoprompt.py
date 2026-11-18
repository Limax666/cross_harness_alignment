from types import SimpleNamespace

import pytest
from flashrsi.algorithms.evoprompt import EvoPrompt
from flashrsi_safe_evoprompt import SafetyEvoPrompt


def test_evoprompt_restores_fixed_policy_and_refuses_tool_edits(monkeypatch):
    algorithm=SafetyEvoPrompt.__new__(SafetyEvoPrompt)
    algorithm._path='prompt.md'
    algorithm._pinned_safety_suffix='## Trusted safety policy\npinned\n'
    class Baseline:
        files={'prompt.md':SimpleNamespace(content=b'original\n## Trusted safety policy\npinned\n',component='prompt',executable=False)}
        def changed_paths(self,candidate): return ('tools.py',)
    algorithm._security_initial=Baseline()
    seen=[]
    monkeypatch.setattr(EvoPrompt,'_candidate',lambda self,base,files: seen.append(files) or object())
    with pytest.raises(ValueError,match='frozen'):
        algorithm._candidate(object(),{'prompt.md':'new text'})
    assert seen[0]['prompt.md'].endswith(algorithm._pinned_safety_suffix)


def test_evoprompt_rejects_changed_policy():
    algorithm=SafetyEvoPrompt.__new__(SafetyEvoPrompt)
    algorithm._path='prompt.md'
    algorithm._pinned_safety_suffix='## Trusted safety policy\npinned\n'
    baseline=object()
    assert algorithm._candidate(baseline,{'prompt.md':'## Trusted safety policy\nchanged\n'}) is baseline
