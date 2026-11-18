from types import SimpleNamespace
import pytest
from flashrsi.algorithms.ace import ACE, _MARKER
from flashrsi_safe_ace import SafetyACE


def test_ace_keeps_playbook_before_immutable_policy(monkeypatch):
    algorithm=SafetyACE.__new__(SafetyACE)
    algorithm._path='prompt.md'
    algorithm._pinned_safety_suffix='## Trusted safety policy\npinned\n'
    class Snapshot:
        files={'prompt.md':SimpleNamespace(content=b'new\n## Trusted safety policy\npinned\n',component='prompt',executable=False)}
        def changed_paths(self,candidate): return ('prompt.md',)
    algorithm._security_initial=Snapshot()
    seen=[]
    monkeypatch.setattr(ACE,'_candidate',lambda self,base,files: seen.append(files) or Snapshot())
    algorithm._candidate(object(),{'prompt.md':'original\n'+algorithm._pinned_safety_suffix+_MARKER+'lesson'})
    assert seen[0]['prompt.md'].endswith(algorithm._pinned_safety_suffix)
    assert seen[0]['prompt.md'].index('lesson') < seen[0]['prompt.md'].index('## Trusted safety policy')
    with pytest.raises(ValueError,match='policy'):
        algorithm._candidate(object(),{'prompt.md':'tampered'})
