import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/verl'))
from verl.utils.reference_placement import should_cpu_offload


def test_legacy_reference_offload_is_preserved(monkeypatch):
    monkeypatch.delenv('VERL_REFERENCE_KEEP_ON_GPU',raising=False)
    assert should_cpu_offload(requested=False,forward_only=True)


def test_optin_keeps_frozen_reference_on_gpu_without_disabling_actor_cpu_policy(monkeypatch):
    monkeypatch.setenv('VERL_REFERENCE_KEEP_ON_GPU','1')
    assert not should_cpu_offload(requested=False,forward_only=True)
    assert should_cpu_offload(requested=True,forward_only=False)
    assert should_cpu_offload(requested=True,forward_only=True)
    assert not should_cpu_offload(requested=False,forward_only=False)
