import pytest
from rl_progress_registry import DEFAULT,resolve

def test_oracle_registry_is_explicit_and_versioned():
    assert resolve(DEFAULT).VERSION==DEFAULT
    assert resolve('agentdojo-workspace-progress-v4-candidate').VERSION=='agentdojo-workspace-progress-v4-candidate'
    with pytest.raises(ValueError):resolve('latest')
    with pytest.raises(ValueError):resolve(None)
