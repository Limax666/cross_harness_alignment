import pytest
from rl_whitebox_validation import validate_config_partition

def test_whitebox_validation_accepts_fixed_disjoint_heldout():
 train=['wb_S0T0O0C0W0'];holdout=['wb_S0T1O0C0W0']
 validate_config_partition(train,holdout)
 with pytest.raises(ValueError):validate_config_partition(train,train)
 with pytest.raises(ValueError):validate_config_partition(train,[])
 with pytest.raises(ValueError):validate_config_partition(train,['wrong'])
