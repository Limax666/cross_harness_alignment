import pytest
from rl_ablation_health import check_pilot_health

def test_pilot_health_stops_high_entropy_and_invalid_signal():
 assert check_pilot_health({'actor/entropy_loss':1.2,'actor/kl_loss':0.01},4,24)==[]
 assert 'entropy' in check_pilot_health({'actor/entropy_loss':4.0,'actor/kl_loss':0.01},4,24)
 assert 'zero_variance' in check_pilot_health({'actor/entropy_loss':1.0,'actor/kl_loss':0.01},1,24)
 assert 'reference_kl' in check_pilot_health({'actor/entropy_loss':1.0,'actor/kl_loss':0.2},4,24)
 assert 'missing_metric' in check_pilot_health({},4,24)
