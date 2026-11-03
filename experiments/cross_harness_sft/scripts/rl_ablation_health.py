"""Conservative engineering-pilot stop guard; thresholds are not claims of divergence."""
from math import isfinite

def check_pilot_health(metrics,variable_groups,groups):
 reasons=[]
 if groups<=0 or not 0<=variable_groups<=groups:raise ValueError('invalid group counts')
 entropy=metrics.get('actor/entropy_loss');kl=metrics.get('actor/kl_loss')
 if any(type(v) not in (float,int) or not isfinite(v) for v in (entropy,kl)):
  return ['missing_metric']
 if entropy>3.0:reasons.append('entropy')
 if kl>0.1:reasons.append('reference_kl')
 if variable_groups/groups <= 0.05:reasons.append('zero_variance')
 return reasons
