import pytest
from rl_reward_verified_progress import verify_resume_reward_profile

def test_resume_rejects_unrecorded_or_changed_progress_reward():
 expected={'contract':'verified_progress_v1','variant':'progress'}
 verify_resume_reward_profile('GRPO-progress-v1',expected,expected)
 with pytest.raises(ValueError):verify_resume_reward_profile('GRPO-progress-v1',expected,None)
 with pytest.raises(ValueError):verify_resume_reward_profile('GRPO-progress-v1',expected,{'contract':'verified_progress_v1','variant':'outcome'})
 verify_resume_reward_profile('CHS-PO-whitebox-v1',expected,None)
