from rl_whitebox_validation import compare_paired

def row(task,config,seed,status='scored',utility=0,attack=False):
 return {'task_id':task,'configuration':config,'seed':seed,'status':status,'partition':'trained_config','official_utility':utility,'official_attack_success':attack,'valid_tool_dispatches':1,'termination':'completed','audit_complete':True}

def test_paired_summary_keeps_errors_in_denominator():
 a=[row('task:clean','h',0),row('task:injection_task_1','h',0,status='error')];b=[row('task:clean','h',0,utility=1),row('task:injection_task_1','h',0)]
 x=compare_paired(a,b);assert x['paired_cases']==2
 assert x['by_model']['sft']['h/injection']['attempted']==1 and x['by_model']['sft']['h/injection']['scored']==0
 assert x['by_model']['candidate']['h/clean']['utility_success']==1

def test_paired_summary_rejects_missing_or_duplicate():
 import pytest
 x=row('task:clean','h',0)
 with pytest.raises(ValueError):compare_paired([x],[x,x])
 with pytest.raises(ValueError):compare_paired([x],[])
