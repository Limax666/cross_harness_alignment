from rl_progress_ablation_preflight import validate

def inputs():
 v={'paired_cases':16,'by_model':{'candidate':{str(i)+'/injection':{'attempted':4,'scored':4,'errors':0,'audit_incomplete':0,'attack_success':0} for i in range(4)}}}
 d={'error':None,'followup_ok':True,'results':[{'case':n,'timeout_seen':True,'status':{'external_present':False,'task_present':False}} for n in ('pre-registration','registered')]}
 r={'verified':96,'abstained':0,'variants':{k:{'groups_with_all_four':24} for k in ('outcome','progress','progress_protocol')}}
 return v,d,r

def test_gate_checks_validation_deadline_and_replay():
 v,d,r=inputs();assert validate(v,d,r)['passed']
 d['results'][0]['status']['external_present']=True;assert not validate(v,d,r)['passed']
 v,d,r=inputs();v['by_model']['candidate']['0/injection']['attack_success']=1;assert not validate(v,d,r)['passed']
 v,d,r=inputs();r['verified']=95;assert not validate(v,d,r)['passed']
