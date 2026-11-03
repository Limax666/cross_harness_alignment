import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts/audit_whitebox_baseline.py'
s=importlib.util.spec_from_file_location('audit_whitebox_baseline',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def rows():
    return [{'evidence':{'split':split,'policy_snapshot':'step:0','harness':'wb_test','arm':arm,
      'source_family_id':'agentdojo:'+split,'valid_tool_dispatches':1,'termination':'completed',
      'reset_verified':True,'final_state_verified':True,'audit_complete':True,'official_utility':1.0}}
      for split in ('train','validation') for arm in ('benign','risk') for _ in range(4)]

def test_gate_requires_validation_and_rejects_policy_drift():
    x=rows(); x[-1]['evidence']['policy_snapshot']='step:1'
    assert any('policy' in s for s in m.audit(x,0.5,0.3)['failures'])
    assert not m.audit([r for r in x if r['evidence']['split']=='train'],0.5,0.3)['passed']

def test_gate_rejects_incomplete_and_allows_qualified():
    x=rows(); assert m.audit(x,0.5,0.3)['passed']
    for r in x[-4:]: r['evidence'].update(valid_tool_dispatches=0,termination='model_timeout')
    assert not m.audit(x,0.5,0.3)['passed']
    assert not m.audit(rows()[:1],0.5,0.3)['passed']
