import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rl_multiharness_batch_gate import require_online_batch


def fixture():
    cells={};rows=[];expected=[]
    for arm in ('benign','risk'):
        cid='wb/'+arm
        cell=dict(cell_id=cid,benchmark='agentdojo',harness_name='wb_S0T0O0C0W0',
                  source_family_id='family',task_id='task:'+arm,arm=arm)
        cells[cid]=cell
        expected.append({**cell,'initial_state_sha256':'a'*64,'fixture_seed':73,
                         'attack_payload_sha256':'b'*64,'task_semantic_sha256':'c'*64})
        for seed in range(4):
            rows.append({**cell,'harness':cell['harness_name'],'rollout_id':f'{arm}/{seed}',
                'split':'train','on_policy_rollout':True,'policy_snapshot':'step:0',
                'reset_verified':True,'harness_contract_verified':True,'audit_complete':True,
                'reward':0.,'reward_path_signal':0.,'seed':seed,'initial_state_sha256':'a'*64,
                'fixture_seed':73,'attack_payload_sha256':'b'*64,'task_semantic_sha256':'c'*64,
                'response_ids':[1,2],'response_mask':[1,0],'token_origins':['assistant','context'],
                'rollout_log_probs':[-1.,0.],'old_log_probs':[-1.,0.]})
    return cells,rows,expected


def test_fixed_assignment_admits_all_constant_groups_without_resampling():
    cells,rows,expected=fixture()
    result=require_online_batch(rows,policy_snapshot='step:0',cells=cells,
        allow_zero_reward_variance=True,assignments=expected)
    assert result['groups']==2 and result['variable_reward_groups']==0


def test_assignment_mismatch_rejected_even_when_group_reset_is_consistent():
    cells,rows,expected=fixture()
    for row in rows:row['fixture_seed']=0
    with pytest.raises(ValueError):
        require_online_batch(rows,policy_snapshot='step:0',cells=cells,
            allow_zero_reward_variance=True,assignments=expected)


def test_officially_verified_safe_zero_with_path_abstention_is_admitted_without_positive_credit():
    cells,rows,expected=fixture()
    row=rows[0]
    row.update(audit_complete=False,final_state_verified=True,path_verified=False,
               official_attack_success=False,reward=0.,reward_path_signal=0.,
               reward_reason='path_audit_abstained')
    result=require_online_batch(rows,policy_snapshot='step:0',cells=cells,
        allow_zero_reward_variance=True,assignments=expected)
    assert result['audit_abstained_rollouts']==1


@pytest.mark.parametrize('changes',[
    {'reward':1.,'reward_reason':'verified_goal'},
    {'official_attack_success':None},
    {'final_state_verified':False},
])
def test_incomplete_audit_without_explicit_safe_zero_is_rejected(changes):
    cells,rows,expected=fixture()
    rows[0].update(audit_complete=False,final_state_verified=True,path_verified=False,
                   official_attack_success=False,reward=0.,reward_path_signal=0.,
                   reward_reason='path_audit_abstained')
    rows[0].update(changes)
    with pytest.raises(ValueError,match='audit'):
        require_online_batch(rows,policy_snapshot='step:0',cells=cells,
            allow_zero_reward_variance=True,assignments=expected)


def test_verified_progress_cannot_mix_variants_or_reward_forgery():
    cells,rows,expected=fixture()
    for row in rows:
        row.update(reward_contract='verified_progress_v1',reward_variant='outcome',reward_outcome='safe_incomplete',
            reward_reason='verified_progress_v1',progress_evidence=None,protocol_error_verified=False,
            termination='completed',official_attack_success=False,official_utility=0.,final_state_verified=True,
            prohibited_persistence=False,harmful_action_executed=False,invalid_tool_dispatched=False)
    require_online_batch(rows,policy_snapshot='step:0',cells=cells,allow_zero_reward_variance=True,assignments=expected)
    rows[0]['reward_variant']='progress'
    with pytest.raises(ValueError,match='mixed reward'):require_online_batch(rows,policy_snapshot='step:0',cells=cells,allow_zero_reward_variance=True,assignments=expected)
    rows[0]['reward_variant']='outcome';rows[0]['reward']=1.
    with pytest.raises(ValueError,match='verified progress reward rejected'):require_online_batch(rows,policy_snapshot='step:0',cells=cells,allow_zero_reward_variance=True,assignments=expected)
