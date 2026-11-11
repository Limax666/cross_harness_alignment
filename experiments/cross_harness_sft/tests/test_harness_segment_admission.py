import pytest
from rl_harness_segments import Recorder, expand_rollouts
from rl_harness_segment_admission import admit_segmented_rollouts


def fixture():
    cell={'benchmark':'agentdojo','source_family_id':'f','task_id':'t','arm':'benign','harness_name':'h'}
    rows=[]; rollouts=[]
    for i in range(4):
        r=Recorder(str(i),'group',float(i % 2)); r.generated([10],[20,21],[-.2,-.3])
        if i == 0: r.generated([80],[22],[-.4])
        rollouts.append(r)
        rows.append(dict(rollout_id=str(i),cell_id='cell',split='train',benchmark='agentdojo',source_family_id='f',task_id='t',arm='benign',harness='h',on_policy_rollout=True,policy_snapshot='step:1',reset_verified=True,harness_contract_verified=True,reward=float(i % 2),audit_complete=True,reward_path_signal=0.,seed=i,initial_state_sha256='a'*64,fixture_seed=1,task_semantic_sha256='b'*64,attack_payload_sha256='c'*64))
    assignment={'cell_id':'cell','initial_state_sha256':'a'*64,'fixture_seed':1,'task_semantic_sha256':'b'*64,'attack_payload_sha256':'c'*64}
    return rollouts,rows,{'cell':cell},[assignment]


def test_admission_preserves_one_rollout_per_group_member_after_expansion():
    rollouts, evidence, cells, assignments=fixture(); segments=expand_rollouts(rollouts)
    old=[list(s.logprobs) + [0.] * (2-len(s.tokens)) for s in segments]
    report=admit_segmented_rollouts(rollouts,evidence,segments,old,policy_snapshot='step:1',cells=cells,assignments=assignments)
    assert report['rollouts']==4 and report['groups']==1 and len(segments)==5


def test_admission_rejects_reordered_missing_or_forged_segments():
    rollouts,evidence,cells,assignments=fixture(); segments=expand_rollouts(rollouts)
    old=[list(s.logprobs)+[0.]*(2-len(s.tokens)) for s in segments]
    with pytest.raises(ValueError):
        admit_segmented_rollouts(rollouts,evidence,segments[1:],old[1:],policy_snapshot='step:1',cells=cells,assignments=assignments)
    evidence[0]['reward']=1.
    with pytest.raises(ValueError):
        admit_segmented_rollouts(rollouts,evidence,segments,old,policy_snapshot='step:1',cells=cells,assignments=assignments)
