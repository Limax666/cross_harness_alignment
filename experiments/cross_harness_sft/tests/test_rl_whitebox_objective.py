import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rl_whitebox_objective import constrain, FixedLedgerSampler
from rl_paired_assignment import AssignmentCursor
from rl_whitebox_harness import build_whitebox_ledger
from rl_whitebox_harness import TRAIN_CONFIGS, config_name

def cells():
    return [dict(benchmark="agentdojo", source_family_id=f"f{f}",arm=a,harness_name=config_name(c),split="train",task_id=f"{f}/{a}",cell_id=f"{f}/{a}/{config_name(c)}",task_semantic_sha256="a"*64,attack_payload_sha256="b"*64,initial_state_sha256="c"*64,fixture_revision="v1",semantic_fixture_seed=0) for f in range(15) for a in ("benign","risk") for c in TRAIN_CONFIGS]


def item(arm='benign',**kw):
    row=dict(benchmark='agentdojo',harness='wb',arm=arm,official_attack_success=False,
             prohibited_persistence=False,invalid_tool_dispatched=False,termination='completed',
             blocked_tool_calls=0,reward_outcome='safe_incomplete')
    row.update(kw)
    return row


def test_blocked_attempt_not_dual_cost_and_unsafe_never_positive():
    rows=[item(),item(),item('risk'),item('risk')]
    rows[0]['blocked_tool_calls']=1
    rows[1]['reward_outcome']='benign_nonengagement'
    rows[2]['official_attack_success']=True
    adv=torch.tensor([[.2],[-.1],[1.],[0.]])
    out,duals,stats=constrain(adv,rows,{})
    assert out[0].item()==pytest.approx(.2)
    assert out[1].item()==pytest.approx(-.4)
    assert out[2].item()<0
    assert stats['agentdojo/wb']['blocked_attempt']==.25
    assert duals['agentdojo/wb']['attack']>.25
    assert duals['agentdojo/wb']['persistence']==.5


def test_nonfinite_advantages_fail_closed():
    with pytest.raises(ValueError):constrain(torch.tensor([[float('nan')]]),[item()],{})


def test_sampler_repeats_until_cursor_commit_and_resumes_exactly():
    rows=cells()
    cursor=AssignmentCursor(build_whitebox_ledger(rows,iterations=4))
    sampler=FixedLedgerSampler(rows,cursor)
    first=list(sampler)
    assert len(first)==24 and list(sampler)==first
    cursor.commit(optimizer_updated=False)
    assert list(sampler)!=first and cursor.optimizer_updates==0
    state=sampler.state_dict()
    other=FixedLedgerSampler(rows,AssignmentCursor(cursor.ledger))
    other.load_state_dict(state)
    assert list(other)==list(sampler)


def test_kl_only_requires_enabled_positive_coefficient_and_observed_difference():
    from rl_whitebox_objective import has_kl_signal
    old=torch.tensor([[-.4,-.2]])
    mask=torch.tensor([[1,0]])
    assert not has_kl_signal(old,old,mask,enabled=True,coefficient=.001)
    assert not has_kl_signal(old,old-.1,mask,enabled=True,coefficient=0)
    assert has_kl_signal(old,old-.1,mask,enabled=True,coefficient=.001)
    changed=old.clone();changed[0,1]-=1
    assert not has_kl_signal(old,changed,mask,enabled=True,coefficient=.001)


def test_path_audit_abstention_is_unknown_cost_and_cannot_receive_positive_advantage():
    rows=[item('risk',audit_complete=False,prohibited_persistence=None,
               invalid_tool_dispatched=None,reward=0.0,reward_reason='path_audit_abstained'),
          item('risk',audit_complete=True,prohibited_persistence=False,
               invalid_tool_dispatched=False,reward=0.0,reward_reason='goal_not_verified'),
          item('benign',audit_complete=True)]
    advantages=torch.tensor([[.7],[.2],[.1]])
    out,duals,stats=constrain(advantages,rows,{})
    assert out[0].item()==pytest.approx(0.0)
    assert stats['agentdojo/wb']['audit_abstention']==pytest.approx(1/3)
    assert stats['agentdojo/wb']['persistence']==pytest.approx(0.0)
    assert duals['agentdojo/wb']['persistence']==pytest.approx(.5)


def test_all_unknown_path_audits_do_not_fabricate_safe_cost_rate():
    rows=[item('risk',audit_complete=False,prohibited_persistence=None,
               invalid_tool_dispatched=None,reward=0.0,reward_reason='path_audit_abstained'),
          item('benign',audit_complete=False,prohibited_persistence=None,
               invalid_tool_dispatched=None,reward=0.0,reward_reason='path_audit_abstained')]
    out,duals,stats=constrain(torch.tensor([[.4],[.3]]),rows,{})
    assert stats['agentdojo/wb']['persistence'] is None
    assert stats['agentdojo/wb']['protocol'] is None
    assert duals['agentdojo/wb']['persistence']==pytest.approx(.5)
    assert duals['agentdojo/wb']['protocol']==pytest.approx(.1)
    assert out[0].item()==pytest.approx(0.0)


def test_stateful_dataloader_follows_next_ledger_phase_per_epoch():
    from torchdata.stateful_dataloader import StatefulDataLoader
    rows=cells()
    cursor=AssignmentCursor(build_whitebox_ledger(rows,iterations=4))
    sampler=FixedLedgerSampler(rows,cursor)
    loader=StatefulDataLoader(list(range(len(rows))),batch_size=24,sampler=sampler,num_workers=0)
    first=next(iter(loader)).tolist()
    cursor.commit(optimizer_updated=True)
    # Drain the old iterator exactly like VeRL's one-batch epoch does.
    # A new iterator must use the changed cursor rather than stale indices.
    assert len(first)==24
    second=list(iter(loader))
    assert len(second)==1 and second[0].tolist()!=first


def test_progress_constraint_does_not_add_legacy_nonengagement_penalty():
    rows=[item('benign',reward_outcome='benign_nonengagement'),item('risk')]
    adv=torch.tensor([[0.2],[0.0]])
    out,_,_=constrain(adv,rows,{},penalize_nonengagement=False)
    assert out[0].item()==pytest.approx(0.2)
    old,_,_=constrain(adv,rows,{},penalize_nonengagement=True)
    assert old[0].item()==pytest.approx(-0.3)


def test_progress_constraint_counts_verified_schema_errors_without_changing_legacy():
    rows=[item('benign',protocol_error_verified=True),item('risk',protocol_error_verified=False)]
    adv=torch.tensor([[0.2],[0.0]])
    out,_,rates=constrain(adv,rows,{},penalize_nonengagement=False,use_verified_protocol=True)
    assert rates['agentdojo/wb']['protocol']==pytest.approx(.5)
    assert out[0].item()<0
    old,_,old_rates=constrain(adv,rows,{})
    assert old_rates['agentdojo/wb']['protocol']==0 and old[0].item()==pytest.approx(.2)
