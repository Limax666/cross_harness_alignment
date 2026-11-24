import pytest

from hsrl_a6000_chspo import Audit, prepare_update
from hsrl_v3.components import config_name
from hsrl_v5.sampler import OA18


def rollout(source, task, utility, risk, index, *, protocol_error=False):
    profile = ('clawgym_official_rl_code_only_v1' if source == 'clawgym'
               else 'hsrl-v5-agentdog-registered-call-goal-v1')
    return dict(key=[source, task], configuration=config_name(OA18[0]), policy_version=0,
                valid=True, slot=index, source_result=dict(valid=True, done=True,
                applicability=dict(legal_utility=utility is not None, registered_call_goal=risk is not None,
                                   persistence=False, broad_judge_safety=False, actual_harm=False),
                metrics=dict(profile=profile), utility=utility, c_source=risk, c_protocol=int(protocol_error),
                persistence=None))


def batch():
    groups, episodes, audits = [], [], []
    for group in range(16):
        source = 'clawgym' if group < 8 else 'agentdog'
        task = str(group)
        groups.append(dict(source=source, id=task, G=8))
        for slot in range(8):
            episodes.append(rollout(source, task, float(slot > 3) if group < 8 else None,
                                    None if group < 8 else float(slot==7), len(episodes)))
            audits.append(Audit(protocol_executed=False, legal_nonparticipation=False, complete=True))
    return dict(step=0, configuration=OA18[0], G=8, groups=groups), episodes, audits


def test_masked_utility_and_stratified_duals():
    schedule, episodes, audits = batch()
    episodes[7]['source_result']['c_protocol'] = 1  # blocked / parse error
    result = prepare_update(schedule, episodes, audits, policy_version=0, duals={})
    assert result['advantages'][7] == pytest.approx(.5/(.5+1e-6))
    assert result['advantages'][64:71] == [0.] * 7
    assert result['advantages'][71] == pytest.approx(-.26)
    cfg = config_name(OA18[0])
    assert result['observations']['clawgym/'+cfg]['attack']['count'] == 0
    assert result['observations']['agentdog/'+cfg]['attack']['count'] == 64
    assert result['observations']['clawgym/'+cfg]['persistence']['count'] == 0
    assert result['next_duals']['clawgym/'+cfg]['protocol'] == pytest.approx(.092)


def test_confirmed_violation_nonparticipation_and_abstention():
    sched, rows, audits = batch()
    audits[4] = Audit(True, True, True)
    audits[5] = Audit(False, True, True)
    audits[6] = Audit(None, None, False)
    result = prepare_update(sched,rows,audits,policy_version=0,duals={})
    assert -.11 < result['advantages'][4] < -.09
    assert result['advantages'][5] == pytest.approx(-.3)
    assert result['advantages'][6] == 0


def test_reject_stale_duplicate_incomplete_or_unscored_without_mutation():
    sched, rows, audits = batch()
    duals = {}
    for modify in ('stale','duplicate','incomplete','unscored'):
        _, es, _ = batch()
        if modify == 'stale': es[-1]['policy_version'] = 1
        if modify == 'duplicate': es[-1]['slot'] -= 1
        if modify == 'incomplete': es.pop()
        if modify == 'unscored': es[0]['source_result']['utility'] = None
        with pytest.raises(ValueError):
            prepare_update(sched,es,audits,policy_version=0,duals=duals)
    assert duals == {}


def test_full_batch_segment_validation_precedes_backward():
    from types import SimpleNamespace
    import torch
    from hsrl_a6000_chspo import optimize_admitted_batch
    from hsrl_a6000_fullparam import HostAdamW
    from hsrl_v3.policy import LOGPROB_CONTRACT
    from hsrl_v5.reward import VERSION
    schedule, rows, audits = batch()
    actor = torch.nn.Linear(4,4)
    actor.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=8))
    opt = HostAdamW(actor.named_parameters())
    for ep in rows:
        ep.update(reward_contract=VERSION, segments=[dict(policy_version=0, prompt_ids=[1],
                  response_ids=[2], old_logprobs=[-1.], reference_logprobs=[-1.], action_mask=[True],
                  logprob_contract=LOGPROB_CONTRACT)])
    rows[-1]['segments'][0]['reference_logprobs'] = [float('nan')]
    with pytest.raises(ValueError, match='unscored action segment'):
        optimize_admitted_batch(actor,opt,schedule,rows,audits,policy_version=0,duals={},lr=1e-5)
    assert not opt.gradients and not opt.state


def test_optimizer_entrypoint_uses_constrained_advantages_and_returns_next_duals():
    """Numerical unit test on CPU, without an environment or training-budget charge."""
    from types import SimpleNamespace
    import torch
    from hsrl_a6000_chspo import optimize_admitted_batch
    from hsrl_a6000_fullparam import HostAdamW
    from hsrl_v3.policy import LOGPROB_CONTRACT
    from hsrl_v5.reward import VERSION

    class Language(torch.nn.Embedding):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=super().forward(input_ids))

    torch.manual_seed(1)
    actor = torch.nn.Module()
    actor.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=4))
    actor.model = torch.nn.Module()
    actor.model.language_model = Language(4, 3)
    actor.lm_head = torch.nn.Linear(3, 4, bias=False)
    opt = HostAdamW(actor.named_parameters())
    before = actor.lm_head.weight.detach().clone()
    schedule, rows, audits = batch()
    for ep in rows:
        ep.update(reward_contract=VERSION, segments=[dict(policy_version=0, prompt_ids=[1],
                  response_ids=[2], old_logprobs=[-1.5], reference_logprobs=[-1.5],
                  action_mask=[True], logprob_contract=LOGPROB_CONTRACT)])
    old_duals = {}
    result = optimize_admitted_batch(actor, opt, schedule, rows, audits, policy_version=0,
                                     duals=old_duals, lr=.01)
    assert result['action_tokens'] == 128 and result['optimizer']['gradient_parameters'] == 24
    assert result['advantages'][-1] < 0 and old_duals == {}
    assert not torch.equal(before, actor.lm_head.weight)
