from copy import deepcopy

import pytest

from a6000_vm.chspo_v2 import credit_partial, credit_safe_success
from hsrl_a6000_chspo import Audit


def sample(source, utility, cost):
    return dict(key=[source, "task"], stop_reason="completed", source_result=dict(
        utility=utility, c_source=cost, c_protocol=0, persistence=None))


def test_only_verified_clean_partial_credit():
    ep = sample("clawgym", .5, None)
    blocked = deepcopy(ep)
    blocked["source_result"]["c_protocol"] = 1
    incomplete = dict(ep, stop_reason="round_limit")
    unknown = Audit(False, None, False)
    audits = credit_partial([ep,ep,ep,blocked,incomplete,ep], [unknown]*5+[Audit(True,None,False)],
                            [.3,None,0,.3,.3,.3])
    assert audits == [Audit(False,False,True)] + [unknown]*4 + [Audit(True,None,False)]


def test_safe_success_floor_only_with_equally_useful_risky_alternative():
    group = [sample("agentdog",u,r) for u,r in [(1,0),(1,1),(1,1),(1,0),(0,1),(0,0),(None,1),(None,0)]]
    audits = [Audit(False,False,True)]*8
    prior = [0,-.4,-.4,1.2,-.4,0,-.4,0]
    result = credit_safe_success(group,audits,prior.copy(),.3)
    assert result == pytest.approx([.2,-.4,-.4,1.2,-.4,0,-.4,0])
    for row in group:
        if row["source_result"]["c_source"] == 1:
            row["source_result"]["utility"] = 0
    assert credit_safe_success(group,audits,prior.copy(),.3) == prior
