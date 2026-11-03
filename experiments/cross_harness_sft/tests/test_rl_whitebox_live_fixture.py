"""Local integration: frozen rows must reproduce the exact runtime contract."""
from pathlib import Path
from functools import partial
import hashlib
import json
import sys
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from rl_native_agentdojo_episode import NativeAgentDojoEpisode
from rl_whitebox_harness import transform_tools


def test_frozen_pool_matches_live_state_and_schema():
    pool=ROOT/'outputs/rl/9b_whitebox_pool_v4/train.parquet'
    if not pool.exists():pytest.skip('local frozen fixtures required')
    rows=pd.read_parquet(pool).to_dict('records')
    driver=AgentDojoDriver(str(ROOT/'configs/agentdojo_worker.yaml'))
    # Every family/arm, all three alias levels, without any generated tokens.
    selected=[r for r in rows if r['harness_config'][1]==2]
    for row in selected:
        config=tuple(int(v) for v in row['harness_config'])
        episode=NativeAgentDojoEpisode(driver,row['task_id'],int(row['semantic_fixture_seed']),
            allowed_cells=frozenset(r['task_id'] for r in rows),
            wire_transform=partial(transform_tools,config=config),path_weight=0.)
        assert episode.initial_state_sha256==row['initial_state_sha256']
        schema=hashlib.sha256(json.dumps(episode.tools,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert schema==row['tool_contract_sha256']


def test_soft_plan_allows_real_audited_tool_after_recorded_model_call():
    from rl_whitebox_harness import plan_gate
    from rl_tool_attempt_audit import parse_strict_tool_turn
    pool=ROOT/'outputs/rl/9b_whitebox_pool_v4/train.parquet'
    diagnostic=ROOT/'outputs/rl/diagnose_step150_backends_20261030/rl_transformers.jsonl'
    if not pool.exists() or not diagnostic.exists():pytest.skip('frozen local diagnostic required')
    outputs=[json.loads(x) for x in diagnostic.read_text().splitlines()]
    sample=next(x for x in outputs if 'user_task_28:clean' in x['cell_id'])
    row=next(r for r in pd.read_parquet(pool).to_dict('records') if r['cell_id']==sample['cell_id'])
    config=tuple(int(v) for v in row['harness_config']);assert config[4]>0
    driver=AgentDojoDriver(str(ROOT/'configs/agentdojo_worker.yaml'))
    def episode():
        return NativeAgentDojoEpisode(driver,row['task_id'],int(row['semantic_fixture_seed']),allowed_cells=frozenset([row['task_id']]),wire_transform=partial(transform_tools,config=config),path_weight=0.)
    strict=episode(); soft=episode();raw=sample['raw_output'];_,calls=parse_strict_tool_turn(raw,soft.tools,allow_schema_errors=True)
    assert calls and all('blocked_reason' not in c for c in calls)
    needed,_=plan_gate(config,False,True,True);assert needed
    strict.turn(raw,[{**c,'blocked_reason':'Workflow requires a tool-free plan before dispatch.'} for c in calls])
    needed,done=plan_gate(config,False,True,False);assert not needed and done
    soft.turn(raw,calls)
    assert len(strict.episode.trace)==0 and len(soft.episode.trace)>0
    assert strict.recorder.records[0].blocked_reason is not None and soft.recorder.records[0].blocked_reason is None
