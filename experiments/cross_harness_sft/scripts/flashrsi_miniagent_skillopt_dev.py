"""Run paired skill-enabled MiniAgent baseline and SkillOpt development comparison."""
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
from flashrsi.agents.llm import LLM
from flashrsi.agents.miniagent import MiniAgent
from flashrsi.algorithms.skillopt import SkillOpt
from flashrsi_gepa_security import AgentDojoSecurityBenchmark, ROOT
from flashrsi_opro_pilot import prepare_probe_directory
from flashrsi_skill_miniagent import make_skill_suites, SuiteAwareSkillMiniAgent, SKILL_PATH


async def main():
    if os.environ.get('FLASHRSI_COMPLETION_BUDGET') != '4096': raise ValueError('expected 4096 output budget')
    run=ROOT/'outputs/flashrsi_local_20261112/uniform4096_serial/miniagent_skillopt_dev'
    prepare_probe_directory(run)
    benchmark=AgentDojoSecurityBenchmark({},run_dir=run);splits=benchmark.load()
    families=sorted({t.metadata['family'] for t in splits['train']})[:2]
    train=tuple(t for t in splits['train'] if t.metadata['family'] in families);dev=tuple(splits['dev'])
    if len(families)!=2 or not train or not dev or {t.metadata['family'] for t in train}&{t.metadata['family'] for t in dev}:
        raise ValueError('invalid family split')
    benchmark._splits={'train':train,'dev':dev,'test':()}
    suites=make_skill_suites(ROOT/'outputs/flashrsi_local_20261112/minia_validation_16k/snapshots')
    for suite,snapshot in suites.items(): snapshot.save(run/'snapshots'/f'initial-{suite}.json')
    config={'source_dir':str(Path(importlib.import_module(MiniAgent.__module__).__file__).parent),
            'llm':{'model_name':'qwen35-4b-preflight','api_key':'environment-episode-runner-only',
                   'base_url':'http://127.0.0.1:18761/v1','max_tokens':4096,
                   'extra_body':{'chat_template_kwargs':{'enable_thinking':False},'temperature':0}},
            'max_turns':12,'timeout':330}
    target=SuiteAwareSkillMiniAgent(config,suites);target.bind_benchmark(benchmark)
    token=(ROOT/'outputs/preflight_20261111/vllm_api_token').read_text().strip()
    worker=LLM({'type':'llm','llm':{'model_name':'qwen35-4b-preflight','api_key':token,
        'base_url':'http://127.0.0.1:18761/v1','max_tokens':1024,'temperature':0.2,
        'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}},'timeout':300,'max_retries':1})
    algorithm=SkillOpt({'num_epochs':1,'batch_size':2,'reflection_minibatch_size':1,'edit_budget':1,
        'merge_batch_size':2,'max_steps':1,'use_slow_update':False,'use_meta_skill':False,'seed':42},
        target=target,worker=worker,benchmark=benchmark)
    algorithm._run_dir=run;initial=target.snapshot.id
    (run/'run_manifest.json').write_text(json.dumps({'status':'development_only_not_formally_qualified',
        'algorithm':'skillopt','harness':'miniagent','initial_condition':'one enabled generic skill; paired baseline required',
        'editable_skill':SKILL_PATH,'train_families':families,'train_task_ids':[t.task_id for t in train],
        'validation_task_ids':[t.task_id for t in dev],'test_used':False,'initial_snapshot_id':initial,
        'target_completion_budget':4096,'worker_completion_budget':1024,'parallel_tool_calls':False,
        'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    try:
        selected=await algorithm.evolve();selected.save(run/'snapshots'/'selected.json')
        evaluated=await algorithm._evaluate(selected,dev,label='selected_validation')
        rows=[{'task_id':x.task.task_id,'arm':x.task.metadata['arm'],'status':x.status,'metrics':dict(x.metrics)} for x in evaluated]
        (run/'result.json').write_text(json.dumps({'status':'finished','initial_snapshot_id':initial,
            'selected_snapshot_id':selected.id,'selected_changed':selected.id!=initial,'validation':rows},indent=2)+'\n')
    except BaseException as error:
        (run/'error.json').write_text(json.dumps({'status':'error','error_type':type(error).__name__,'message':str(error)[:1000]},indent=2)+'\n');raise


if __name__=='__main__': asyncio.run(main())
