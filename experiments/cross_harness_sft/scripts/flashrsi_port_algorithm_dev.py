"""Bounded OPRO or GEPA development comparison for a FlashRSI Python port."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from flashrsi.agents.llm import LLM

from flashrsi_gepa_security import ROOT
from flashrsi_miniagent_algorithm_dev import PrefixSafetyGEPA
from flashrsi_opro_pilot import PrefixOnlyWorker, prepare_probe_directory
from flashrsi_port_agentdojo import PORTS, PortBenchmark, build_port
from flashrsi_native_agentdojo import build_native
from flashrsi_safe_opro import SafetyOPRO
from flashrsi_safe_evoprompt import SafetyEvoPrompt
from flashrsi_safe_ace import SafetyACE


async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--harness',choices=(*PORTS,'codex','claude_code'),required=True)
    parser.add_argument('--algorithm',choices=('opro','gepa','evoprompt','ace'),required=True)
    args=parser.parse_args()
    if os.environ.get('FLASHRSI_COMPLETION_BUDGET') != '4096':
        raise ValueError('this development comparison requires 4096 output tokens')
    run=ROOT/'outputs/flashrsi_local_20261112/uniform4096_serial'/f'{args.harness}_{args.algorithm}_dev'
    prepare_probe_directory(run)
    benchmark=PortBenchmark({},run_dir=run,host=PORTS[args.harness][4] if args.harness in PORTS else 'flashrsi_native_host.py')
    splits=benchmark.load()
    families=sorted({t.metadata['family'] for t in splits['train']})[:2]
    train=tuple(t for t in splits['train'] if t.metadata['family'] in families)
    dev=tuple(splits['dev'])
    if len(families)!=2 or not train or not dev or {t.metadata['family'] for t in train}&{t.metadata['family'] for t in dev}:
        raise ValueError('invalid family split')
    benchmark._splits={'train':train,'dev':dev,'test':()}
    llm={'model_name':'qwen35-4b-preflight','api_key':'placeholder','base_url':'http://127.0.0.1:1/v1','max_tokens':4096}
    if args.harness=='pi': llm.update(provider_name='local',model_api='openai-completions')
    config={'llm':llm,'max_turns':12,'timeout':300}
    policy=(ROOT/'configs/flashrsi_b0_safety_policy.txt').read_text()
    target,suites=(build_port(args.harness,benchmark,config,policy) if args.harness in PORTS
                   else build_native(args.harness,benchmark))
    for name,snapshot in suites.items(): snapshot.save(run/'snapshots'/f'initial-{name}.json')
    token=(ROOT/'outputs/preflight_20261111/vllm_api_token').read_text().strip()
    worker_type=LLM if args.algorithm=='ace' else PrefixOnlyWorker
    worker=worker_type({'type':'llm','llm':{'model_name':'qwen35-4b-preflight','api_key':token,
        'base_url':'http://127.0.0.1:18761/v1','max_tokens':1024,'temperature':0.2,
        'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}},'timeout':300,'max_retries':1})
    if args.algorithm=='opro':
        algorithm=SafetyOPRO({'max_steps':1,'num_candidates':1,'history_size':2,'few_shot_size':1},target=target,worker=worker,benchmark=benchmark,safety_policy=policy)
    elif args.algorithm=='gepa':
        algorithm=PrefixSafetyGEPA({'num_epochs':1,'reflection_minibatch_size':3,'max_metric_calls':2*len(dev)+6,
            'skip_perfect_score':True,'acceptance_criterion':'strict_improvement','candidate_selection_strategy':'pareto',
            'max_evidence_chars':8000,'seed':42,'use_merge':False},target=target,worker=worker,benchmark=benchmark,safety_policy=policy)
    elif args.algorithm=='evoprompt':
        algorithm=SafetyEvoPrompt({'variant':'ga','population_size':2,'max_steps':1,'selection':'random',
            'survivor_selection':'topk','seed':42},target=target,worker=worker,benchmark=benchmark,safety_policy=policy)
    else:
        algorithm=SafetyACE({'mode':'offline','num_epochs':1,'batch_size':1,'max_steps':1,'edit_budget':1,
            'eval_steps':1,'max_reflection_rounds':1,'seed':42},target=target,worker=worker,benchmark=benchmark,safety_policy=policy)
    algorithm._run_dir=run
    initial=target.snapshot.id
    (run/'run_manifest.json').write_text(json.dumps({'status':'development_only_not_formally_qualified',
        'harness':args.harness,'algorithm':args.algorithm,'project_adaptation':'pinned policy, prompt-only scope; prefix-only worker except ACE',
        'train_families':families,'train_task_ids':[x.task_id for x in train],
        'validation_task_ids':[x.task_id for x in dev],'test_used':False,'initial_snapshot_id':initial,
        'target_completion_budget':4096,'worker_completion_budget':1024,'max_model_len':16384,'parallel_tool_calls':False,
        'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    try:
        selected=await algorithm.evolve(); selected.save(run/'snapshots'/'selected.json')
        evaluated=await algorithm._evaluate(selected,dev,label='selected_validation')
        rows=[{'task_id':x.task.task_id,'arm':x.task.metadata['arm'],'status':x.status,'metrics':dict(x.metrics)} for x in evaluated]
        (run/'result.json').write_text(json.dumps({'status':'finished','initial_snapshot_id':initial,
            'selected_snapshot_id':selected.id,'selected_changed':selected.id!=initial,'validation':rows},indent=2)+'\n')
    except BaseException as error:
        (run/'error.json').write_text(json.dumps({'status':'error','error_type':type(error).__name__,'message':str(error)[:1000]},indent=2)+'\n')
        raise


if __name__=='__main__': asyncio.run(main())
