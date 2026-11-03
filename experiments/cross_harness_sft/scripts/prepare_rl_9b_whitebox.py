#!/usr/bin/env python3
"""Freeze audited AgentDojo train tasks and a K=2 whitebox assignment ledger."""
from __future__ import annotations
import hashlib
import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rl_paired_assignment import digest
from rl_whitebox_harness import TRAIN_CONFIGS,build_whitebox_ledger,config_name,system_context,transform_tools


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--iterations',type=int,default=20)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import pandas as pd
    from transformers import AutoTokenizer
    from cross_harness_sft.backends.agentdojo import AgentDojoDriver
    from rl_native_agentdojo_episode import _digest
    from rl_agentdojo_path_audit import GENERALIZED_FAMILIES,READ_TOOLS,_ground_truth_tools
    from rl_multiharness_chat_template import BENCHMARK_MCP_CHAT_TEMPLATE
    release=ROOT/'data/sft/teacher_release_v7/family_split_manifest.jsonl'
    allow={r['source_family_id'] for line in release.read_text().splitlines() if line.strip()
           for r in [json.loads(line)] if r['benchmark_namespace']=='agentdojo' and r['split']=='train'}
    old=pd.read_parquet(ROOT/'outputs/rl/multiharness_agentdojo_agentharm_pool_v5.parquet')
    arms=defaultdict(set)
    for r in old.to_dict('records'):
        if r['benchmark']=='agentdojo':arms[r['source_family_id']].add(r['arm'])
    families=sorted(f for f,a in arms.items() if a=={'benign','risk'} and f in allow)
    driver=AgentDojoDriver(str(ROOT/'configs/agentdojo_worker.yaml'))
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tok.chat_template=BENCHMARK_MCP_CHAT_TEMPLATE
    rows=[];excluded=[]
    for family in families:
        taskroot=family.removeprefix('agentdojo:')
        pending=[]
        for arm,suffix in [('benign','clean'),('risk','injection_task_1')]:
            task=taskroot+':'+suffix
            try:
                started=driver.start(task,0);ep=started['episode']
                plan=_ground_truth_tools(ep)
                if plan is None or not (taskroot in GENERALIZED_FAMILIES or plan<=READ_TOOLS):
                    raise ValueError('path audit not qualified')
                state=_digest(ep.environment)
                again=driver.start(task,0)['episode']
                if _digest(again.environment)!=state:
                    raise ValueError('independent reset mismatch')
                attack={}
                if ep.injection_task is not None:
                    from agentdojo.attacks.baseline_attacks import DirectAttack
                    attack=DirectAttack(ep.suite,target_pipeline=None).attack(ep.user_task,ep.injection_task)
                for config in TRAIN_CONFIGS:
                    name=config_name(config);tools,_=transform_tools(started['tools'],config)
                    messages=[{'role':'system','content':system_context(config)},
                              {'role':'user','content':ep.user_task.PROMPT}]
                    ids=tok.apply_chat_template(messages,tools=tools,tokenize=True,add_generation_prompt=True)
                    if isinstance(ids,Mapping):ids=ids['input_ids']
                    if hasattr(ids,'tolist'):ids=ids.tolist()
                    if ids and isinstance(ids[0],list):ids=ids[0]
                    if len(ids)>8192:raise ValueError(f'full schema exceeds prompt budget: {len(ids)}')
                    pending.append({'benchmark':'agentdojo','harness_name':name,'harness_config':list(config),
                        'source_family_id':family,'task_id':task,'cell_id':f'agentdojo/{name}/{task}',
                        'arm':arm,'split':'train','policy_snapshot':'step:0','prompt':messages,
                        'prompt_tokens':len(ids),'tool_contract_sha256':hashlib.sha256(json.dumps(tools,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                        'task_semantic_sha256':digest({'task':task,'prompt':ep.user_task.PROMPT,'tools':started['tools']}),
                        'attack_payload_sha256':digest(attack),'initial_state_sha256':state,
                        'semantic_fixture_seed':0,'fixture_revision':driver.version,
                        'agent_name':'whitebox_agentdojo_v1','data_source':'agentdojo_whitebox_train'})
            except Exception as e:
                excluded.append({'family':family,'task':task,'reason':str(e)})
                pending=[];break
        rows.extend(pending)
    ledger=build_whitebox_ledger(rows,iterations=args.iterations)
    args.output.mkdir(parents=True)
    path=args.output/'train.parquet';pd.DataFrame(rows).to_parquet(path,index=False)
    manifest={'status':'whitebox_engineering_pool_not_native_cli','output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'family_manifest_sha256':hashlib.sha256(release.read_bytes()).hexdigest(),
              'qualified_families':len({r['source_family_id'] for r in rows}),'cells':len(rows),
              'configurations':len(TRAIN_CONFIGS),'context_compaction':False,'excluded':excluded,
              'sft_9b_actual_view':'unknown; conservative existing v7 train-family allowlist',
              'rl_heldout_products':['codex','claude_code','hermes','nanobot','openclaw'],
              'max_prompt_tokens':max(r['prompt_tokens'] for r in rows)}
    (args.output/'train.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (args.output/'assignment_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
