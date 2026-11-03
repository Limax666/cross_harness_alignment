#!/usr/bin/env python3
"""Re-execute recorded turns in isolated AgentDojo fixtures and verify state equality."""
import argparse,gzip,json,os,sys
from pathlib import Path
from functools import partial
from collections import Counter,defaultdict
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'scripts'),str(root/'src')]
os.environ['VERL_QWEN_SCHEMA_STRING_PARAMS']='1'
from rl_whitebox_harness import transform_tools,plan_gate
from rl_tool_attempt_audit import parse_strict_tool_turn
from rl_verifier import EpisodeEvidence,TaskKind
from rl_reward_verified_progress import score_verified_progress,verify_progress_reward_record


def replay(original,row,driver,progress_oracle_version):
 from rl_native_agentdojo_episode import NativeAgentDojoEpisode
 config=tuple(int(v) for v in row['harness_config'])
 e=NativeAgentDojoEpisode(driver,row['task_id'],int(row['semantic_fixture_seed']),allowed_cells=frozenset([row['task_id']]),wire_transform=partial(transform_tools,config=config),path_weight=0.,reward_contract='verified_progress_v1',progress_oracle_version=progress_oracle_version)
 if e.initial_state_sha256!=original['initial_state_sha256']:raise ValueError('initial state mismatch')
 turns=original['raw_assistant_turns'];planning_done=False;termination=original['termination']
 for i,raw in enumerate(turns):
  last=i==len(turns)-1
  if last and termination=='model_timeout' and not any(raw.rstrip().endswith(t) for t in ('</tool_call>','<|im_end|>','<|endoftext|>')):
   e.reject_incomplete_turn(raw);continue
  try:content,calls=parse_strict_tool_turn(raw,e.tools,allow_schema_errors=True)
  except ValueError:
   if last and termination=='protocol_error':e.reject_protocol_turn(raw);continue
   raise
  needs_plan,planning_done=plan_gate(config,planning_done,bool(calls),original.get('workflow_plan_policy','strict')=='strict')
  if needs_plan and calls:calls=[{**c,'blocked_reason':'Workflow requires a tool-free plan before dispatch.'} for c in calls]
  final=content
  if not calls:
   for suffix in ('<|im_end|>','<|endoftext|>'):
    if final.endswith(suffix):final=final[:-len(suffix)].rstrip()
   final=final.rsplit('</think>',1)[-1].strip()
  e.turn(raw,calls,final_content=final if not calls else None)
  if needs_plan and not calls:planning_done=True
 result=e.finish(termination=termination)
 for key in ('initial_state_sha256','final_state_sha256','official_utility','official_attack_success','audit_complete','reset_verified','tool_calls','blocked_tool_calls','valid_tool_dispatches'):
  if result.get(key)!=original.get(key):raise ValueError(f'replay mismatch {key}: {result.get(key)!r} != {original.get(key)!r}')
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--max',type=int,default=0);p.add_argument('--pool',type=Path,default=root/'outputs/rl/9b_whitebox_pool_v4');p.add_argument('--progress-oracle-version',default='agentdojo-workspace-progress-v3');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import pandas as pd
 from cross_harness_sft.backends.agentdojo import AgentDojoDriver
 from rl_progress_registry import resolve
 progress_oracle_version=resolve(a.progress_oracle_version).VERSION
 rows={r['cell_id']:r for r in pd.read_parquet(a.pool/'train.parquet').to_dict('records')};driver=AgentDojoDriver(str(root/'configs/agentdojo_worker.yaml'))
 with gzip.open(a.source,'rt') as f:archive=[json.loads(l)['evidence'] for l in f if l.strip()]
 if a.max:archive=archive[:a.max]
 output=[];cells={v:defaultdict(list) for v in ('outcome','progress','progress_protocol')};counts={v:Counter() for v in cells}
 for r in archive:
  try:
   new=replay(r,rows[r['cell_id']],driver,progress_oracle_version)
   ee=EpisodeEvidence(harness=r['harness'],family_id=r['source_family_id'],task_kind=TaskKind.BENIGN if r['arm']=='benign' else TaskKind.INJECTION,termination=new['termination'],final_state_verified=new['final_state_verified'],path_verified=new['path_verified'],legitimate_goal_complete=bool(new['official_utility']) if new['official_utility'] is not None else None,attacker_goal_achieved=new['official_attack_success'],prohibited_persistence=new['prohibited_persistence'],harmful_action_executed=new['harmful_action_executed'],invalid_tool_dispatched=new['invalid_tool_dispatched'])
   values={};q=new['progress_evidence']['fraction'] if new['progress_evidence'] is not None else None
   for variant in cells:
    v=score_verified_progress(ee,q,new['protocol_error_verified'],variant=variant)
    merged={**r,**new,'reward_contract':'verified_progress_v1','reward_variant':variant,'reward':v.score,'reward_outcome':v.outcome,'reward_reason':v.reason,'reward_path_signal':v.path_signal}
    if v.score is not None:verify_progress_reward_record(merged);cells[variant][r['cell_id']].append(v.score)
    counts[variant][str(v.score)]+=1;values[variant]=v.score
   output.append({'rollout_id':r['rollout_id'],'cell_id':r['cell_id'],'status':'verified_replay','progress':new['progress_evidence'],'protocol_error_verified':new['protocol_error_verified'],'scores':values})
  except Exception as exc:
   output.append({'rollout_id':r['rollout_id'],'cell_id':r['cell_id'],'status':'abstained','error':repr(exc)})
 result={'source':str(a.source),'pool':str(a.pool),'progress_oracle_version':progress_oracle_version,'rows':len(archive),'verified':sum(x['status']=='verified_replay' for x in output),'abstained':sum(x['status']=='abstained' for x in output),'variants':{v:{'scores':dict(counts[v]),'groups_with_all_four':sum(len(s)==4 for s in cells[v].values()),'variable_groups_with_all_four':sum(len(s)==4 and len(set(s))>1 for s in cells[v].values())} for v in cells},'records':output}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
