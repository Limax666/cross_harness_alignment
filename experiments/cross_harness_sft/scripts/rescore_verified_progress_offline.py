#!/usr/bin/env python3
"""Replay archived evidence without inventing unrecorded process proofs."""
import argparse,collections,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rl_verifier import EpisodeEvidence,TaskKind
from rl_reward_verified_progress import score_verified_progress,verify_progress_reward_record

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 with gzip.open(a.source,'rt') as f:rows=[json.loads(l)['evidence'] for l in f if l.strip()]
 if not rows:raise ValueError('empty evidence')
 result={}
 for variant in ('outcome','progress','progress_protocol'):
  cells=collections.defaultdict(list);counts=collections.Counter();outputs=[]
  for r in rows:
   kind=TaskKind.BENIGN if r['arm']=='benign' else TaskKind.INJECTION
   e=EpisodeEvidence(harness=r['harness'],family_id=r['source_family_id'],task_kind=kind,
    termination=r['termination'],final_state_verified=r['final_state_verified'],path_verified=r['path_verified'],
    legitimate_goal_complete=bool(r['official_utility']) if r['official_utility'] is not None else None,
    attacker_goal_achieved=r['official_attack_success'],prohibited_persistence=r['prohibited_persistence'],
    harmful_action_executed=r['harmful_action_executed'],invalid_tool_dispatched=r['invalid_tool_dispatched'])
   # Archived trajectories do not contain independent tool-result evidence.
   q=None; protocol=r['termination']=='protocol_error'
   verdict=score_verified_progress(e,q,protocol,variant=variant)
   out=dict(r,reward_contract='verified_progress_v1',reward_variant=variant,progress_evidence=None,
    protocol_error_verified=protocol,reward=verdict.score,reward_outcome=verdict.outcome,
    reward_reason=verdict.reason,reward_path_signal=verdict.path_signal)
   if verdict.score is not None:verify_progress_reward_record(out);cells[r['cell_id']].append(verdict.score)
   counts[str(verdict.score)]+=1
   outputs.append({'rollout_id':r['rollout_id'],'cell_id':r['cell_id'],'score':verdict.score,'outcome':verdict.outcome,'progress_status':'unavailable_archived_tool_results','protocol_evidence':'termination_only' if protocol else 'unknown_historical_blocks'})
  result[variant]={'scores':dict(counts),'groups':len(cells),'variable_groups':sum(len(set(v))>1 for v in cells.values()),'groups_with_four_scored':sum(len(v)==4 for v in cells.values()),'records':outputs}
 payload={'source':str(a.source),'rows':len(rows),'warning':'Q unavailable on archived rows; schema-block costs can be undercounted; not a proxy for live full reward', 'variants':result}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 for v,d in result.items():print(v,'scores',d['scores'],'variable_groups',d['variable_groups'],'/',d['groups'])
if __name__=='__main__':main()
