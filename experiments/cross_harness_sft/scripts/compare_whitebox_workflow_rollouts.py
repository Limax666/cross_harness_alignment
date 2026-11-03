#!/usr/bin/env python3
"""Compare strict and soft workflow evidence by W factor and task arm."""
import argparse,collections,gzip,json
from pathlib import Path

def read(path):
 with gzip.open(path,'rt') as f:return [json.loads(line)['evidence'] for line in f if line.strip()]

def summarize(rows):
 groups=collections.defaultdict(list)
 for r in rows:
  name=r['harness']; w=name.rsplit('W',1)[-1]
  if w not in ('0','1','2'):raise ValueError('unknown workflow')
  groups[(w,r['arm'])].append(r)
 out={}
 for (w,arm),rs in sorted(groups.items()):
  cells=collections.defaultdict(list)
  for r in rs:cells[r['cell_id']].append(float(r['reward']))
  out[f'W{w}/{arm}']={'n':len(rs),'valid_tool_episodes':sum(r.get('valid_tool_dispatches',0)>0 for r in rs),'calls':sum(r.get('tool_calls',0) for r in rs),'blocked_calls':sum(r.get('blocked_tool_calls',0) for r in rs),'completed':sum(r['termination']=='completed' for r in rs),'model_timeout':sum(r['termination']=='model_timeout' for r in rs),'protocol_error':sum(r['termination']=='protocol_error' for r in rs),'utility':sum(bool(r['official_utility']) for r in rs),'attack_success':sum(bool(r['official_attack_success']) for r in rs),'variable_reward_groups':sum(len(set(v))>1 for v in cells.values()),'groups':len(cells)}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('strict',type=Path);p.add_argument('soft',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x,y=read(a.strict),read(a.soft)
 if len(x)!=96 or len(y)!=96:raise ValueError('each batch must contain 96 trajectories')
 if collections.Counter(r['cell_id'] for r in x)!=collections.Counter(r['cell_id'] for r in y):raise ValueError('batch assignments do not match')
 if any(r.get('workflow_plan_policy')!='soft' for r in y):raise ValueError('soft policy evidence missing')
 result={'strict':summarize(x),'soft':summarize(y),'note':'same cells, independent rollouts and seeds; descriptive engineering comparison; do not infer model safety improvement'}
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
