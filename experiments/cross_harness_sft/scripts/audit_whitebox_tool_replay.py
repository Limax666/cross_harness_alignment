#!/usr/bin/env python3
"""Reparse recorded whitebox turns against their original live schema, read-only."""
import argparse,collections,gzip,json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'scripts'),str(root/'src')]
os.environ['VERL_QWEN_SCHEMA_STRING_PARAMS']='1'
from rl_tool_attempt_audit import parse_strict_tool_turn
from rl_whitebox_harness import transform_tools

def main():
 p=argparse.ArgumentParser();p.add_argument('rollouts',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 import pandas as pd
 from cross_harness_sft.backends.agentdojo import AgentDojoDriver
 rows={r['cell_id']:r for r in pd.read_parquet(root/'outputs/rl/9b_whitebox_pool_v4/train.parquet').to_dict('records')};driver=AgentDojoDriver(str(root/'configs/agentdojo_worker.yaml'));cache={};records=[];counts=collections.Counter()
 with gzip.open(a.rollouts,'rt') as f:
  for line in f:
   e=json.loads(line)['evidence'];cell=e['cell_id'];row=rows[cell]
   if cell not in cache:
    started=driver.start(row['task_id'],int(row['semantic_fixture_seed']));cache[cell]=transform_tools(started['tools'],tuple(int(x) for x in row['harness_config']))[0]
   for i,raw in enumerate(e['raw_assistant_turns']):
    try:
     _,calls=parse_strict_tool_turn(raw,cache[cell],allow_schema_errors=True)
     blocks=[x.get('blocked_reason') for x in calls if x.get('blocked_reason')]
     status='schema_blocked' if blocks else 'valid_calls' if calls else 'text_only'
     detail=blocks
    except ValueError as ex:status='parser_rejected';detail=str(ex)
    counts[status]+=1;records.append({'cell_id':cell,'rollout_id':e['rollout_id'],'turn':i,'status':status,'detail':detail})
 result={'source':str(a.rollouts),'turn_counts':dict(counts),'note':'raw turn replay only, does not establish tool execution or score','turns':records};a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result['turn_counts']))
if __name__=='__main__':main()
