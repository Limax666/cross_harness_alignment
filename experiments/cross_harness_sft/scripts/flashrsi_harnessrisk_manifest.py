"""Freeze HarnessRisk case identities for FlashRSI paired full-set evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

HARNESSES=('miniagent','claude_code','codex','opencode','openclaw','hermes','pi')
ALGORITHMS=('opro','gepa','evoprompt','ace','skillopt','evo_harness','ahe','adas',
            'darwinx','meta_harness','modular_rsi','rrsi','arbor')


def build_manifest(case_dir, *, expected_count=128):
    files=sorted(Path(case_dir).glob('*.json'))
    if len(files)!=expected_count:
        raise ValueError(f'expected {expected_count} cases, got {len(files)}')
    records=[]
    for path in files:
        raw=path.read_bytes();data=json.loads(raw)
        case_id=data.get('case_id')
        if not isinstance(case_id,str) or case_id!=path.stem:
            raise ValueError(f'case identity mismatch: {path.name}')
        task=data.get('task',{});messages=task.get('user_messages') or [task.get('user_message')]
        if not all(isinstance(x,str) and x.strip() for x in messages):
            raise ValueError(f'missing owner messages: {case_id}')
        records.append({'case_id':case_id,'phase':data.get('phase'),
                        'sha256':hashlib.sha256(raw).hexdigest(),'owner_turns':len(messages)})
    return {'benchmark':'HarnessRisk','full_evaluation_count':len(records),'cases':records,
            'model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
            'flashrsi_revision':'778e2d01a995491fe63db210dcf46db1723a4430',
            'harnesses':list(HARNESSES),'algorithms':list(ALGORITHMS),
            'evaluation_scope':'all 128 cases per harness and snapshot; training-exposed cases reported separately',
            'execution_status':'not_started_bridges_not_qualified'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    value=build_manifest(args.case_dir);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(f"Frozen {len(value['cases'])} cases to {args.output}")


if __name__=='__main__':main()
