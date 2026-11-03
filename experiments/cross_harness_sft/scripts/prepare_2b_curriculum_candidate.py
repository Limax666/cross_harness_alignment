#!/usr/bin/env python3
"""Prepare, but do not train on, an immutable K=2 curriculum candidate."""
import argparse,hashlib,json,shutil
from pathlib import Path
import pandas as pd
from rl_curriculum_schedule import make_snapshot,curriculum_cohorts
from rl_whitebox_harness import build_whitebox_ledger

def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metrics',type=Path,action='append',required=True);p.add_argument('--iterations',type=int,default=20);p.add_argument('--seed',type=int,default=267);p.add_argument('--progress-oracle-version',default='agentdojo-workspace-progress-v4-candidate');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from rl_progress_registry import resolve
    oracle=resolve(a.progress_oracle_version)
    data=a.pool/'train.parquet'; rows=pd.read_parquet(data).to_dict('records')
    if any(r.get('split')!='train' or r.get('benchmark')!='agentdojo' for r in rows):raise ValueError('train-only AgentDojo pool required')
    families={r['source_family_id'] for r in rows};snapshot=make_snapshot(a.metrics,families)
    schedule=curriculum_cohorts(families,a.iterations//2,a.seed,snapshot)
    snapshot_raw=json.dumps(snapshot,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode();snapshot_sha=hashlib.sha256(snapshot_raw).hexdigest()
    ledger=build_whitebox_ledger(rows,iterations=a.iterations,seed=a.seed,family_schedule=schedule,schedule_metadata={'snapshot_sha256':snapshot_sha,'version':snapshot['version']})
    a.output.mkdir(parents=True);shutil.copyfile(data,a.output/'train.parquet')
    (a.output/'train.manifest.json').write_text(json.dumps({'output_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'source':'frozen curriculum candidate','split':'train'},indent=2)+'\n')
    (a.output/'family_priority_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    (a.output/'assignment_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    (a.output/'manifest.json').write_text(json.dumps({'source_pool':str(a.pool.resolve()),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'snapshot_sha256':snapshot_sha,'ledger_sha256':ledger['ledger_sha256'],'families':len(families),'iterations':a.iterations,'progress_oracle_version':oracle.VERSION,'status':'frozen candidate; requires a fresh online pilot'},indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'families':len(families),'ledger_sha256':ledger['ledger_sha256'],'snapshot_sha256':snapshot_sha}))
if __name__=='__main__':main()
