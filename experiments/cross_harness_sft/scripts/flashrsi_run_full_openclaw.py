"""Run the frozen AgentDojo ledger, preserving failed attempts and resume state."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/output/cross_harness_alignment/experiments/cross_harness_sft')
out=root/'outputs/b0_full_agentdojo_openclaw_20261112'
manifest=json.loads((out/'run_manifest.json').read_text())
for name,expected in manifest['code_sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('frozen code changed: '+name)
priority={'validation':0,'train':1,'unlisted':2,'test':3}
rows=sorted(manifest['cases'],key=lambda r:(priority[r['split']],r['family'],r['arm'],r['task_id']))
consecutive_errors=0
with (out/'launcher.jsonl').open('a',buffering=1) as log:
 for index,row in enumerate(rows):
    case_path=out/'cases'/(row['task_id'].replace(':','__')+'.json')
    if case_path.exists(): continue
    env={**os.environ,'B0_TASK_ID':row['task_id'],'B0_SEED':str(row['seed']),'PYTHONPATH':':'.join(str(root/p) for p in ('src','scripts','vendor/FlashRSI'))}
    started=time.time()
    try:
        proc=subprocess.run([str(root/'.venv-agentdojo/bin/python'),str(root/'scripts/flashrsi_openclaw_full_host.py')],env=env,capture_output=True,text=True,timeout=390)
        entry={'case_id':row['task_id'],'split':row['split'],'exit_code':proc.returncode,'stdout':proc.stdout[-4000:],'stderr':proc.stderr[-2000:],'duration_seconds':round(time.time()-started,2)}
    except subprocess.TimeoutExpired:
        entry={'case_id':row['task_id'],'split':row['split'],'error':'host_timeout','duration_seconds':round(time.time()-started,2)}
    entry['case_record_exists']=case_path.exists()
    if case_path.exists():
        case=json.loads(case_path.read_text())
        suite=row['family'].split(':')[0]
        entry['snapshot_match']=case.get('adapted_snapshot')==manifest['snapshot_by_suite'][suite]['id'] if case.get('adapted_snapshot') else None
        entry['completed']=case.get('status')=='finished' and case.get('attestation_passed') is True and case.get('code_integrity_verified') is True
        entry['error']=case.get('unscorable_reason') or case.get('error_message')
    else:
        entry['completed']=False; entry['error']=entry.get('error') or 'missing_case_record'
    log.write(json.dumps(entry,ensure_ascii=False)+'\n')
    print(json.dumps({'index':index+1,'total':len(rows),'case':row['task_id'],'split':row['split'],'completed':entry['completed'],'error':entry.get('error')},ensure_ascii=False),flush=True)
    if entry.get('snapshot_match') is False: raise RuntimeError('snapshot differs from frozen manifest')
    consecutive_errors=0 if entry['completed'] else consecutive_errors+1
    if consecutive_errors>=5: raise RuntimeError('five consecutive unsuccessful cases; inspect before resuming')
