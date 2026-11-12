"""Collect the frozen AgentHarm ledger; preserve unscored and failed cases."""
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/output/cross_harness_alignment/experiments/cross_harness_sft'); out=root/'outputs/b0_full_agentharm_20261112'
m=json.loads((out/'run_manifest.json').read_text())
for name,expected in m['code_sha256'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected: raise RuntimeError('frozen code changed: '+name)
failures=0
with (out/'launcher.jsonl').open('a',buffering=1) as log:
 for index,row in enumerate(m['cases']):
  case=out/'cases'/(row['split']+'__'+hashlib.sha256(row['task_id'].encode()).hexdigest()+'.json')
  if case.exists(): continue
  env={**os.environ,'B0_TASK_ID':row['task_id'],'B0_SPLIT':row['split'],'PYTHONPATH':':'.join(str(root/p) for p in ('src','scripts','vendor/FlashRSI'))}
  start=time.time()
  try:
   p=subprocess.run([str(root/'.venv-agentharm/bin/python'),str(root/'scripts/flashrsi_agentharm_full_host.py')],env=env,capture_output=True,text=True,timeout=390)
   item={'case_id':row['task_id'],'split':row['split'],'exit_code':p.returncode,'stdout':p.stdout[-4000:],'stderr':p.stderr[-2000:]}
  except subprocess.TimeoutExpired: item={'case_id':row['task_id'],'split':row['split'],'error':'host_timeout'}
  item['elapsed_seconds']=round(time.time()-start,2); item['record_exists']=case.exists()
  if case.exists():
   result=json.loads(case.read_text()); item['completed']=result.get('status')=='finished' and result.get('attestation_passed') is True and result.get('observation_attestation_passed') is True and result.get('code_integrity_verified') is True
   item['error']=result.get('unscorable_reason') or result.get('error_message')
   if result.get('adapted_snapshot') and result['adapted_snapshot']!=row['snapshot_id']: raise RuntimeError('snapshot differs from manifest')
  else: item['completed']=False; item['error']=item.get('error') or 'missing_case_record'
  log.write(json.dumps(item,ensure_ascii=False)+'\n')
  print(json.dumps({'index':index+1,'total':len(m['cases']),'split':row['split'],'completed':item['completed'],'error':item.get('error')}),flush=True)
  failures=0 if item['completed'] else failures+1
  if failures>=5: raise RuntimeError('five consecutive unsuccessful cases; inspect before resuming')
