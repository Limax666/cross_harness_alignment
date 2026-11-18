"""HarnessRisk official case environments and scoring for FlashRSI algorithms."""
import asyncio, hashlib, json, os
from collections.abc import Mapping
from pathlib import Path
from flashrsi.core import Task, Trajectory
from flashrsi.core.benchmark import Benchmark
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parents[1]/'HarnessRisk/data/HarnessRisk'
MANIFEST=Path(os.environ.get('HR_OUTPUT_ROOT',str(ROOT/'outputs/flashrsi_harnessrisk_4b_20261113'))).resolve()/'case_manifest.json'

class HarnessRiskBenchmark(Benchmark):
 name='harnessrisk';description='HarnessRisk environments with official deterministic scoring';primary_metric='utility_score'
 metadata={'benchmark':'HarnessRisk','split':'deterministic phase-stratified; full 128 case result includes evolution-exposed cases',
           'scoring':'official deterministic rule evaluator, no LLM judge'}
 def __init__(self,run_dir):
  super().__init__();self.run_dir=Path(run_dir);self._sequence=0;self._records={};self.full=();self.reject_unscorable=False
 def load(self):
  if self._splits is not None:return self._splits
  manifest=json.loads(MANIFEST.read_text());groups={}
  for row in manifest['cases']:
   path=DATA/(row['case_id']+'.json');raw=path.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('case hash mismatch')
   data=json.loads(raw);owner=data['task'].get('user_messages') or [data['task']['user_message']]
   task=Task(task_id=row['case_id'],question='\n'.join(owner),env='harnessrisk',metadata={'phase':row['phase'],'case_sha256':row['sha256']})
   groups.setdefault(row['phase'],[]).append(task)
  parts={'train':[],'dev':[],'test':[]}
  for phase,items in sorted(groups.items()):
   for i,t in enumerate(sorted(items,key=lambda t:hashlib.sha256(('flashrsi-42:'+t.task_id).encode()).hexdigest())):
    parts['dev' if i%5==0 else 'test' if i%5==1 else 'train'].append(t)
  self._splits={k:tuple(v) for k,v in parts.items()};self.full=tuple(sorted(sum(parts.values(),[]),key=lambda t:t.task_id))
  if len(self.full)!=128:raise ValueError('expected 128 cases')
  return self._splits
 async def env_runner(self,target,task):
  attempts=int(os.environ.get('HR_EVOLUTION_CASE_ATTEMPTS','1')) if self.reject_unscorable else 1
  if attempts not in (1,2):raise ValueError('unsupported evolution attempt limit')
  for index in range(attempts):
   trajectory=await self._run_once(target,task)
   with (self.run_dir/'case_retry_events.jsonl').open('a') as stream:
    stream.write(json.dumps({'case_id':task.task_id,'snapshot_id':target.snapshot.id,'attempt':index+1,
       'attempt_limit':attempts,'status':trajectory.status,'evidence_sequence':self._sequence})+'\n')
   if trajectory.status=='finished':return trajectory
  return trajectory
 async def _run_once(self,target,task):
  self._sequence+=1; seq=self._sequence;folder=self.run_dir/'evidence'/f'{seq:08d}';folder.mkdir(parents=True)
  snap=target.snapshot;snapfile=folder/'snapshot.json';snap.save(snapfile);cases=folder/'cases';cases.mkdir()
  proc=await asyncio.create_subprocess_exec(__import__('sys').executable,str(ROOT/'scripts'/os.environ.get('HR_HOST_SCRIPT','flashrsi_harnessrisk_host.py')),
      task.task_id,str(snapfile),snap.id,str(cases),cwd=str(ROOT),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
  try:out,err=await asyncio.wait_for(proc.communicate(),timeout=2460)
  except TimeoutError:
   proc.kill();await proc.wait();return Trajectory(status='error',error_message='HarnessRisk host timeout')
  path=cases/task.task_id; evaluation=path/'official_rule_evaluation.json';resultfile=path/'harness_result.json'
  if proc.returncode or not evaluation.exists() or not resultfile.exists():
   return Trajectory(status='error',error_message=err.decode(errors='replace')[-2000:] or 'HarnessRisk episode failed')
  result=json.loads(resultfile.read_text());grade=json.loads(evaluation.read_text())['runs'][0]
  if result.get('snapshot_id')!=snap.id or grade.get('case_id')!=task.task_id or result.get('status')!='completed':
   return Trajectory(status='error',error_message='snapshot identity or episode completion mismatch')
  self._records[task.task_id]=grade
  turns=result.get('turns',[]);return Trajectory(status='finished',output=turns[-1]['response_text'] if turns else '',
    messages=tuple(dict(m) if isinstance(m,Mapping) else {'role':'event','content':str(m)}
                   for t in turns for m in t.get('messages',[])))
 def grade(self,trajectories):
  if self.reject_unscorable:
   invalid=[t.task.task_id for t in trajectories if t.status!='finished' or t.task.task_id not in self._records]
   if invalid:raise RuntimeError('unscorable trajectories cannot enter candidate ranking: '+','.join(invalid))
  result=[]
  for t in trajectories:
   row=self._records.pop(t.task.task_id,None)
   metrics={'utility_score':0.0,'verifier_coverage':0.0,'attack_success':0.0,'attack_partial_score':0.0,
            'persistence_score':0.0,'detection_score':0.0}
   if row is not None and t.status=='finished':
    m=row['metrics'];metrics.update(utility_score=float(m['utility_score']),verifier_coverage=1.0,
      attack_success=float(row['details']['attack_success']['passed']),attack_partial_score=float(m['attack_success_rate']),
      persistence_score=float(m['persistence_score']),detection_score=float(m['detection_score']))
   result.append(Trajectory(task=t.task,status=t.status,output=t.output,messages=t.messages,metrics=metrics,error_message=t.error_message,usage=t.usage))
  return tuple(result)
