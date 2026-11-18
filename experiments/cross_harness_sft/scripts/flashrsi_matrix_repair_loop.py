#!/usr/bin/env python3
"""Evidence-driven durable matrix and serialized autonomous Codex repair worker."""
import argparse, collections, datetime, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
OUTPUTS=ROOT/'outputs'
MONITOR=OUTPUTS/'flashrsi_harnessrisk_4b_private_v1_20261113/monitor'
STATE=MONITOR/'matrix_loop'
HARNESSES=('miniagent','opencode','openclaw','hermes','pi','claude_code','codex')
ALGORITHMS=('opro','gepa','evoprompt','ace','skillopt','evo_harness')

def now():return datetime.datetime.now().astimezone().isoformat()
def read(path, default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return default

def save(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def event(kind,**data):
 with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps({'at':now(),'event':kind,**data},ensure_ascii=False)+'\n')

def active_runs():
 runs={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   args=(p/'cmdline').read_bytes().split(b'\0')
   if not any(a.endswith(b'/flashrsi_harnessrisk_algorithm.py') or a==b'scripts/flashrsi_harnessrisk_algorithm.py' for a in args):continue
   env={x.split(b'=',1)[0]:x.split(b'=',1)[1] for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x}
   base=env.get(b'HR_OUTPUT_ROOT',b'').decode();h=env.get(b'HR_HARNESS',b'miniagent').decode()
   i=args.index(b'--algorithm');a=args[i+1].decode()
   if base:runs[str((Path(base)/f'{h}_{a}_full').resolve())]=int(p.name)
  except (OSError,ValueError,IndexError):continue
 return runs

def queued_launches(roots):
 """Track real waiting launchers without treating them as model inference."""
 queued={}
 for base in roots:
  protocol=read(base/'repair_protocol.json',{});state=read(base/'launch_state.json',{})
  script=protocol.get('queue_launcher');job=protocol.get('job_id');pid=state.get('launcher_pid')
  if not script or not job or not isinstance(pid,int) or state.get('status')!='waiting_endpoint_capacity':continue
  try:
   args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
   if not any(a.decode(errors='replace') in (script,str(ROOT/script)) for a in args):continue
  except OSError:continue
  queued.setdefault(job,[]).append({'path':str(base),'launcher_pid':pid,'pid':None,
      'status':'waiting_endpoint_capacity','protocol':protocol.get('protocol'),
      'last_update':state.get('at'),'clients':state.get('clients',[])})
 return queued

def inspect_run(run,expected,pid):
 reasons=[];result=read(run/'result.json');manifest=read(run/'run_manifest.json',{})
 err=read(run/'error.json');selected=read(run/'snapshots/selected.json',{})
 row={'path':str(run),'pid':pid,'protocol':manifest.get('host_script'), 'error':err,
      'scoring_contract':manifest.get('scoring_contract'),'valid':0,'attempted':0,'judge_valid':0}
 progress=[run/'result.json',run/'error.json',run/'run_manifest.json']
 # Small per-case audit files provide progress without loading large transcripts.
 progress+=list(run.glob('evidence/*/cases/*/official_rule_evaluation.json'))
 row['last_progress_age_seconds']=round(time.time()-max((p.stat().st_mtime for p in progress if p.exists()),default=run.stat().st_mtime))
 if result:
  rows=result.get('results',[]);ids=[x.get('case_id') for x in rows];row['attempted']=len(rows)
  good=[x for x in rows if x.get('status')=='finished' and x.get('metrics',{}).get('verifier_coverage')==1]
  row['valid']=len(good);sid=selected.get('id')
  if result.get('status')!='finished' or len(ids)!=128 or len(set(ids))!=128 or set(ids)!=expected:reasons.append('final_case_set_mismatch')
  if len(good)!=128:reasons.append('incomplete_final_scoring')
  if not sid or sid!=result.get('selected_snapshot_id'):reasons.append('selected_snapshot_identity_mismatch')
  for cid in ids:
   if not isinstance(cid,str) or cid not in expected:continue
   case=run/'selected_cases'/cid;h=read(case/'harness_result.json',{});audit=read(case/'privacy_audit.json',{})
   official=read(case/'official_rule_evaluation.json',{}).get('runs',[])
   if h.get('snapshot_id')!=sid or h.get('status')!='completed':reasons.append('case_snapshot_or_completion_mismatch');break
   if audit.get('protocol')!='harnessrisk-private-case-v1':reasons.append('missing_privacy_audit');break
   if len(official)!=1 or official[0].get('case_id')!=cid or official[0].get('status')!='completed':reasons.append('missing_official_rule_score');break
  # Legacy error-zero evolution is admissible only with no failed evolution episodes.
  if manifest.get('scoring_contract')!='reject_unscorable_during_evolution':
   for p in run.glob('evidence/*/cases/*/harness_result.json'):
    if read(p,{}).get('status')!='completed':reasons.append('failed_episode_in_legacy_zero_score_evolution');break
  judges=read(run/'selected_cases/llm_judge_multi_harness/manifest.json',[])
  judged=[x for x in judges if x.get('status') in ('judged','existing')]
  row['judge_valid']=len(judged)
  judge_ok=len(judges)==128 and len({x.get('case_id') for x in judged})==128 and {x.get('case_id') for x in judged}==expected
  if not reasons and not judge_ok:row['status']='needs_judge'
  elif not reasons:row['status']='complete'
  else:row['status']='needs_repair'
 elif pid:
  row['status']='suspected_stall' if row['last_progress_age_seconds']>2700 else 'running'
 else:row['status']='needs_repair';reasons.append('failed_run' if err else 'interrupted_run')
 row['reasons']=sorted(set(reasons))
 if err:row['failure_fingerprint']=hashlib.sha256(str(err).encode()).hexdigest()[:16]
 return row

def _scan():
 STATE.mkdir(parents=True,exist_ok=True)
 manifest=read(OUTPUTS/'flashrsi_harnessrisk_4b_private_v1_20261113/case_manifest.json',{})
 expected={x['case_id'] for x in manifest.get('cases',[])}
 if len(expected)!=128:raise RuntimeError('trusted case manifest must contain 128 IDs')
 active=active_runs();roots=sorted(OUTPUTS.glob('flashrsi_harnessrisk_4b_private*'));queued=queued_launches(roots)
 jobs=[]
 for h in HARNESSES:
  for a in ALGORITHMS:
   item={'harness':h,'algorithm':a,'job_id':h+'/'+a,'runs':[]}
   if h=='opencode' and a in ('skillopt','evo_harness'):item.update(status='not_applicable',reason='upstream Snapshot has no skill component');jobs.append(item);continue
   for base in roots:
    run=base/f'{h}_{a}_full'
    if run.is_dir():item['runs'].append(inspect_run(run,expected,active.get(str(run.resolve()))))
   item['queued_launches']=queued.get(item['job_id'],[])
   states={x['status'] for x in item['runs']}
   if 'complete' in states:item['status']='complete'
   elif 'running' in states:item['status']='running'
   elif 'suspected_stall' in states:item['status']='suspected_stall'
   elif 'needs_judge' in states:item['status']='needs_judge'
   elif item['queued_launches']:item['status']='queued'
   elif item['runs']:item['status']='needs_repair'
   else:item['status']='pending'
   item['failure_fingerprints']=dict(collections.Counter(x.get('failure_fingerprint') for x in item['runs'] if x.get('failure_fingerprint')))
   jobs.append(item)
 counts=dict(collections.Counter(x['status'] for x in jobs))
 data={'at':now(),'jobs':jobs,'counts':counts,'applicable_jobs':40,'all_complete':all(x['status'] in ('complete','not_applicable') for x in jobs)}
 save(STATE/'matrix.json',data)
 tickets=[x for x in jobs if x['status'] not in ('complete','not_applicable','running')]
 save(STATE/'repair_tickets.json',{'at':now(),'tickets':tickets})
 lines=['# FlashRSI 算法矩阵实时任务表','',data['at'],'','| Harness | '+' | '.join(ALGORITHMS)+' |','|---|'+'---|'*len(ALGORITHMS)]
 for h in HARNESSES:lines.append('| '+h+' | '+' | '.join(x['status'] for x in jobs if x['harness']==h)+' |')
 (STATE/'matrix.md').write_text('\n'.join(lines)+'\n')
 event('scan',counts=counts)
 if data['all_complete']:
  (STATE/'complete').write_text(now()+'\n');(MONITOR/'disabled').write_text('all 40 algorithm cells verified '+now()+'\n')
 return data

def scan():
 STATE.mkdir(parents=True,exist_ok=True)
 with (STATE/'scan.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _scan()

PROMPT='''用户明确授权你持续修复 FlashRSI × HarnessRisk 4B 的全部 harness×RSI算法，不要停留在巡检总结。你是独立串行维修执行器。先读仓库 AGENT.md、实验 experiments/cross_harness_sft/docs/flashrsi_matrix_repair_loop_20261114.md、experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_v1_20261113/monitor/matrix_loop/matrix.json 和 repair_tickets.json，以及 repair_journal.md（存在则读）。
每轮选择一个需要修复/补judge/未启动的单元，定位真实根因，实施必要代码或配置修复，保存旧错误及来源、代码哈希、协议；然后启动/恢复真实完整评测和 judge，核查进程与首次实际证据。不要只列待办；做完一个后继续解决能独立处理的工单。优先完成可靠的同协议算法对照，已有完整产物不重复调用 judge。现有初始Snapshot可引用，非必要不重跑初始基线；执行协议变化影响配对时需明确登记及同协议对照需求。既有活进程不得重复启动，检查进程树/资源和GPU0/3，每个端点有限并发，不碰别人的GPU。修改运行中的shell必须新文件加原子替换，避免原位重写。
不可评分不能当0分排序；错误分清工程故障、正常模型预算失败与攻击成功。严禁降低验收要求、删难例、截断后假称成功、伪造Snapshot、放松沙箱/隐私门禁、暴露密钥、拼接不同协议的结果。不要跑模拟/冒烟测试，使用真实任务证据和完整评测。重试失败后回到诊断；同一故障反复三次必须提出并实施新的诊断，不得只增token或重复旧命令。重复输出、上下文、消息/工具协议等必须以真实证据分析。资源和服务使修复暂不能完成时，写精确证据和下一条可执行诊断，下轮继续，不能把任务标完成。
把诊断→变更→启动→实际验收/失败和后续动作追加到 experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_v1_20261113/monitor/matrix_loop/repair_journal.md，同时写 machine-readable action_last.json（job_id、root_cause、changes、run_dir、pid、outcome、next_action，不含密钥）。完成本轮前重新运行 experiments/cross_harness_sft/scripts/flashrsi_matrix_repair_loop.py --scan。初始基线不是当前主要工作，重点交付算法对照。只有所有适用单元通过扫描器验收才结束整体任务。'''

def worker():
 STATE.mkdir(parents=True,exist_ok=True)
 with (STATE/'worker.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  if (STATE/'disabled').exists():return
  data=scan()
  if data['all_complete']:return
  stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');out=STATE/('codex_repair_'+stamp+'.log')
  save(STATE/'worker.json',{'status':'running','pid':os.getpid(),'started':now(),'log':str(out)})
  event('repair_started',log=str(out))
  env={k:v for k,v in os.environ.items() if k not in ('CODEX_THREAD_ID','CODEX_SESSION_ID','CODEX_TOOL_CALL_ID')}
  try:
   with out.open('w') as log:
    p=subprocess.Popen(['/data/home/liumingxiao/.local/bin/codex','exec','-C',str(REPO),'--sandbox','danger-full-access','-c','approval_policy="never"','-o',str(STATE/('reply_'+stamp+'.md')),'-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env)
    save(STATE/'worker.json',{'status':'running','pid':os.getpid(),'codex_pid':p.pid,'started':now(),'log':str(out)})
    p.communicate(PROMPT.encode());code=p.returncode
   event('repair_finished',exit_code=code,log=str(out));save(STATE/'worker.json',{'status':'finished','at':now(),'exit_code':code,'log':str(out)})
  except Exception as e:
   event('repair_launcher_failed',error=type(e).__name__);raise
  finally:scan()

def tick(launch):
 STATE.mkdir(parents=True,exist_ok=True)
 data=scan()
 print(json.dumps({'counts':data['counts'],'all_complete':data['all_complete']}))
 if not launch or data['all_complete'] or (STATE/'disabled').exists():return
 # Worker itself owns the lock for the full Codex lifetime; duplicate launchers exit.
 with (STATE/'worker.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  if not any(x['status'] in ('needs_repair','pending','needs_judge','suspected_stall') for x in data['jobs']):return
 with (STATE/'launcher.log').open('a') as log:
  subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--repair-worker'],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scan',action='store_true');p.add_argument('--tick',action='store_true');p.add_argument('--repair-worker',action='store_true');a=p.parse_args()
 if a.repair_worker:worker()
 else:tick(a.tick)
