"""Full 128-case HarnessRisk initial FlashRSI MiniAgent baseline."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from flashrsi.agents.miniagent import MiniAgent
from flashrsi.agents.opencode import OpenCode
from flashrsi.agents.openclaw import OpenClaw
from flashrsi.agents.hermes import Hermes
from flashrsi.agents.pi_agent import Pi
from flashrsi.agents.claude_code import ClaudeCode
from flashrsi.agents.codex import Codex
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('HR_OUTPUT_ROOT',str(ROOT/'outputs/flashrsi_harnessrisk_4b_20261113'))).resolve()
DATA=ROOT.parents[1]/'HarnessRisk/data/HarnessRisk'

def main():
 manifest=json.loads((OUT/'case_manifest.json').read_text()); cases=manifest['cases']
 if len(cases)!=128 or len({x['case_id'] for x in cases})!=128:raise ValueError('invalid case manifest')
 for row in cases:
  if hashlib.sha256((DATA/(row['case_id']+'.json')).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('case hash changed')
 run=OUT/os.environ.get('HR_RUN_NAME','miniagent_initial_full');resume=os.environ.get('HR_RESUME')=='1'
 if resume:
  if not (run/'initial_snapshot.json').is_file() or not (run/'attempts.jsonl').is_file():raise ValueError('incomplete resume state')
 else:
  run.mkdir(parents=True,exist_ok=False);(run/'cases').mkdir()
 harness=os.environ.get('HR_HARNESS','miniagent')
 classes={'miniagent':MiniAgent,'opencode':OpenCode,'openclaw':OpenClaw,'hermes':Hermes,'pi':Pi,'claude_code':ClaudeCode,'codex':Codex}
 if harness not in classes:raise ValueError('unsupported harness')
 agent=classes[harness]({'llm':{'model_name':'qwen35-4b-preflight','api_key':'placeholder','base_url':'http://127.0.0.1:1/v1'},'max_turns':12,'timeout':360})
 snapshot=agent.snapshot
 if resume:
  from flashrsi.core import Snapshot
  frozen=Snapshot.load(run/'initial_snapshot.json');old=json.loads((run/'run_manifest.json').read_text())
  if frozen.id!=snapshot.id or frozen.agent_type!=harness or old.get('host_script','flashrsi_harnessrisk_host.py')!=os.environ.get('HR_HOST_SCRIPT','flashrsi_harnessrisk_host.py') or old.get('privacy_protocol')!='harnessrisk-private-case-v1' or old.get('harness',harness)!=harness or old.get('model_port',18761)!=int(os.environ.get('HARNESSRISK_MODEL_PORT','18761')) or old['case_manifest_sha256']!=hashlib.sha256((OUT/'case_manifest.json').read_bytes()).hexdigest():raise ValueError('resume conditions changed')
 else:
  snapshot.save(run/'initial_snapshot.json')
  (run/'run_manifest.json').write_text(json.dumps({'status':'running','scope':'full 128 HarnessRisk cases; initial official FlashRSI snapshot','harness':harness,
  'snapshot_id':snapshot.id,'model_revision':manifest['model_revision'],'flashrsi_revision':manifest['flashrsi_revision'],
  'case_manifest_sha256':hashlib.sha256((OUT/'case_manifest.json').read_bytes()).hexdigest(),'max_turns_per_owner_turn':12,
  'temperature':0,'max_tokens_per_turn':4096,'model_port':int(os.environ.get('HARNESSRISK_MODEL_PORT','18761')),'owner_turn_protocol':'sequential, previous owner/assistant text replay; environment state preserved',
  'host_script':os.environ.get('HR_HOST_SCRIPT','flashrsi_harnessrisk_host.py'),'privacy_protocol':'harnessrisk-private-case-v1','rules':'HarnessRisk official deterministic evaluator','path_audit_qualification':'not yet established','case_count':len(cases)},indent=2)+'\n')
 attempts=[json.loads(line) for line in (run/'attempts.jsonl').read_text().splitlines() if line.strip()] if resume else []
 seen={x['case_id'] for x in attempts}
 if len(seen)!=len(attempts) or not seen.issubset({x['case_id'] for x in cases}):raise ValueError('invalid previous attempts')
 with (run/'attempts.jsonl').open('a' if resume else 'x') as log:
  for index,row in enumerate(cases,1):
   started=time.time();case=row['case_id']
   if case in seen:continue
   proc=subprocess.run([sys.executable,str(ROOT/'scripts'/os.environ.get('HR_HOST_SCRIPT','flashrsi_harnessrisk_host.py')),case,str(run/'initial_snapshot.json'),snapshot.id,str(run/'cases')],cwd=ROOT)
   result={'case_id':case,'index':index,'exit_code':proc.returncode,'elapsed_seconds':round(time.time()-started,2),
           'rule_evaluation_present':(run/'cases'/case/'official_rule_evaluation.json').is_file()}
   log.write(json.dumps(result)+'\n');log.flush();print(json.dumps(result),flush=True)
   attempts.append(result)
 scored=sum(x['exit_code']==0 and x['rule_evaluation_present'] for x in attempts)
 (run/'completed.json').write_text(json.dumps({'status':'all_cases_attempted','attempts':len(attempts),'completed_scored':scored,
   'failed_or_unscored':len(attempts)-scored,'snapshot_id':snapshot.id},indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
