"""Retry infrastructure failures once and preserve all original evidence."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)!=2:raise SystemExit('usage: retry_infrastructure RUN_DIR')
run=Path(sys.argv[1]).resolve()
with (run/'retry.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if (run/'retry_finished.json').exists():raise SystemExit(0)
 attempts=[json.loads(x) for x in (run/'attempts.jsonl').read_text().splitlines() if x.strip()]
 if len(attempts)!=128 or len({x['case_id'] for x in attempts})!=128:raise SystemExit('initial attempt set incomplete')
 snap=json.loads((run/'run_manifest.json').read_text())['snapshot_id'];retry=run/'infrastructure_retries';retry.mkdir(exist_ok=False)
 paired=run/'paired_cases';paired.mkdir(exist_ok=False);records=[]
 for row in attempts:
  case=row['case_id'];original=run/'cases'/case;source=original
  if row['exit_code']==1:
   process=subprocess.run([sys.executable,str(ROOT/'scripts'/os.environ.get('HR_HOST_SCRIPT','flashrsi_harnessrisk_host.py')),case,str(run/'initial_snapshot.json'),snap,str(retry)],cwd=ROOT)
   source=retry/case
   records.append({'case_id':case,'exit_code':process.returncode,'rule_evaluation_present':(source/'official_rule_evaluation.json').is_file()})
   if process.returncode!=0:source=original
  if (source/'trajectory'/'trajectory.json').is_file() and (source/'official_rule_evaluation.json').is_file():
   (paired/case).symlink_to(source.resolve(),target_is_directory=True)
 (run/'retry_finished.json').write_text(json.dumps({'retried':records,'paired_case_directories':len(list(paired.iterdir())),
  'note':'Only infrastructure exit 1 was retried once; original evidence retained; model output-limit errors not retried.'},indent=2)+'\n')
