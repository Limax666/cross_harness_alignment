"""Summarize completed official HarnessRisk case evidence; retain missing denominator."""
import argparse,json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
manifest=json.loads((args.run/'run_manifest.json').read_text());case_root=args.run/'cases';rows=[]
for file in sorted(case_root.glob('*/official_rule_evaluation.json')):
 data=json.loads(file.read_text());rows.extend(data['runs'])
completed=[r for r in rows if r.get('status')=='completed'];n=len(completed)
keys=('utility_score','utility_oracle_score','utility_final_answer_fallback_score','persistence_score','detection_score')
result={'planned':manifest.get('case_count',128),'official_records':len(rows),'completed_scored':n,
        'status_counts':dict(Counter(str(r.get('status')) for r in rows)),
        'metric_averages':{k:sum(float(r['metrics'][k]) for r in completed)/n if n else None for k in keys},
        'attack_success_binary':sum(bool(r['details']['attack_success']['passed']) for r in completed),
        'attack_partial_score_mean':sum(float(r['metrics']['attack_success_rate']) for r in completed)/n if n else None,
        'note':'HarnessRisk rule attack_success_rate can be fractional: binary success uses details.attack_success.passed. No LLM judge.'}
print(json.dumps(result,ensure_ascii=False,indent=2))
