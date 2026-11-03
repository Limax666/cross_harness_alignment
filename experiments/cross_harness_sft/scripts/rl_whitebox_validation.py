"""Small shared guards for fixed whitebox held-out validation."""
import re

def validate_config_partition(training,holdout):
 if not training or not holdout or len(set(training))!=len(training) or len(set(holdout))!=len(holdout):raise ValueError('empty or duplicate configuration')
 if set(training)&set(holdout):raise ValueError('held-out configuration overlaps training')
 if any(not re.fullmatch(r'wb_S[012]T[012]O[012]C0W[012]',n) for n in training+holdout):raise ValueError('invalid or unsupported whitebox configuration')

def parse_name(name):
 m=re.fullmatch(r'wb_S([012])T([012])O([012])C(0)W([012])',name)
 if not m:raise ValueError('invalid configuration')
 return tuple(map(int,m.groups()))


def compare_paired(sft,candidate):
 from collections import defaultdict
 def index(rows):
  result={}
  for r in rows:
   key=(r['configuration'],r['task_id'],r['seed'])
   if key in result:raise ValueError('duplicate paired case')
   result[key]=r
  return result
 x,y=index(sft),index(candidate)
 if not x or x.keys()!=y.keys():raise ValueError('paired case identities differ')
 summary={}
 for model,records in (('sft',x),('candidate',y)):
  groups=defaultdict(list)
  for row in records.values():
   arm='clean' if row['task_id'].endswith(':clean') else 'injection'
   groups[row['configuration']+'/'+arm].append(row)
  summary[model]={}
  for key,rows in sorted(groups.items()):
   scored=[r for r in rows if r.get('status')=='scored']
   summary[model][key]={'attempted':len(rows),'scored':len(scored),'errors':len(rows)-len(scored),'utility_success':sum(r.get('official_utility')==1 for r in scored),'attack_success':sum(r.get('official_attack_success') is True for r in scored),'tool_episodes':sum(r.get('valid_tool_dispatches',0)>0 for r in scored),'completed':sum(r.get('termination')=='completed' for r in scored),'audit_incomplete':sum(not r.get('audit_complete',False) for r in scored)}
 return {'paired_cases':len(x),'by_model':summary,'note':'attempted includes errors; per-configuration small-sample engineering validation, not final test'}
