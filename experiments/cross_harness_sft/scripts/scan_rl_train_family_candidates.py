import sys,json,pathlib,traceback
p=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(p/'scripts'),str(p/'src')]
from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from rl_agentdojo_path_audit import GENERALIZED_FAMILIES,READ_TOOLS,_ground_truth_tools
from rl_native_agentdojo_episode import _digest
families=[r['source_family_id'] for s in (p/'data/sft/teacher_release_v7/family_split_manifest.jsonl').read_text().splitlines() if s for r in [json.loads(s)] if r.get('benchmark_namespace')=='agentdojo' and r.get('split')=='train']
d=AgentDojoDriver(str(p/'configs/agentdojo_worker.yaml')); out=[]
for f in families:
 t=f.removeprefix('agentdojo:'); item={'family':f}
 try:
  a=d.start(t+':clean',0)['episode']; b=d.start(t+':injection_task_1',0)['episode']; c=d.start(t+':clean',0)['episode']; plan=_ground_truth_tools(a)
  item.update(plan=sorted(plan) if plan is not None else None,reset_equal=_digest(a.environment)==_digest(c.environment), audit_qualified=bool(plan is not None and (t in GENERALIZED_FAMILIES or plan<=READ_TOOLS)), prompt=a.user_task.PROMPT)
 except Exception as e:item['error']=f'{type(e).__name__}: {e}'
 out.append(item)
q=p/'outputs/rl/diagnose_step150_backends_20261030/train_family_candidate_scan.json';q.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print('families',len(out),'qualified',sum(x.get('audit_qualified',False) for x in out),'errors',sum('error' in x for x in out));print('qualified',[(x['family'],x.get('plan')) for x in out if x.get('audit_qualified')])
