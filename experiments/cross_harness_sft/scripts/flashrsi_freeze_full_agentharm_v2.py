"""Freeze official AgentHarm task identities and MiniAgent tool snapshots."""
import hashlib,importlib.metadata,json
from collections import Counter
from pathlib import Path
from cross_harness_sft.backends.agentharm import AgentHarmDriver
from flashrsi.agents.miniagent import MiniAgent
from flashrsi_restricted_snapshot import make_restricted_snapshot
from inspect_evals.agentharm.utils import AGENTHARM_DATASET_REVISION
root=Path('/output/cross_harness_alignment/experiments/cross_harness_sft'); out=root/'outputs/b0_full_agentharm_20261112_v2'
policy=(root/'configs/flashrsi_b0_safety_policy.txt').read_text(); snapshots={}; rows=[]
for split,conf in [('val','configs/agentharm_worker_val.yaml'),('test_public','configs/agentharm_worker_test_public.yaml')]:
 d=AgentHarmDriver(str(root/conf))
 for task_id in sorted(d.samples):
  started=d.start(task_id,0); ep=started['episode']; tools=started['tools']; schema=hashlib.sha256(json.dumps(tools,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  if schema not in snapshots:
   agent=MiniAgent({'llm':{'model_name':'qwen35-4b-preflight','api_key':'placeholder','base_url':'http://127.0.0.1:1/v1','max_tokens':512},'max_turns':12,'timeout':300})
   snapshots[schema]=make_restricted_snapshot(agent,tools,safety_policy=policy).id
  rows.append({'task_id':task_id,'split':split,'kind':ep.task_name,'source_id':str(ep.sample.id),'prompt_sha256':hashlib.sha256(str(ep.sample.input).encode()).hexdigest(),'schema_sha256':schema,'snapshot_id':snapshots[schema]})
files=['scripts/flashrsi_final_turn_proxy.py', 'scripts/flashrsi_freeze_full_agentharm_v2.py','scripts/flashrsi_run_full_agentharm_v2.py','scripts/flashrsi_agentharm_full_supervisor_v2.py','scripts/flashrsi_agentharm_full_host_v2.py','scripts/flashrsi_gateway.py','scripts/flashrsi_landlock.py','scripts/flashrsi_model_relay.py','scripts/flashrsi_restricted_snapshot.py','scripts/flashrsi_sandbox_exec.py','scripts/flashrsi_seccomp.py','scripts/flashrsi_tool_attestation.py','src/cross_harness_sft/backends/agentharm.py','configs/flashrsi_b0_safety_policy.txt','configs/agentharm_worker_val.yaml','configs/agentharm_worker_test_public.yaml']
manifest={'scope':'AgentHarm complete val and test_public; FlashRSI MiniAgent; scoring pending independent judge','dataset':'ai-safety-institute/AgentHarm','dataset_revision':AGENTHARM_DATASET_REVISION,'inspect_evals_version':importlib.metadata.version('inspect-evals'),'inspect_ai_version':importlib.metadata.version('inspect-ai'),'model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','flashrsi_revision':'778e2d01a995491fe63db210dcf46db1723a4430','seed':0,'temperature':0,'max_turns':12,'max_tokens_per_turn':512,'max_model_len':8192,'counts':dict(Counter(row['split']+':'+row['kind'] for row in rows)),'code_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},'cases':rows}
with (out/'run_manifest.json').open('x') as f: json.dump(manifest,f,indent=2,ensure_ascii=False)
print('Frozen AgentHarm cases:',len(rows),'unique tool snapshots:',len(snapshots),'counts:',manifest['counts'],flush=True)
