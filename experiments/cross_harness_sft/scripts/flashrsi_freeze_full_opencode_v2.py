"""Freeze all AgentDojo conditions before the first full baseline rollout."""
import hashlib
import importlib.metadata
import json
import subprocess
from collections import Counter
from pathlib import Path

from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from flashrsi.agents.opencode import OpenCode
from flashrsi_opencode_agentdojo_snapshot import make_opencode_snapshot
from flashrsi_full_baseline_ledger import make_ledger

ROOT = Path('/output/cross_harness_alignment/experiments/cross_harness_sft')
OUT = ROOT / 'outputs/b0_full_agentdojo_opencode_20261112_v2'
SOURCE = Path('/output/cross_harness_alignment/hf_transfer/extracted/data/sft/teacher_release_v7/family_split_manifest.jsonl')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
driver=AgentDojoDriver(str(ROOT/'configs/agentdojo_worker.yaml'))
splits={}
for line in SOURCE.read_text().splitlines():
    row=json.loads(line)
    if row.get('benchmark_namespace')=='agentdojo':
        family=row['source_family_id']
        if family in splits: raise ValueError('duplicate family split')
        splits[family]=row['split']
ledger=make_ledger(driver.cases(),splits)
for row in ledger: row['harness']='flashrsi_opencode_python_port'
policy=(ROOT/'configs/flashrsi_b0_safety_policy.txt').read_text()
snapshots={}
for suite in driver.suite_names:
    case=next(k for k,(s,u,i) in driver._case_map.items() if s.name==suite and i is None)
    tools=driver.start(case,0)['tools']
    agent=OpenCode({'llm':{'model_name':'qwen35-4b-preflight','api_key':'placeholder','base_url':'http://127.0.0.1:1/v1','max_tokens':512},'max_turns':12,'timeout':300})
    snapshots[suite]={'id':make_opencode_snapshot(agent,tools,policy).id,
      'tool_schema_sha256':hashlib.sha256(json.dumps(tools,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'tool_count':len(tools)}
files=['scripts/flashrsi_final_turn_proxy.py','scripts/flashrsi_opencode_agentdojo_snapshot.py','scripts/flashrsi_opencode_entry_v2.py','scripts/flashrsi_run_full_opencode_v2.py','scripts/flashrsi_opencode_full_supervisor_v2.py','scripts/flashrsi_opencode_full_host_v2.py','scripts/flashrsi_full_baseline_ledger.py','scripts/flashrsi_freeze_full_opencode_v2.py','scripts/flashrsi_gateway.py','scripts/flashrsi_landlock.py','scripts/flashrsi_model_relay.py','scripts/flashrsi_restricted_snapshot.py','scripts/flashrsi_sandbox_exec.py','scripts/flashrsi_seccomp.py','scripts/flashrsi_tool_attestation.py','scripts/rl_agentdojo_path_audit.py','src/cross_harness_sft/backends/agentdojo.py','configs/agentdojo_worker.yaml','configs/flashrsi_b0_safety_policy.txt','outputs/preflight_20261111/qwen4b_model_manifest.json']
manifest={'scope':'complete AgentDojo benchmark conditions for FlashRSI OpenCode Python port; other harnesses and benchmarks separate','model':'Qwen/Qwen3.5-4B','model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','flashrsi_revision':subprocess.check_output(['git','-C',str(ROOT/'vendor/FlashRSI'),'rev-parse','HEAD'],text=True).strip(),'agentdojo_revision':subprocess.check_output(['git','-C',str(ROOT/'vendor/agentdojo'),'rev-parse','HEAD'],text=True).strip(),'agentdojo_version':driver.version,'seed':0,'temperature':0,'max_turns':12,'max_tokens_per_turn':512,'max_model_len':8192,'timeout_seconds':300,'attack':'AgentDojo DirectAttack at all official ground-truth-observed injection vectors','snapshot_by_suite':snapshots,'family_split_manifest_sha256':sha(SOURCE),'split_policy':'unlisted is unknown, never train; test outcomes isolated from training and selection','code_sha256':{f:sha(ROOT/f) for f in files},'case_counts':dict(Counter(r['split']+':'+r['arm'] for r in ledger)),'cases':ledger}
OUT.mkdir(exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
with (OUT/'run_manifest.json').open('x') as f: json.dump(manifest,f,indent=2,ensure_ascii=False)
print('frozen',len(ledger),'cases',manifest['case_counts'],flush=True)
