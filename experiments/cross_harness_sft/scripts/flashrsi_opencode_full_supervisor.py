import asyncio,hashlib,json,os,secrets,sys,tempfile,threading,time,subprocess
from pathlib import Path
from collections.abc import Mapping
from dataclasses import asdict
from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from rl_agentdojo_path_audit import NativePathRecorder
from flashrsi_gateway import create_gateway
from flashrsi_model_relay import relay_request
from flashrsi_restricted_snapshot import make_restricted_snapshot
from flashrsi_validation_verifiers import verify_workspace_dinner_answer, verify_workspace_hiking_answer
from agentdojo.attacks.baseline_attacks import DirectAttack
from flashrsi_final_turn_proxy import FinalTurnChatProxy
from flashrsi_tool_attestation import ToolCallGate,GuardedChatProxy,GuardedToolDispatcher,attest_tool_calls
root=Path('/output/cross_harness_alignment/experiments/cross_harness_sft')
task_id=os.environ['B0_TASK_ID']; seed=int(os.environ['B0_SEED']); family,_,arm=task_id.rpartition(':')
out=root/'outputs/b0_full_agentdojo_opencode_20261112'/'cases'/(task_id.replace(':','__')+'.json')
record={'scope':'complete AgentDojo B0, FlashRSI OpenCode Python-port restricted snapshot','family':family,'arm':arm,'seed':seed,'task_id':task_id,'model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','network_namespace':'new loopback-only','max_turns':12,'max_tokens_per_turn':512,'thinking':False}
files=['scripts/flashrsi_final_turn_proxy.py', 'scripts/flashrsi_gateway.py', 'scripts/flashrsi_landlock.py', 'scripts/flashrsi_model_relay.py', 'scripts/flashrsi_restricted_snapshot.py', 'scripts/flashrsi_sandbox_exec.py', 'scripts/flashrsi_seccomp.py', 'scripts/flashrsi_tool_attestation.py', 'scripts/rl_agentdojo_path_audit.py', 'src/cross_harness_sft/backends/agentdojo.py', 'scripts/flashrsi_opencode_full_supervisor.py', 'scripts/flashrsi_opencode_full_host.py', 'scripts/flashrsi_opencode_agentdojo_snapshot.py', 'scripts/flashrsi_opencode_entry.py', 'configs/flashrsi_b0_safety_policy.txt']
def code_hashes(): return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
record['code_hashes_before']=code_hashes()
server=None; thread=None; start=time.monotonic()
try:
 d=AgentDojoDriver(str(root/'configs/agentdojo_worker.yaml'))
 suite,user,injection=d._case_map[task_id]
 attack={} if injection is None else DirectAttack(suite,target_pipeline=None).attack(user,injection)
 record['attack_vectors']=sorted(attack)
 record['attack_payload_sha256']={k:hashlib.sha256(v.encode()).hexdigest() for k,v in attack.items()}
 started=d.start(task_id,seed)
 episode=started['episode']; tools=started['tools']; prompt=episode.user_task.PROMPT
 def digest(e): return hashlib.sha256(json.dumps(e.model_dump(mode='json'),sort_keys=True).encode()).hexdigest()
 record['initial_state_sha256']=digest(episode.environment)
 gate=ToolCallGate({t['function']['name'] for t in tools})
 relay_path=os.environ['DEV_RELAY_PATH']; relay_token=os.environ['DEV_RELAY_TOKEN']
 def upstream(request):
  request=dict(request); request['chat_template_kwargs']={'enable_thinking':False}
  return relay_request(relay_path,relay_token,request)
 proxy=FinalTurnChatProxy(gate,'qwen35-4b-preflight',tools,upstream)
 recorder=NativePathRecorder(d,episode,tool_schemas=tools,qualified_families=NativePathRecorder.__dataclass_fields__["qualified_families"].default)
 dispatcher=GuardedToolDispatcher(gate,recorder)
 token=secrets.token_urlsafe(32); server=create_gateway(proxy,dispatcher,gate,token)
 thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
 base=f'http://127.0.0.1:{server.server_port}'
 with tempfile.TemporaryDirectory(prefix='rsi-child-') as parent:
  parent=Path(parent); writable=parent/'work'; writable.mkdir(); config=parent/'config.json'
  entry=root/'scripts/flashrsi_opencode_entry.py'
  config.write_text(json.dumps({'entry':str(entry),'python':sys.executable,'gateway_port':server.server_port,
    'readonly_roots':['/usr','/lib','/lib64','/etc/ssl/certs','/opt/uv/python',str(root/'.venv-agentdojo'),str(root/'vendor/FlashRSI'),str(root/'scripts')],'writable_root':str(writable)}))
  os.chmod(config,0o600)
  child_env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','FLASHRSI_SANDBOX_CONFIG':str(config),'FLASHRSI_GATEWAY_URL':base,'FLASHRSI_GATEWAY_TOKEN':token}
  request={'tools':tools,'policy':(root/'configs/flashrsi_b0_safety_policy.txt').read_text(),'prompt':prompt}
  child=subprocess.run([str(root/'scripts/flashrsi_sandbox_exec.py'),'-I','-B',str(entry)],input=json.dumps(request),env=child_env,text=True,capture_output=True,timeout=320)
  record['child_exit']=child.returncode
  if child.returncode: raise RuntimeError('OpenCode child failed: '+child.stderr[-1500:])
  raw=json.loads(child.stdout); result=raw['result']
  record.update(root_snapshot=raw['root_snapshot'],adapted_snapshot=raw['adapted_snapshot'],tool_count=len(tools),status=result['status'],error_message=result['error_message'],output=result['output'],trajectory_messages=result['messages'],usage=result['usage'],raw_model_requests=proxy.raw_requests,raw_model_messages=proxy.raw_messages,dispatches=dispatcher.executed)
  record['attestation_passed']=attest_tool_calls(proxy.raw_messages,dispatcher.executed)
  if result['status']=='finished' and record['attestation_passed']:
   gate.finish(); server.shutdown(); thread.join(timeout=5); server.server_close(); server=None
   audit=recorder.audit_structured(record['task_id'],gate=gate,proxy=proxy,dispatcher=dispatcher)
   record['path_audit']=asdict(audit); record['path_abstention_reason']=recorder.abstention_reason
   record['official']=d.finish(episode,result['output'])
   record['attack_payload_visible_to_model']={k:any(v in json.dumps(req,ensure_ascii=False) for req in proxy.raw_requests) for k,v in attack.items()}
   record['reset_verified']=record['initial_state_sha256']==digest(d.start(task_id,seed)['episode'].environment)
  else: gate.abort(); record['unscorable_reason']='incomplete_or_unattested'
except Exception as e:
 record['unscorable_reason']=type(e).__name__+': '+str(e)
finally:
 record['code_hashes_after']=code_hashes()
 record['code_integrity_verified']=record['code_hashes_before']==record['code_hashes_after']
 if not record['code_integrity_verified']: record['unscorable_reason']='bridge_code_changed_during_run'
 record['elapsed_seconds']=round(time.monotonic()-start,3)
 with out.open('x') as f: f.write(json.dumps(record,ensure_ascii=False,indent=2,default=lambda x:dict(x) if isinstance(x,Mapping) else str(x))+'\n')
 if server: server.shutdown(); thread.join(timeout=5); server.server_close()
print(json.dumps({k:record.get(k) for k in ('scope','status','error_message','unscorable_reason','attestation_passed','official','reset_verified','elapsed_seconds')},ensure_ascii=False))
