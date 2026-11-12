import asyncio,hashlib,json,os,secrets,sys,tempfile,threading,time
from pathlib import Path
from collections.abc import Mapping
from dataclasses import asdict
from cross_harness_sft.backends.agentharm import AgentHarmDriver
from flashrsi.agents.miniagent import MiniAgent,_ENTRY
from flashrsi_gateway import create_gateway
from flashrsi_model_relay import relay_request
from flashrsi_restricted_snapshot import make_restricted_snapshot
from flashrsi_validation_verifiers import verify_workspace_dinner_answer, verify_workspace_hiking_answer
from flashrsi_tool_attestation import ToolCallGate,GuardedChatProxy,GuardedToolDispatcher,attest_tool_calls,attest_tool_observations
root=Path('/output/cross_harness_alignment/experiments/cross_harness_sft')
task_id=os.environ['B0_TASK_ID']; split=os.environ['B0_SPLIT']; seed=0
out=root/'outputs/b0_full_agentharm_20261112'/'cases'/(split+'__'+hashlib.sha256(task_id.encode()).hexdigest()+'.json')
record={'scope':'AgentHarm full task trajectory collection; scoring pending','split':split,'seed':seed,'task_id':task_id,'model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','max_turns':12,'max_tokens_per_turn':512,'thinking':False}
files=['scripts/flashrsi_agentharm_full_supervisor.py','scripts/flashrsi_agentharm_full_host.py','scripts/flashrsi_gateway.py','scripts/flashrsi_landlock.py','scripts/flashrsi_model_relay.py','scripts/flashrsi_restricted_snapshot.py','scripts/flashrsi_sandbox_exec.py','scripts/flashrsi_seccomp.py','scripts/flashrsi_tool_attestation.py','src/cross_harness_sft/backends/agentharm.py','configs/flashrsi_b0_safety_policy.txt']
def code_hashes(): return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
record['code_hashes_before']=code_hashes()
class Recorder:
 def __init__(self,driver,episode): self.driver=driver; self.episode=episode
 def tool(self,name,arguments): return self.driver.tool(self.episode,name,arguments)
server=None; thread=None; start=time.monotonic()
try:
 config=root/('configs/agentharm_worker_val.yaml' if split=='val' else 'configs/agentharm_worker_test_public.yaml')
 d=AgentHarmDriver(str(config)); started=d.start(task_id,seed)
 episode=started['episode']; tools=started['tools']; prompt=str(episode.sample.input)
 record['tool_schema_sha256']=hashlib.sha256(json.dumps(tools,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 gate=ToolCallGate({t['function']['name'] for t in tools})
 relay_path=os.environ['DEV_RELAY_PATH']; relay_token=os.environ['DEV_RELAY_TOKEN']
 def upstream(request): return relay_request(relay_path,relay_token,request)
 proxy=GuardedChatProxy(gate,'qwen35-4b-preflight',tools,upstream)
 recorder=Recorder(d,episode)
 dispatcher=GuardedToolDispatcher(gate,recorder)
 token=secrets.token_urlsafe(32); server=create_gateway(proxy,dispatcher,gate,token)
 thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
 base=f'http://127.0.0.1:{server.server_port}'
 with tempfile.TemporaryDirectory(prefix='rsi-child-') as parent:
  parent=Path(parent); writable=parent/'work'; writable.mkdir(); config=parent/'config.json'
  config.write_text(json.dumps({'entry':str(_ENTRY),'python':sys.executable,'gateway_port':server.server_port,
    'readonly_roots':['/usr','/lib','/lib64','/etc/ssl/certs','/opt/uv/python',str(root/'.venv-agentharm'),str(_ENTRY.parent)],'writable_root':str(writable)}))
  os.chmod(config,0o600); os.environ['FLASHRSI_SANDBOX_CONFIG']=str(config); os.environ['FLASHRSI_GATEWAY_URL']=base; os.environ['FLASHRSI_GATEWAY_TOKEN']=token
  tempfile.tempdir=str(writable)
  agent=MiniAgent({'llm':{'model_name':'qwen35-4b-preflight','api_key':token,'base_url':base+'/v1','max_tokens':512,'extra_body':{'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'seed':seed}},'max_turns':12,'timeout':300})
  record['root_snapshot']=agent.snapshot.id; snapshot=make_restricted_snapshot(agent,tools,safety_policy=(root/'configs/flashrsi_b0_safety_policy.txt').read_text()); agent.update(snapshot)
  record['adapted_snapshot']=snapshot.id; record['tool_count']=len(tools)
  sys.executable=str(root/'scripts/flashrsi_sandbox_exec.py')
  result=asyncio.run(agent.run(prompt))
  record.update(status=result.status,error_message=result.error_message,output=result.output,trajectory_messages=result.messages,usage=result.usage,raw_model_requests=proxy.raw_requests,raw_model_messages=proxy.raw_messages,dispatches=dispatcher.executed)
  record['attestation_passed']=attest_tool_calls(proxy.raw_messages,dispatcher.executed)
  if result.status=='finished' and record['attestation_passed']:
   gate.finish(); server.shutdown(); thread.join(timeout=5); server.server_close(); server=None
   record['observation_attestation_passed']=attest_tool_observations(proxy.raw_messages,proxy.raw_requests,dispatcher.observations)
   record['native_tool_trace']=episode.trace
   record['scoring_status']='not_scored_requires_independent_agentharm_judge'
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
