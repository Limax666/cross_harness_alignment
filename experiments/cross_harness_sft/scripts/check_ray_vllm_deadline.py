#!/usr/bin/env python3
"""Actual Ray RPC -> vLLM engine deadline/abort test; no model updates."""
import argparse,asyncio,json,os,time
from pathlib import Path
import ray
from omegaconf import OmegaConf
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer,_track_generation_task

@ray.remote(num_gpus=1,max_concurrency=32)
class EngineActor:
 def __init__(self,model):self.model=model;self.node_rank=0;self.engine=None;self.events=[]
 async def init(self):
  from vllm import AsyncEngineArgs
  from vllm.v1.engine.async_llm import AsyncLLM
  self.engine=AsyncLLM.from_engine_args(AsyncEngineArgs(model=self.model,tokenizer=self.model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.65,enforce_eager=True,skip_mm_profiling=True,max_num_seqs=1))
  return True
 @_track_generation_task
 async def generate(self,prompt_ids,sampling_params,request_id,**kwargs):
  from vllm import SamplingParams
  self.events.append({'event':'start','id':request_id})
  if request_id=='pre-registration':await asyncio.sleep(15)
  last=None
  async for last in self.engine.generate({'prompt_token_ids':prompt_ids},SamplingParams(temperature=0.8,max_tokens=2048,ignore_eos=True),request_id):pass
  return last
 async def abort_request(self,request_id,reset_prefix_cache=False):
  ids=list(self.engine.output_processor.external_req_ids.get(request_id,[]))
  result=await vLLMHttpServer.abort_request(self,request_id,reset_prefix_cache=reset_prefix_cache)
  self.events.append({'event':'abort','id':request_id,'registered_internal_ids':ids,'result':result})
  return result
 async def status(self,request_id):
  return {'external_present':bool(self.engine.output_processor.external_req_ids.get(request_id)), 'task_present':request_id in getattr(self,'_generation_tasks',{}),'events':self.events}
 async def followup(self,ids):
  from vllm import SamplingParams
  async for result in self.engine.generate({'prompt_token_ids':ids},SamplingParams(temperature=0,max_tokens=4),'followup'):
   if result.finished:return bool(result.outputs[0].token_ids)
  return False
 async def close(self):
  if self.engine:self.engine.shutdown()

a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
if args.output.exists():raise FileExistsError(args.output)

async def main():
 from transformers import AutoTokenizer
 ray.init(num_cpus=4,num_gpus=1,object_store_memory=100*1024*1024,include_dashboard=False,log_to_driver=False)
 actor=EngineActor.remote(args.model);results=[];error=None
 try:
  await actor.init.remote()
  ids=AutoTokenizer.from_pretrained(args.model,local_files_only=True).encode('Explain this topic in detail.',add_special_tokens=True)
  config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'name':'vllm','full_determinism':True}}})
  client=LLMServerClient(config=config,load_balancer_handle=None)
  async def acquire(*a,**kw):return 'ray-engine',actor
  client._acquire_server=acquire;client._release_server=lambda *a,**kw:None
  for name,seconds in [('pre-registration',1),('registered',10)]:
   os.environ['VERL_GENERATION_TIMEOUT_SECONDS']=str(seconds)
   timed_out=False
   try:await client.generate(name,prompt_ids=ids,sampling_params={})
   except TimeoutError:timed_out=True
   status=await actor.status.remote(name)
   if not timed_out or status['external_present'] or status['task_present']:raise RuntimeError(f'{name} did not cancel cleanly: {status}')
   results.append({'case':name,'timeout_seen':timed_out,'status':status})
  followup=await actor.followup.remote(ids)
  if not followup:raise RuntimeError('post-cancellation request failed')
 except Exception as e:error=repr(e);followup=False
 finally:
  try:await actor.close.remote()
  except Exception:pass
  ray.kill(actor,no_restart=True);ray.shutdown()
 payload={'results':results,'followup_ok':followup,'error':error,'scope':'Ray async actor RPC and actual vLLM engine with project abort method; actor initialization is an integration harness'}
 args.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2));return 1 if error else 0
if __name__=='__main__':raise SystemExit(asyncio.run(main()))
