#!/usr/bin/env python3
"""Exercise opt-in client deadline against a real vLLM engine, without RL updates."""
import argparse,asyncio,json,os,time
from pathlib import Path
from types import SimpleNamespace

async def run(model,output):
 from omegaconf import OmegaConf
 from transformers import AutoTokenizer
 from vllm import AsyncEngineArgs,SamplingParams
 from vllm.v1.engine.async_llm import AsyncLLM
 from verl.workers.rollout.llm_server import LLMServerClient
 from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
 args=AsyncEngineArgs(model=model,tokenizer=model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.65,enforce_eager=True,skip_mm_profiling=True,max_num_seqs=1)
 engine=AsyncLLM.from_engine_args(args)
 server_impl=object.__new__(vLLMHttpServer);server_impl.node_rank=0;server_impl.engine=engine
 events=[];backend_id='deadline-live-check'
 async def generate(**kwargs):
  events.append({'event':'generate','request_id':kwargs['request_id']})
  prompt={'prompt_token_ids':kwargs['prompt_ids']};params=SamplingParams(temperature=0.8,max_tokens=2048,ignore_eos=True)
  last=None
  async for last in engine.generate(prompt,params,kwargs['request_id']):pass
  return last
 async def abort(request_id,reset_prefix_cache):
  events.append({'event':'pre_abort_registry','external_ids':list(engine.output_processor.external_req_ids.keys()),'internal_ids':list(engine.output_processor.request_states.keys())})
  result=await server_impl.abort_request(request_id,reset_prefix_cache=reset_prefix_cache)
  events.append({'event':'abort','request_id':request_id,'result':result});return result
 server=SimpleNamespace(generate=SimpleNamespace(remote=generate),abort_request=SimpleNamespace(remote=abort))
 client=LLMServerClient(config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'name':'vllm','full_determinism':False}}}),load_balancer_handle=None)
 async def acquire(*args,**kwargs):return 'local-real-vllm',server
 client._acquire_server=acquire;client._vllm_request_id=lambda _:backend_id
 client._release_server=lambda sid,request_id=None:events.append({'event':'release','id':sid,'request_id':request_id})
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True);ids=tok.encode('Explain a long technical subject in detail:',add_special_tokens=True)
 start=time.monotonic();timeout_seen=False;after_ok=False;remaining=None;error=None
 try:
  try:await client.generate('sticky',prompt_ids=ids,sampling_params={})
  except TimeoutError:timeout_seen=True
  else:raise AssertionError('request finished without triggering deadline')
  remaining=(backend_id in engine.output_processor.request_states or
             bool(engine.output_processor.external_req_ids.get(backend_id)))
  if remaining:raise AssertionError('backend request still registered after abort')
  params=SamplingParams(temperature=0,max_tokens=4)
  async for result in engine.generate({'prompt_token_ids':ids},params,'after-deadline-check'):
   if result.finished:after_ok=bool(result.outputs[0].token_ids)
  if not after_ok:raise AssertionError('subsequent request did not complete')
 except Exception as e:error=repr(e)
 finally:engine.shutdown()
 payload={'model':model,'deadline_seconds':os.environ['VERL_GENERATION_TIMEOUT_SECONDS'],'elapsed_seconds':time.monotonic()-start,'timeout_seen':timeout_seen,'backend_request_remaining':remaining,'subsequent_request_ok':after_ok,'events':events,'error':error,'scope':'real vLLM engine and vLLMHttpServer abort method with in-process RPC shim; distributed Ray transport not tested'}
 output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2));return 1 if error else 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 return asyncio.run(run(a.model,a.output))
if __name__=='__main__':raise SystemExit(main())
