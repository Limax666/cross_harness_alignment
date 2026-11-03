#!/usr/bin/env python3
"""Compare step-0 recorded logprobs with frozen HF model on exact first-turn token IDs."""
import argparse,gzip,hashlib,json,os,sys
from pathlib import Path
from collections.abc import Mapping
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'scripts'),str(root/'src')]
os.environ['VERL_QWEN_SCHEMA_STRING_PARAMS']='1'
from rl_multiharness_chat_template import benchmark_mcp_chat_template
from rl_whitebox_harness import transform_tools

def main():
 p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=4);p.add_argument('--last-segment',action='store_true');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import torch,pandas as pd
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from cross_harness_sft.backends.agentdojo import AgentDojoDriver
 rows={r['cell_id']:r for r in pd.read_parquet(root/'outputs/rl/9b_whitebox_pool_v4/train.parquet').to_dict('records')}
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.chat_template=benchmark_mcp_chat_template()
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,use_kernels=False,device_map={'':0}).eval()
 driver=AgentDojoDriver(str(root/'configs/agentdojo_worker.yaml'))
 seen=set()
 with gzip.open(a.rollouts,'rt') as f,a.output.open('x') as out:
  for line in f:
   r=json.loads(line);e=r['evidence']
   if e['cell_id'] in seen:continue
   if len(seen)>=a.count:break
   seen.add(e['cell_id']);row=rows[e['cell_id']];started=driver.start(row['task_id'],int(row['semantic_fixture_seed']));tools,_=transform_tools(started['tools'],tuple(int(v) for v in row['harness_config']))
   h=hashlib.sha256(json.dumps(tools,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
   if h!=row['tool_contract_sha256']:raise ValueError('schema mismatch')
   ids=tok.apply_chat_template(list(row['prompt']),tools=tools,tokenize=True,add_generation_prompt=True)
   if isinstance(ids,Mapping):ids=ids['input_ids']
   if hasattr(ids,'tolist'):ids=ids.tolist()
   if ids and isinstance(ids[0],list):ids=ids[0]
   response=r['response_ids'];mask=r['response_mask'];segments=[];start=None
   for i,m in enumerate(mask+[0]):
    if m==1 and start is None:start=i
    if m!=1 and start is not None:segments.append((start,i));start=None
   if not segments:raise ValueError('no assistant tokens')
   start,end=segments[-1] if a.last_segment else segments[0];n=min(end-start,64)
   full=torch.tensor([ids+response[:start+n]],device='cuda')
   with torch.inference_mode():logits=model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False,logits_to_keep=n+1).logits[0,:-1].float()
   targets=torch.tensor(response[start:start+n],device='cuda')
   hf1=torch.log_softmax(logits,dim=-1).gather(1,targets[:,None]).squeeze(1).cpu().tolist()
   hf08=torch.log_softmax(logits/0.8,dim=-1).gather(1,targets[:,None]).squeeze(1).cpu().tolist()
   result={'rollout_id':e['rollout_id'],'cell_id':e['cell_id'],'segment_start':start,'prompt_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'tokens':response[start:start+n],'rollout_logprobs':r['rollout_log_probs'][start:start+n],'old_logprobs':r['old_log_probs'][start:start+n],'hf_logprobs_temperature_1':hf1,'hf_logprobs_temperature_08':hf08}
   out.write(json.dumps(result,allow_nan=False)+'\n');out.flush();print('evaluated',e['cell_id'],n,flush=True)
if __name__=='__main__':main()
