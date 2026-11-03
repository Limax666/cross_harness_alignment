#!/usr/bin/env python3
"""Fixed first-turn greedy diagnostic; no policy update, one backend/model per process."""
import argparse,hashlib,json,os,sys
from collections.abc import Mapping
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
os.environ['VERL_QWEN_SCHEMA_STRING_PARAMS']='1'
from rl_multiharness_chat_template import benchmark_mcp_chat_template
from rl_whitebox_harness import transform_tools
from rl_tool_attempt_audit import parse_strict_tool_turn

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--backend',choices=['transformers','vllm'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=4);p.add_argument('--max-tokens',type=int,default=192);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import pandas as pd
 from transformers import AutoTokenizer
 from cross_harness_sft.backends.agentdojo import AgentDojoDriver
 ledger=json.loads((ROOT/'outputs/rl/9b_whitebox_pool_v4/assignment_ledger.json').read_text()); assigned=ledger['rounds'][0]['assignments'][:a.cases]
 rows={r['cell_id']:r for r in pd.read_parquet(ROOT/'outputs/rl/9b_whitebox_pool_v4/train.parquet').to_dict('records')}
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.chat_template=benchmark_mcp_chat_template()
 driver=AgentDojoDriver(str(ROOT/'configs/agentdojo_worker.yaml')); prompts=[]
 for assignment in assigned:
  row=rows[assignment['cell_id']]; raw=driver.start(row['task_id'],int(row['semantic_fixture_seed'])); tools,_=transform_tools(raw['tools'],tuple(int(v) for v in row['harness_config']))
  actual=hashlib.sha256(json.dumps(tools,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
  if actual!=row['tool_contract_sha256']:raise ValueError('tool schema mismatch')
  messages=list(row['prompt']);ids=tok.apply_chat_template(messages,tools=tools,tokenize=True,add_generation_prompt=True)
  if isinstance(ids,Mapping):ids=ids['input_ids']
  if hasattr(ids,'tolist'):ids=ids.tolist()
  if ids and isinstance(ids[0],list):ids=ids[0]
  prompts.append((assignment,tools,ids))
 if a.backend=='transformers':
  import torch
  from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
  stops=[tok.encode(x,add_special_tokens=False) for x in ('</tool_call>','<|im_end|>')]
  class Boundary(StoppingCriteria):
   def __call__(self,input_ids,scores,**kwargs):
    values=input_ids[0].tolist()
    return any(x and values[-len(x):]==x for x in stops)
  model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,use_kernels=False,device_map={'':0}).eval()
 else:
  from vllm import LLM,SamplingParams
  model=LLM(model=a.model,tokenizer=a.model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.70,enforce_eager=True,trust_remote_code=False,skip_mm_profiling=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as out:
  for assignment,tools,ids in prompts:
   if a.backend=='transformers':
    import torch
    tensor=torch.tensor([ids],device='cuda')
    with torch.inference_mode():result=model.generate(tensor,attention_mask=torch.ones_like(tensor),do_sample=False,max_new_tokens=a.max_tokens,return_dict_in_generate=True,output_scores=True,stopping_criteria=StoppingCriteriaList([Boundary()]),pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen=result.sequences[0,len(ids):].tolist(); lps=[float(torch.log_softmax(score.float(),dim=-1)[0,t].item()) for score,t in zip(result.scores,gen)]
   else:
    from vllm import SamplingParams
    x=model.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=a.max_tokens,logprobs=1,stop=['</tool_call>','<|im_end|>'],include_stop_str_in_output=True),use_tqdm=False)[0].outputs[0]
    gen=x.token_ids;lps=[float(v[t].logprob) for v,t in zip(x.logprobs,gen)]
   raw=tok.decode(gen,skip_special_tokens=False)
   try: content,calls=parse_strict_tool_turn(raw,tools,allow_schema_errors=True); parsed={'calls':calls,'content':content[:500]}
   except ValueError as e:parsed={'error':str(e)}
   rec={'cell_id':assignment['cell_id'],'harness':assignment['harness_name'],'model':a.model,'backend':a.backend,'prompt_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'prompt_token_ids':ids,'generated_token_ids':list(gen),'token_logprobs':lps,'raw_output':raw,'parsed':parsed}
   out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();print(json.dumps({'cell_id':rec['cell_id'],'tokens':len(gen),'parsed':parsed},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
