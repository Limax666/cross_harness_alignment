#!/usr/bin/env python3
"""Compare frozen, greedy first-turn outputs; never infer multi-turn performance."""
import argparse,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args(); root=a.directory
 groups={}
 for role in ('sft','rl'):
  for backend in ('transformers','vllm'):
   f=root/f'{role}_{backend}.jsonl';rows=[json.loads(x) for x in f.open() if x.strip()];groups[role,backend]={x['cell_id']:x for x in rows}
 result=[]
 for role in ('sft','rl'):
  left,right=groups[role,'transformers'],groups[role,'vllm']
  if left.keys()!=right.keys():raise ValueError('case sets differ')
  for cid in sorted(left):
   x,y=left[cid],right[cid]
   if x['prompt_token_ids']!=y['prompt_token_ids']:raise ValueError('prompt tokens differ')
   a1,a2=x['generated_token_ids'],y['generated_token_ids'];common=0
   for aa,bb in zip(a1,a2):
    if aa!=bb:break
    common+=1
   diff=[abs(x['token_logprobs'][i]-y['token_logprobs'][i]) for i in range(min(common,len(x['token_logprobs']),len(y['token_logprobs'])))]
   result.append({'model':role,'cell_id':cid,'same_prompt':True,'same_output_tokens':a1==a2,'common_prefix_tokens':common,'tf_tokens':len(a1),'vllm_tokens':len(a2),'max_logprob_delta_common_prefix':max(diff) if diff else None,'tf_parsed':x['parsed'],'vllm_parsed':y['parsed']})
 out=root/'comparison.json';out.write_text(json.dumps({'method':'greedy first-turn; standalone vLLM; not live VeRL sync','comparisons':result},ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
