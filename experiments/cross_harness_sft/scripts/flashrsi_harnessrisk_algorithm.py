"""Run an upstream FlashRSI method and score its snapshot on all HarnessRisk cases."""
import argparse,asyncio,json,sys
from pathlib import Path
from flashrsi.agents.llm import LLM
from flashrsi.agents.miniagent import MiniAgent
from flashrsi.agents.opencode import OpenCode
from flashrsi.agents.openclaw import OpenClaw
from flashrsi.agents.hermes import Hermes
from flashrsi.agents.pi_agent import Pi
from flashrsi.agents.claude_code import ClaudeCode
from flashrsi.agents.codex import Codex
import os
from flashrsi.algorithms.opro import OPRO
from flashrsi.algorithms.gepa import GEPA
from flashrsi.algorithms.evoprompt import EvoPrompt
from flashrsi.algorithms.ace import ACE
from flashrsi.algorithms.skillopt import SkillOpt
from flashrsi.algorithms.evo_harness import EvoHarness
from flashrsi_evo_skill import RestrictedSkillWorker
from flashrsi.core import Snapshot
from flashrsi_harnessrisk_benchmark import HarnessRiskBenchmark,ROOT

ALGORITHMS={
 'opro':(OPRO,{'max_steps':1,'num_candidates':1,'history_size':2,'few_shot_size':1}),
 'gepa':(GEPA,{'num_epochs':1,'reflection_minibatch_size':3,'skip_perfect_score':True,'use_merge':False}),
 'evoprompt':(EvoPrompt,{'variant':'ga','population_size':2,'max_steps':1,'selection':'random','survivor_selection':'topk','seed':42}),
 'ace':(ACE,{'mode':'offline','num_epochs':1,'batch_size':1,'max_steps':1,'edit_budget':1,'eval_steps':1,'max_reflection_rounds':1,'seed':42}),
 'skillopt':(SkillOpt,{'num_epochs':1,'batch_size':4,'reflection_minibatch_size':3,'max_steps':1,'edit_budget':1,'seed':42}),
 'evo_harness':(EvoHarness,{'batch_size':4,'max_steps':1,'success_threshold':1.0}),
}
def resolve_algorithm(name):
 cls,config=ALGORITHMS[name]
 if name=='gepa' and os.environ.get('HR_GEPA_REFLECTION_PROTOCOL')=='hermes-v10-contextfit':
  from flashrsi_hermes_gepa_contextfit_v10 import HermesContextfitGEPA
  cls=HermesContextfitGEPA
 if name=='gepa' and os.environ.get('HR_GEPA_REFLECTION_PROTOCOL')=='hermes-v6-quoted-evidence':
  from flashrsi_hermes_gepa_reflection_v6 import HermesQuotedEvidenceGEPA
  cls=HermesQuotedEvidenceGEPA
 return cls,config
async def main():
 p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=ALGORITHMS,required=True);args=p.parse_args()
 harness=os.environ.get('HR_HARNESS','miniagent')
 classes={'miniagent':MiniAgent,'opencode':OpenCode,'openclaw':OpenClaw,'hermes':Hermes,'pi':Pi,'claude_code':ClaudeCode,'codex':Codex}
 if harness not in classes:raise ValueError('unsupported harness')
 base=Path(os.environ.get('HR_OUTPUT_ROOT',str(ROOT/'outputs/flashrsi_harnessrisk_4b_20261113'))).resolve();initial=base/('miniagent_initial_full_v2' if harness=='miniagent' else f'{harness}_initial_full')
 if not (initial/'completed.json').exists():raise RuntimeError('initial full baseline must complete before evolution')
 if json.loads((initial/'run_manifest.json').read_text()).get('privacy_protocol')!='harnessrisk-private-case-v1':raise RuntimeError('initial privacy protocol mismatch')
 run=base/f'{harness}_{args.algorithm}_full';run.mkdir(exist_ok=False);(run/'snapshots').mkdir();(run/'trajectories').mkdir()
 benchmark=HarnessRiskBenchmark(run);splits=benchmark.load()
 target=classes[harness]({'llm':{'model_name':'qwen35-4b-preflight','api_key':'placeholder','base_url':'http://127.0.0.1:1/v1'},'max_turns':12,'timeout':360})
 snapshot=Snapshot.load(initial/'initial_snapshot.json');target.update(snapshot);target._env_runner=benchmark.env_runner
 key=(ROOT/'outputs/preflight_20261111/vllm_api_token').read_text().strip()
 worker_cls=RestrictedSkillWorker if args.algorithm=='evo_harness' else LLM
 worker_port=int(os.environ.get('HARNESSRISK_MODEL_PORT','18761'))
 if not 1<=worker_port<=65535:raise ValueError('invalid model port')
 worker=worker_cls({'type':'llm','llm':{'model_name':'qwen35-4b-preflight','api_key':key,'base_url':f'http://127.0.0.1:{worker_port}/v1',
         'max_tokens':int(os.environ.get('HR_WORKER_MAX_TOKENS','1024')),'temperature':0.2,'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}},'timeout':300,'max_retries':1})
 cls,config=resolve_algorithm(args.algorithm);algorithm=cls(config,target=target,worker=worker,benchmark=benchmark);algorithm._run_dir=run
 (run/'run_manifest.json').write_text(json.dumps({'algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,'evolution_config':config,
    'privacy_protocol':'harnessrisk-private-case-v1','target':'qwen35-4b-preflight','worker':'qwen35-4b-preflight','model_port':worker_port,'objective':'official rule utility_score',
    'scoring_contract': 'reject_unscorable_during_evolution' if os.environ.get('HR_STRICT_EVOLUTION','1')=='1' else 'legacy_errors_score_zero',
    'host_script':os.environ.get('HR_HOST_SCRIPT','flashrsi_harnessrisk_host.py'),'worker_max_tokens':int(os.environ.get('HR_WORKER_MAX_TOKENS','1024')),'evolution_case_attempts':int(os.environ.get('HR_EVOLUTION_CASE_ATTEMPTS','1')),
    'worker_note':'restricted JSON skill editor adaptation' if args.algorithm=='evo_harness' else 'upstream LLM worker',
    'splits':{k:[t.task_id for t in v] for k,v in splits.items()},'evaluation_case_ids':[t.task_id for t in benchmark.full],
    'full_evaluation_exposure':'includes train and dev cases; not an independent held-out test'},indent=2)+'\n')
 try:
  benchmark.reject_unscorable=os.environ.get('HR_STRICT_EVOLUTION','1')=='1'
  selected=await algorithm.evolve();selected.save(run/'snapshots'/'selected.json')
  benchmark.reject_unscorable=False
  first_selected_sequence=benchmark._sequence+1
  scored=await algorithm._evaluate(selected,benchmark.full,label='selected_full_128')
  rows=[{'case_id':x.task.task_id,'status':x.status,'metrics':dict(x.metrics),'error_message':x.error_message} for x in scored]
  selected_cases=run/'selected_cases';selected_cases.mkdir()
  for offset,task in enumerate(benchmark.full):
   case_path=run/'evidence'/f'{first_selected_sequence+offset:08d}'/'cases'/task.task_id
   if rows[offset]['status']=='finished' and (case_path/'trajectory'/'trajectory.json').is_file():
    (selected_cases/task.task_id).symlink_to(case_path.resolve(),target_is_directory=True)
  (run/'result.json').write_text(json.dumps({'status':'finished','initial_snapshot_id':snapshot.id,'selected_snapshot_id':selected.id,
   'selected_changed':snapshot.id!=selected.id,'results':rows},indent=2)+'\n')
 except BaseException as e:
  (run/'error.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e)[:2000]},indent=2)+'\n');raise
if __name__=='__main__':asyncio.run(main())
