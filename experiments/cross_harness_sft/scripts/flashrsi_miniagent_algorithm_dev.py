"""Run one bounded MiniAgent RSI development comparison on AgentDojo."""
import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path

from flashrsi.agents.miniagent import MiniAgent
from flashrsi.agents.llm import LLM
from flashrsi.core import Snapshot
from flashrsi_gepa_security import AgentDojoSecurityBenchmark, ROOT, SafetyGEPA, SuiteAwareMiniAgent
from flashrsi_opro_pilot import PrefixOnlyWorker, prepare_probe_directory
from flashrsi_safe_opro import SafetyOPRO, protect_policy_suffix
from flashrsi_safe_evoprompt import SafetyEvoPrompt
from flashrsi_safe_ace import SafetyACE


class PrefixSafetyGEPA(SafetyGEPA):
    def _candidate(self, baseline, files):
        if self._path in files:
            try:
                files = {**files, self._path: protect_policy_suffix(files[self._path], self._pinned_safety_suffix)}
            except ValueError:
                return baseline
        return super()._candidate(baseline, files)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=('opro', 'gepa', 'evoprompt', 'ace'), required=True)
    args = parser.parse_args()
    if os.environ.get('FLASHRSI_COMPLETION_BUDGET') != '4096':
        raise ValueError('this comparison requires completion budget 4096')
    run = ROOT / 'outputs/flashrsi_local_20261112/uniform4096_serial' / f'miniagent_{args.algorithm}_dev'
    prepare_probe_directory(run)
    benchmark = AgentDojoSecurityBenchmark({}, run_dir=run)
    splits = benchmark.load()
    families = sorted({t.metadata['family'] for t in splits['train']})[:2]
    train = tuple(t for t in splits['train'] if t.metadata['family'] in families)
    dev = tuple(splits['dev'])
    if len(families) != 2 or not train or not dev or {t.metadata['family'] for t in train} & {t.metadata['family'] for t in dev}:
        raise ValueError('invalid development split')
    benchmark._splits = {'train': train, 'dev': dev, 'test': ()}
    snapshot_dir = ROOT / 'outputs/flashrsi_local_20261112/minia_validation_16k/snapshots'
    snapshots = {suite: Snapshot.load(snapshot_dir / f'{suite}.json') for suite in ('workspace','travel','banking','slack')}
    config = {'source_dir': str(Path(importlib.import_module(MiniAgent.__module__).__file__).parent),
              'llm': {'model_name':'qwen35-4b-preflight','api_key':'environment-episode-runner-only',
                      'base_url':'http://127.0.0.1:18761/v1','max_tokens':4096,
                      'extra_body':{'chat_template_kwargs':{'enable_thinking':False},'temperature':0}},
              'max_turns':12,'timeout':330}
    target = SuiteAwareMiniAgent(config, snapshots)
    target.bind_benchmark(benchmark)
    token = (ROOT / 'outputs/preflight_20261111/vllm_api_token').read_text().strip()
    worker_type = LLM if args.algorithm == 'ace' else PrefixOnlyWorker
    worker = worker_type({'type':'llm','llm':{'model_name':'qwen35-4b-preflight','api_key':token,
                               'base_url':'http://127.0.0.1:18761/v1','max_tokens':1024,'temperature':0.2,
                               'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}},'timeout':300,'max_retries':1})
    policy = (ROOT / 'configs/flashrsi_b0_safety_policy.txt').read_text()
    if args.algorithm == 'opro':
        algorithm = SafetyOPRO({'max_steps':1,'num_candidates':1,'history_size':2,'few_shot_size':1},target=target,worker=worker,benchmark=benchmark,safety_policy=policy)
    elif args.algorithm == 'gepa':
        algorithm = PrefixSafetyGEPA({'num_epochs':1,'reflection_minibatch_size':3,'max_metric_calls':2*len(dev)+6,
                                     'skip_perfect_score':True,'acceptance_criterion':'strict_improvement',
                                     'candidate_selection_strategy':'pareto','max_evidence_chars':8000,'seed':42,'use_merge':False},
                                    target=target,worker=worker,benchmark=benchmark,safety_policy=policy)
    elif args.algorithm == 'evoprompt':
        algorithm = SafetyEvoPrompt({'variant':'ga','population_size':2,'max_steps':1,'selection':'random',
                                     'survivor_selection':'topk','seed':42},
                                    target=target,worker=worker,benchmark=benchmark,safety_policy=policy)
    else:
        algorithm = SafetyACE({'mode':'offline','num_epochs':1,'batch_size':1,'max_steps':1,'edit_budget':1,
                               'eval_steps':1,'max_reflection_rounds':1,'seed':42},
                              target=target,worker=worker,benchmark=benchmark,safety_policy=policy)
    algorithm._run_dir = run
    initial = target.snapshot.id
    manifest = {'status':'development_only_not_formally_qualified','algorithm':args.algorithm,
                'project_adaptation':'pinned policy and safety admission; prefix-only worker except ACE',
                'train_families':families,'train_task_ids':[t.task_id for t in train],
                'validation_task_ids':[t.task_id for t in dev],'test_used':False,
                'initial_snapshot_id':initial,'target_completion_budget':4096,'worker_completion_budget':1024,
                'max_model_len':16384,'parallel_tool_calls':False,
                'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        selected = await algorithm.evolve()
        selected.save(run/'snapshots'/'selected.json')
        evaluated = await algorithm._evaluate(selected, dev, label='selected_validation')
        rows = [{'task_id':x.task.task_id,'arm':x.task.metadata['arm'],'status':x.status,
                 'metrics':dict(x.metrics)} for x in evaluated]
        (run/'result.json').write_text(json.dumps({'status':'finished','initial_snapshot_id':initial,
            'selected_snapshot_id':selected.id,'selected_changed':selected.id != initial,'validation':rows},indent=2)+'\n')
    except BaseException as error:
        (run/'error.json').write_text(json.dumps({'status':'error','error_type':type(error).__name__,'message':str(error)[:1000]},indent=2)+'\n')
        raise


if __name__ == '__main__':
    asyncio.run(main())
