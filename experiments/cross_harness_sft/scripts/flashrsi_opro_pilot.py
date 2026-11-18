"""Two-case, train-only OPRO integration probe; not a formal baseline."""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path

from flashrsi.agents.llm import LLM
from flashrsi.agents.miniagent import MiniAgent
from flashrsi.core import Snapshot
from flashrsi_gepa_security import AgentDojoSecurityBenchmark, ROOT, SuiteAwareMiniAgent
from flashrsi_safe_opro import SafetyOPRO

RUN = Path(os.environ.get('FLASHRSI_OPRO_RUN', str(ROOT / 'outputs/flashrsi_local_20261112/opro_miniagent_train_probe_r4')))
SNAPSHOTS = ROOT / 'outputs/flashrsi_local_20261112/minia_validation_16k/snapshots'


class PrefixOnlyWorker(LLM):
    async def ask(self, prompt):
        return await super().ask(prompt + '\n\nReturn only the editable instruction in the requested tags. '
                                 'Do not include or rewrite the Trusted safety policy or its heading; '
                                 'trusted code appends that immutable policy after your output.')


def select_training_pair(tasks):
    selected = [task for task in tasks if task.metadata.get('family') == 'workspace:user_task_0'
                and task.metadata.get('source_split') == 'train']
    clean = [task for task in selected if task.metadata.get('arm') == 'clean']
    attacks = sorted((task for task in selected if task.metadata.get('arm') == 'attack'),
                     key=lambda task: task.task_id)
    if len(clean) != 1 or not attacks:
        raise ValueError('training family lacks a clean/attack pair')
    return (clean[0], attacks[0])


def prepare_probe_directory(run):
    run.mkdir(parents=True, exist_ok=False)
    (run / 'snapshots').mkdir()
    (run / 'trajectories').mkdir()


async def main():
    prepare_probe_directory(RUN)
    benchmark = AgentDojoSecurityBenchmark({}, run_dir=RUN)
    splits = benchmark.load()
    pair = select_training_pair(splits['train'])
    benchmark._splits = {'train': pair, 'dev': (), 'test': ()}
    snapshots = {suite: Snapshot.load(SNAPSHOTS / f'{suite}.json')
                 for suite in ('workspace', 'travel', 'banking', 'slack')}
    config = {'source_dir': str(Path(importlib.import_module(MiniAgent.__module__).__file__).parent),
              'llm': {'model_name': 'qwen35-4b-preflight', 'api_key': 'environment-episode-runner-only',
                      'base_url': 'http://127.0.0.1:18761/v1', 'max_tokens': 512,
                      'extra_body': {'chat_template_kwargs': {'enable_thinking': False}, 'temperature': 0}},
              'max_turns': 12, 'timeout': 330}
    target = SuiteAwareMiniAgent(config, snapshots)
    target.bind_benchmark(benchmark)
    worker = PrefixOnlyWorker({'type': 'llm', 'llm': {'model_name': 'qwen35-4b-preflight', 'api_key': 'local-placeholder',
                                        'base_url': 'http://127.0.0.1:18761/v1', 'max_tokens': 1024,
                                        'temperature': 0.2,
                                        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}},
                  'timeout': 300, 'max_retries': 1})
    policy = (ROOT / 'configs/flashrsi_b0_safety_policy.txt').read_text(encoding='utf-8')
    algorithm = SafetyOPRO({'max_steps': 1, 'num_candidates': 1, 'history_size': 2, 'few_shot_size': 1},
                           target=target, worker=worker, benchmark=benchmark, safety_policy=policy)
    algorithm._run_dir = RUN
    initial_id = target.snapshot.id
    (RUN / 'pilot_manifest.json').write_text(json.dumps({
        'status': 'development_only_not_formally_qualified', 'algorithm': 'OPRO + project safety gate',
        'tasks': [task.task_id for task in pair], 'split': 'train', 'test_used': False,
        'max_steps': 1, 'num_candidates': 1, 'initial_snapshot_id': initial_id,
        'policy_handling': 'trusted code appends pinned suffix if worker omits marker; modified marker is rejected',
        'max_model_len': 16384, 'target_completion_budget': 512, 'worker_completion_budget': 1024,
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'snapshot_guard_sha256': hashlib.sha256((ROOT / 'scripts/flashrsi_snapshot_guard.py').read_bytes()).hexdigest(),
        'policy_sha256': hashlib.sha256((ROOT / 'configs/flashrsi_b0_safety_policy.txt').read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    try:
        selected = await algorithm.evolve()
        (RUN / 'result.json').write_text(json.dumps({'status': 'finished', 'initial_snapshot_id': initial_id,
                                                    'selected_snapshot_id': selected.id,
                                                    'selected_changed': selected.id != initial_id}, indent=2) + '\n')
    except Exception as error:
        (RUN / 'error.json').write_text(json.dumps({'status': 'error', 'error_type': type(error).__name__,
                                                   'message': str(error)[:1000]}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    asyncio.run(main())
