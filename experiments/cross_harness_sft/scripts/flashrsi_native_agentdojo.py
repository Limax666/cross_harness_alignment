"""Development bridge from FlashRSI native agents to AgentDojo episodes."""
import os
import inspect
from pathlib import Path
from flashrsi.agents.codex import Codex
from flashrsi.agents.claude_code import ClaudeCode
from flashrsi_native_snapshot import make_native_snapshot
from flashrsi_port_agentdojo import PortBenchmark
from flashrsi_snapshot_guard import validate_snapshot_candidate


def build_native(name, benchmark):
    if name not in ('codex', 'claude_code'):
        raise ValueError('unsupported native harness')
    os.environ['NATIVE_HARNESS'] = name
    benchmark.host = 'flashrsi_native_host.py'
    config = {'llm': {'model_name':'qwen35-4b-preflight','api_key':'placeholder',
                      'base_url':'http://127.0.0.1:1/v1'}, 'timeout':300}
    if name == 'claude_code':
        config.update(cli_path='/data/home/liumingxiao/.npm-global/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe',max_turns=12)
    initial=make_native_snapshot(name,config)
    prompt='AGENTS.md' if name=='codex' else 'CLAUDE.md'
    base_class={'codex':Codex,'claude_code':ClaudeCode}[name]

    class NativeTarget(base_class):
        def __init__(self):
            super().__init__({**config,'source_dir':str(Path(inspect.getfile(base_class)).parent)})
            _ = self.snapshot
            self.update(initial)
            self._env_runner=benchmark.env_runner
            self._canonical_candidate_id=None

        async def rollout(self,tasks,*,snapshot=None):
            candidate=self.snapshot if snapshot is None else snapshot
            validate_snapshot_candidate(initial,candidate,{prompt},prompt)
            self._canonical_candidate_id=candidate.id
            return await super().rollout(tasks,snapshot=candidate)

    return NativeTarget(), {'all_suites':initial}
