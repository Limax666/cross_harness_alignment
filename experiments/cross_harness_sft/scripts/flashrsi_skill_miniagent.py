"""MiniAgent skill-condition development target for AgentDojo."""
import hashlib
from pathlib import Path
import yaml
from flashrsi.agents.miniagent import MiniAgent
from flashrsi.core import Snapshot
from flashrsi_gepa_security import ROOT
from flashrsi_snapshot_guard import validate_snapshot_candidate

SKILL_PATH='skills/rsi.md'
PROMPT_PATH='prompts/system.md'
INITIAL_SKILL='Use the available tools to complete the user task carefully.\n'


def make_skill_suites(snapshot_dir):
    suites={}
    for suite in ('workspace','travel','banking','slack'):
        original=Snapshot.load(Path(snapshot_dir)/f'{suite}.json')
        with original.check_out() as directory:
            config_path=directory/'agent.yaml'; config=yaml.safe_load(config_path.read_text())
            if config.get('skills') != []:
                raise ValueError('expected restricted initial Snapshot without skills')
            config['skills']=['rsi'];config_path.write_text(yaml.safe_dump(config,sort_keys=True))
            skill=directory/SKILL_PATH;skill.parent.mkdir(exist_ok=True);skill.write_text(INITIAL_SKILL)
            suites[suite]=original.commit(directory)
    return suites


class SuiteAwareSkillMiniAgent(MiniAgent):
    def __init__(self, config, suite_snapshots):
        super().__init__(config)
        self._suite_snapshots=dict(suite_snapshots)
        self._canonical_base=self._suite_snapshots[sorted(self._suite_snapshots)[0]]
        self._authorized_effective={}
        self._canonical_candidate_id=None
        self.update(self._canonical_base)

    def bind_benchmark(self, benchmark):
        self._env_runner=benchmark.env_runner

    def _validate_snapshot(self, snapshot, *, baseline=None):
        authorized=getattr(self,'_authorized_effective',{}).get(snapshot.id)
        if authorized is None:
            if hasattr(self,'_canonical_base'):
                validate_snapshot_candidate(self._canonical_base,snapshot,{SKILL_PATH},PROMPT_PATH)
                if SKILL_PATH not in snapshot.files:
                    raise ValueError('required skill file is missing')
            return super()._validate_snapshot(snapshot,baseline=baseline)
        suite,digest=authorized
        validate_snapshot_candidate(self._suite_snapshots[suite],snapshot,{SKILL_PATH},PROMPT_PATH)
        if hashlib.sha256(snapshot.files[SKILL_PATH].content).hexdigest()!=digest:
            raise ValueError('effective skill identity mismatch')

    async def rollout(self,tasks,*,snapshot=None):
        candidate=self.snapshot if snapshot is None else snapshot
        validate_snapshot_candidate(self._canonical_base,candidate,{SKILL_PATH},PROMPT_PATH)
        content=candidate.files[SKILL_PATH].content;digest=hashlib.sha256(content).hexdigest()
        self._canonical_candidate_id=candidate.id;rows=[]
        for task in tuple(tasks):
            suite=task.metadata.get('suite')
            if suite not in self._suite_snapshots: raise ValueError('unknown AgentDojo suite')
            base=self._suite_snapshots[suite]
            with base.check_out() as directory:
                (directory/SKILL_PATH).write_bytes(content)
                effective=base.commit(directory)._with_scope(candidate._scope)
            self._authorized_effective[effective.id]=(suite,digest)
            rows.extend(await super().rollout((task,),snapshot=effective))
        return tuple(rows)
