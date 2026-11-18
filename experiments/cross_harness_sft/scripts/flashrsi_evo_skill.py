"""Restricted directory skills and a text-only editing worker for Evo-Harness."""
import asyncio
import hashlib
import json
import re
import shutil
import os
import sys
from pathlib import Path
import yaml
from flashrsi.agents.llm import LLM
from flashrsi.agents.miniagent import MiniAgent
from flashrsi.core import Snapshot, Trajectory
from flashrsi_gepa_security import AgentDojoSecurityBenchmark, ROOT, _safe_name
from flashrsi_restricted_snapshot import _TOOL_SOURCE
from flashrsi_snapshot_guard import validate_snapshot_candidate

PROMPT='prompts/system.md'
SKILL=re.compile(r'skills/[A-Za-z0-9_-]+/SKILL\.md\Z')
READ_SCHEMA={'name':'read_skill','description':'Read a named directory skill available in this workspace.',
             'parameters':{'type':'object','properties':{'skill':{'type':'string'}},'required':['skill'],'additionalProperties':False}}


def make_evo_suites(snapshot_dir):
    suites={}
    for suite in ('workspace','travel','banking','slack'):
        baseline=Snapshot.load(Path(snapshot_dir)/f'{suite}.json')
        with baseline.check_out() as directory:
            path=directory/'agent.yaml';config=yaml.safe_load(path.read_text())
            if config.get('skills') != [] or 'read_skill' in config.get('tools',[]):
                raise ValueError('expected original restricted baseline')
            config['tools'].append('read_skill');path.write_text(yaml.safe_dump(config,sort_keys=True))
            (directory/'tools'/'read_skill.py').write_text('TOOL_NAME = '+repr('read_skill')+'\n'+_TOOL_SOURCE)
            (directory/'tool_descriptions'/'read_skill.json').write_text(json.dumps(READ_SCHEMA))
            suites[suite]=baseline.commit(directory)
    return suites


def validate_evo_candidate(baseline,candidate):
    changed=set(baseline.changed_paths(candidate))
    if any(not SKILL.fullmatch(path) for path in changed):
        raise ValueError('candidate modifies frozen resources or unregistered skill paths')
    if any(len(item.content)>16384 for path,item in candidate.files.items() if SKILL.fullmatch(path)):
        raise ValueError('skill exceeds size limit')
    validate_snapshot_candidate(baseline,candidate,changed,PROMPT)


class SuiteAwareEvoMiniAgent(MiniAgent):
    def __init__(self,config,suites):
        super().__init__(config);self._suites=dict(suites);self._canonical=self._suites[sorted(suites)[0]]
        self._authorized={};self._canonical_candidate_id=None;self.update(self._canonical)

    def bind_benchmark(self,benchmark): self._env_runner=benchmark.env_runner

    def _validate_snapshot(self,snapshot,*,baseline=None):
        authorized=getattr(self,'_authorized',{}).get(snapshot.id)
        if authorized is None:
            if hasattr(self,'_canonical'):validate_evo_candidate(self._canonical,snapshot)
            return super()._validate_snapshot(snapshot,baseline=baseline)
        suite,digest=authorized;validate_evo_candidate(self._suites[suite],snapshot)
        if _skill_digest(snapshot)!=digest:raise ValueError('effective skill identity mismatch')

    async def rollout(self,tasks,*,snapshot=None):
        candidate=self.snapshot if snapshot is None else snapshot;validate_evo_candidate(self._canonical,candidate)
        skills={path:item.content for path,item in candidate.files.items() if SKILL.fullmatch(path)}
        digest=_skill_digest(candidate);self._canonical_candidate_id=candidate.id;rows=[]
        for task in tuple(tasks):
            suite=task.metadata.get('suite')
            if suite not in self._suites:raise ValueError('unknown AgentDojo suite')
            base=self._suites[suite]
            with base.check_out() as directory:
                skill_dir=directory/'skills'
                if skill_dir.exists():shutil.rmtree(skill_dir)
                for path,content in skills.items():
                    file=directory/path;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(content)
                effective=base.commit(directory)._with_scope(candidate._scope)
            self._authorized[effective.id]=(suite,digest)
            rows.extend(await super().rollout((task,),snapshot=effective))
        return tuple(rows)


def _skill_digest(snapshot):
    payload=[(path,hashlib.sha256(item.content).hexdigest()) for path,item in sorted(snapshot.files.items()) if SKILL.fullmatch(path)]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


class RestrictedSkillWorker(LLM):
    def __init__(self,config=None):
        super().__init__(config)
        # Unlike the base text-only LLM, this Worker implements bounded edit().
        self.config.type='agent'

    async def edit(self,working_dir,prompt):
        reply=await self.ask(prompt+'\nReturn only JSON: {"files": {"skills/NAME/SKILL.md": "complete file content"}, "summary": "..."}. Do not edit any other path. Return an empty files object if no edit is justified.')
        try:
            value=json.loads(reply.strip().removeprefix('```json').removesuffix('```').strip())
            files=value['files']
            if not isinstance(files,dict) or len(files)>4:raise ValueError('invalid file mapping')
            edits=[]
            for name,content in files.items():
                if not isinstance(name,str) or not SKILL.fullmatch(name) or not isinstance(content,str) or len(content.encode())>16384:
                    raise ValueError('invalid skill edit')
                path=Path(working_dir)/name
                if any(part.is_symlink() for part in (path,path.parent,path.parent.parent)):
                    raise ValueError('symlink denied')
                if path.resolve().parent.parent.parent != Path(working_dir).resolve():raise ValueError('skill path escapes checkout')
                edits.append((path,content))
            for path,content in edits:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(content)
            return str(value.get('summary','skill edits applied'))[:1000]
        except (ValueError,TypeError,KeyError,json.JSONDecodeError):
            return 'No edit: worker response did not satisfy the restricted skill protocol.'


class EvoBenchmark(AgentDojoSecurityBenchmark):
    async def env_runner(self,target,task):
        with self._lock:
            self._case_sequence+=1;sequence=self._case_sequence
        evidence=self.run_dir/'evidence';evidence.mkdir(parents=True,exist_ok=True)
        name=f'{sequence:08d}-{_safe_name(task.task_id)}'
        case=evidence/(name+'.json');snapshot_path=evidence/(name+'.snapshot.json');snapshot=target.snapshot
        snapshot.save(snapshot_path)
        env={**os.environ,'B1_TASK_ID':task.task_id,'B1_SEED':str(task.metadata['seed']),
             'B1_CASE_OUT':str(case),'B1_SNAPSHOT_JSON':str(snapshot_path),
             'B1_EXPECTED_SNAPSHOT_ID':snapshot.id,'B1_CANONICAL_SNAPSHOT_ID':target._canonical_candidate_id}
        proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'scripts/flashrsi_evo_skill_host.py'),
             cwd=ROOT,env=env,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        try:stdout,stderr=await asyncio.wait_for(proc.communicate(),timeout=390)
        except TimeoutError:
            proc.kill();await proc.wait();return Trajectory(status='error',error_message='Evo-Harness host timeout')
        if proc.returncode or not case.is_file():
            return Trajectory(status='error',error_message=stderr.decode(errors='replace')[-2000:] or 'Evo-Harness host failed')
        record=json.loads(case.read_text())
        if record.get('task_id')!=task.task_id or record.get('effective_snapshot_id')!=snapshot.id or record.get('candidate_snapshot_id')!=target._canonical_candidate_id:
            return Trajectory(status='error',error_message='Evo-Harness Snapshot identity mismatch')
        self._latest_records[task.task_id]=record
        return Trajectory(status=record.get('status','error'),output=record.get('output',''),
            messages=tuple(record.get('trajectory_messages',())),error_message=record.get('error_message') or record.get('unscorable_reason'),
            usage=record.get('usage',{}))
