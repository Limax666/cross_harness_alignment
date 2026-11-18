import asyncio
import json
import pytest
from flashrsi_evo_skill import RestrictedSkillWorker, validate_evo_candidate
from flashrsi.core import Snapshot


def test_skill_worker_validates_all_paths_before_writing(tmp_path,monkeypatch):
    worker=RestrictedSkillWorker.__new__(RestrictedSkillWorker)
    async def proposal(self,prompt):
        return json.dumps({'files':{'skills/ok/SKILL.md':'content','../../outside':'bad'}})
    monkeypatch.setattr(RestrictedSkillWorker,'ask',proposal)
    asyncio.run(worker.edit(tmp_path,'edit'))
    assert not (tmp_path/'skills').exists()


def test_skill_worker_writes_an_allowed_directory_skill(tmp_path,monkeypatch):
    worker=RestrictedSkillWorker.__new__(RestrictedSkillWorker)
    async def proposal(self,prompt):
        return json.dumps({'files':{'skills/ok/SKILL.md':'bounded content'},'summary':'created'})
    monkeypatch.setattr(RestrictedSkillWorker,'ask',proposal)
    assert asyncio.run(worker.edit(tmp_path,'edit'))=='created'
    assert (tmp_path/'skills/ok/SKILL.md').read_text()=='bounded content'


def test_evo_candidate_changes_only_directory_skills(tmp_path):
    (tmp_path/'prompts').mkdir();(tmp_path/'prompts/system.md').write_text('prompt\n## Trusted safety policy\npinned\n')
    (tmp_path/'tools.py').write_text('fixed')
    base=Snapshot.from_directory(tmp_path,agent_type='fixture',component_rules={'prompt':('prompts/**',),'skill':('skills/**',),'tool':('tools.py',)})
    with base.check_out() as directory:
        skill=directory/'skills/test/SKILL.md';skill.parent.mkdir(parents=True);skill.write_text('test')
        candidate=base.commit(directory)
    validate_evo_candidate(base,candidate)
    with base.check_out() as directory:
        (directory/'tools.py').write_text('changed');invalid=base.commit(directory)
    with pytest.raises(ValueError,match='frozen'):
        validate_evo_candidate(base,invalid)
