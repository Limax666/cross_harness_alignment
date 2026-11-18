from types import SimpleNamespace
import asyncio

import pytest

from flashrsi_opro_pilot import PrefixOnlyWorker, prepare_probe_directory, select_training_pair
from flashrsi.agents.llm import LLM


def task(task_id, arm, split='train'):
    return SimpleNamespace(task_id=task_id, metadata={'family': 'workspace:user_task_0', 'arm': arm, 'source_split': split})


def test_selects_same_family_training_pair_only():
    rows = [task('clean', 'clean'), task('attack', 'attack'), task('dev', 'attack', 'validation')]
    assert [x.task_id for x in select_training_pair(rows)] == ['clean', 'attack']


def test_fails_when_attack_condition_missing():
    with pytest.raises(ValueError):
        select_training_pair([task('clean', 'clean')])


def test_probe_directory_prepares_flashrsi_record_locations(tmp_path):
    run = tmp_path / 'new-run'
    prepare_probe_directory(run)
    assert (run / 'snapshots').is_dir()
    assert (run / 'trajectories').is_dir()
    with pytest.raises(FileExistsError):
        prepare_probe_directory(run)


def test_worker_is_told_to_return_only_editable_instruction(monkeypatch):
    seen = []
    async def fake_ask(self, prompt):
        seen.append(prompt)
        return 'candidate'
    monkeypatch.setattr(LLM, 'ask', fake_ask)
    worker = PrefixOnlyWorker.__new__(PrefixOnlyWorker)
    assert asyncio.run(worker.ask('original request')) == 'candidate'
    assert 'original request' in seen[0]
    assert 'Do not include' in seen[0] and 'Trusted safety policy' in seen[0]
