"""Filesystem transaction checks; no model, task/source, Docker or API runs."""
import copy
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_hsrl_v5 as trainer
from hsrl_v5.io import file_hash


def fixture_episode(tmp_path):
    out = tmp_path / 'run'
    episode = out / 'episodes/step_0000/rollout_000'
    (episode / 'private/grader').mkdir(parents=True)
    (episode / 'private/grader/response.json').write_text('{"valid":true}\n')
    (episode / 'episode.json').write_text('{"valid":true}\n')
    state = {'attempts': 1, 'inflight': {'directory': str(episode), 'attempt': 1}, 'episodes': []}
    entry = {'episode_path': str(episode / 'episode.json'), 'sha256': file_hash(episode / 'episode.json')}
    return out, episode, state, entry


def test_every_private_file_and_bottom_up_directory_is_synced_before_completion(tmp_path, monkeypatch):
    out, episode, state, entry = fixture_episode(tmp_path)
    synced = []
    original = os.fsync
    def fsync(fd):
        synced.append(Path(os.readlink('/proc/self/fd/' + str(fd))))
        original(fd)
    monkeypatch.setattr(trainer.os, 'fsync', fsync)
    journal = []
    monkeypatch.setattr(trainer, 'persist_journal', lambda *args: journal.append(copy.deepcopy(state)))
    trainer.commit_completed_attempt(out, state, entry, None)
    assert state['inflight'] is None and state['episodes'] == [entry]
    assert journal == [state]
    expected = [episode / 'private/grader/response.json', episode / 'private/grader',
                episode / 'private', episode, episode.parent, episode.parent.parent, out, out.parent]
    assert all(p in synced for p in expected)
    positions = [synced.index(p) for p in expected]
    assert positions == sorted(positions)


def test_failed_barrier_retains_reservation_without_completed_journal(tmp_path, monkeypatch):
    out, episode, state, entry = fixture_episode(tmp_path)
    before = copy.deepcopy(state)
    def fail(fd):
        raise OSError('actual barrier boundary rejected')
    monkeypatch.setattr(trainer.os, 'fsync', fail)
    journal = []
    monkeypatch.setattr(trainer, 'persist_journal', lambda *args: journal.append(True))
    with pytest.raises(OSError, match='barrier'):
        trainer.commit_completed_attempt(out, state, entry, None)
    assert state == before and not journal
    assert (episode / 'episode.json').exists()


def test_inflight_identity_and_symbolic_links_cannot_seal_other_evidence(tmp_path, monkeypatch):
    out, episode, state, entry = fixture_episode(tmp_path)
    (episode / 'private/unsafe').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='symlink'):
        trainer.commit_completed_attempt(out, state, entry, None)
    assert state['inflight']
