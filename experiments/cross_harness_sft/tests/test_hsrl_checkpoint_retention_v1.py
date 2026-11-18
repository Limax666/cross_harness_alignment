"""Owned filesystem tests, never experimental training."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def checkpoint(out, name):
    from hsrl_v5.io import atomic_json, file_hash
    path = out / 'checkpoints' / name
    (path / 'adapter').mkdir(parents=True)
    (path / 'adapter/adapter_model.safetensors').write_bytes(b'adapter-' + name.encode())
    (path / 'resume.pt').write_text(json.dumps({'optimizer': {'fixture': name}, 'rng': {},
        'scheduler': {'last_epoch': 3}, 'state': {'checkpoint': str(path)}, 'version': 3,
        'adapter_hash': 'fixture-adapter-hash', 'frozen_signature': 'fixture-metadata-guard'}))
    atomic_json(path / 'COMPLETE.json', {'schema': 'hsrl-atomic-checkpoint/v1',
        'files': {p.relative_to(path).as_posix(): file_hash(p) for p in path.rglob('*') if p.is_file()}})
    (out / 'journal.pt').write_text(json.dumps({'state': {'checkpoint': str(path)}}))
    return path


def read_recovery(path):
    return json.loads(path.read_text())


def owned(tmp_path):
    import hsrl_checkpoint_retention_v1 as retention
    (tmp_path / 'run_manifest.json').write_text('{}')
    retention.initialize(tmp_path)
    return retention


def test_retains_initial_last_two_anchors_and_every_adapter(tmp_path):
    retention = owned(tmp_path)
    points = []
    for i in range(7):
        path = checkpoint(tmp_path, 'initial' if i == 0 else f'update_{i:04d}_1')
        points.append(path)
        retention.after_commit(tmp_path, path, load_recovery=read_recovery, anchors=points[2:3] if i >= 2 else ())
    for i, path in enumerate(points):
        assert (path / 'adapter/adapter_model.safetensors').read_bytes().startswith(b'adapter-')
        if i in (0, 2, 5, 6):
            from hsrl_v3.policy import validate_checkpoint_files
            validate_checkpoint_files(path)
        else:
            assert not (path / 'resume.pt').exists()
            assert not (path / 'COMPLETE.json').exists()
            receipt = json.loads((path / 'ARCHIVED.json').read_text())
            assert receipt['recovery_unavailable'] is True
            assert receipt['action'] == 'resume.pt_unlinked'
            assert receipt['removed']['size_bytes'] > 0
            assert len(receipt['removed']['sha256']) == 64
            assert (path / 'RECOVERY_MANIFEST.json').is_file()
            metadata = json.loads((path / 'ARCHIVED_STATE.json').read_text())
            assert metadata['state']['checkpoint'] == str(path)
            assert metadata['scheduler'] == {'last_epoch': 3}
            assert 'optimizer' not in metadata and 'rng' not in metadata
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    retention.after_commit(tmp_path, points[-1], load_recovery=read_recovery, anchors=[points[2]])
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_unowned_incomplete_symlink_changed_and_uncommitted_preserved(tmp_path):
    retention = owned(tmp_path)
    initial = checkpoint(tmp_path, 'initial')
    retention.after_commit(tmp_path, initial, load_recovery=read_recovery)
    older = checkpoint(tmp_path, 'update_0001_1')
    retention.after_commit(tmp_path, older, load_recovery=read_recovery)
    (older / 'resume.pt').write_bytes(b'drift')
    uncommitted = checkpoint(tmp_path, 'update_0002_1')
    (tmp_path / 'checkpoints/link').symlink_to(uncommitted, target_is_directory=True)
    for i in range(3, 6):
        retention.after_commit(tmp_path, checkpoint(tmp_path, f'update_{i:04d}_1'), load_recovery=read_recovery)
    assert (older / 'resume.pt').read_bytes() == b'drift'
    assert (uncommitted / 'resume.pt').exists()
    outside = checkpoint(tmp_path.parent / (tmp_path.name + '-outside'), 'initial')
    with pytest.raises(ValueError):
        retention.after_commit(tmp_path, outside, load_recovery=read_recovery)
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        retention.after_commit(legacy, initial, load_recovery=read_recovery)


@pytest.mark.parametrize('failure', ['intent_fsync', 'rename', 'rename_fsync', 'unlink', 'unlink_fsync', 'final_marker'])
def test_interruption_never_deletes_before_invalidating_complete(tmp_path, monkeypatch, failure):
    retention = owned(tmp_path)
    points = []
    for i in range(3):
        points.append(checkpoint(tmp_path, 'initial' if i == 0 else f'update_{i:04d}_1'))
        retention.after_commit(tmp_path, points[-1], load_recovery=read_recovery)
    victim = points[1]
    rename, unlink, atomic, sync = retention.os.rename, Path.unlink, retention.atomic_json, retention._sync
    def fail_rename(source, destination):
        if Path(source) == victim / 'COMPLETE.json':
            raise OSError('injected')
        return rename(source, destination)
    def fail_unlink(path, *args, **kwargs):
        if path == victim / 'resume.pt':
            assert not (victim / 'COMPLETE.json').exists()
            raise OSError('injected')
        return unlink(path, *args, **kwargs)
    def fail_marker(path, value):
        if Path(path) == victim / 'ARCHIVED.json' and value['action'] == 'resume.pt_unlinked':
            raise OSError('injected')
        return atomic(path, value)
    def fail_sync(path):
        if path == victim and (victim / 'ARCHIVED.json').exists():
            complete = (victim / 'COMPLETE.json').exists()
            blob = (victim / 'resume.pt').exists()
            if ((failure == 'intent_fsync' and complete)
                    or (failure == 'rename_fsync' and not complete and blob)
                    or (failure == 'unlink_fsync' and not blob)):
                raise OSError('injected')
        return sync(path)
    monkeypatch.setattr(retention, '_sync', fail_sync)
    monkeypatch.setattr(retention.os, 'rename', fail_rename if failure == 'rename' else rename)
    monkeypatch.setattr(Path, 'unlink', fail_unlink if failure == 'unlink' else unlink)
    monkeypatch.setattr(retention, 'atomic_json', fail_marker if failure == 'final_marker' else atomic)
    current = checkpoint(tmp_path, 'update_0003_1')
    with pytest.raises(OSError, match='injected'):
        retention.after_commit(tmp_path, current, load_recovery=read_recovery)
    if not (victim / 'resume.pt').exists():
        assert not (victim / 'COMPLETE.json').exists()
    retention.after_commit(tmp_path, current, load_recovery=read_recovery)
    from hsrl_v3.policy import validate_checkpoint_files
    validate_checkpoint_files(current)
    validate_checkpoint_files(points[2])


def test_unknown_payload_and_changed_journal_fail_closed(tmp_path):
    retention = owned(tmp_path)
    paths = []
    for i in range(3):
        paths.append(checkpoint(tmp_path, 'initial' if i == 0 else f'update_{i:04d}_1'))
        retention.after_commit(tmp_path, paths[-1], load_recovery=read_recovery)
    current = checkpoint(tmp_path, 'update_0003_1')
    def unknown_payload(path):
        value = read_recovery(path)
        if path == paths[1] / 'resume.pt':
            value['unexpected_tensor_or_field'] = 'unrecognized'
        return value
    with pytest.raises(ValueError, match='unknown_recovery_payload'):
        retention.after_commit(tmp_path, current, load_recovery=unknown_payload)
    assert (paths[1] / 'resume.pt').exists()
    assert (paths[1] / 'COMPLETE.json').exists()
    (tmp_path / 'journal.pt').write_text(json.dumps({'state': {'checkpoint': str(paths[0])}}))
    with pytest.raises(ValueError, match='authoritative_journal_pointer'):
        retention.after_commit(tmp_path, current, load_recovery=read_recovery)
    assert (paths[1] / 'resume.pt').exists()


def test_real_trainer_commit_serialization_and_resume_hashes(tmp_path, monkeypatch):
    """Real save/journal/retention path with CPU serialization and owned file fixtures."""
    import torch
    import train_hsrl_v5 as trainer
    from hsrl_v3.policy import Policy, validate_checkpoint_files
    from hsrl_v5.io import file_hash
    from types import SimpleNamespace

    retention = owned(tmp_path)
    class AdapterFiles:
        def save_pretrained(self, directory, **kwargs):
            directory.mkdir(parents=True)
            (directory / 'adapter_model.safetensors').write_bytes(b'owned-fixture-adapter')
    policy = object.__new__(Policy)
    policy.torch, policy.model = torch, AdapterFiles()
    policy.version, policy.frozen_signature = 0, 'owned-fixture-guard'
    policy.adapter_hash, policy.rng_state = lambda: 'owned-fixture-hash', lambda: {'fixture': True}
    optimizer = SimpleNamespace(state_dict=lambda: {'fixture_optimizer': True})
    scheduler = SimpleNamespace(state_dict=lambda: {'last_epoch': 0})
    original_after_commit = retention.after_commit
    observed = []
    def after_commit(out, current, **kwargs):
        saved = torch.load(out / 'journal.pt', map_location='cpu', weights_only=False)
        assert saved['state']['checkpoint'] == str(current)
        validate_checkpoint_files(Path(saved['state']['checkpoint']))
        observed.append(file_hash(out / 'journal.pt'))
        return original_after_commit(out, current, **kwargs)
    monkeypatch.setattr(retention, 'after_commit', after_commit)
    points = []
    for i in range(5):
        path = tmp_path / 'checkpoints' / ('initial' if i == 0 else f'update_{i:04d}_1')
        points.append(path)
        state = {'checkpoint': str(path), 'step': i, 'last_update': {'fixture_stat': i}}
        trainer.commit_checkpoint(tmp_path, policy, optimizer, scheduler, state)
    assert len(observed) == 5
    saved = torch.load(tmp_path / 'journal.pt', map_location='cpu', weights_only=False)
    recovery = torch.load(Path(saved['state']['checkpoint']) / 'resume.pt', map_location='cpu', weights_only=False)
    assert recovery['state'] == saved['state']
    assert recovery['optimizer'] == {'fixture_optimizer': True}
    assert recovery['scheduler'] == {'last_epoch': 0}
    archived = json.loads((points[1] / 'ARCHIVED_STATE.json').read_text())
    assert archived['state']['last_update'] == {'fixture_stat': 1}
    assert archived['scheduler'] == {'last_epoch': 0}
    assert archived['adapter_hash'] == 'owned-fixture-hash'
    (points[-1] / 'resume.pt').write_bytes(b'drift')
    with pytest.raises(ValueError, match='incomplete_or_changed_checkpoint'):
        validate_checkpoint_files(points[-1])


def test_last_two_complete_points_survive_later_checkpoint_drift(tmp_path):
    retention = owned(tmp_path)
    points = []
    for i in range(4):
        points.append(checkpoint(tmp_path, 'initial' if i == 0 else f'update_{i:04d}_1'))
        retention.after_commit(tmp_path, points[-1], load_recovery=read_recovery)
    (points[-1] / 'resume.pt').write_bytes(b'drift')
    current = checkpoint(tmp_path, 'update_0004_1')
    retention.after_commit(tmp_path, current, load_recovery=read_recovery)
    from hsrl_v3.policy import validate_checkpoint_files
    validate_checkpoint_files(current)
    validate_checkpoint_files(points[2])
    assert (points[3] / 'resume.pt').read_bytes() == b'drift'


def test_symlinked_and_incomplete_old_checkpoints_never_pruned(tmp_path):
    retention = owned(tmp_path)
    points = []
    for i in range(3):
        points.append(checkpoint(tmp_path, 'initial' if i == 0 else f'update_{i:04d}_1'))
        retention.after_commit(tmp_path, points[-1], load_recovery=read_recovery)
    blob = points[1] / 'resume.pt'
    original = blob.read_bytes()
    blob.unlink()
    blob.symlink_to(points[2] / 'resume.pt')
    (points[2] / 'COMPLETE.json').unlink()
    current = checkpoint(tmp_path, 'update_0003_1')
    retention.after_commit(tmp_path, current, load_recovery=read_recovery)
    assert blob.is_symlink()
    assert (points[2] / 'resume.pt').read_bytes() == original.replace(b'update_0001', b'update_0002')
    assert not (points[1] / 'ARCHIVED.json').exists()
    assert not (points[2] / 'ARCHIVED.json').exists()


def test_drift_during_metadata_extraction_preserves_recovery_blob(tmp_path):
    retention = owned(tmp_path)
    points = []
    for i in range(3):
        points.append(checkpoint(tmp_path, 'initial' if i == 0 else f'update_{i:04d}_1'))
        retention.after_commit(tmp_path, points[-1], load_recovery=read_recovery)
    current = checkpoint(tmp_path, 'update_0003_1')
    def mutate_adapter(path):
        metadata = read_recovery(path)
        if path == points[1] / 'resume.pt':
            (points[1] / 'adapter/adapter_model.safetensors').write_bytes(b'drift-during-extraction')
        return metadata
    with pytest.raises(ValueError, match='checkpoint_changed_before_unlink'):
        retention.after_commit(tmp_path, current, load_recovery=mutate_adapter)
    assert (points[1] / 'resume.pt').exists()
    assert not (points[1] / 'COMPLETE.json').exists()
    assert json.loads((points[1] / 'ARCHIVED.json').read_text())['action'] == 'prune_pending'


def test_failed_journal_commit_never_enters_retention(tmp_path, monkeypatch):
    import train_hsrl_v5 as trainer
    from types import SimpleNamespace
    owned(tmp_path)
    current = tmp_path / 'checkpoints/update_0001_1'
    policy = SimpleNamespace(save=lambda *args: None)
    def journal_failure(*args):
        raise OSError('owned-journal-barrier-failure')
    def forbidden_retention(*args, **kwargs):
        pytest.fail('retention entered before journal was durable')
    monkeypatch.setattr(trainer, 'persist_journal', journal_failure)
    monkeypatch.setattr(trainer.checkpoint_retention, 'after_commit', forbidden_retention)
    with pytest.raises(OSError, match='owned-journal-barrier-failure'):
        trainer.commit_checkpoint(tmp_path, policy, None, None, {'checkpoint': str(current)})
    assert json.loads((tmp_path / 'checkpoint_retention.json').read_text())['committed'] == []
