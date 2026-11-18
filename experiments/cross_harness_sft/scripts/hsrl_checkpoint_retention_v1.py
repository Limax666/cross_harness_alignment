"""Bounded recovery retention for fresh owned runs; adapters/evidence are immutable.

Called only after the trainer durably commits its authoritative journal pointer.
An interrupted archival is left untouched for explicit reconciliation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat

from hsrl_v5.io import atomic_json, file_hash
from hsrl_v3.policy import validate_checkpoint_files

POLICY = {'schema': 'hsrl-checkpoint-retention/v1', 'keep_last': 2,
          'keep_initial': True, 'keep_explicit_anchors': True,
          'remove_only': ['resume.pt'], 'interrupted_archive': 'preserve_without_retry'}
LEDGER = 'checkpoint_retention.json'


def _sync(path):
    fd = os.open(path, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _owned_path(out, checkpoint):
    path = Path(checkpoint).absolute()
    if (out.resolve() != out or path.resolve() != path
            or path.parent != out / 'checkpoints'
            or not re.fullmatch(r'initial|update_[0-9]{4,}_[0-9]+', path.name)):
        raise ValueError('retention_requires_owned_direct_checkpoint')
    return path


def _validate(path):
    for entry in [path, *path.rglob('*')]:
        if entry.is_symlink() or not (stat.S_ISREG(entry.stat().st_mode) or entry.is_dir()):
            raise ValueError('retention_symlink_or_special_file')
    validate_checkpoint_files(path)


def initialize(out):
    """Issue ownership once, before this fresh run has any checkpoint/journal."""
    out = Path(out).absolute()
    if (out.resolve() != out or any((out / name).exists() for name in
            (LEDGER, 'checkpoints', 'journal.pt', 'state.json'))):
        raise ValueError('retention_initialization_requires_fresh_run')
    atomic_json(out / LEDGER, {'policy': POLICY,
        'manifest_sha256': file_hash(out / 'run_manifest.json'),
        'committed': [], 'anchors': []})
    _sync(out)
    _sync(out.parent)


def after_commit(out, current, *, anchors=(), load_recovery):
    """Register the durable pointer, then archive only older validated receipts."""
    out = Path(out).absolute()
    current = _owned_path(out, current)
    ledger_path = out / LEDGER
    if ledger_path.is_symlink() or (out / 'run_manifest.json').is_symlink():
        raise ValueError('retention_symlink_identity')
    ledger = json.loads(ledger_path.read_text())
    if (ledger['policy'] != POLICY or ledger['manifest_sha256'] != file_hash(out / 'run_manifest.json')):
        raise ValueError('retention_ownership_or_manifest_changed')
    journal = out / 'journal.pt'
    if journal.is_symlink() or Path(load_recovery(journal)['state']['checkpoint']).absolute() != current:
        raise ValueError('retention_requires_authoritative_journal_pointer')
    _validate(current)  # Pointed recovery always remains fully hash verified.
    receipt = {'name': current.name, 'complete_sha256': file_hash(current / 'COMPLETE.json')}
    committed = ledger['committed']
    existing = next((r for r in committed if r['name'] == current.name), None)
    if existing is not None and existing != receipt:
        raise ValueError('retention_committed_checkpoint_drift')
    changed = False
    if existing is None:
        committed.append(receipt)
        changed = True
    protected = set(ledger['anchors'])
    for anchor in anchors:
        path = _owned_path(out, anchor)
        _validate(path)
        protected.add(path.name)
    if sorted(protected) != ledger['anchors']:
        ledger['anchors'] = sorted(protected)
        changed = True
    if changed:
        atomic_json(ledger_path, ledger)
        _sync(out)  # durable ownership receipt before mutation
    complete = []
    for recorded in reversed(committed):
        try:
            _validate(_owned_path(out, out / "checkpoints" / recorded["name"]))
            complete.append(recorded["name"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(complete) == POLICY["keep_last"]:
            break
    protected.update(['initial', current.name, *complete])
    for recorded in committed:
        if recorded['name'] in protected:
            continue
        path = _owned_path(out, out / 'checkpoints' / recorded['name'])
        # Any archival intent (including an interrupted one) is a permanent
        # conservative stop. Never infer successful deletion or retry it.
        if (path / 'ARCHIVED.json').exists() or (path / 'RECOVERY_MANIFEST.json').exists():
            continue
        try:
            _validate(path)
            if file_hash(path / 'COMPLETE.json') != recorded['complete_sha256']:
                continue
            blob = path / 'resume.pt'
            removed = {'path': 'resume.pt', 'sha256': file_hash(blob),
                       'size_bytes': blob.stat().st_size}
        except (OSError, ValueError, KeyError, TypeError):
            continue  # incomplete/unrecognized/drifted evidence is never removed
        metadata = dict(load_recovery(blob))
        if set(metadata) != {'optimizer', 'rng', 'scheduler', 'state', 'version',
                             'adapter_hash', 'frozen_signature'}:
            raise ValueError('retention_unknown_recovery_payload_no_prune')
        del metadata['optimizer'], metadata['rng']
        atomic_json(path / 'ARCHIVED_STATE.json', metadata)
        archive = {'schema': 'hsrl-archived-checkpoint/v1', 'recovery_unavailable': True,
            'original_complete_sha256': recorded['complete_sha256'],
            'retention_policy': POLICY, 'removed': removed, 'action': 'prune_pending',
            'retained_metadata_sha256': file_hash(path / 'ARCHIVED_STATE.json')}
        atomic_json(path / 'ARCHIVED.json', archive)
        _sync(path)
        os.rename(path / 'COMPLETE.json', path / 'RECOVERY_MANIFEST.json')
        _sync(path)  # COMPLETE can no longer falsely advertise full recovery
        # Recheck the complete original inventory after metadata extraction.
        original = path / 'RECOVERY_MANIFEST.json'
        files = json.loads(original.read_text())['files']
        actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
        if (file_hash(original) != recorded['complete_sha256']
                or actual != set(files) | {'RECOVERY_MANIFEST.json', 'ARCHIVED.json', 'ARCHIVED_STATE.json'}
                or any(p.is_symlink() for p in path.rglob('*'))
                or any(file_hash(path / name) != sha for name, sha in files.items())
                or file_hash(path / 'ARCHIVED_STATE.json') != archive['retained_metadata_sha256']):
            raise ValueError('retention_checkpoint_changed_before_unlink')
        blob.unlink()
        _sync(path)
        archive['action'] = 'resume.pt_unlinked'
        atomic_json(path / 'ARCHIVED.json', archive)
        _sync(path)
        _sync(path.parent)
