"""Materialize the official selected ClawGym tasks from pinned HF bytes, including .log inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .backends import sha256_file, tree_digest, write_json

HF_REVISION = 'bd9043cff88b913922a165d9c022a4796aee3f3a'
HF_SHA256 = '0455f13c4f7a4839d39c3c1da9704c4959046ed529040d44f46f8f9bd07f7774'
SELECTION_REVISION = '0e8526d8f5e0999896c74cee4d6b998fb8a0429e'


def _source_path(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in name:
        raise ValueError('unsafe_HF_asset_path')
    return path


def materialize(raw: Path, selected_root: Path, output: Path):
    if sha256_file(raw) != HF_SHA256:
        raise ValueError('HF_raw_bytes_differ_from_frozen_revision')
    repo = selected_root.parents[2]
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SELECTION_REVISION:
        raise ValueError('official_selection_revision_mismatch')
    selected = {}
    for path in sorted(selected_root.glob('*/data_entry.json')):
        relative = path.relative_to(repo).as_posix()
        committed = subprocess.check_output(['git', '-C', str(repo), 'show', f'{SELECTION_REVISION}:{relative}'])
        if committed != path.read_bytes():
            raise ValueError('modified_official_selection_entry')
        entry = json.loads(committed)
        key = hashlib.sha256(entry['user_query'].strip().encode()).hexdigest()
        if key in selected:
            raise ValueError('ambiguous_official_selection_prompt')
        selected[key] = (path.parent.name, entry)
    if len(selected) != 2000:
        raise ValueError('official_selection_must_contain_2000')
    output.mkdir(parents=True, exist_ok=False)
    tasks = output / 'tasks'
    tasks.mkdir()
    records, missing_files = [], []
    with raw.open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = hashlib.sha256(row['prompt'].strip().encode()).hexdigest()
            if key not in selected:
                continue
            task_id, entry = selected[key]
            target = tasks / task_id
            target.mkdir(exist_ok=False)
            write_json(target / 'data_entry.json', entry)
            seen, file_hashes = set(), {}
            for asset in row['input_files']:
                name = _source_path(asset['file_path'])
                if name.as_posix() in seen:
                    raise ValueError('duplicate_HF_asset')
                seen.add(name.as_posix())
                if name.parts[0] == 'reward':
                    name = Path('reward') / ('check.py' if name.parts[1:] == ('test.py',) else Path(*name.parts[1:]))
                else:
                    name = Path('input_files') / name
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = asset['content'].encode()
                path.write_bytes(data)
                file_hashes[name.as_posix()] = sha256_file(path)
                old = selected_root / task_id / name
                if not old.exists():
                    missing_files.append({'task': task_id, 'path': name.as_posix(), 'sha256': sha256_file(path)})
                elif old.read_bytes() != data:
                    raise ValueError('official_selected_asset_differs_from_HF')
            # Keep the official code-only reward wrapper byte-for-byte; the actual
            # private grader invokes unchanged check.py without exposing this path.
            wrapper_rel = f'RL/data/clawgym_train/{task_id}/reward/reward.sh'
            wrapper = subprocess.check_output(['git', '-C', str(repo), 'show', f'{SELECTION_REVISION}:{wrapper_rel}'])
            (target / 'reward/reward.sh').write_bytes(wrapper)
            records.append({'id': task_id, 'hf_line': number, 'prompt_sha256': key,
                            'hf_record_sha256': hashlib.sha256(line.rstrip('\r\n').encode()).hexdigest(),
                            'files': file_hashes, 'tree_sha256': tree_digest(target)})
    if len(records) != 2000 or len({row['id'] for row in records}) != 2000:
        raise ValueError('selected_HF_materialization_not_complete_unique_2000')
    manifest = {'schema': 'clawgym-selected2k-HF-materialization/v1', 'repo': 'RUC-AIBOX/ClawGym-Task',
                'hf_revision': HF_REVISION, 'raw_path': str(raw.resolve()), 'raw_sha256': HF_SHA256,
                'selection_revision': SELECTION_REVISION, 'selection_root': str(selected_root.resolve()),
                'selection_count': len(records), 'records': records, 'restored_HF_input_files': missing_files,
                'status': 'static materialization; no source execution or model calls',
                'reward_contract': 'official_selected_RL_code_only_no_LLM_rubric'}
    write_json(output / 'HF_MATERIALIZATION.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--selected-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.raw, args.selected_root, args.output)
    print(json.dumps({'selected': result['selection_count'], 'restored_inputs': len(result['restored_HF_input_files']),
                      'hf_revision': result['hf_revision']}))
