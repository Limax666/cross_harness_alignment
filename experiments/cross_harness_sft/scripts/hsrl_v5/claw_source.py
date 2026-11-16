"""Prepare all pinned ClawGym HF records without executing source code.

Run with ``PYTHONPATH=scripts python -m hsrl_v5.claw_source --raw ... --out ...
--benchmark-root ...``. Output is new-only and remains host-side; only each
``public_assets_path`` and the public user query may enter an episode.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import copy
import shutil
import itertools
import re
import subprocess
import unicodedata

from hsrl_v3.backends import sha256_file, tree_digest, write_json
from hsrl_v3.claw_backend import (HF_REVISION, HF_SHA256, PINNED_REVISION,
                                  REVIEWED_WRITABLE_OUTPUTS, source_contract)
from hsrl_v3.pool import _query_features

HF_COUNT = 15402
SEED = 20261116
PROFILE = 'clawgym_official_rl_code_only_v1'
MAIN_EXP = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')
DEFAULT_BENCH = MAIN_EXP / 'vendor/clawgym_agents_hsrl_v3_0e8526d/RL/data/clawgym_eval'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def normalized_prompt(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _asset_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise ValueError('unsafe_HF_asset_path')
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {'', '.', '..'} for part in name.split('/')) or path.as_posix() != name:
        raise ValueError('unsafe_HF_asset_path')
    return path


def _git_bytes(repo: Path, relative: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(repo), 'show', f'{PINNED_REVISION}:{relative}'])


@lru_cache(maxsize=4)
def _public_entries(root_string: str, expected: int) -> tuple[dict, dict]:
    """Verify *only public entry files* against the pinned Git blob bytes."""
    root = Path(root_string).resolve(strict=True)
    repo = root.parents[2]
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PINNED_REVISION:
        raise ValueError('official_public_source_revision_mismatch')
    relative = root.relative_to(repo).as_posix()
    names = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', PINNED_REVISION, '--', relative], text=True).splitlines()
    names = sorted(name for name in names if name.endswith('/data_entry.json'))
    actual = sorted(path.relative_to(repo).as_posix() for path in root.glob('*/data_entry.json'))
    if len(names) != expected or names != actual:
        raise ValueError('official_public_entry_inventory_mismatch')
    entries, hashes = {}, {}
    for name in names:
        path = repo / name
        data = path.read_bytes()
        if path.is_symlink() or data != _git_bytes(repo, name):
            raise ValueError('modified_official_public_entry')
        entry = json.loads(data)
        # Read only public user instructions, never grader/oracle fields.
        entries[path.parent.name] = entry['user_query']
        hashes[name] = _sha(data)
    return entries, hashes


def public_benchmark_hashes(root: Path) -> dict:
    entries, entry_hashes = _public_entries(str(Path(root).resolve()), 200)
    # Benchmark input bytes are public. Never traverse or hash reward/ files.
    root = Path(root).resolve(strict=True)
    repo = root.parents[2]
    input_hashes = {}
    for name in sorted(entries):
        public = root / name / 'input_files'
        for path in sorted(public.rglob('*')):
            if path.is_symlink():
                raise ValueError('benchmark_public_input_symlink')
            if path.is_file():
                relative = path.relative_to(repo).as_posix()
                data = path.read_bytes()
                if data != _git_bytes(repo, relative):
                    raise ValueError('modified_official_public_input')
                input_hashes[relative] = _sha(data)
    return {'revision': PINNED_REVISION, 'count': len(entries),
            'exact_prompt_sha256': sorted({_sha(prompt.encode()) for prompt in entries.values()}),
            'normalized_prompt_sha256': sorted({_sha(normalized_prompt(prompt).encode()) for prompt in entries.values()}),
            'public_entry_sha256': entry_hashes, 'public_input_sha256': input_hashes,
            'consumed_public_paths': sorted([*entry_hashes, *input_hashes]),
            'scope': 'public_user_query_and_input_files_only_no_eval_graders_or_answers'}


@lru_cache(maxsize=1)
def _reviewed_prompt_ids() -> dict[str, str]:
    entries, _ = _public_entries(str(DEFAULT_BENCH.parent / 'clawgym_train'), 2000)
    return {_sha(query.encode()): task_id for task_id, query in entries.items()
            if task_id in REVIEWED_WRITABLE_OUTPUTS}


def _contract(entry_path: Path, source_row: dict) -> tuple[dict, dict, list[str]]:
    try:
        contract = source_contract(entry_path)
    except (SyntaxError, UnicodeError, ValueError, OSError) as exc:
        reason = 'checker_contract_parse_pending'
        return ({'version': 'clawgym_full_hf_contract_v1', 'status': 'pending',
                 'error': {'type': type(exc).__name__, 'message': str(exc)}},
                {'status': 'pending_or_unsupported', 'capability_evidence': [{'reason': reason, 'message': str(exc)}]}, [reason])
    contract['upstream_contract_version'] = contract['version']
    contract['version'] = 'clawgym_full_hf_source_capability_and_input_roles_v1'
    # v3 reviewed task IDs are selection IDs, while the full source keeps its
    # own IDs. Transfer only exact public-query matches, never a bare ID grant.
    selected_id = _reviewed_prompt_ids().get(_sha(source_row['prompt'].encode()))
    if selected_id:
        for name, review in REVIEWED_WRITABLE_OUTPUTS[selected_id].items():
            if name not in contract['pending_writable_inputs']:
                continue
            contract['writable_input_evidence'][name] = {
                'source_text': contract['pending_writable_inputs'].pop(name),
                'reviewed_gold_contract': review, 'official_selection_id': selected_id,
                'matched_public_query_sha256': _sha(source_row['prompt'].encode())}
            contract['writable_inputs'].append(name)
            contract['immutable_inputs'].remove(name)
    contract['writable_inputs'] = sorted(contract['writable_inputs'])
    contract['immutable_gold_reference'] = {
        'copy_from_original_source': True,
        'original_public_assets_path': str(entry_path.parent / 'input_files'),
        'initial_input_hashes': contract['initial_input_hashes'],
        'policy': 'private_grader_uses_immutable_original_reference_and_separate_episode_outputs; writable_scopes_are_not_gold'}
    contract['reward_profile'] = PROFILE
    contract['original_validation_mode'] = source_row['meta_data'].get('validation_mode')
    contract['original_reward_aggregation'] = source_row['meta_data'].get('reward_aggregation')
    contract['rubric_applicable'] = bool(source_row.get('rules'))
    contract['rubric_not_in_training_reward'] = bool(source_row.get('rules'))
    contract['reward_scope'] = 'official_RL_code_verifier_component_only_not_original_full_composite_utility'
    requirements = contract['capability_requirements']
    reasons = ['requires_unsupported_' + name for name in requirements]
    if contract['pending_writable_inputs']:
        reasons.append('writable_input_gold_contract_pending_review')
    deps = contract['dependency_contract']
    if deps['unavailable_required_checker_modules'] or deps['unresolved_requested_execution_modules']:
        reasons.append('required_third_party_dependency_contract_pending')
    capability = {'version': 'clawgym_full_hf_static_capability_v1',
                  'status': 'pending_or_unsupported' if reasons else 'structurally_supported_runtime_unverified',
                  'capability_requirements': requirements,
                  'capability_evidence': contract['capability_evidence'],
                  'dependency_contract': deps,
                  'pending_writable_inputs': contract['pending_writable_inputs'],
                  'failure_semantics': 'invalid_pending_not_model_utility_zero'}
    return contract, capability, reasons


def materialize_claw_record(row: dict, line_number: int, line_bytes: bytes, tasks: Path) -> dict:
    """Extract one original record; caller proves the enclosing fixed raw pin.

    This boundary also supports small real-source regression tests. It does
    not assert the full enclosing pin; only prepare_claw_source does that.
    """
    if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
        raise ValueError('missing_user_query')
    seen, file_assets = set(), []
    for asset in row['input_files']:
        name = _asset_name(asset['file_path'])
        if str(name) in seen:
            raise ValueError('duplicate_HF_asset')
        seen.add(str(name))
        if not isinstance(asset['content'], str):
            raise ValueError('non_text_HF_asset')
        file_assets.append((name, asset['content'].encode('utf-8')))
    # Prefix conflicts could otherwise partly write a task before extraction.
    if any(str(parent) in seen for name, _ in file_assets for parent in name.parents if str(parent) != '.'):
        raise ValueError('duplicate_HF_asset_prefix')
    if json.loads(line_bytes) != row:
        raise ValueError('HF_record_does_not_match_consumed_line')
    canonical = _sha(_canonical(row))
    source_id = row.get('meta_data', {}).get('task_id')
    task_id = source_id if source_id else f'hf_line_{line_number:05d}'
    directory = Path(tasks).resolve() / f'hf_{line_number:05d}'
    directory.mkdir(parents=True, exist_ok=False)
    public = directory / 'input_files'
    private = directory / 'reward'
    public.mkdir()
    private.mkdir()
    file_hashes, public_hashes = {}, {}
    for name, data in file_assets:
        destination = directory / name if name.parts[0] == 'reward' else public / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        file_hashes[destination.relative_to(directory).as_posix()] = _sha(data)
        if name.parts[0] != 'reward':
            public_hashes[str(name)] = _sha(data)
    checker = private / 'test.py'
    reasons = []
    if checker.is_file():
        # Compatibility alias is byte-identical. The runtime executes test.py.
        if (private / 'check.py').exists():
            raise ValueError('reserved_checker_alias_collision')
        (private / 'check.py').write_bytes(checker.read_bytes())
    else:
        reasons.append('missing_original_checker')
    entry = {'task_id': task_id, 'user_query': row['prompt'],
             'metadata': row.get('meta_data', {}).get('metadata', {}),
             'meta_data': row.get('meta_data', {}), 'rules': row.get('rules'),
             'hook_code': row.get('hook_code'), 'hook_lang': row.get('hook_lang'),
             'input_mount_dir': '/root/.openclaw/workspace'}
    write_json(directory / 'data_entry.json', entry)
    (directory / 'source_record.json').write_bytes(line_bytes)
    if row.get('hook_lang') != 'bash' or row.get('hook_code') != 'python3 reward/test.py /root/.openclaw/workspace':
        reasons.append('original_hook_contract_pending')
    if checker.is_file():
        contract, capability, failures = _contract(directory / 'data_entry.json', row)
        reasons.extend(failures)
    else:
        contract, capability = {}, {'status': 'pending_or_unsupported', 'capability_evidence': []}
    identity = {'format': 'hf_full_corpus', 'repo': 'RUC-AIBOX/ClawGym-Task',
                'hf_revision': HF_REVISION, 'hf_sha256': HF_SHA256,
                'hf_line': line_number, 'source_task_id': source_id,
                'source_id_kind': 'original_task_id' if source_id else 'stable_HF_line_identity_missing_source_task_id',
                'hf_record_sha256': _sha(line_bytes.rstrip(b'\r\n')),
                'hf_record_canonical_sha256': canonical,
                'raw_record_path': str(directory / 'source_record.json'),
                'enclosing_raw_pin_verified': False}
    record = {'source': 'clawgym', 'id': task_id, 'family': None, 'split': None,
            'condition': 'benign', 'source_path': str(directory / 'data_entry.json'),
            'source_sha256': sha256_file(directory / 'data_entry.json'),
            'source_tree_sha256': tree_digest(directory), 'source_identity': identity,
            'source_contract': contract, 'capability_contract': capability,
            'public_assets_path': str(public), 'private_grader_path': str(directory),
            'private_checker_path': str(checker),
            'original_asset_sha256': file_hashes, 'public_asset_sha256': public_hashes,
            'public_asset_tree_sha256': _sha(_canonical(public_hashes)),
            'prompt_sha256': _sha(row['prompt'].encode()),
            'normalized_prompt_sha256': _sha(normalized_prompt(row['prompt']).encode()),
            'source_metadata': row.get('meta_data', {}),
            'qualified': not reasons, 'structurally_admissible': not reasons,
            'exclusion_reasons': sorted(set(reasons)), 'runtime_ready': False,
            'admission_status': 'static_only_runtime_validation_pending',
            'reward_profile': PROFILE, 'historical_exposure': 'unknown'}
    from .claw_contracts import apply_execution_contract
    apply_execution_contract(record)
    return record


def verify_prepared_record(record: dict) -> None:
    path = Path(record['source_path'])
    if sha256_file(path) != record['source_sha256']:
        raise ValueError('source_hash_changed')
    if tree_digest(path.parent) != record['source_tree_sha256']:
        raise ValueError('source_tree_hash_changed')
    raw = Path(record['source_identity']['raw_record_path']).read_bytes()
    if _sha(raw.rstrip(b'\r\n')) != record['source_identity']['hf_record_sha256']:
        raise ValueError('consumed_HF_record_hash_changed')


def group_and_split(records: list[dict]) -> dict:
    """Whole metadata/duplicate/template connected components, then 90/10.

    v3's entity-normalized trigram Jaccard >= .35 is retained. A rare-token
    prefix filter finds every pair meeting that threshold, without an O(N²)
    scan over common trigrams. Known nonempty identical public source trees
    are also grouped, conservatively. Semantic paraphrases remain unproven.
    """
    records.sort(key=lambda row: row['source_identity']['hf_line'])
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parents[max(a, b)] = min(a, b)
    keys = {}
    features = []
    frequencies = Counter()
    metadata_counts = Counter()
    for i, row in enumerate(records):
        metadata = row['source_metadata']
        nested = metadata.get('metadata', {})
        grouping = [('prompt', row['normalized_prompt_sha256'])]
        if row['public_asset_sha256']:
            grouping.append(('nonempty_public_asset_tree', row['public_asset_tree_sha256']))
        if nested.get('primary_skill_id'):
            grouping.append(('primary_skill', nested['primary_skill_id']))
        for container in (metadata, nested):
            for key in ('origin_template', 'template_id', 'source_template', 'source_family', 'origin_id'):
                if container.get(key):
                    grouping.append((key, _sha(_canonical(container[key]))))
        metadata_counts['named_task_id' if metadata.get('task_id') else 'missing_task_id'] += 1
        metadata_counts['primary_skill_id_present' if nested.get('primary_skill_id') else 'primary_skill_id_absent'] += 1
        for key in grouping:
            if key in keys:
                union(i, keys[key])
            else:
                keys[key] = i
        query = json.loads(Path(row['source_path']).read_text())['user_query']
        terms = _query_features(query)
        features.append(terms)
        frequencies.update(terms)
    index = defaultdict(list)
    candidate_pairs, linked_pairs = 0, 0
    # Prefix theorem: a Jaccard>=t pair must share a token in prefixes of
    # length len(S)-ceil(t*len(S))+1 under one global token ordering.
    for i, terms in enumerate(features):
        ordered = sorted(terms, key=lambda term: (frequencies[term], term))
        prefix = ordered[:len(ordered) - math.ceil(.35 * len(ordered)) + 1] if ordered else []
        candidates = {j for term in prefix for j in index[term]}
        for j in candidates:
            if min(len(terms), len(features[j])) < .35 * max(len(terms), len(features[j])):
                continue
            candidate_pairs += 1
            if find(i) == find(j):
                continue
            other = features[j]
            intersection = len(terms & other)
            denominator = len(terms) + len(other) - intersection
            if denominator and intersection / denominator >= .35:
                union(i, j)
                linked_pairs += 1
        for term in prefix:
            index[term].append(i)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    sizes = []
    for group in groups.values():
        signature = _sha(_canonical(sorted(records[i]['source_identity']['hf_record_canonical_sha256'] for i in group)))
        family = 'clawgym:full_source_family:' + signature[:24]
        sizes.append(len(group))
        for i in group:
            records[i]['family'] = family
            records[i]['family_members'] = len(group)
            records[i]['family_method'] = 'source_metadata_normalized_prompt_nonempty_public_asset_tree_trigram_jaccard_ge_0.35_transitive_v1'
    split_map = {}
    # Stratify only using source-side static eligibility, before any outcomes.
    qualified = {r['family'] for r in records if r['qualified']}
    for number, families in enumerate((qualified, {r['family'] for r in records} - qualified)):
        names = sorted(families)
        random.Random(f'{SEED}:clawgym:{number}').shuffle(names)
        train_end = round(len(names) * .9)
        split_map.update({name: 'train' if n < train_end else 'dev' for n, name in enumerate(names)})
    for row in records:
        row['split'] = split_map[row['family']]
    return {'method': records[0]['family_method'] if records else None, 'seed': SEED,
            'family_count': len(groups), 'largest_family_records': max(sizes, default=0),
            'family_size_histogram': dict(sorted(Counter(sizes).items())),
            'metadata_counts': dict(metadata_counts), 'jaccard_candidate_pairs': candidate_pairs,
            'jaccard_union_links': linked_pairs,
            'near_duplicate_scope': 'all_pairs_above_entity_normalized_token_trigram_Jaccard_0.35_plus_known_metadata_and_exact_public_assets; semantic_independence_not_proven'}


def apply_benchmark_guard(records: list[dict], proof: dict) -> None:
    exact = set(proof['exact_prompt_sha256'])
    normalized = set(proof['normalized_prompt_sha256'])
    for row in records:
        if row['prompt_sha256'] in exact or row['normalized_prompt_sha256'] in normalized:
            row['exclusion_reasons'] = sorted({*row['exclusion_reasons'], 'official_bench200_public_prompt_collision'})
            row['qualified'] = row['structurally_admissible'] = False
            row['benchmark_collision'] = {'exact': row['prompt_sha256'] in exact,
                                          'normalized': row['normalized_prompt_sha256'] in normalized}


def prepare_claw_source(raw: Path, out: Path, benchmark_root: Path | None = None) -> dict:
    raw, out = Path(raw).resolve(strict=True), Path(out).absolute()
    if sha256_file(raw) != HF_SHA256:
        raise ValueError('HF_raw_bytes_differ_from_frozen_revision')
    proof = public_benchmark_hashes(benchmark_root or DEFAULT_BENCH)
    # Reserve before writes; no output directory can be reused or overwritten.
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'preparation_status.json', {'status': 'started', 'raw': str(raw), 'raw_sha256': HF_SHA256})
    rows = []
    raw_digest = hashlib.sha256()
    try:
        with raw.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                raw_digest.update(line)
                row = materialize_claw_record(json.loads(line), number, line, out / 'tasks')
                row['source_identity']['enclosing_raw_pin_verified'] = True
                rows.append(row)
                if number % 1000 == 0:
                    print(json.dumps({'processed': number, 'static_accepted_so_far': sum(r['qualified'] for r in rows)}), flush=True)
        if raw_digest.hexdigest() != HF_SHA256 or len(rows) != HF_COUNT:
            raise ValueError('consumed_HF_bytes_or_record_count_differ_from_frozen_source')
        ids = [row['id'] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate_source_task_identity')
        apply_benchmark_guard(rows, proof)
        grouping = group_and_split(rows)
        for row in rows:
            verify_prepared_record(row)
        files = {'inventory.jsonl': rows, 'accepted.jsonl': [row for row in rows if row['qualified']],
                 'exclusions.jsonl': [row for row in rows if not row['qualified']]}
        for name, records in files.items():
            with (out / name).open('w') as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        write_json(out / 'benchmark_public_hashes.json', proof)
        write_json(out / 'split_provenance.json', {'grouping': grouping,
                   'families': [{'family': family, 'split': members[0]['split'], 'ids': [r['id'] for r in members]}
                                for family, members in _families(rows).items()]})
        accepted = files['accepted.jsonl']
        stats = {'inventory_records': len(rows), 'accepted_records': len(accepted),
                 'accepted_by_split': dict(Counter(r['split'] for r in accepted)),
                 'inventory_by_split': dict(Counter(r['split'] for r in rows)),
                 'exclusion_reasons': dict(Counter(reason for r in rows for reason in r['exclusion_reasons'])),
                 'rubric_applicable_records': sum(bool(r['source_contract'].get('rubric_applicable')) for r in rows),
                 'original_source_task_ids': sum(bool(r['source_identity']['source_task_id']) for r in rows),
                 'family_count': grouping['family_count'],
                 'meets_7000_structural_train_target': sum(r['split'] == 'train' for r in accepted) >= 7000,
                 'runtime_ready_records': 0}
        write_json(out / 'stats.json', stats)
        manifest = {'schema': 'clawgym-full-HF-preparation/v1', 'source': 'clawgym',
                    'hf_revision': HF_REVISION, 'raw_sha256': HF_SHA256, 'raw_path': str(raw),
                    'consumed_raw_sha256': raw_digest.hexdigest(), 'raw_bytes': raw.stat().st_size,
                    'source_count': len(rows), 'stats': stats, 'grouping': grouping,
                    'reward_profile': PROFILE, 'public_benchmark_revision': PINNED_REVISION,
                    'status': 'static_preparation_complete_runtime_admission_pending',
                    'runtime_ready': False, 'source_code_executed': False,
                    'file_sha256': {name: sha256_file(out / name) for name in [*files, 'benchmark_public_hashes.json', 'split_provenance.json', 'stats.json']}}
        write_json(out / 'source_manifest.json', manifest)
        write_json(out / 'fingerprint.json', {'raw_sha256': HF_SHA256, 'manifest_sha256': sha256_file(out / 'source_manifest.json'),
                   'inventory_sha256': sha256_file(out / 'inventory.jsonl'),
                   'accepted_sha256': sha256_file(out / 'accepted.jsonl'),
                   'consumed_task_trees_sha256': _sha(_canonical({r['id']: r['source_tree_sha256'] for r in rows}))})
        write_json(out / 'preparation_status.json', {'status': manifest['status'], 'stats': stats})
        return manifest
    except Exception as exc:
        # Partial source/evidence survives failure. No cleanup erases it.
        write_json(out / 'preparation_status.json', {'status': 'failed_preserved', 'processed_records': len(rows),
                   'error': {'type': type(exc).__name__, 'message': str(exc)}})
        raise


def _families(rows: list[dict]) -> dict[str, list[dict]]:
    families = defaultdict(list)
    for row in rows:
        families[row['family']].append(row)
    return dict(sorted(families.items()))


def copy_prepared_claw_record(record: dict, row: dict, line: bytes, tasks: Path) -> dict:
    """Revalidate all original bytes, independently copy, refine capabilities.

    A previous static source pass caches the reviewed v3 AST analysis only.
    Original raw/assets/tree bytes are independently verified again; previous
    family/split identities survive dependency-profile repairs unchanged.
    """
    if json.loads(line) != row:
        raise ValueError('HF_record_does_not_match_consumed_line')
    identity = record['source_identity']
    if _sha(line.rstrip(b'\r\n')) != identity['hf_record_sha256'] or _sha(_canonical(row)) != identity['hf_record_canonical_sha256']:
        raise ValueError('prepared_record_does_not_match_fixed_raw_line')
    verify_prepared_record(record)
    old = Path(record['source_path']).parent
    if (old / 'source_record.json').read_bytes() != line:
        raise ValueError('prepared_raw_record_bytes_changed')
    for asset in row['input_files']:
        name = _asset_name(asset['file_path'])
        source = old / name if name.parts[0] == 'reward' else old / 'input_files' / name
        if source.read_bytes() != asset['content'].encode():
            raise ValueError('prepared_asset_differs_from_fixed_raw')
    target = Path(tasks).resolve() / old.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old, target)  # independent files, never hardlinks to evidence
    copied = copy.deepcopy(record)
    copied['source_path'] = str(target / 'data_entry.json')
    copied['public_assets_path'] = str(target / 'input_files')
    copied['private_grader_path'] = str(target)
    copied['private_checker_path'] = str(target / 'reward/test.py')
    copied['source_identity']['raw_record_path'] = str(target / 'source_record.json')
    reference = copied['source_contract'].get('immutable_gold_reference')
    if reference:
        reference['original_public_assets_path'] = copied['public_assets_path']
    from .claw_contracts import apply_execution_contract
    apply_execution_contract(copied)
    verify_prepared_record(copied)
    return copied


def refine_claw_source(raw: Path, prepared_source: Path, out: Path, benchmark_root: Path | None = None) -> dict:
    """Full fixed-raw pass with cached v3 AST contracts and frozen splits."""
    raw, prepared_source, out = Path(raw).resolve(strict=True), Path(prepared_source).resolve(strict=True), Path(out).absolute()
    if sha256_file(raw) != HF_SHA256:
        raise ValueError('HF_raw_bytes_differ_from_frozen_revision')
    prior_path = prepared_source / 'source_manifest.json'
    prior = json.loads(prior_path.read_text())
    if prior.get('raw_sha256') != HF_SHA256 or prior.get('source_count') != HF_COUNT:
        raise ValueError('prior_full_preparation_pin_or_count_mismatch')
    fingerprint = json.loads((prepared_source / 'fingerprint.json').read_text())
    if sha256_file(prior_path) != fingerprint['manifest_sha256']:
        raise ValueError('prior_source_manifest_hash_changed')
    for name, expected in prior['file_sha256'].items():
        if sha256_file(prepared_source / name) != expected:
            raise ValueError('prior_preparation_file_hash_changed:' + name)
    proof = public_benchmark_hashes(benchmark_root or DEFAULT_BENCH)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'preparation_status.json', {'status': 'started', 'mode': 'refine_frozen_splits', 'raw_sha256': HF_SHA256})
    rows, digest = [], hashlib.sha256()
    try:
        with raw.open('rb') as stream, (prepared_source / 'inventory.jsonl').open() as inventory:
            for number, (line, previous_line) in enumerate(itertools.zip_longest(stream, inventory), 1):
                if line is None or previous_line is None:
                    raise ValueError('prior_inventory_and_fixed_raw_count_differ')
                digest.update(line)
                previous = json.loads(previous_line)
                if previous['source_identity']['hf_line'] != number or previous['split'] not in {'train', 'dev'} or not previous['family']:
                    raise ValueError('prior_frozen_family_or_line_identity_invalid')
                record = copy_prepared_claw_record(previous, json.loads(line), line, out / 'tasks')
                if (record['family'], record['split']) != (previous['family'], previous['split']):
                    raise ValueError('dependency_refinement_changed_frozen_family_split')
                rows.append(record)
                if number % 1000 == 0:
                    print(json.dumps({'processed': number, 'static_accepted_so_far': sum(r['qualified'] for r in rows), 'split_policy': 'reuse_frozen_original_family_split'}), flush=True)
        if digest.hexdigest() != HF_SHA256 or len(rows) != HF_COUNT or len({r['id'] for r in rows}) != HF_COUNT:
            raise ValueError('refined_consumed_bytes_count_or_identity_mismatch')
        apply_benchmark_guard(rows, proof)
        files = {'inventory.jsonl': rows, 'accepted.jsonl': [r for r in rows if r['qualified']], 'exclusions.jsonl': [r for r in rows if not r['qualified']]}
        for name, records in files.items():
            with (out / name).open('w') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        write_json(out / 'benchmark_public_hashes.json', proof)
        split_proof = json.loads((prepared_source / 'split_provenance.json').read_text())
        write_json(out / 'split_provenance.json', split_proof)
        contracts = [r['source_contract'].get('checker_execution_contract', {}) for r in rows]
        execution = {'source_records': len(rows), 'checker_process_call_records': sum(bool(c.get('process_calls')) for c in contracts),
                     'checker_execution_status': dict(Counter(c.get('status', 'missing_checker') for c in contracts)),
                     'unavailable_or_unverified_programs': dict(Counter(program for c in contracts for program in c.get('unavailable_or_unverified_programs', []))),
                     'dynamic_command_records': sum(bool(c.get('dynamic_command_evidence')) for c in contracts),
                     'requested_script_or_binary_records': sum(bool(c.get('requested_script_and_binary_evidence')) for c in contracts),
                     'records': [{'id': r['id'], 'hf_line': r['source_identity']['hf_line'], 'qualified': r['qualified'], 'contract': c} for r, c in zip(rows, contracts)]}
        write_json(out / 'checker_execution_audit.json', execution)
        accepted = files['accepted.jsonl']
        stats = {'inventory_records': len(rows), 'accepted_records': len(accepted),
                 'accepted_by_split': dict(Counter(r['split'] for r in accepted)),
                 'inventory_by_split': dict(Counter(r['split'] for r in rows)),
                 'exclusion_reasons': dict(Counter(reason for r in rows for reason in r['exclusion_reasons'])),
                 'rubric_applicable_records': sum(bool(r['source_contract'].get('rubric_applicable')) for r in rows),
                 'original_source_task_ids': sum(bool(r['source_identity']['source_task_id']) for r in rows),
                 'family_count': prior['grouping']['family_count'],
                 'meets_7000_structural_train_target': sum(r['split'] == 'train' for r in accepted) >= 7000,
                 'runtime_ready_records': 0}
        write_json(out / 'stats.json', stats)
        names = [*files, 'benchmark_public_hashes.json', 'split_provenance.json', 'checker_execution_audit.json', 'stats.json']
        manifest = {'schema': 'clawgym-full-HF-preparation/v1', 'source': 'clawgym', 'hf_revision': HF_REVISION,
                    'raw_sha256': HF_SHA256, 'consumed_raw_sha256': digest.hexdigest(), 'raw_path': str(raw), 'raw_bytes': raw.stat().st_size,
                    'source_count': len(rows), 'stats': stats, 'grouping': prior['grouping'], 'reward_profile': PROFILE,
                    'public_benchmark_revision': PINNED_REVISION, 'status': 'static_preparation_complete_runtime_admission_pending',
                    'runtime_ready': False, 'source_code_executed': False,
                    'prior_preparation': {'path': str(prepared_source), 'manifest_sha256': sha256_file(prior_path),
                        'reuse': 'v3_AST_contracts_only_all_original_raw_assets_checker_trees_revalidated_and_independently_copied',
                        'frozen_split_sha256': sha256_file(prepared_source / 'split_provenance.json')},
                    'file_sha256': {name: sha256_file(out / name) for name in names}}
        write_json(out / 'source_manifest.json', manifest)
        write_json(out / 'fingerprint.json', {'raw_sha256': HF_SHA256, 'manifest_sha256': sha256_file(out / 'source_manifest.json'),
                   'inventory_sha256': sha256_file(out / 'inventory.jsonl'), 'accepted_sha256': sha256_file(out / 'accepted.jsonl'),
                   'consumed_task_trees_sha256': _sha(_canonical({r['id']: r['source_tree_sha256'] for r in rows}))})
        write_json(out / 'preparation_status.json', {'status': manifest['status'], 'stats': stats})
        return manifest
    except Exception as exc:
        write_json(out / 'preparation_status.json', {'status': 'failed_preserved', 'processed_records': len(rows), 'error': {'type': type(exc).__name__, 'message': str(exc)}})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--benchmark-root', type=Path, default=DEFAULT_BENCH)
    parser.add_argument('--reuse-static-source', type=Path, help='revalidate/copy full fixed raw using cached v3 contracts and frozen family splits')
    args = parser.parse_args()
    manifest = (refine_claw_source(args.raw, args.reuse_static_source, args.out, args.benchmark_root)
                if args.reuse_static_source else prepare_claw_source(args.raw, args.out, args.benchmark_root))
    print(json.dumps({'out': str(args.out), 'status': manifest['status'], 'stats': manifest['stats']}, sort_keys=True))


if __name__ == '__main__':
    main()
