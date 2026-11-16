"""Pre-model AgentDoG v6 component recipe; static data only, no tool execution.

The HF catalog has no official folds. The retired v5 internal research folds
were never used for model training, inference, or selection. This correction
joins base families through published selected-scenario queries before any
qualification or candidate selection. External benchmark holdouts are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import copy
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata

import jsonschema

from .agentdog_source import (REVISION, CATALOG_SHA256, PAIRING_SHA256,
                             pairing_records, public_contract)
from .io import atomic_json, canonical, digest, file_hash

SEED = '20261116'
PROTOTYPE_MANIFEST_SHA256 = '7fead9f6a7d0e543fd1d3ba182599b0d0010e4ebe3fef2d6e08e1f4cd492611f'
PROTOTYPE_INVENTORY_SHA256 = '0f31d4ff5eb83fd11694efefbafcaf2cdba6ed72805b6036f55ff665d63cfcd1'
PROTOTYPE_ACCEPTED_SHA256 = 'e1d2273c84051663cab790dbd8d1b34b38f7862581b2e43b6a9974a9d3c355be'
SCHEMA = 'hsrl-v6-agentdog-component-source-preparation/v1'
ROLE = {'attack': 'attacked_benign', 'clean': 'matched_clean_control', 'query_target': 'pure_malicious'}
QUOTAS = {'attack': 5500, 'clean': 1000, 'query_target': 500}


def query_hash(query):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', query).casefold().split()).encode()).hexdigest()


def _pin(path, expected, label):
    if file_hash(path) != expected:
        raise ValueError(label)
    return {'path': str(Path(path).resolve()), 'sha256': expected, 'bytes': Path(path).stat().st_size}


def load_verified_inventory(prepared, raw):
    """Verify externally fixed pins, lineage, and every private bundle identity."""
    prepared, raw = Path(prepared), Path(raw)
    pins = {'prototype_manifest': _pin(prepared / 'source_manifest.json', PROTOTYPE_MANIFEST_SHA256,
                                     'prototype_manifest_hash_mismatch')}
    for name, expected in [('runtime_catalog.json.gz', CATALOG_SHA256),
                           ('semantic_local_pairing.jsonl.gz', PAIRING_SHA256)]:
        pins[name] = _pin(raw / name, expected, 'source_hash_mismatch:' + name)
    pins['prototype_inventory'] = _pin(prepared / 'inventory.jsonl', PROTOTYPE_INVENTORY_SHA256,
                                        'prototype_inventory_hash_mismatch')
    pins['prototype_accepted'] = _pin(prepared / 'accepted.jsonl', PROTOTYPE_ACCEPTED_SHA256,
                                       'prototype_accepted_hash_mismatch')
    manifest = json.loads((prepared / 'source_manifest.json').read_text())
    if (manifest['revision'] != REVISION or manifest['inventory_sha256'] != PROTOTYPE_INVENTORY_SHA256
            or manifest['accepted_sha256'] != PROTOTYPE_ACCEPTED_SHA256):
        raise ValueError('prototype_manifest_identity_mismatch')
    paired, rows, seen = pairing_records(raw / 'semantic_local_pairing.jsonl.gz'), [], set()
    with (prepared / 'inventory.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row['id']
            if identity in seen:
                raise ValueError('duplicate_inventory_identity')
            seen.add(identity)
            if identity in paired:
                lineage = paired[identity]
                if (row['family'] != lineage['base_family_key'] or row['split'] != lineage['split']
                        or row['scenario'] != lineage['scenario'] or row['condition'] != lineage['catalog_task_type']):
                    raise ValueError('prototype_lineage_mismatch:' + identity)
                path = Path(row['source_path'])
                if path.is_symlink() or not path.resolve().is_relative_to((prepared / 'private_assets').resolve()):
                    raise ValueError('prototype_asset_path_escape')
                if file_hash(path) != row['source_sha256']:
                    raise ValueError('private_bundle_hash_mismatch:' + identity)
                bundle = json.loads(path.read_text())
                if digest(bundle) != row['source_record_canonical_sha256'] or bundle['task_spec']['task_id'] != identity:
                    raise ValueError('private_bundle_canonical_identity_mismatch:' + identity)
                row['prototype_split'] = row['split']
                row['normalized_prompt_sha256'] = query_hash(bundle['execution_bundle']['scenarios'][row['scenario']]['user_query'])
            elif row.get('family') or row['exclusion_reasons'] != ['outside_frozen_pairing_lineage']:
                raise ValueError('unpaired_lineage_mismatch')
            rows.append(row)
    if len(rows) != 15774 or set(paired) - seen:
        raise ValueError('prototype_inventory_count_mismatch')
    # Check that accepted is exactly the qualified projection, not merely a
    # separately well-hashed but inconsistent inventory.
    accepted = {json.loads(line)['id'] for line in (prepared / 'accepted.jsonl').open()}
    if accepted != {r['id'] for r in rows if r['qualified']}:
        raise ValueError('prototype_accepted_projection_mismatch')
    return rows, pins


def component_assignment(rows):
    """Union ALL paired base-family vertices through equal selected-query edges."""
    parent = {r['family']: r['family'] for r in rows}

    def find(family):
        while parent[family] != family:
            parent[family] = parent[parent[family]]
            family = parent[family]
        return family

    first_query = {}
    for row in rows:
        family = row['family']
        previous = first_query.setdefault(row['normalized_prompt_sha256'], family)
        a, b = find(previous), find(family)
        if a != b:
            parent[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for family in sorted(parent):
        members[find(family)].append(family)
    components = {digest(sorted(families)): families for families in members.values()}
    ordered = sorted(components, key=lambda key: hashlib.sha256((SEED + '|' + key).encode()).digest())
    n = len(ordered)
    return {family: {'split_group': group, 'split': 'train' if i < n*8//10 else 'dev' if i < n*9//10 else 'test'}
            for i, group in enumerate(ordered) for family in components[group]}


def capability_contract(bundle, row):
    """Validate full advertised JSON parameter schemas without executing tools."""
    public = public_contract(bundle, row['scenario'])
    schemas, errors = {}, []
    for tool in public['tools']:
        fn = tool['function']
        schema = fn.get('parameters')
        schemas[fn['name']] = digest(schema)
        try:
            jsonschema.Draft7Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            errors.append({'tool': fn['name'], 'schema_path': list(exc.absolute_path), 'message': exc.message})
    return {'version': 'hsrl-v6-agentdog-capability-contract/v1',
            'status': 'structurally_supported_runtime_unverified' if row['qualified'] and not errors else 'pending_or_unsupported',
            'schema_validation': {'validator': 'jsonschema.Draft7Validator',
                'dialect': 'http://json-schema.org/draft-07/schema#',
                'validation_scope': 'complete_advertised_function_parameters_without_rewriting',
                'all_public_parameter_schemas_valid': not errors, 'tool_schema_sha256': schemas, 'errors': errors},
            'public_contract_sha256': digest(public),
            'source_rule_contract': copy.deepcopy(row['source_contract']),
            'source_execution_performed': False, 'runtime_admission_verified': False,
            'public_interface': 'published_selected_scenario_user_query_and_advertised_tools_only',
            'failure_semantics': 'unsupported_or_unverified_contract_is_not_runtime_ready'}


def _round_robin(rows):
    families = defaultdict(list)
    for row in rows:
        families[row['family']].append(row)
    order = lambda key: hashlib.sha256((SEED + '|' + key).encode()).digest()
    queues = {family: deque(sorted(items, key=lambda r: order(r['id']))) for family, items in families.items()}
    pending = sorted(queues, key=order)
    while pending:
        remaining = []
        for family in pending:
            yield queues[family].popleft()
            if queues[family]:
                remaining.append(family)
        pending = remaining


def candidate_index(rows):
    pool = [r for r in rows if r['qualified'] and r['split'] == 'train']
    attacks = list(_round_robin(r for r in pool if r['condition'] == 'attack'))[:QUOTAS['attack']]
    attack_families = {r['family'] for r in attacks}
    clean = list(_round_robin(r for r in pool if r['condition'] == 'clean' and r['family'] in attack_families))[:QUOTAS['clean']]
    queries = list(_round_robin(r for r in pool if r['condition'] == 'query_target'))[:QUOTAS['query_target']]
    selected = attacks + clean + queries
    if Counter(r['condition'] for r in selected) != QUOTAS:
        raise ValueError('candidate_role_or_paired_family_capacity_insufficient')
    if len({r['id'] for r in selected}) != 7000 or len({r['source_sha256'] for r in selected}) != 7000:
        raise ValueError('candidate_duplicate_identity_or_source_bytes')
    tasks = [{'task_id': r['id'], 'base_family_key': r['family'], 'split_group': r['split_group'],
              'catalog_task_type': r['condition'], 'scenario': r['scenario'], 'split': r['split'],
              'training_role': ROLE[r['condition']], 'source_sha256': r['source_sha256'],
              'source_record_canonical_sha256': r['source_record_canonical_sha256']} for r in selected]
    index = {'schema': 'hsrl-v6-agentdog-static-candidate-index/v1', 'seed': SEED, 'seed_trials': 1,
             'selection': 'qualified_train_base_family_round_robin_then_same_base_family_clean_controls',
             'runtime_ready': False, 'tasks': tasks}
    index['index_sha256'] = digest(index)
    return index


def audit_partition(rows):
    """Independent adjacency traversal and direct cross-fold set intersection."""
    queries, families, adjacency = defaultdict(set), defaultdict(set), defaultdict(set)
    by_query = defaultdict(list)
    for row in rows:
        if not row.get('family'):
            continue
        families[row['family']].add(row['split'])
        queries[row['normalized_prompt_sha256']].add(row['split'])
        by_query[row['normalized_prompt_sha256']].append(row['family'])
        adjacency[row['family']]
    for values in by_query.values():
        first = values[0]
        for family in values[1:]:
            adjacency[first].add(family); adjacency[family].add(first)
    mapping = {r['family']: r['split_group'] for r in rows if r.get('family')}
    unseen, components = set(adjacency), []
    while unseen:
        start = min(unseen)
        pending, members = [start], set()
        while pending:
            family = pending.pop()
            if family in members:
                continue
            members.add(family)
            pending.extend(adjacency[family] - members)
        unseen -= members
        if {mapping[f] for f in members} != {digest(sorted(members))}:
            raise ValueError('independent_component_topology_mismatch')
        components.append(members)
    result = {'schema': 'hsrl-v6-agentdog-independent-partition-audit/v1',
              'method': 'independent_adjacency_graph_traversal_and_direct_all_paired_fold_sets',
              'paired_records': sum(bool(r.get('family')) for r in rows),
              'components': len(components), 'base_families': len(families), 'normalized_public_queries': len(queries),
              'cross_fold_base_families': sum(len(v)>1 for v in families.values()),
              'cross_fold_normalized_public_queries': sum(len(v)>1 for v in queries.values())}
    if result['cross_fold_base_families'] or result['cross_fold_normalized_public_queries']:
        raise ValueError('cross_fold_overlap')
    return result


def prepare_source(prepared, raw, out):
    rows, pins = load_verified_inventory(prepared, raw)
    paired = [r for r in rows if r.get('family')]
    assignment = component_assignment(paired)
    for row in paired:
        row.update(assignment[row['family']])
    counts = Counter(r['split'] + ':' + r['condition'] for r in paired if r['qualified'])
    component_counts = Counter({v['split_group']: v['split'] for v in assignment.values()}.values())
    if component_counts != {'train':1164,'dev':146,'test':146} or any(
            counts['train:' + condition] != expected for condition,expected in [('attack',6843),('clean',2073),('query_target',1530)]):
        raise ValueError('published_expected_topology_or_capacity_mismatch')
    index = candidate_index(paired)
    selected = {r['task_id']: r for r in index['tasks']}
    audit = audit_partition(rows)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    private = out / 'private_assets'; private.mkdir(mode=0o700)
    mapping = []
    with ((out / 'inventory.jsonl').open('w') as inventory,
          (out / 'accepted.jsonl').open('w') as accepted,
          (out / 'exclusions.jsonl').open('w') as exclusions):
        for i,row in enumerate(rows):
            if row.get('family'):
                previous = Path(row['source_path'])
                target = private / previous.name
                # copyfile never shares writable hardlink state with the prototype.
                shutil.copyfile(previous, target)
                target.chmod(0o600)
                target_stat, previous_stat = target.stat(), previous.stat()
                if (file_hash(target) != row['source_sha256'] or
                        (target_stat.st_dev, target_stat.st_ino) == (previous_stat.st_dev, previous_stat.st_ino)):
                    raise ValueError('copied_bundle_identity_or_independence_mismatch')
                bundle = json.loads(target.read_text())
                if digest(bundle) != row['source_record_canonical_sha256']:
                    raise ValueError('copied_bundle_canonical_mismatch')
                row['prototype_source_path'] = str(previous)
                row['source_path'] = str(target)
                row['selected_candidate'] = row['id'] in selected
                row['training_role'] = selected.get(row['id'], {}).get('training_role')
                row['runtime_ready'] = False
                row['structurally_admissible'] = row['qualified']
                row['capability_contract'] = capability_contract(bundle, row)
                if row['qualified'] and not row['capability_contract']['schema_validation']['all_public_parameter_schemas_valid']:
                    raise ValueError('qualified_schema_contract_changed')
                row['original_asset_sha256'] = {'bundle.json': row['source_sha256']}
                row['asset_identity'] = {'version': 'single_bundle_json_v1', 'asset_key': 'bundle.json',
                    'sha256': row['source_sha256'], 'canonical_sha256': row['source_record_canonical_sha256'],
                    'scope': 'single_source_bundle_file_not_private_assets_parent'}
                mapping.append({k:row[k] for k in ('id','family','prototype_split','split_group','split','normalized_prompt_sha256')})
                if row['qualified']:
                    accepted.write(canonical(row) + '\n')
            inventory.write(canonical(row) + '\n')
            if not row['qualified']:
                exclusions.write(canonical(row) + '\n')
            if (i+1) % 500 == 0:
                print(canonical({'processed': i+1, 'copied': len(mapping)}), flush=True)
    atomic_json(out / 'candidate_index.json', index)
    atomic_json(out / 'fold_mapping.json', {'schema':'hsrl-v6-agentdog-fold-mapping/v1','records':mapping})
    atomic_json(out / 'partition_audit.json', audit)
    files = {name:file_hash(out/name) for name in ['inventory.jsonl','accepted.jsonl','exclusions.jsonl','candidate_index.json','fold_mapping.json','partition_audit.json']}
    manifest = {'schema': SCHEMA, 'revision': REVISION, 'catalog_gzip_sha256': CATALOG_SHA256,
        'raw_input_pins_and_prototype_references': pins,
        'partition_policy': {'version':'pre_model_component_correction_v1','seed': SEED,'seed_trials':1,
            'official_hf_folds':False,'prototype_folds_used_for_model_training_inference_or_selection':False,
            'external_benchmark_holdouts_changed':False,
            'edges':'all_paired_original_base_family_plus_equal_selected_scenario_NFKC_casefold_whitespace_query',
            'component_identity':'SHA256(canonical_JSON(sorted(base_family_ids)))',
            'ordering':'SHA256(UTF8(20261116|component_sha256))',
            'boundaries':'floor(0.8*N), floor(0.9*N), remainder; independent_of_qualification'},
        'source_count':len(rows),'paired_count':len(paired),'unpaired_count':len(rows)-len(paired),
        'original_base_family_count':len(assignment),'split_group_count':len(set(v['split_group'] for v in assignment.values())),
        'split_group_counts':dict(component_counts),
        'prototype_base_family_counts':dict(Counter({r['family']:r['prototype_split'] for r in paired}.values())),
        'new_base_family_counts':dict(Counter({r['family']:r['split'] for r in paired}.values())),
        'inventory_counts':dict(Counter(r['split']+':'+r['condition'] for r in paired)),
        'accepted_counts':dict(counts),
        'exclusion_counts':dict(Counter(reason for r in rows for reason in r['exclusion_reasons'])),
        'accepted_selected_candidates':len(selected),'candidate_index_sha256':index['index_sha256'],
        'inventory_sha256':files['inventory.jsonl'],'accepted_sha256':files['accepted.jsonl'],
        'file_sha256':files,'topology_and_fold_mapping_sha256':files['fold_mapping.json'],
        'asset_copy_policy':'independent_byte_identical_files_no_hardlinks',
        'capability_contract_version':'hsrl-v6-agentdog-capability-contract/v1',
        'runtime_ready':False,'runtime_admission_verified':False,'source_execution_performed':False,
        'status':'static_candidates_not_rl_ready','old_quality_evidence_reused':False}
    atomic_json(out / 'source_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-source',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    print(canonical(prepare_source(args.prepared_source,args.source_root,args.out)))
