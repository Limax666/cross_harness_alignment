"""Frozen AgentDoG source inventory. No tool implementation is executed here."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re

import jsonschema

from .io import atomic_json, canonical, digest, file_hash

REVISION = '7c263216d05afcd2a31185f6d102b35504418b3c'
CATALOG_SHA256 = 'd6d51b20a6a7d6597305af13e1a775233b94fd4c6da7a4ad79b3ad1632e7ee0c'
PAIRING_SHA256 = 'bebb181aca11e16da11d0658b9cd6f36b307d8d6d6213b66b0b9db11eb9f79b3'
INDEX_SHA256 = '69c289ffabf39c8ea0c88754743b15658f84dfbb0faf0090f81843e112dd75ea'
UTILITY_RULES = {'state_subtree_only_matching_records_changed', 'state_subtree_any_match', 'state_subtree_new_any_match'}
RISK_RULES = {'history_call_matches', 'history_call_sequence_contains', 'never'}
OPERATORS = {'$startswith', '$contains', '$contains_all', '$contains_any', '$contains_ci',
             '$contains_all_ci', '$contains_any_ci', '$equals_ci'}


def iter_catalog(path):
    """Stream values from the pinned top-level runtime_catalog object."""
    path = Path(path)
    if file_hash(path) != CATALOG_SHA256:
        raise ValueError('source_hash_mismatch:agentdog_catalog')
    decoder = json.JSONDecoder()
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        buffer = stream.read(65536)
        start = re.match(r'\s*\{\s*"runtime_catalog"\s*:\s*\{', buffer)
        if not start:
            raise ValueError('unexpected_catalog_wrapper')
        buffer = buffer[start.end():]
        eof = False
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith('}'):
                return
            if buffer.startswith(','):
                buffer = buffer[1:].lstrip()
            try:
                identity, pos = decoder.raw_decode(buffer)
                colon = pos
                while colon < len(buffer) and buffer[colon].isspace():
                    colon += 1
                if colon >= len(buffer):
                    raise json.JSONDecodeError('incomplete key', buffer, colon)
                if buffer[colon] != ':':
                    raise ValueError('catalog_key_missing_colon')
                value_start = colon + 1
                while value_start < len(buffer) and buffer[value_start].isspace():
                    value_start += 1
                bundle, end = decoder.raw_decode(buffer, value_start)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError('incomplete_catalog_json')
                block = stream.read(262144)
                eof = not block
                buffer += block
                if len(buffer) > 64 * 1024 * 1024:
                    raise ValueError('catalog_record_size_limit')
                continue
            if not isinstance(identity, str) or not isinstance(bundle, dict):
                raise ValueError('catalog_record_type_mismatch')
            yield identity, bundle
            buffer = buffer[end:]
            if len(buffer) < 4096 and not eof:
                block = stream.read(65536)
                eof = not block
                buffer += block


def pairing_records(path):
    if file_hash(path) != PAIRING_SHA256:
        raise ValueError('source_hash_mismatch:agentdog_pairing')
    families, identities = set(), {}
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            family = row['base_family_key']
            families.add(family)
            for key, category, scenario in [('clean', 'clean', 'clean'),
                                           ('attacks', 'attack', 'attacked'),
                                           ('queries', 'query_target', 'clean')]:
                items = [row[key]] if isinstance(row[key], dict) else row[key]
                for item in items:
                    identity = item['task_id']
                    record = {'task_id': identity, 'base_family_key': family,
                              'catalog_task_type': category, 'scenario': scenario}
                    if identity in identities and identities[identity] != record:
                        raise ValueError('task_identity_family_conflict')
                    identities[identity] = record
    ordered = sorted(families, key=lambda key: hashlib.sha256(('20261116|' + key).encode()).digest())
    if len(ordered) != 2931 or len(identities) != 13212:
        raise ValueError('pairing_inventory_mismatch')
    assignment = {family: 'train' if i < 2344 else 'dev' if i < 2637 else 'test'
                  for i, family in enumerate(ordered)}
    for row in identities.values():
        row['split'] = assignment[row['base_family_key']]
    return identities


def load_index(index_path, pairing_path):
    value = json.loads(Path(index_path).read_text())
    expected = value.pop('index_sha256')
    if digest(value) != expected or expected != INDEX_SHA256:
        raise ValueError('candidate_index_hash_mismatch')
    pairing = pairing_records(pairing_path)
    rows = value['tasks']
    if len(rows) != 7000 or len({row['task_id'] for row in rows}) != 7000:
        raise ValueError('candidate_inventory_mismatch')
    for row in rows:
        actual = pairing[row['task_id']]
        if any(row.get(key) != actual[key] for key in actual) or row['split'] != 'train':
            raise ValueError('candidate_split_or_source_mismatch')
    if Counter(row['training_role'] for row in rows) != {'attacked_benign': 5500, 'matched_clean_control': 1000, 'pure_malicious': 500}:
        raise ValueError('candidate_role_quota_mismatch')
    return rows


def public_contract(bundle, scenario):
    """Whitelist only actual user request and advertised function interfaces."""
    source = bundle['execution_bundle']['scenarios'][scenario]
    registry = bundle['tool_registry_view']
    tools = []
    for tool in registry['tool_schemas']:
        fn = tool['function']
        if fn['name'] not in registry['allowed_tool_names']:
            continue
        tools.append({'type': 'function', 'function': {key: copy.deepcopy(fn[key])
                      for key in ('name', 'description', 'parameters') if key in fn}})
    return {'messages': [{'role': 'user', 'content': source['user_query']}], 'tools': tools}


def _operator_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.startswith('$'):
                yield key
            yield from _operator_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _operator_keys(child)


def source_contract(bundle, paired):
    reasons, details = [], {}
    identity = bundle.get('task_spec', {}).get('task_id')
    if identity != paired['task_id']:
        reasons.append('source_identity_mismatch')
    try:
        public = public_contract(bundle, paired['scenario'])
        if not public['messages'][0]['content'].strip() or not public['tools']:
            reasons.append('empty_public_task_or_tools')
        registry = bundle['tool_registry_view']
        names = {tool['function']['name'] for tool in public['tools']}
        if names != set(registry['allowed_tool_names']) or len(names) != len(public['tools']):
            reasons.append('public_schema_dispatch_mismatch')
        dispatch = bundle['server_adapter_manifest']['tool_dispatch_table']
        sources = bundle['execution_bundle']['tool_impl_sources']
        for tool in public['tools']:
            name = tool['function']['name']
            jsonschema.Draft7Validator.check_schema(tool['function']['parameters'])
            entry = dispatch[name]
            code = sources[entry['source_key']]
            tree = ast.parse(code)
            if len(code) > 12000 or not any(isinstance(node, ast.FunctionDef) and node.name == entry['entrypoint'] for node in ast.walk(tree)):
                reasons.append('unsupported_tool_source_contract')
        evaluation = bundle['evaluation_bundle']
        utility, risk = evaluation['success_eval_rule'], evaluation['risk_success_eval_rule']
        if utility.get('type') not in UTILITY_RULES:
            reasons.append('non_state_utility_requires_review')
        if risk.get('type') not in RISK_RULES:
            reasons.append('unknown_risk_rule')
        if paired['catalog_task_type'] == 'clean' and risk.get('type') != 'never':
            reasons.append('clean_risk_applicability_mismatch')
        if paired['catalog_task_type'] != 'clean' and risk.get('type') == 'never':
            reasons.append('missing_applicable_risk_rule')
        unknown = sorted(set(_operator_keys([utility, risk])) - OPERATORS)
        if unknown:
            reasons.append('unsupported_source_rule_operator')
            details['unsupported_rule_operators'] = unknown
        details.update({'utility_rule_type': utility.get('type'), 'risk_rule_type': risk.get('type'),
                        'rule_semantics': 'pinned_source_state_and_registered_call_goal_v1',
                        'tool_source_hashes': {key: hashlib.sha256(code.encode()).hexdigest() for key, code in sources.items()},
                        'public_contract_sha256': digest(public), 'public_tool_count': len(public['tools']),
                        'risk_effect_oracle_applicable': False,
                        'legal_utility_applicable': paired['catalog_task_type'] != 'query_target'})
    except (KeyError, TypeError, ValueError, SyntaxError, jsonschema.SchemaError) as exc:
        reasons.append('malformed_source_contract:' + type(exc).__name__ + ':' + str(exc)[:300])
    return {'qualified': not reasons, 'exclusion_reasons': sorted(set(reasons)), **details}


def prepare_source(source_root, index_path, out):
    source_root, out = Path(source_root), Path(out)
    selected = {row['task_id']: row for row in load_index(index_path, source_root / 'semantic_local_pairing.jsonl.gz')}
    paired = pairing_records(source_root / 'semantic_local_pairing.jsonl.gz')
    out.mkdir(parents=True, exist_ok=False)
    private = out / 'private_assets'
    private.mkdir(mode=0o700)
    inventory, seen, count = [], set(), 0
    with (out / 'inventory.jsonl').open('w') as all_rows, (out / 'accepted.jsonl').open('w') as accepted:
        for identity, bundle in iter_catalog(source_root / 'runtime_catalog.json.gz'):
            if identity in seen:
                raise ValueError('duplicate_source_task_id')
            seen.add(identity); count += 1
            if identity not in paired:
                row = {'source': 'agentdog', 'id': identity, 'qualified': False,
                       'exclusion_reasons': ['outside_frozen_pairing_lineage']}
            else:
                lineage = paired[identity]
                contract = source_contract(bundle, lineage)
                task_path = private / (hashlib.sha256(identity.encode()).hexdigest() + '.json')
                atomic_json(task_path, bundle)
                row = {'source': 'agentdog', 'id': identity, 'family': lineage['base_family_key'],
                       'split': lineage['split'], 'condition': lineage['catalog_task_type'],
                       'scenario': lineage['scenario'], 'selected_candidate': identity in selected,
                       'training_role': selected.get(identity, {}).get('training_role'),
                       'source_path': str(task_path.resolve()), 'source_sha256': file_hash(task_path),
                       'source_tree_sha256': None, 'source_record_canonical_sha256': digest(bundle),
                       'source_identity': {'format': 'agentdog_hf_catalog_v1', 'revision': REVISION,
                                           'catalog_gzip_sha256': CATALOG_SHA256},
                       'source_contract': contract, 'qualified': contract['qualified'],
                       'exclusion_reasons': contract['exclusion_reasons'], 'runtime_ready': False}
                if row['qualified']:
                    accepted.write(canonical(row) + '\n')
            all_rows.write(canonical(row) + '\n')
            inventory.append({key: row.get(key) for key in ('id', 'split', 'condition', 'qualified', 'selected_candidate', 'exclusion_reasons')})
            if count % 500 == 0:
                print(json.dumps({'processed': count, 'paired': sum(item['split'] is not None for item in inventory)}), flush=True)
    if count != 15774 or set(selected) - seen or set(paired) - seen:
        raise ValueError('source_catalog_inventory_mismatch')
    summary = {'schema': 'hsrl-v5-agentdog-source-preparation/v1', 'revision': REVISION,
               'catalog_gzip_sha256': CATALOG_SHA256, 'source_count': count,
               'paired_count': len(paired), 'candidate_index_sha256': INDEX_SHA256,
               'accepted_counts': dict(Counter(item['split'] + ':' + item['condition'] for item in inventory if item['qualified'])),
               'exclusion_counts': dict(Counter(reason for item in inventory for reason in item['exclusion_reasons'])),
               'accepted_selected_candidates': sum(item['qualified'] and bool(item['selected_candidate']) for item in inventory),
               'inventory_sha256': file_hash(out / 'inventory.jsonl'),
               'accepted_sha256': file_hash(out / 'accepted.jsonl'), 'runtime_admission_verified': False}
    atomic_json(out / 'source_manifest.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--candidate-index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(canonical(prepare_source(args.source_root, args.candidate_index, args.out)))
