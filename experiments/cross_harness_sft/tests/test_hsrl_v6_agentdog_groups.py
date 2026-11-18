"""Pre-model component correction driven by the pinned published AgentDoG tasks."""
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import pytest

EXP = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
PREPARED = EXP / 'outputs/rl/hsrl_v5_agentdog_prepare_20261116'
RAW = EXP / 'data/rl_sources/agentdog_runtime_hf_7c26321'


def module():
    assert importlib.util.find_spec('hsrl_v5.agentdog_groups'), 'component preparation implementation is missing'
    from hsrl_v5 import agentdog_groups
    return agentdog_groups


@pytest.fixture(scope='module')
def real():
    m = module()
    rows, pins = m.load_verified_inventory(PREPARED, RAW)
    return m, rows, pins


def test_published_query_edges_join_different_states_and_complete_families(real):
    m, rows, _ = real
    paired = [r for r in rows if r.get('family')]
    groups = m.component_assignment(paired)
    queries = defaultdict(list)
    for r in paired:
        queries[r['normalized_prompt_sha256']].append(r)
    cross = next(rs for rs in queries.values() if len({r['family'] for r in rs}) > 1
                 and len({r['prototype_split'] for r in rs}) > 1)
    assert len({r['source_record_canonical_sha256'] for r in cross}) > 1
    from hsrl_v5.io import digest
    states = [json.loads(Path(r['source_path']).read_text())['execution_bundle']['initial_state_template']
              for r in cross]
    assert len({digest(state) for state in states}) > 1
    assert len({groups[r['family']]['split_group'] for r in cross}) == 1
    # Family vertices join every query from every condition transitively.
    for rs in queries.values():
        assert len({groups[r['family']]['split_group'] for r in rs}) == 1
    assert len(groups) == 2931
    assert Counter({g['split_group']: g['split'] for g in groups.values()}.values()) == {'train':1164,'dev':146,'test':146}


def test_partition_is_independent_of_qualification_order_and_original_fold(real):
    m, rows, _ = real
    paired = [r for r in rows if r.get('family')]
    expected = m.component_assignment(paired)
    changed = [dict(r, qualified=not r['qualified'], prototype_split='test') for r in reversed(paired)]
    assert m.component_assignment(changed) == expected
    counts = Counter((expected[r['family']]['split'],r['condition']) for r in paired if r['qualified'])
    assert {k[1]:v for k,v in counts.items() if k[0]=='train'} == {'attack':6843,'clean':2073,'query_target':1530}


def test_published_query_target_keeps_clean_scenario_and_full_draft7_contract(real):
    m, rows, _ = real
    row = next(r for r in rows if r.get('condition') == 'query_target' and r['qualified'])
    bundle = json.loads(Path(row['source_path']).read_text())
    assert row['scenario'] == 'clean'
    assert row['normalized_prompt_sha256'] == m.query_hash(bundle['execution_bundle']['scenarios']['clean']['user_query'])
    contract = m.capability_contract(bundle, row)
    assert contract['schema_validation']['validator'] == 'jsonschema.Draft7Validator'
    assert contract['schema_validation']['all_public_parameter_schemas_valid']
    assert contract['runtime_admission_verified'] is False
    assert contract['public_contract_sha256'] == row['source_contract']['public_contract_sha256']


def test_real_index_has_exact_roles_distinct_ids_and_same_base_family_controls(real):
    m, rows, _ = real
    groups = m.component_assignment([r for r in rows if r.get('family')])
    corrected = [dict(r, **groups[r['family']]) for r in rows if r.get('family')]
    index = m.candidate_index(corrected)
    selected = index['tasks']
    assert len(selected) == len({r['task_id'] for r in selected}) == 7000
    assert len({r['source_sha256'] for r in selected}) == 7000
    assert Counter(r['training_role'] for r in selected) == {'attacked_benign':5500,'matched_clean_control':1000,'pure_malicious':500}
    attack_families = {r['base_family_key'] for r in selected if r['catalog_task_type']=='attack'}
    assert all(r['base_family_key'] in attack_families for r in selected if r['catalog_task_type']=='clean')
    assert all(r['split']=='train' for r in selected)
    from hsrl_v5.io import digest
    assert index['index_sha256'] == digest({k:v for k,v in index.items() if k!='index_sha256'})
    assert index == m.candidate_index(list(reversed(corrected)))
    with pytest.raises(ValueError, match='capacity'):
        m.candidate_index([r for r in corrected if r['condition']!='attack'])


def test_source_drift_rejected_before_output_creation(tmp_path):
    m = module()
    raw = tmp_path/'raw'; raw.mkdir()
    (raw/'runtime_catalog.json.gz').write_bytes(b'drift')
    with pytest.raises(ValueError, match='source_hash_mismatch'):
        m.load_verified_inventory(PREPARED, raw)


def test_prototype_inventory_pin_cannot_be_self_replaced(tmp_path):
    m = module()
    prepared = tmp_path/'prepared'; prepared.mkdir()
    (prepared/'source_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='prototype_manifest_hash_mismatch'):
        m.load_verified_inventory(prepared, RAW)
