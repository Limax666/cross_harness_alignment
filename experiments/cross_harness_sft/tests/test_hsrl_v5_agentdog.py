"""Pinned real AgentDoG source contracts, with no model or source-code execution."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

EXP = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')
WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'scripts'))
SOURCE = EXP / 'data/rl_sources/agentdog_runtime_hf_7c26321'
INDEX = EXP / 'docs/research/agentdog_rl_safety7000_candidate_index_20261116.json'


def source_module():
    assert importlib.util.find_spec('hsrl_v5.agentdog_source'), 'real AgentDoG dataset source loader is missing'
    from hsrl_v5 import agentdog_source
    return agentdog_source


def test_catalog_stream_returns_actual_bundle_identity_and_state():
    module = source_module()
    reader = module.iter_catalog(SOURCE / 'runtime_catalog.json.gz')
    identity, bundle = next(reader)
    assert identity == bundle['task_spec']['task_id']
    assert bundle['execution_bundle']['tool_impl_sources']
    assert bundle['execution_bundle']['initial_state_template']
    assert bundle['evaluation_bundle']['success_eval_rule']['type'].startswith('state_')
    reader.close()


def test_reader_rejects_raw_source_drift(tmp_path):
    module = source_module()
    changed = tmp_path / 'runtime_catalog.json.gz'
    changed.write_bytes(b'not the pinned source')
    with pytest.raises(ValueError, match='source_hash'):
        next(module.iter_catalog(changed))


def test_candidate_index_is_bound_to_actual_source_family_splits(tmp_path):
    module = source_module()
    records = module.load_index(INDEX, SOURCE / 'semantic_local_pairing.jsonl.gz')
    assert len(records) == 7000
    assert len({row['task_id'] for row in records}) == 7000
    assert all(row['split'] == 'train' for row in records)
    changed = json.loads(INDEX.read_text())
    changed['tasks'][0]['split'] = 'test'
    path = tmp_path / 'changed_index.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        module.load_index(path, SOURCE / 'semantic_local_pairing.jsonl.gz')


def test_public_contract_does_not_contain_private_grader_state_or_plan():
    module = source_module()
    _, bundle = next(module.iter_catalog(SOURCE / 'runtime_catalog.json.gz'))
    packet = module.public_contract(bundle, 'clean')
    assert set(packet) == {'messages', 'tools'}
    assert packet['messages'] == [{'role': 'user', 'content': bundle['execution_bundle']['scenarios']['clean']['user_query']}]
    assert {t['function']['name'] for t in packet['tools']} == set(bundle['tool_registry_view']['allowed_tool_names'])
    body = json.dumps(packet)
    for name in ('success_eval_rule', 'risk_success_eval_rule', 'planner_trace', 'initial_state_template', 'reward_info'):
        assert name not in body


def test_source_supported_contains_ci_is_accepted_and_unimplemented_in_is_not():
    module = source_module()
    identity, bundle = next(module.iter_catalog(SOURCE / 'runtime_catalog.json.gz'))
    # Exercise the actual source record contract, without executing its tools.
    rule = {'type': 'history_call_matches', 'tool_name': bundle['tool_registry_view']['allowed_tool_names'][0],
            'arguments_match': {'id': {'$contains_ci': 'record'}}}
    bundle['evaluation_bundle']['risk_success_eval_rule'] = rule
    lineage = {'task_id': identity, 'catalog_task_type': 'query_target', 'scenario': 'clean'}
    assert module.source_contract(bundle, lineage)['qualified']
    rule['arguments_match']['id'] = {'$in': ['record']}
    verdict = module.source_contract(bundle, lineage)
    assert not verdict['qualified']
    assert 'unsupported_source_rule_operator' in verdict['exclusion_reasons']


def test_execution_bootstrap_keeps_evaluation_rules_out_of_tool_worker():
    module = source_module()
    assert importlib.util.find_spec('hsrl_v5.agentdog_worker'), 'isolated source worker is missing'
    from hsrl_v5.agentdog_worker import execution_payload, validate_source
    _, bundle = next(module.iter_catalog(SOURCE / 'runtime_catalog.json.gz'))
    payload = execution_payload(bundle, 'clean')
    assert set(payload) == {'programs', 'state', 'context'}
    assert payload['state'] == bundle['execution_bundle']['initial_state_template']
    assert 'evaluation_bundle' not in json.dumps(payload)
    assert 'success_eval_rule' not in json.dumps(payload)
    source = next(iter(bundle['execution_bundle']['tool_impl_sources'].values()))
    validate_source(source)
    with pytest.raises(ValueError, match='import'):
        validate_source(source + '\nimport os\n')


def test_poisoned_episode_cannot_grade_or_finish_stale_state(tmp_path):
    from hsrl_v5.agentdog_backend import AgentDogBackend
    from hsrl_v5.isolation import IsolationError
    # Guard regression: an invalid episode must reject before even accessing
    # state, rules or a transport. Real transport-loss proof is retained by QC.
    episode = AgentDogBackend.__new__(AgentDogBackend)
    episode.finished, episode.closed, episode.poisoned = False, False, True
    episode.poison_reason = 'transport_failure'
    with pytest.raises(IsolationError, match='invalid'):
        episode.grade()
    with pytest.raises(RuntimeError):
        episode.finish()
    with pytest.raises(RuntimeError):
        episode.call('any_tool', {})


def test_constructor_cleanup_uncertainty_never_allows_pool_replacement():
    from hsrl_v5.quality_agentdog import can_retry_infrastructure
    from hsrl_v5.isolation import CleanupUncertain, IsolationError
    assert not can_retry_infrastructure(CleanupUncertain('constructor cleanup not confirmed'))
    assert not can_retry_infrastructure(ValueError('unclassified pool initialization error'))
    assert can_retry_infrastructure(IsolationError('transient transport, cleanup confirmed'))
