"""Algorithm-neutral masks and K2 schedule; no fake actor rollouts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
MAIN = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')


def actual_safety_result():
    root = MAIN / 'outputs/rl/hsrl_v6_agentdog_real_quality_20261116_r1'
    for path in root.glob('cases/*/quality.json'):
        proof = json.loads(path.read_text())
        if proof['ready'] and proof['condition'] == 'attack':
            return proof['positive']
    raise AssertionError('actual source-quality positive proof missing')


def test_algorithm_components_preserve_actual_legal_and_risk_masks():
    assert importlib.util.find_spec('hsrl_v5.reward'), 'common RL reward contract missing'
    from hsrl_v5.reward import components, grpo_score
    actual = actual_safety_result()
    vector = components(actual)
    assert vector['utility'] == actual['utility']
    assert vector['source_cost'] == actual['c_source']
    assert vector['persistence_cost'] is None
    assert not vector['masks']['persistence_cost']
    assert grpo_score(actual, 'safety') == actual['utility'] - actual['c_source'] - .1 * actual['c_protocol']


def test_missing_applicable_cost_and_invalid_episodes_never_become_zero():
    from hsrl_v5.reward import components
    actual = actual_safety_result()
    actual['c_source'] = None
    with pytest.raises(ValueError, match='source_cost'):
        components(actual)
    actual['valid'] = False
    with pytest.raises(ValueError, match='invalid'):
        components(actual)


def test_oa18_covers_every_two_factor_pair_and_peer_is_distinct():
    assert importlib.util.find_spec('hsrl_v5.sampler'), 'algorithm-neutral K2 schedule missing'
    from hsrl_v5.sampler import OA18, configuration_pair, role_slots
    assert len(OA18) == len(set(OA18)) == 18
    for a in range(5):
        for b in range(a + 1, 5):
            assert {(row[a], row[b]) for row in OA18} == {(i, j) for i in range(3) for j in range(3)}
    for block in range(18):
        first, second = configuration_pair(block)
        assert first in OA18 and second in OA18 and first != second
    from collections import Counter
    assert Counter(role for b in range(7) for role in role_slots(b)) == {'attack': 44, 'clean': 8, 'query_target': 4}


def test_prepared_candidates_cannot_be_loaded_as_ready_training_release():
    from hsrl_v5.dataset import Dataset
    with pytest.raises(ValueError, match='not_ready'):
        Dataset(MAIN / 'outputs/rl/hsrl_v5_agentdog_prepare_20261116')


def test_release_relocation_cannot_change_real_source_scenario_or_role():
    from hsrl_v5.dataset import assert_preserved_semantics
    import copy
    path = MAIN / 'outputs/rl/hsrl_v6_agentdog_components_prepare_20261116/accepted.jsonl'
    original = next(row for row in map(json.loads, path.open()) if row['condition'] == 'query_target')
    relocated = copy.deepcopy(original)
    relocated['source_path'] = 'assets/agentdog/task.json'
    relocated['runtime_ready'] = True
    assert_preserved_semantics(original, relocated)
    relocated['scenario'] = 'attacked'
    with pytest.raises(ValueError, match='semantics'):
        assert_preserved_semantics(original, relocated)

V6_SOURCE = MAIN / 'outputs/rl/hsrl_v6_agentdog_components_prepare_20261116'
V6_QUALITY = MAIN / 'outputs/rl/hsrl_v6_agentdog_real_quality_20261116_r1'
CLAW_SOURCE = Path(__file__).resolve().parents[1] / 'outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116'


def v6_real_proof():
    records = {r['id']: r for r in map(json.loads, (V6_SOURCE / 'accepted.jsonl').open())}
    for path in sorted(V6_QUALITY.glob('cases/*/quality.json')):
        proof = json.loads(path.read_text())
        if proof.get('ready') is True:
            return records[proof['id']], proof, path
    raise AssertionError('new v6 real retained source proof missing')


def test_release_requires_real_proof_identity_and_private_grader_evidence():
    from hsrl_v5.release import validate_proof
    import copy
    record, proof, path = v6_real_proof()
    validate_proof(record, proof, path, V6_QUALITY)
    changed = copy.deepcopy(record)
    changed['scenario'] = 'unrelated'
    with pytest.raises(ValueError, match='identity'):
        validate_proof(changed, proof, path, V6_QUALITY)
    changed = copy.deepcopy(proof)
    changed.pop('reset_replay')
    with pytest.raises(ValueError, match='reset'):
        validate_proof(record, changed, path, V6_QUALITY)
    changed = copy.deepcopy(proof)
    changed['attempt_directory'] = '/tmp'
    with pytest.raises(ValueError, match='evidence'):
        validate_proof(record, changed, path, V6_QUALITY)


def test_release_status_rejects_incomplete_new_quality_without_output(tmp_path):
    from hsrl_v5.release import inspect_source
    state, admitted = inspect_source('agentdog', V6_SOURCE, V6_QUALITY)
    assert state['source_manifest_sha256'] == '6fea44557eb505fdbba646a6f75abbf61003033729b851b87b03225e3f85ee91'
    assert state['planned'] == 10446
    assert state['completed'] <= state['planned']
    if not (V6_QUALITY / 'summary.json').exists():
        assert state['complete'] is False
    assert not list(tmp_path.iterdir())


def test_whole_claw_family_benchmark_exclusion_uses_all_inventory():
    from hsrl_v5.release import excluded_benchmark_families
    excluded = excluded_benchmark_families(CLAW_SOURCE)
    assert 'clawgym:full_source_family:35c7ebf6dc3a23022c6b0cb5' in excluded


def test_selection_round_robin_is_stable_and_clean_requires_same_base_family():
    from hsrl_v5.release import family_round_robin, select_agentdog
    rows = [r for r in map(json.loads, (V6_SOURCE / 'accepted.jsonl').open()) if r['split'] == 'train']
    selected = select_agentdog(rows, attack_count=20, clean_count=5, query_count=3)
    assert len(selected) == 28
    attacks = {r['family'] for r in selected if r['condition'] == 'attack'}
    assert all(r['family'] in attacks for r in selected if r['condition'] == 'clean')
    assert [r['id'] for r in selected] == [r['id'] for r in select_agentdog(list(reversed(rows)), attack_count=20, clean_count=5, query_count=3)]
    assert len({r['family'] for r in family_round_robin(rows, 20)}) == 20


def test_sampler_rejects_negative_block_before_accessing_pools():
    from hsrl_v5.sampler import BalancedSampler
    sampler = BalancedSampler.__new__(BalancedSampler)
    with pytest.raises(ValueError, match='invalid_K_block'):
        sampler.block(-1)


def test_candidate_verification_view_cannot_enter_training_objectives():
    from hsrl_v5.dataset import VerificationDataset
    view = VerificationDataset.__new__(VerificationDataset)
    with pytest.raises(ValueError, match='candidate_not_training_ready'):
        view.grpo_reward({})
    with pytest.raises(ValueError, match='candidate_not_training_ready'):
        view.chspo_components({})


def test_unfinished_admission_cannot_stage_or_publish(tmp_path):
    from hsrl_v5.release import require_admission
    with pytest.raises(ValueError, match='source_quality'):
        require_admission({'sources': {'agentdog': {'complete': False}}, 'shortfall': {}})
    assert not list(tmp_path.iterdir())


def test_reward_masks_reject_error_na_and_nonfinite_values():
    from hsrl_v5.reward import components, grpo_score
    import copy
    actual = actual_safety_result()
    for value in ('NA', 'error', float('nan'), True):
        changed = copy.deepcopy(actual)
        changed['c_source'] = value
        with pytest.raises(ValueError, match='source_cost'):
            components(changed)
    record, proof, path = v6_real_proof()
    changed = copy.deepcopy(proof['positive'])
    changed['applicability']['legal_utility'] = False
    changed['utility'] = None
    assert grpo_score(changed, 'utility') is None


def test_certificates_cannot_claim_success_for_other_candidate(tmp_path):
    from hsrl_v5.release import validate_certificates, SOURCE_PINS
    from hsrl_v5.io import digest, file_hash
    (tmp_path / 'audits').mkdir()
    certificate = {'passed': True, 'selection_sha256': 'wrong', 'source_manifest_sha256': SOURCE_PINS,
                   'artifact_file_sha256_sha256': digest({}), 'records': 14000}
    (tmp_path / 'audits/context.json').write_text(json.dumps(certificate))
    manifest = {'file_sha256': {'audits/context.json': file_hash(tmp_path / 'audits/context.json')},
                'artifact_file_sha256_sha256': digest({}), 'selection_sha256': 'expected'}
    with pytest.raises(ValueError, match='audit_identity'):
        validate_certificates(tmp_path, manifest)


def test_packaged_source_execution_code_must_match_real_qc_protocol():
    from hsrl_v5.release import verify_execution_code
    protocol = json.loads((V6_QUALITY / 'protocol.json').read_text())
    frozen = {'hsrl_v5/' + key: value for key, value in protocol['code_sha256'].items()}
    verify_execution_code('agentdog', protocol, frozen)
    frozen['hsrl_v5/agentdog_worker.py'] = 'changed'
    with pytest.raises(ValueError, match='execution_code'):
        verify_execution_code('agentdog', protocol, frozen)


def test_successful_attack_proof_requires_actual_registered_risk_control():
    from hsrl_v5.release import validate_proof
    records = {r['id']: r for r in map(json.loads, (V6_SOURCE / 'accepted.jsonl').open())}
    for path in V6_QUALITY.glob('cases/*/quality.json'):
        proof = json.loads(path.read_text())
        if proof.get('ready') and proof['condition'] == 'attack':
            proof.pop('risk_control')
            with pytest.raises(ValueError, match='risk_control'):
                validate_proof(records[proof['id']], proof, path, V6_QUALITY)
            break
    else:
        raise AssertionError('fresh successful attack source proof missing')


def test_record_masks_match_real_result_vector_for_every_agentdog_role():
    from hsrl_v5.release import record_masks
    from hsrl_v5.reward import components
    records = {r['id']: r for r in map(json.loads, (V6_SOURCE / 'accepted.jsonl').open())}
    seen = set()
    for path in V6_QUALITY.glob('cases/*/quality.json'):
        proof = json.loads(path.read_text())
        if proof.get('ready') and proof['condition'] not in seen:
            assert record_masks(records[proof['id']]) == components(proof['positive'])['masks']
            seen.add(proof['condition'])
        if len(seen) == 3:
            break
    assert seen == {'attack', 'clean', 'query_target'}


def test_read_only_status_cannot_be_combined_with_mutating_modes(capsys):
    from hsrl_v5.release import main
    with pytest.raises(SystemExit) as error:
        main(['--status', '--stage'])
    assert error.value.code == 2
    assert 'not allowed with argument' in capsys.readouterr().err
