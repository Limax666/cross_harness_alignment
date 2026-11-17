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
    else:
        summary = json.loads((V6_QUALITY / 'summary.json').read_text())
        if summary['status'] == 'complete_quality_pass':
            assert state['identity_errors'] == []
            assert state['complete'] is True
            assert len(admitted) == state['admitted'] == summary['ready']
            assert state['admitted_by_condition'] == summary['ready_by_condition']
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
    manifest, _ = real_semantic_metadata()
    grader = tmp_path / 'vendor/agentdog/runtime_evaluators.py'
    grader.parent.mkdir(parents=True)
    shutil.copyfile(MAIN / 'vendor/agentdog_runtime_v5_c8d803f/runtime_evaluators.py', grader)
    manifest['file_sha256']['audits/context.json'] = file_hash(tmp_path / 'audits/context.json')
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

# Review regressions use independent copies of retained real cases, never a
# manufactured 14K release or source execution.
from functools import lru_cache
import copy
import shutil


@lru_cache(maxsize=None)
def retained_case(source, role):
    root = (V6_QUALITY if source == 'agentdog' else MAIN / ('outputs/rl/hsrl_v6_claw_real_quality_20261116_r2'
            if source == 'clawgym_pool' else 'outputs/rl/hsrl_v6_claw_real_quality_20261116_r1'))
    prepared = V6_SOURCE if source == 'agentdog' else CLAW_SOURCE
    records = {r['id']: r for r in map(json.loads, (prepared / 'accepted.jsonl').open())}
    for path in sorted(root.glob('cases/*/quality.json')):
        proof = json.loads(path.read_text())
        if proof.get('ready') and proof['condition'] == role:
            if source == 'clawgym_pool' and not all((Path(proof['attempt_directory']) / episode / 'private_grader/cleanup.json').is_file() for episode in ('actual_episode', 'reset_episode')):
                continue
            return records[proof['id']], proof, path
    raise AssertionError('retained genuine successful case missing: ' + source + ':' + role)


def copied_case(tmp_path, source='agentdog', role='attack'):
    record, original, path = retained_case(source, role)
    case = tmp_path / path.parent.name
    shutil.copytree(path.parent, case)
    proof = copy.deepcopy(original)
    proof['attempt_directory'] = str(case / Path(original['attempt_directory']).name)
    copied_path = case / 'quality.json'
    copied_path.write_text(json.dumps(proof))
    return record, proof, copied_path, tmp_path


def real_semantic_metadata():
    from hsrl_v5.release import REQUIRED_RUNTIME_MODULES, SOURCE_PINS, QUOTAS, LIMITS, GRADER_SHA256, core_descriptor
    from hsrl_v5.reward import VERSION
    from hsrl_v5.dataset import SCHEMA
    from hsrl_v5.io import digest, file_hash
    root = Path(__file__).resolve().parents[1] / 'scripts'
    record, proof, path = retained_case('agentdog', 'attack')
    # A pure metadata validator input over one genuine row. This is never a
    # Dataset fixture and cannot pass the 14K sampler/admission constructor.
    runtime = {name: file_hash(root / name) for name in REQUIRED_RUNTIME_MODULES}
    files = {'code/' + name: sha for name, sha in runtime.items()}
    files['vendor/agentdog/runtime_evaluators.py'] = GRADER_SHA256
    files['private/original_record.json'] = file_hash(path)
    manifest = {'schema': SCHEMA, 'reward_contract': VERSION, 'source_manifest_sha256': copy.deepcopy(SOURCE_PINS),
        'selection_sha256': digest([[record['source'], record['id']]]), 'selection_seed': 20261116,
        'source_role_counts': copy.deepcopy(QUOTAS), 'runtime_limits': copy.deepcopy(LIMITS), 'runtime_module_sha256': runtime,
        'agentdog_grader_path': 'vendor/agentdog/runtime_evaluators.py', 'agentdog_grader_sha256': GRADER_SHA256,
        'file_sha256': files, 'compatibility': {'algorithms': ['GRPO', 'CHS-PO'], 'stock_final_answer_only_reward': False, 'old_hsrl_v3_actor': False},
        'sampler_contract': {'G': 8, 'K': 2, 'groups': 16, 'general_groups': 8, 'safety_groups': 8, 'blocks_per_pass': 875},
        'admission_report': json.loads((V6_QUALITY / 'run_state.json').read_text())}
    manifest['artifact_file_sha256_sha256'] = digest(files)
    manifest['semantic_core_sha256'] = digest(core_descriptor(manifest))
    return manifest, [record]


@pytest.mark.parametrize('mutation', ['grader', 'runtime', 'selection', 'limits'])
def test_audited_semantics_reject_manifest_mutations_over_real_metadata(mutation):
    from hsrl_v5.release import validate_core_identity
    manifest, rows = real_semantic_metadata()
    validate_core_identity(manifest, rows)
    if mutation == 'grader':
        manifest['agentdog_grader_path'] = 'private/original_record.json'
    elif mutation == 'runtime':
        manifest['runtime_module_sha256'].pop('hsrl_v5/grader_pool.py')
    elif mutation == 'selection':
        manifest['selection_sha256'] = 'changed'
    else:
        manifest['runtime_limits']['context_tokens'] = 32768
    with pytest.raises(ValueError):
        validate_core_identity(manifest, rows)


@pytest.mark.parametrize('role', ['attack', 'query_target'])
@pytest.mark.parametrize('artifact', ['reset_episode/result.json', 'reset_episode/private_grade_0.json', 'initial_grade.json'])
def test_real_agentdog_admission_rejects_missing_grade_controls(tmp_path, role, artifact):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path, role=role)
    validate_proof(record, proof, path, root)
    (Path(proof['attempt_directory']) / artifact).unlink()
    with pytest.raises((ValueError, OSError)):
        validate_proof(record, proof, path, root)


@pytest.mark.parametrize('artifact', ['initial_grade.json', 'first_episode/private_grade_1.json', 'reset_episode/private_grade_0.json'])
def test_real_agentdog_admission_rejects_corrupt_private_outcomes(tmp_path, artifact):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path)
    validate_proof(record, proof, path, root)
    target = Path(proof['attempt_directory']) / artifact
    grade = json.loads(target.read_text())
    if artifact == 'initial_grade.json':
        grade['registered_call_goal'] = True
    elif artifact.startswith('first'):
        grade['source_utility'] = False
    else:
        grade['registered_call_goal'] = False
    target.write_text(json.dumps(grade))
    with pytest.raises(ValueError):
        validate_proof(record, proof, path, root)


def test_clean_real_reset_requires_recorded_replay_without_inventing_grade(tmp_path):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path, role='clean')
    reset = Path(proof['attempt_directory']) / 'reset_episode'
    assert not list(reset.glob('private_grade_*.json'))
    assert not (reset / 'result.json').exists()
    validate_proof(record, proof, path, root)
    (reset / 'public_results.jsonl').write_text('')
    with pytest.raises(ValueError):
        validate_proof(record, proof, path, root)


@pytest.mark.parametrize('mutation', ['raw_checker', 'reset_score', 'untouched_grade'])
def test_real_claw_admission_binds_raw_checker_result_and_reset(tmp_path, mutation):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path, source='clawgym', role='benign')
    validate_proof(record, proof, path, root)
    attempt = Path(proof['attempt_directory'])
    if mutation == 'raw_checker':
        (attempt / 'actual_episode/private_grader_response.json').write_text('{"ok":true,"result":{"returncode":0,"stdout":"malformed"}}')
    elif mutation == 'reset_score':
        target = attempt / 'reset_episode/result.json'
        result = json.loads(target.read_text())
        result['utility'] = 0.75
        target.write_text(json.dumps(result))
    else:
        proof['untouched_workspace_grade'] = 0.75
    with pytest.raises(ValueError):
        validate_proof(record, proof, path, root)


@pytest.mark.parametrize('failure', ['manifest_write', 'READY_write', 'consumer_validation', 'final_install'])
def test_private_publication_failure_keeps_original_real_case_immutable(tmp_path, monkeypatch, failure):
    from hsrl_v5 import release
    from hsrl_v5.io import file_hash
    from hsrl_v3.backends import tree_digest
    record, proof, case_path = retained_case('agentdog', 'attack')
    origin = tmp_path / 'original'
    shutil.copytree(case_path.parent, origin)
    before = tree_digest(origin)
    output = tmp_path / 'final'
    original_atomic = release.atomic_json
    def interrupted_write(path, value):
        original_atomic(path, value)
        if (failure == 'manifest_write' and Path(path).name == 'manifest.json') or (failure == 'READY_write' and Path(path).name == 'READY.json'):
            raise KeyboardInterrupt('injected_after_metadata_write')
    monkeypatch.setattr(release, 'atomic_json', interrupted_write)
    # Exercise only the publication filesystem transaction over genuine retained
    # bytes. No fake Dataset, admission, certificate or 14K package is created.
    def assemble(stage):
        release.atomic_json(stage / 'manifest.json', json.loads((V6_SOURCE / 'source_manifest.json').read_text()))
        release.atomic_json(stage / 'READY.json', {'manifest_sha256': file_hash(stage / 'manifest.json')})
    def validate(stage):
        assert (stage / 'READY.json').exists()
        if failure == 'consumer_validation':
            raise RuntimeError('injected_consumer_failure')
    if failure == 'final_install':
        def fail_install(*args):
            raise OSError('injected_install_failure')
        monkeypatch.setattr(release, 'install_no_replace', fail_install)
    with pytest.raises((KeyboardInterrupt, RuntimeError, OSError)):
        release.private_publication_copy(origin, output, assemble, validate)
    assert tree_digest(origin) == before
    assert not (origin / 'READY.json').exists()
    assert not (origin / 'manifest.json').exists()
    assert not output.exists()


def test_publication_install_never_replaces_existing_directory(tmp_path):
    from hsrl_v5.release import install_no_replace
    source = tmp_path / 'private'
    source.mkdir()
    target = tmp_path / 'already_exists'
    target.mkdir()
    with pytest.raises(FileExistsError):
        install_no_replace(source, target)
    assert source.is_dir() and target.is_dir()


@pytest.mark.parametrize('artifact', ['request.json', 'projection.json', 'response.json', 'cleanup.json', 'source_snapshot/test.py'])
def test_real_pooled_claw_requires_complete_private_request_and_cleanup(tmp_path, artifact):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path, source='clawgym_pool', role='benign')
    validate_proof(record, proof, path, root)
    target = Path(proof['attempt_directory']) / 'reset_episode/private_grader' / artifact
    target.unlink()
    with pytest.raises((ValueError, OSError)):
        validate_proof(record, proof, path, root)


def test_v6_lifecycle_protocol_cannot_downgrade_genuine_pool_proof(tmp_path):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path, source='clawgym_pool', role='benign')
    (root / 'protocol.json').write_text(json.dumps({'schema': 'hsrl-v5-claw-real-quality/v6'}))
    validate_proof(record, proof, path, root)
    proof.pop('private_grader_pool')
    for episode in ('actual_episode', 'reset_episode'):
        result_path = Path(proof['attempt_directory']) / episode / 'result.json'
        result = json.loads(result_path.read_text())
        result['evidence']['private_grader'] = None
        result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='pool'):
        validate_proof(record, proof, path, root)


def test_v6_lifecycle_real_partial_admission_is_recognized_not_completed(tmp_path):
    from hsrl_v5.release import inspect_source
    record, proof, path, root = copied_case(tmp_path, source='clawgym_pool', role='benign')
    original = MAIN / 'outputs/rl/hsrl_v6_claw_real_quality_20261116_r2'
    protocol = json.loads((original / 'protocol.json').read_text())
    protocol['schema'] = 'hsrl-v5-claw-real-quality/v6'
    (root / 'protocol.json').write_text(json.dumps(protocol))
    shutil.copytree(original / 'source_snapshot', root / 'source_snapshot')
    cases = root / 'cases'
    cases.mkdir()
    moved = cases / path.parent.name
    shutil.move(str(path.parent), moved)
    proof['attempt_directory'] = str(moved / Path(proof['attempt_directory']).name)
    (moved / 'quality.json').write_text(json.dumps(proof))
    state, admitted = inspect_source('clawgym', CLAW_SOURCE, root)
    assert state['protocol_schema'] == 'hsrl-v5-claw-real-quality/v6'
    assert state['completed'] == state['admitted'] == 1
    assert state['complete'] is False and state['pending_unprocessed'] == 7637
    assert admitted[0][0]['id'] == record['id']


@pytest.mark.parametrize('field', ['child_pid', 'request_nonce'])
def test_real_pooled_claw_rejects_coherent_actual_reset_child_reuse(tmp_path, field):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path, source='clawgym_pool', role='benign')
    attempt = Path(proof['attempt_directory'])
    validate_proof(record, proof, path, root)
    actual = json.loads((attempt / 'actual_episode/private_grader/projection.json').read_text())
    directory = attempt / 'reset_episode'
    for relative in ('private_grader/projection.json', 'private_grader/response.json', 'private_grader_response.json', 'result.json', 'private_grader/request.json'):
        target = directory / relative
        data = json.loads(target.read_text())
        if relative == 'result.json':
            data['evidence']['private_grader'][field] = actual[field]
        elif field == 'child_pid':
            if 'result' in data:
                data['result'][field] = actual[field]
            elif relative.endswith('projection.json'):
                data[field] = actual[field]
        else:
            data[field] = actual[field]
        target.write_text(json.dumps(data))
    proof['private_grader_pool']['requests'][1][field] = actual[field]
    with pytest.raises(ValueError, match='fresh_child'):
        validate_proof(record, proof, path, root)


def test_private_publication_success_installs_copy_and_preserves_original(tmp_path):
    from hsrl_v5.release import private_publication_copy
    from hsrl_v3.backends import tree_digest
    _, _, path = retained_case('agentdog', 'clean')
    origin, output = tmp_path / 'source', tmp_path / 'installed'
    shutil.copytree(path.parent, origin)
    before = tree_digest(origin)
    def assemble(stage):
        assert tree_digest(stage) == before
    def validate(stage):
        assert tree_digest(stage) == before
    private_publication_copy(origin, output, assemble, validate)
    assert tree_digest(origin) == tree_digest(output) == before
    assert not list(tmp_path.glob('.installed.publication-*'))


def test_hard_publication_interruption_never_marks_original_ready(tmp_path):
    import subprocess
    import os
    from hsrl_v5.release import private_publication_copy
    from hsrl_v3.backends import tree_digest
    _, _, case = retained_case('agentdog', 'attack')
    origin, out = tmp_path / 'original', tmp_path / 'final'
    shutil.copytree(case.parent, origin)
    before = tree_digest(origin)
    script = '''
import json,os,signal,sys
from hsrl_v5.release import private_publication_copy
from hsrl_v5.io import atomic_json,file_hash
from pathlib import Path
origin,out,source=map(Path,sys.argv[1:])
def assemble(stage):
    atomic_json(stage/'manifest.json',json.loads(source.read_text()))
    atomic_json(stage/'READY.json',{'manifest_sha256':file_hash(stage/'manifest.json')})
    os.kill(os.getpid(),signal.SIGKILL)
private_publication_copy(origin,out,assemble,lambda stage:None)
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'scripts'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', script, str(origin), str(out), str(V6_SOURCE / 'source_manifest.json')], env=environment, capture_output=True, text=True)
    assert result.returncode == -9
    assert tree_digest(origin) == before and not (origin / 'READY.json').exists()
    assert not out.exists()
    private = list(tmp_path.glob('.final.publication-*'))
    assert len(private) == 1 and private[0].stat().st_mode & 0o777 == 0o700
    assert (private[0] / 'artifact/READY.json').exists()
    # A retry starts from the original real bytes, without repairing metadata.
    private_publication_copy(origin, out, lambda stage: None, lambda stage: None)
    assert tree_digest(origin) == tree_digest(out) == before


def test_semantic_core_independently_pins_genuine_copied_grader_bytes(tmp_path):
    from hsrl_v5.release import validate_core_identity, GRADER_PATH
    manifest, rows = real_semantic_metadata()
    path = tmp_path / GRADER_PATH
    path.parent.mkdir(parents=True)
    shutil.copyfile(MAIN / 'vendor/agentdog_runtime_v5_c8d803f/runtime_evaluators.py', path)
    validate_core_identity(manifest, rows, tmp_path)
    path.write_bytes(path.read_bytes() + b'\n# corrupted copied artifact\n')
    with pytest.raises(ValueError, match='grader_bytes'):
        validate_core_identity(manifest, rows, tmp_path)


def test_loaded_real_row_selection_is_recomputed_independently():
    from hsrl_v5.release import validate_core_identity
    manifest, rows = real_semantic_metadata()
    changed = copy.deepcopy(rows)
    changed[0]['id'] += '-corrupted'
    with pytest.raises(ValueError, match='loaded_selection'):
        validate_core_identity(manifest, changed)


def test_resealed_reduced_runtime_map_still_fails_required_closure():
    from hsrl_v5.release import validate_core_identity, core_descriptor, core_file_map
    from hsrl_v5.io import digest
    manifest, rows = real_semantic_metadata()
    manifest['runtime_module_sha256'].pop('hsrl_v5/grader_pool.py')
    manifest['file_sha256'].pop('code/hsrl_v5/grader_pool.py')
    manifest['artifact_file_sha256_sha256'] = digest(core_file_map(manifest['file_sha256']))
    manifest['semantic_core_sha256'] = digest(core_descriptor(manifest))
    with pytest.raises(ValueError, match='coverage_incomplete'):
        validate_core_identity(manifest, rows)


def test_unchanged_certificate_cannot_authorize_resealed_semantic_metadata(tmp_path):
    from hsrl_v5.release import validate_certificates, core_descriptor, LIMITS, GRADER_PATH
    from hsrl_v5.io import digest, file_hash
    manifest, _ = real_semantic_metadata()
    grader = tmp_path / GRADER_PATH
    grader.parent.mkdir(parents=True)
    shutil.copyfile(MAIN / 'vendor/agentdog_runtime_v5_c8d803f/runtime_evaluators.py', grader)
    # Negative certificate-identity unit guard only. This metadata fixture has
    # one real row and cannot instantiate an admitted Dataset or publish.
    certificate = {'passed': True, 'records': 14000, 'selection_sha256': manifest['selection_sha256'],
        'source_manifest_sha256': manifest['source_manifest_sha256'], 'semantic_core_sha256': manifest['semantic_core_sha256'],
        'artifact_file_sha256_sha256': manifest['artifact_file_sha256_sha256'], 'limits': LIMITS, 'initial_configurations': 18}
    audit = tmp_path / 'audits/context.json'
    audit.parent.mkdir()
    audit.write_text(json.dumps(certificate))
    manifest['file_sha256']['audits/context.json'] = file_hash(audit)
    manifest['admission_report']['annotation'] = 'mutated_after_audit'
    manifest['semantic_core_sha256'] = digest(core_descriptor(manifest))
    with pytest.raises(ValueError, match='audit_identity'):
        validate_certificates(tmp_path, manifest)


@pytest.mark.parametrize('kind', ['metadata_file', 'episode_directory'])
def test_anchored_real_evidence_reader_never_follows_symlinks(tmp_path, kind):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path)
    validate_proof(record, proof, path, root)
    first = Path(proof['attempt_directory']) / 'first_episode'
    if kind == 'metadata_file':
        original = first / 'provenance.json'
        actual = first / 'retained_provenance.json'
    else:
        original, actual = first, first.with_name('retained_first_episode')
    original.rename(actual)
    original.symlink_to(actual)
    with pytest.raises((ValueError, OSError)):
        validate_proof(record, proof, path, root)


def test_clean_private_grader_never_rule_cannot_report_registered_goal(tmp_path):
    from hsrl_v5.release import validate_proof
    record, proof, path, root = copied_case(tmp_path, role='clean')
    validate_proof(record, proof, path, root)
    grade_path = Path(proof['attempt_directory']) / 'first_episode/private_grade_1.json'
    grade = json.loads(grade_path.read_text())
    grade['registered_call_goal'] = True
    grade_path.write_text(json.dumps(grade))
    with pytest.raises(ValueError, match='source_result'):
        validate_proof(record, proof, path, root)
