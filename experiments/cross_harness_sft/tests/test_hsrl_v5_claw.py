"""Real pinned ClawGym data-path regressions. Never execute source payloads."""
from collections import defaultdict
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

EXP = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')
RAW = EXP / 'data/rl_sources/clawgym_task_hf_bd9043c/syn_task.jsonl'
BENCH = EXP / 'vendor/clawgym_agents_hsrl_v3_0e8526d/RL/data/clawgym_eval'
SELECTED = BENCH.parent / 'clawgym_train'


def api():
    try:
        return importlib.import_module('hsrl_v5.claw_source')
    except ModuleNotFoundError:
        pytest.fail('full pinned ClawGym preparation API is absent')


@pytest.fixture(scope='module')
def actual_rows():
    # Pin source rows by source line and official public query, independently
    # of production extraction/grouping logic.
    queries = {json.loads((SELECTED / task / 'data_entry.json').read_text())['user_query']: task
               for task in ('task_0102', 'task_0486', 'task_1895', 'task_0398')}
    result = {}
    for number, line in enumerate(RAW.open('rb'), 1):
        row = json.loads(line)
        if number in (1, 2, 4074) or row['prompt'] in queries:
            result[number] = (row, line)
            if row['prompt'] in queries:
                result[queries[row['prompt']]] = (number, row, line)
    return result


def materialized(module, tmp_path, actual_rows, key=1):
    if isinstance(key, int):
        row, line = actual_rows[key]
        number = key
    else:
        number, row, line = actual_rows[key]
    return module.materialize_claw_record(row, number, line, tmp_path)


def test_fixed_raw_drift_rejected_before_output_creation(tmp_path, actual_rows):
    module = api()
    changed = tmp_path / 'changed.jsonl'
    changed.write_bytes(actual_rows[1][1])
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='HF_raw_bytes_differ'):
        module.prepare_claw_source(changed, out, BENCH)
    assert not out.exists()


def test_preserves_real_source_identity_original_checker_and_private_metadata(tmp_path, actual_rows):
    module = api()
    record = materialized(module, tmp_path, actual_rows)
    row, line = actual_rows[1]
    assert record['id'] == 'arc_security_audit_358537a6060c'
    assert record['source_identity']['hf_line'] == 1
    assert record['source_identity']['hf_record_sha256'] == hashlib.sha256(line.rstrip(b'\r\n')).hexdigest()
    original = next(asset['content'] for asset in row['input_files'] if asset['file_path'] == 'reward/test.py').encode()
    private = Path(record['private_grader_path'])
    assert (private / 'reward/test.py').read_bytes() == original
    assert (private / 'reward/check.py').read_bytes() == original
    public = Path(record['public_assets_path'])
    assert (public / 'input/skills_audit_input.json').is_file()
    assert not (public / 'reward').exists()
    assert not (public / 'data_entry.json').exists()
    assert json.loads(Path(record['source_path']).read_text())['meta_data'] == row['meta_data']
    assert not record['runtime_ready']


def test_consumed_tree_drift_and_missing_real_input_are_rejected(tmp_path, actual_rows):
    module = api()
    record = materialized(module, tmp_path, actual_rows)
    module.verify_prepared_record(record)
    asset = Path(record['public_assets_path']) / 'input/skills_audit_input.json'
    before = asset.read_bytes()
    asset.unlink()
    with pytest.raises(ValueError, match='source_tree_hash_changed'):
        module.verify_prepared_record(record)
    asset.write_bytes(before)
    grader = Path(record['private_grader_path']) / 'reward/test.py'
    grader.write_text(grader.read_text() + '\n# untracked change\n')
    with pytest.raises(ValueError, match='source_tree_hash_changed'):
        module.verify_prepared_record(record)


@pytest.mark.parametrize('bad', ['../escape', '/absolute', 'reward/../escape', 'input\\escape', 'input//alias'])
def test_unsafe_asset_aliases_never_materialize(tmp_path, actual_rows, bad):
    module = api()
    row, line = actual_rows[1]
    changed = copy.deepcopy(row)
    changed['input_files'][0]['file_path'] = bad
    with pytest.raises(ValueError, match='unsafe_HF_asset_path'):
        module.materialize_claw_record(changed, 1, line, tmp_path)
    assert not list(tmp_path.rglob('test.py'))


def test_duplicate_asset_names_rejected_before_writing(tmp_path, actual_rows):
    module = api()
    row, line = actual_rows[1]
    changed = copy.deepcopy(row)
    changed['input_files'].append(copy.deepcopy(changed['input_files'][0]))
    with pytest.raises(ValueError, match='duplicate_HF_asset'):
        module.materialize_claw_record(changed, 1, line, tmp_path)
    assert not list(tmp_path.iterdir())


def test_skill_family_and_normalized_duplicate_grouping_have_no_crosssplit(tmp_path, actual_rows):
    module = api()
    records = [materialized(module, tmp_path, actual_rows, key) for key in (1, 2, 4074)]
    module.group_and_split(records)
    assert records[0]['family'] == records[2]['family']
    assert records[0]['family'] != records[1]['family']
    assert records[0]['split'] == records[2]['split']
    first = {r['id']: (r['family'], r['split']) for r in records}
    module.group_and_split(list(reversed(records)))
    assert first == {r['id']: (r['family'], r['split']) for r in records}
    assert module.normalized_prompt('  Hello\u00a0WORLD!\n') == module.normalized_prompt('hello world!')
    assert all(r['id'] != r['family'] for r in records)


def test_real_bench_public_prompts_excluded_without_reading_private_graders(tmp_path, actual_rows):
    module = api()
    proof = module.public_benchmark_hashes(BENCH)
    assert proof['count'] == 200
    entry = json.loads(next(BENCH.glob('*/data_entry.json')).read_text())
    assert hashlib.sha256(entry['user_query'].encode()).hexdigest() in proof['exact_prompt_sha256']
    assert all('/reward/' not in name for name in proof['consumed_public_paths'])
    # Guard-boundary regression using real public benchmark hashes; this
    # deliberately colliding metadata is a unit negative, not source evidence.
    records = [materialized(module, tmp_path, actual_rows)]
    records[0]['prompt_sha256'] = hashlib.sha256(entry['user_query'].encode()).hexdigest()
    module.apply_benchmark_guard(records, proof)
    assert 'official_bench200_public_prompt_collision' in records[0]['exclusion_reasons']
    assert not records[0]['qualified']


def test_real_capability_exclusion_keeps_evidence_and_legitimate_edits(tmp_path, actual_rows):
    module = api()
    network = materialized(module, tmp_path, actual_rows, 'task_0398')
    assert any('network' in reason for reason in network['exclusion_reasons'])
    assert network['capability_contract']['capability_evidence']
    assert network['capability_contract']['status'] == 'pending_or_unsupported'
    for key, name in (('task_0102', 'pipeline/config.yaml'), ('task_0486', 'input/indexing_config.yaml')):
        record = materialized(module, tmp_path, actual_rows, key)
        contract = record['source_contract']
        assert name in contract['writable_inputs']
        assert name not in contract['immutable_inputs']
        assert name in contract['initial_input_hashes']
        assert contract['immutable_gold_reference']['copy_from_original_source']
        assert record['source_contract']['upstream_contract_version'] == 'clawgym_source_capability_and_input_roles_v4_fix1'
    unresolved = materialized(module, tmp_path, actual_rows, 'task_1895')
    assert 'writable_input_gold_contract_pending_review' in unresolved['exclusion_reasons']


def test_existing_output_never_overwritten(tmp_path, actual_rows):
    module = api()
    record = materialized(module, tmp_path, actual_rows)
    before = Path(record['source_path']).read_bytes()
    with pytest.raises(FileExistsError):
        materialized(module, tmp_path, actual_rows)
    assert Path(record['source_path']).read_bytes() == before


def test_real_checker_binary_and_dynamic_commands_are_pending_not_accepted(tmp_path):
    module = api()
    selected = {}
    for number, line in enumerate(RAW.open('rb'), 1):
        if number in (9142, 77, 6160):
            selected[number] = (json.loads(line), line)
        if len(selected) == 3:
            break
    row, line = selected[9142]
    record = module.materialize_claw_record(row, 9142, line, tmp_path)
    contract = record['source_contract'].get('checker_execution_contract')
    assert contract is not None, 'subprocess executable requirements were omitted from admission'
    assert 'node' in contract['unavailable_or_unverified_programs']
    assert not record['qualified']
    assert 'checker_executable_or_script_contract_pending' in record['exclusion_reasons']
    assert contract['process_calls']
    assert contract['source_code_executed'] is False
    row, line = selected[77]
    dynamic = module.materialize_claw_record(row, 77, line, tmp_path)
    assert dynamic['source_contract']['checker_execution_contract']['dynamic_command_evidence']
    assert 'checker_executable_or_script_contract_pending' in dynamic['exclusion_reasons']
    row, line = selected[6160]
    python = module.materialize_claw_record(row, 6160, line, tmp_path)
    execution = python['source_contract']['checker_execution_contract']
    assert 'sys.executable' in {c['program_or_expression'] for c in execution['process_calls']}
    assert not execution['unavailable_or_unverified_programs']


def test_refinement_copies_real_source_and_freezes_identity_split_without_mutating_old(tmp_path, actual_rows):
    module = api()
    original = materialized(module, tmp_path / 'old', actual_rows)
    module.group_and_split([original])
    original_bytes = Path(original['source_path']).read_bytes()
    row, line = actual_rows[1]
    copied = module.copy_prepared_claw_record(original, row, line, tmp_path / 'new')
    assert Path(copied['source_path']).read_bytes() == original_bytes
    assert copied['id'] == original['id']
    assert (copied['family'], copied['split']) == (original['family'], original['split'])
    assert Path(copied['source_path']).parent != Path(original['source_path']).parent
    source_asset = Path(original['public_assets_path']) / 'input/skills_audit_input.json'
    copied_asset = Path(copied['public_assets_path']) / 'input/skills_audit_input.json'
    copied_asset.write_text('independent copied asset changed')
    assert source_asset.read_text() == row['input_files'][0]['content']
    with pytest.raises(ValueError, match='source_tree_hash_changed'):
        module.verify_prepared_record(copied)
    changed = copy.deepcopy(row)
    changed['prompt'] += ' altered goal'
    with pytest.raises(ValueError, match='HF_record_does_not_match_consumed_line'):
        module.copy_prepared_claw_record(original, changed, line, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
