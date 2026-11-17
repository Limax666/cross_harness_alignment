"""Continuation-state and immutable-lineage gates; no model/source execution."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

M = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')
A = M / 'outputs/rl/hsrl_hourly_monitor_20261117/audits/20261118_0600_19d7f87f-410c-4fa7-a8a3-40b018420843'


def module():
    assert importlib.util.find_spec('hsrl_source_recovery_v1'), 'verified continuation gates are missing'
    return importlib.import_module('hsrl_source_recovery_v1')


@pytest.fixture(scope='module')
def boundary():
    import torch
    failed = json.loads((A / 'state.json').read_text())
    saved = torch.load(Path(failed['checkpoint']) / 'resume.pt', map_location='cpu', weights_only=False)['state']
    receipt = {'all_inflight_accounted': True, 'source_cleanup_confirmed': True,
        'parent_attempts': failed['attempts'], 'parent_updates': failed['updates'],
        'quarantined_current_step_attempts': 120, 'parent_journal_sha256': '1' * 64}
    return failed, saved, receipt


def test_resume_preserves_spent_budget_and_accepted_policy_without_partial_protocol_mix(boundary):
    failed, saved, receipt = boundary
    result = module().make_continuation_state(failed, saved, receipt,
        manifest_sha256='a' * 64, checkpoint='/tmp/new-owned-run/checkpoints/initial')
    assert result['updates'] == result['effective_updates'] == result['step'] == result['policy_version'] == 4
    assert result['attempts'] == 632
    assert result['adapter_hash'] == saved['adapter_hash']
    assert result['episodes'] == [] and result['inflight'] is None
    assert result['assignment'] is None and 'failure' not in result
    assert result['covered_task_conditions'] == saved['covered_task_conditions']
    assert result['manifest_sha256'] == 'a' * 64
    assert failed['inflight'] is not None and len(failed['episodes']) == 112
    assert module().max_accepted_updates(result, 38400) == 299


@pytest.mark.parametrize('flag', ['all_inflight_accounted', 'source_cleanup_confirmed'])
def test_unknown_inflight_or_uncertain_cleanup_refuses_continuation(boundary, flag):
    failed, saved, receipt = copy.deepcopy(boundary)
    receipt[flag] = False
    with pytest.raises(ValueError, match='reconciliation'):
        module().make_continuation_state(failed, saved, receipt,
            manifest_sha256='a' * 64, checkpoint='/tmp/new-owned-run/checkpoints/initial')


def test_unbound_checkpoint_or_attempt_reset_is_rejected(boundary):
    failed, saved, receipt = copy.deepcopy(boundary)
    saved['adapter_hash'] = 'unrelated'
    with pytest.raises(ValueError):
        module().make_continuation_state(failed, saved, receipt,
            manifest_sha256='a' * 64, checkpoint='/tmp/new-owned-run/checkpoints/initial')
    failed, saved, receipt = copy.deepcopy(boundary)
    receipt['parent_attempts'] = 512
    with pytest.raises(ValueError):
        module().make_continuation_state(failed, saved, receipt,
            manifest_sha256='a' * 64, checkpoint='/tmp/new-owned-run/checkpoints/initial')


def test_lineage_pin_drift_is_rejected(tmp_path):
    mod = module()
    path = tmp_path / 'receipt.json'
    path.write_text('original')
    pins = {str(path): mod.file_hash(path)}
    mod.verify_pins(pins)
    path.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        mod.verify_pins(pins)


def test_real_nested_failure_and_every_aborted_slot_have_recognized_typed_cause():
    mod = module()
    failed = json.loads((A / 'state.json').read_text())
    assert mod.known_source_failure(failed['failure'])
    for attempt in failed['inflight']['attempts']:
        episode = json.loads((Path(attempt['directory']) / 'episode.json').read_text())
        assert mod.known_source_failure(episode['error'])
    assert not mod.known_source_failure({'type': 'RuntimeError', 'detail': "untyped name 'ValueError' is not defined"})
    assert not mod.known_source_failure({'ok': False, 'error_type': 'ValueError', 'error': 'different fault'})


def test_v9_keeps_the_entire_v8_collection_and_optimizer_body():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    assert (scripts / 'train_hsrl_v9.py').exists(), 'versioned formal continuation entry is missing'
    old = (scripts / 'train_hsrl_v8.py').read_text()
    new = (scripts / 'train_hsrl_v9.py').read_text()
    marker = '        try:\n            while state[\'step\'] < CONFIG[\'max_steps\']:'
    assert old[old.index(marker):old.index('\n\ndef argument_parser')].encode() == new[
        new.index(marker):new.index('\n\ndef argument_parser')].encode()
