"""CPU/filesystem resource boundary checks; never launch CUDA or a model."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_hsrl_v5 as trainer


@pytest.fixture
def handoff(tmp_path):
    ledger = tmp_path / 'ledger.json'
    ledger.write_text('{"tasks":"preserved"}')
    lease = dict(schema='hsrl-gpu1-training-handoff/v1', physical_gpu=1,
                 exclusive=True, forbidden_gpus=[2, 3],
                 boundary_status='stages_ended_preserved_unresolved_work',
                 baseline_status='paused_with_preserved_results',
                 old_service_pids=[999999999], old_controller_pids=[],
                 baseline_ledger_path=str(ledger), baseline_ledger_sha256=trainer.file_hash(ledger),
                 lock_path=str(tmp_path / 'lease.lock'))
    path = tmp_path / 'lease.json'
    def save(**changes):
        path.write_text(json.dumps({**lease, **changes}))
        return path
    return lease, save


def test_gpu1_and_legacy_gpu0_lease(handoff):
    lease, save = handoff
    assert trainer.validate_lease(save(), 1) == lease
    legacy = {**lease, 'schema': 'hsrl-gpu0-training-handoff/v1', 'physical_gpu': 0,
              'gpu1_baselines_preserved': True}
    assert trainer.validate_lease(save(**legacy)) == legacy
    with pytest.raises(ValueError, match='selected_GPU'):
        trainer.validate_lease(save(), 0)


@pytest.mark.parametrize('changes', [
    {'physical_gpu': 0}, {'exclusive': False}, {'forbidden_gpus': []},
    {'schema': 'hsrl-gpu0-training-handoff/v1'}, {'baseline_status': 'running'},
    {'gpu1_baselines_preserved': True}, {'old_service_pids': []},
    {'old_service_pids': ['999999999']}, {'old_controller_pids': None},
    {'old_controller_pids': [os.getpid()]}, {'old_service_pids': [os.getpid()]},
    {'baseline_ledger_sha256': 'wrong'}, {'lock_path': 'relative'},
])
def test_unsafe_handoff_rejected(handoff, changes):
    _, save = handoff
    with pytest.raises(ValueError):
        trainer.validate_lease(save(**changes), 1)


def test_changed_ledger_and_malformed_lease(handoff):
    lease, save = handoff
    path = save()
    Path(lease['baseline_ledger_path']).write_text('changed task ledger')
    with pytest.raises(ValueError, match='ledger_identity'):
        trainer.validate_lease(path, 1)
    path.write_text('[]')
    with pytest.raises(ValueError, match='malformed'):
        trainer.validate_lease(path, 1)


@pytest.mark.parametrize('gpu', [2, 3, -1, True, '1'])
def test_forbidden_gpu_before_any_work(gpu, tmp_path):
    with pytest.raises(ValueError, match='must_be_0_or_1'):
        trainer.run(SimpleNamespace(physical_gpu=gpu))
    with pytest.raises(ValueError, match='must_be_0_or_1'):
        trainer.resource_gate(tmp_path, initial=True, physical_gpu=gpu)


@pytest.mark.parametrize('gpu,visible', [(0, '1'), (1, '0'), (1, '1,0'), (1, ''), (1, '01')])
def test_environment_mismatch_before_import_or_filesystem(gpu, visible, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(ValueError, match=f'exactly_physical_{gpu}'):
        trainer.run(SimpleNamespace(physical_gpu=gpu))


@pytest.mark.parametrize('gpu', [0, 1])
def test_initial_gate_queries_only_selected_device(tmp_path, monkeypatch, gpu):
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **k:
                        'MemAvailable: 67108864 kB\n' if str(p) == '/proc/meminfo'
                        else original_read(p, *a, **k))
    monkeypatch.setattr(trainer.shutil, 'disk_usage', lambda p: SimpleNamespace(free=200 * 2**30))
    query = Mock(return_value=SimpleNamespace(stdout=''))
    monkeypatch.setattr(trainer.subprocess, 'run', query)
    assert trainer.resource_gate(tmp_path, initial=True, physical_gpu=gpu)['run_bytes'] == 0
    assert query.call_args.args[0] == ['nvidia-smi', f'--id={gpu}',
                                      '--query-compute-apps=pid', '--format=csv,noheader,nounits']
    query.return_value.stdout = '12345\n'
    with pytest.raises(RuntimeError, match=f'GPU{gpu}_has_compute_processes'):
        trainer.resource_gate(tmp_path, initial=True, physical_gpu=gpu)


@pytest.mark.parametrize('selection', [[], ['--physical-gpu', '1']])
def test_real_cli_selection_reaches_environment_guard(selection, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['train_hsrl_v5.py', '--dataset', '/not/read',
                                    '--model', '/not/read', '--out', '/not/created',
                                    '--lease', '/not/read', *selection])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'invalid')
    expected = 1 if selection else 0
    with pytest.raises(ValueError, match=f'exactly_physical_{expected}'):
        trainer.main()


@pytest.mark.parametrize('selection', ['2', '3'])
def test_cli_forbidden_gpu_rejected_by_parser(selection, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train_hsrl_v5.py', '--dataset', '/not/read',
                                    '--model', '/not/read', '--out', '/not/created',
                                    '--lease', '/not/read', '--physical-gpu', selection])
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2
