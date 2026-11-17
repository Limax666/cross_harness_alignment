"""Original six RSI algorithms with strict matched native-owner H0 admission."""
from __future__ import annotations

import json
import os
from pathlib import Path

from flashrsi_pi_owner_profile_v6 import PROFILE, CONTRACT, ProfileError, byte_hash, validate_case, snapshot_source_map

ROOT = Path(__file__).resolve().parents[1]


def validate_reference(initial):
    initial = Path(initial).resolve(strict=True)
    manifest = json.loads((initial / 'run_manifest.json').read_text())
    completed = json.loads((initial / 'completed.json').read_text())
    if (manifest.get('execution_protocol') != PROFILE or manifest.get('contract') != CONTRACT
            or manifest.get('model_port') != 18764 or completed.get('all128_measurement_valid') is not True
            or completed.get('execution_protocol') != PROFILE or completed.get('attempts') != 128
            or completed.get('completed_scored') != 128 or completed.get('failed_or_unscored') != 0):
        raise ProfileError('fresh_matched_native_owner_full128_H0_required')
    from flashrsi_harnessrisk_pi_native_owner_initial_v6 import source_pins, DATA
    if source_pins() != manifest['code_sha256']:
        raise ProfileError('H0_and_RSI_profile_code_changed')
    rows_file = initial.parent / 'case_manifest.json'
    if byte_hash(rows_file.read_bytes()) != manifest['case_manifest_sha256']:
        raise ProfileError('H0_source_manifest_changed')
    current_base = os.environ.get('HR_OUTPUT_ROOT')
    if current_base is None or byte_hash((Path(current_base) / 'case_manifest.json').read_bytes()) != manifest['case_manifest_sha256']:
        raise ProfileError('RSI_source_manifest_differs_from_matched_H0')
    snapshot_file = initial / 'initial_snapshot.json'
    snapshot_record = json.loads(snapshot_file.read_text())
    if (byte_hash(snapshot_file.read_bytes()) != manifest['initial_snapshot_sha256']
            or snapshot_record.get('id') != manifest['snapshot_id']
            or completed.get('snapshot_id') != manifest['snapshot_id']):
        raise ProfileError('H0_initial_Snapshot_identity_changed')
    rows = json.loads(rows_file.read_text())['cases']
    if len(rows) != 128 or len({r['case_id'] for r in rows}) != 128:
        raise ProfileError('all128_original_cases_required')
    for row in rows:
        raw = (DATA / (row['case_id'] + '.json')).read_bytes()
        if byte_hash(raw) != row['sha256']:
            raise ProfileError('original_case_bytes_changed')
        task = json.loads(raw)['task']
        owners = task.get('user_messages') or [task['user_message']]
        validate_case(initial / 'cases' / row['case_id'], row['case_id'], manifest['snapshot_id'], owners, row['sha256'],
                      native_source=snapshot_source_map(snapshot_record))
    return initial


def build_source():
    source = (ROOT / 'scripts/flashrsi_harnessrisk_algorithm.py').read_text()
    changes = {
        'from flashrsi_harnessrisk_benchmark import HarnessRiskBenchmark,ROOT':
            'from flashrsi_harnessrisk_pi_native_owner_benchmark_v6 import HarnessRiskBenchmark,ROOT',
        "initial=base/('miniagent_initial_full_v2' if harness=='miniagent' else f'{harness}_initial_full')":
            "initial=validate_reference(Path(os.environ['HR_INITIAL_REFERENCE']))",
        "{'algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,":
            "{'target_execution_protocol':PROFILE,'initial_reference_dir':str(initial),'algorithm':args.algorithm,'initial_snapshot_id':snapshot.id,",
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ProfileError('original_algorithm_transform_not_exact:' + old)
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    if (os.environ.get('HR_HARNESS') != 'pi' or os.environ.get('HARNESSRISK_MODEL_PORT') != '18764'
            or os.environ.get('HR_HOST_SCRIPT') != 'flashrsi_harnessrisk_host_pi_native_owner_v6.py'
            or os.environ.get('HR_STRICT_EVOLUTION') != '1'
            or os.environ.get('HR_EVOLUTION_CASE_ATTEMPTS', '1') != '1'
            or os.environ.get('HR_WORKER_MAX_TOKENS', '1024') != '1024'):
        raise ProfileError('Pi_native_owner_profile_and_strict_evolution_required')
    exec(compile(build_source(), __file__, 'exec'))
