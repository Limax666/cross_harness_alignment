"""Fresh full128 H0 using only the new native Claude owner profile; never old scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from flashrsi_claude_owner_profile_v9 import CONTRACT, PROFILE, ProfileError, byte_hash, save_json, validate_case, snapshot_source_map

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
DATA = ROOT.parents[1] / 'HarnessRisk/data/HarnessRisk'
HOST = 'flashrsi_harnessrisk_host_claude_native_owner_v9.py'


def source_pins():
    """Bind all selected adapters/algorithms, native SDK/CLI and reused helpers."""
    from flashrsi_claude_owner_profile_v9 import CLI
    sdk=ROOT/'.venv-rl-pilot/lib/python3.11/site-packages/claude_agent_sdk'
    paths=list((ROOT/'scripts').glob('flashrsi*claude*owner*v9.py'))
    paths += [ROOT/'scripts'/name for name in (
        'flashrsi_pi_owner_profile_v6.py','flashrsi_pi_owner_mock_barrier_v6.py',
        'flashrsi_pi_owner_mock_manager_v6.py','flashrsi_harnessrisk_sandbox.py',
        'flashrsi_harnessrisk_privacy.py','flashrsi_gateway.py','flashrsi_model_relay.py',
        'flashrsi_native_protocol_relay.py','flashrsi_landlock.py','flashrsi_native_seccomp.py',
        'flashrsi_native_sandbox_exec.py','flashrsi_harnessrisk_benchmark.py',
        'flashrsi_harnessrisk_algorithm.py','flashrsi_evo_skill.py')]
    paths += list((ROOT/'vendor/FlashRSI/flashrsi').rglob('*.py'))
    paths += list(sdk.rglob('*.py'))
    paths += [Path(CLI),Path(CLI).parents[1]/'package.json']
    paths += [PROJECT/'HarnessRisk/harness_adapter'/name for name in ('harness_adapter.py','scripts/evaluate_run.py')]
    paths += list((PROJECT/'HarnessRisk/services').rglob('*.py'))
    paths += [p for p in (PROJECT/'HarnessRisk/harness_adapter/templates').rglob('*') if p.is_file()]
    pins={str(p.relative_to(PROJECT)) if p.is_relative_to(PROJECT) else str(p):byte_hash(p.read_bytes())
          for p in sorted(set(paths))}
    expected={str(Path(CLI)):'dabafec82ca1319fa43d8be36a10fc61b1d11821cb98ad5df04ad95e199b8eed',
        str((sdk/'_version.py').relative_to(PROJECT)):'679044a399f18e84d0243e4669e51ddf96a701f3fcf4f725cf59f693b361b4cf',
        str((sdk/'_internal/transport/subprocess_cli.py').relative_to(PROJECT)):'c532d7a6421854cc09f51b64750a998b4542ab08e3195a1fee88c5e669d4318e',
        str((sdk/'_internal/query.py').relative_to(PROJECT)):'e7d9d31d070d55e1f6d94a6b4331ef22ccd8742282e0823e1564aa1c382c8f4c',
        str((sdk/'types.py').relative_to(PROJECT)):'47ae0b7f91dd95d099a0cedacc51bc61b0a320ebe1dd2ab3d4543691b2afdfa6',
        str((ROOT/'vendor/FlashRSI/flashrsi/agents/claude_code/__init__.py').relative_to(PROJECT)):'95ef35bf3fe69931cef4bc594ef0f41f348a0f80b100ac85687e826d822d6ae6'}
    if any(pins.get(path)!=expected_hash for path,expected_hash in expected.items()):
        raise ProfileError('pinned_native_SDK_CLI_or_original_adapter_changed')
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('HR_HARNESS') != 'claude_code' or os.environ.get('HARNESSRISK_MODEL_PORT') != '18764':
        raise ProfileError('explicit_Claude_GPU1_endpoint_required')
    from flashrsi.core import Snapshot
    base = args.base.resolve(strict=True)
    manifest_file = base / 'case_manifest.json'
    manifest = json.loads(manifest_file.read_text())
    if manifest.get('model_revision') != '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a':
        raise ProfileError('frozen_official_4B_revision_required')
    rows = manifest['cases']
    if len(rows) != 128 or len({r['case_id'] for r in rows}) != 128:
        raise ProfileError('full128_original_manifest_required')
    for row in rows:
        if byte_hash((DATA / (row['case_id'] + '.json')).read_bytes()) != row['sha256']:
            raise ProfileError('source_case_bytes_changed')
    snapshot = Snapshot.load(args.snapshot)
    if snapshot.agent_type != 'claude_code':
        raise ProfileError('original_Claude_Snapshot_required')
    run = base / 'claude_code_initial_full'
    run.mkdir(mode=0o700, exist_ok=False)
    (run / 'cases').mkdir()
    snapshot.save(run / 'initial_snapshot.json')
    pins = source_pins()
    save_json(run / 'run_manifest.json', {'status': 'running', 'execution_protocol': PROFILE,
        'scope': 'fresh full128 native owner-schedule measurements, not assumed task successes', 'case_count': 128,
        'privacy_protocol': 'harnessrisk-private-case-v1', 'harness': 'claude_code', 'host_script': HOST,
        'max_turns_per_owner_turn': 16, 'native_timeout_seconds_per_owner_turn': 600,
        'sandbox_wall_seconds_per_owner_turn': 900, 'max_tokens_per_turn': 4096, 'temperature': 0,
        'flashrsi_revision': manifest.get('flashrsi_revision'),
        'contract': CONTRACT, 'snapshot_id': snapshot.id, 'model_port': 18764,
        'initial_snapshot_sha256': byte_hash((run / 'initial_snapshot.json').read_bytes()),
        'model_revision': manifest['model_revision'], 'case_manifest_sha256': byte_hash(manifest_file.read_bytes()),
        'code_sha256': pins, 'source_status_ruling': CONTRACT['source_status_ruling'],
        'comparison': 'fresh matched H0 and RSI only; old strict case scores never mixed'})
    valid, attempts = 0, []
    with (run / 'attempts.jsonl').open('x') as ledger:
        for row in rows:
            if source_pins() != pins:
                raise ProfileError('frozen_profile_code_changed_during_full128')
            case_id = row['case_id']
            folder = run / 'host_observability' / case_id
            folder.mkdir(parents=True)
            command = [sys.executable, str(ROOT / 'scripts' / HOST), case_id,
                       str(run / 'initial_snapshot.json'), snapshot.id, str(run / 'cases')]
            started = time.monotonic()
            proof = {'profile': PROFILE, 'command': command, 'timeout_seconds': 2460}
            try:
                with (folder / 'host.stdout.log').open('wb') as out, (folder / 'host.stderr.log').open('wb') as err:
                    proc = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, timeout=2460)
                proof['returncode'] = proc.returncode
                host_receipt = run / 'cases' / case_id / 'native_owner_host.json'
                host = json.loads(host_receipt.read_text()) if host_receipt.is_file() else {}
                if host.get('namespace_exited') is not True or host.get('model_requests_settled') is not True:
                    proof.update(cleanup_confirmed=False, measurement_valid=False)
                    raise ProfileError('case_namespace_cleanup_uncertain_no_next_case')
                if proc.returncode:
                    raise ProfileError('actual_native_case_invalid')
                source_task = json.loads((DATA / (case_id + '.json')).read_text())['task']
                owners = source_task.get('user_messages') or [source_task['user_message']]
                receipt = validate_case(run / 'cases' / case_id, case_id, snapshot.id, owners, row['sha256'],
                                        native_source=snapshot_source_map(snapshot))
                proof.update(measurement_valid=True, bounded_owners=receipt['bounded_owners'])
                valid += 1
            except subprocess.TimeoutExpired as exc:
                proof.update(timed_out=True, cleanup_confirmed=False, measurement_valid=False)
                raise ProfileError('host_cleanup_uncertain_no_next_case') from exc
            except ProfileError as exc:
                proof.update(measurement_valid=False, error=str(exc))
                if proof.get('cleanup_confirmed') is False:
                    raise
            finally:
                proof.update(case_id=case_id, elapsed_seconds=time.monotonic() - started)
                save_json(folder / 'host_execution.json', proof)
                ledger.write(json.dumps(proof) + '\n')
                ledger.flush()
                os.fsync(ledger.fileno())
                attempts.append(proof)
    save_json(run / 'completed.json', {'status': 'all_cases_attempted', 'execution_protocol': PROFILE,
        'attempts': len(attempts), 'completed_scored': valid, 'failed_or_unscored': len(attempts) - valid,
        'snapshot_id': snapshot.id, 'all128_measurement_valid': valid == 128})
    return 0 if valid == 128 else 2


if __name__ == '__main__':
    raise SystemExit(main())
