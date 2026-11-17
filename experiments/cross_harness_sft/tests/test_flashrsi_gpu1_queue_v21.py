"""Pure future v21 integration fixtures; no native, task, model or process runs."""
import json
import os
from pathlib import Path

import pytest

import flashrsi_matrix_gpu1_queue_v21 as queue
from flashrsi.core.snapshot import Snapshot


def resource(tmp_path, monkeypatch, harness):
    tree = tmp_path / (harness + '-tree')
    (tree / 'runtime/pi_native').mkdir(parents=True)
    (tree / 'runtime/pi_native/loop.py').write_text('# inert controller fixture\n')
    snapshot = Snapshot.from_directory(tree, agent_type=harness, component_rules={'runtime': ('runtime/**',)})
    path = tmp_path / (harness + '-resource.json'); snapshot.save(path)
    monkeypatch.setitem(queue.NATIVE_PROFILES, harness, {**queue.NATIVE_PROFILES[harness],
        'snapshot': path, 'snapshot_sha256': queue.sha(path), 'snapshot_id': snapshot.id})
    return path, snapshot


def native_h0(tmp_path, monkeypatch, harness):
    _, snapshot = resource(tmp_path, monkeypatch, harness)
    base = tmp_path / harness
    run = base / (harness + '_initial_full'); run.mkdir(parents=True)
    data = tmp_path / 'original-cases'; data.mkdir(exist_ok=True)
    rows = []
    for index in range(128):
        cid = f'case{index}'
        path = data / (cid + '.json'); path.write_text(json.dumps({'task': {'user_messages': ['original owner']}}))
        rows.append({'case_id': cid, 'sha256': queue.sha(path)})
        queue.save(run / 'cases' / cid / 'harness_result.json', {})
    manifest = {'model_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                'flashrsi_revision': 'unit-runtime', 'cases': rows}
    queue.save(base / 'case_manifest.json', manifest)
    snapshot.save(run / 'initial_snapshot.json')
    contract = queue.expected_initial_contract((base / 'case_manifest.json').read_bytes(), harness)
    queue.save(run / 'run_manifest.json', {**contract, 'snapshot_id': snapshot.id,
        'initial_snapshot_sha256': queue.sha(run / 'initial_snapshot.json'), 'code_sha256': {'unit': 'pin'}})
    queue.save(run / 'completed.json', {'status': 'all_cases_attempted', 'execution_protocol': contract['execution_protocol'],
        'snapshot_id': snapshot.id, 'all128_measurement_valid': True, 'attempts': 128,
        'completed_scored': 128, 'failed_or_unscored': 0})
    attempts = [{'case_id': row['case_id'], 'returncode': 0, 'measurement_valid': True} for row in rows]
    (run / 'attempts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in attempts))
    monkeypatch.setattr(queue, 'DATA', data)
    return run, {row['case_id'] for row in rows}, contract, data


def test_v21_registry_and_commands_use_actual_interfaces():
    assert tuple(queue.NATIVE_PROFILES) == ('pi', 'claude_code')
    assert queue.model_harnesses() == ('opencode', 'openclaw', 'codex', 'pi', 'claude_code')
    assert queue.deferred_profiles() == {}
    profile = queue.NATIVE_PROFILES['claude_code']
    assert profile['profile'] == queue.CLAUDE_PROFILE
    assert profile['contract'] == queue.CLAUDE_CONTRACT
    assert queue.HOSTS['claude_code'] == 'flashrsi_harnessrisk_host_claude_native_owner_v9.py'
    assert queue.initial_command('claude_code', Path('/fresh')) == [
        'scripts/flashrsi_harnessrisk_claude_native_owner_initial_v9.py', '--base', '/fresh', '--snapshot', str(profile['snapshot'])]
    assert all(queue.algorithm_entry('claude_code', a) == 'flashrsi_harnessrisk_claude_native_owner_reference_v9.py'
               for a in queue.ALGORITHMS)
    raw = b'{"model_revision":"model","flashrsi_revision":"runtime"}'
    contract = queue.expected_initial_contract(raw, 'claude_code')
    assert contract['max_turns_per_owner_turn'] == 16
    assert contract['native_timeout_seconds_per_owner_turn'] == 600
    assert contract['sandbox_wall_seconds_per_owner_turn'] == 900
    assert contract['max_tokens_per_turn'] == 4096 and contract['model_port'] == 18764
    assert contract['contract'] == queue.CLAUDE_CONTRACT
    assert contract['execution_protocol'] == queue.CLAUDE_PROFILE


def test_composed_metadata_and_native_reference_schema():
    source = queue.build_source(); compile(source, '<future-v21>', 'exec')
    assert 'gpu1-finite-strict-matrix/v21' in source
    assert 'scripts/flashrsi_matrix_gpu1_queue_v21.py' in source
    assert '.flashrsi_gpu1_finite_queue_v21.lock' in source
    assert "'target_contract'" in source
    from flashrsi_harnessrisk_claude_native_owner_reference_v9 import build_source
    reference = build_source(); compile(reference, '<Claude-reference-schema>', 'exec')
    assert "'target_execution_protocol':PROFILE,'target_contract':CONTRACT,'initial_reference_dir':str(initial)" in reference
    assert "initial=validate_reference(Path(os.environ['HR_INITIAL_REFERENCE']))" in reference
    initial = (queue.ROOT / 'scripts/flashrsi_harnessrisk_claude_native_owner_initial_v9.py').read_text()
    for field in ("'max_turns_per_owner_turn': 16", "'native_timeout_seconds_per_owner_turn': 600",
                  "'sandbox_wall_seconds_per_owner_turn': 900", "proof['returncode'] = proc.returncode",
                  "measurement_valid=True", "'status': 'all_cases_attempted'", "'all128_measurement_valid': valid == 128"):
        assert field in initial


@pytest.mark.parametrize('harness', ['pi', 'claude_code'])
def test_actual_registry_reference_walks_all128_and_restores_environment(tmp_path, monkeypatch, harness):
    run, expected, contract, data = native_h0(tmp_path, monkeypatch, harness)
    if harness == 'pi':
        import flashrsi_harnessrisk_pi_native_owner_initial_v6 as initial
        import flashrsi_harnessrisk_pi_native_owner_reference_v6 as reference
    else:
        import flashrsi_harnessrisk_claude_native_owner_initial_v9 as initial
        import flashrsi_harnessrisk_claude_native_owner_reference_v9 as reference
    monkeypatch.setattr(initial, 'DATA', data)
    monkeypatch.setattr(initial, 'source_pins', lambda: {'unit': 'pin'})
    calls = []
    monkeypatch.setattr(reference, 'validate_case', lambda path, cid, sid, owners, source, **kw:
                        calls.append((cid, sid, owners, source, kw)))
    monkeypatch.setenv('HR_OUTPUT_ROOT', '/unrelated/prior')
    assert queue.initial_acceptance(run, expected, contract)
    assert len(calls) == 128 and {row[0] for row in calls} == expected
    assert all(row[2] == ['original owner'] and row[4]['native_source'] for row in calls)
    assert os.environ['HR_OUTPUT_ROOT'] == '/unrelated/prior'
    monkeypatch.setattr(initial, 'source_pins', lambda: {'unit': 'drift'})
    assert not queue.initial_acceptance(run, expected, contract)
    assert os.environ['HR_OUTPUT_ROOT'] == '/unrelated/prior'


@pytest.mark.parametrize('mutation', ['host', 'contract', 'model', 'turns', 'port', 'attempt', 'duplicate', 'completed', 'resource'])
def test_claude_native_admission_rejects_drift(tmp_path, monkeypatch, mutation):
    run, expected, contract, _ = native_h0(tmp_path, monkeypatch, 'claude_code')
    monkeypatch.setattr(queue, 'validate_claude_reference', lambda initial: initial)
    if mutation in ('host', 'contract', 'model', 'turns', 'port'):
        row = queue.read(run / 'run_manifest.json')
        row[{'host': 'host_script', 'contract': 'contract', 'model': 'model_revision',
             'turns': 'max_turns_per_owner_turn', 'port': 'model_port'}[mutation]] = 'wrong'
        queue.save(run / 'run_manifest.json', row)
    elif mutation in ('attempt', 'duplicate'):
        rows = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines()]
        if mutation == 'attempt': rows[-1]['measurement_valid'] = False
        else: rows[-1] = rows[0]
        (run / 'attempts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif mutation == 'completed': queue.save(run / 'completed.json', {'completed_scored': 128})
    else: queue.NATIVE_PROFILES['claude_code']['snapshot'].write_text('{}')
    assert not queue.initial_acceptance(run, expected, contract)


def plan(tmp_path, monkeypatch):
    for harness in queue.NATIVE_PROFILES: resource(tmp_path, monkeypatch, harness)
    data = tmp_path / 'cases'; data.mkdir()
    manifest = {'model_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'flashrsi_revision': 'unit', 'cases': []}
    for i in range(128):
        path = data / f'case{i}.json'; path.write_text('{}')
        manifest['cases'].append({'case_id': f'case{i}', 'sha256': queue.sha(path)})
    ref = tmp_path / 'reference'; ref.mkdir(); (ref / 'initial_snapshot.json').write_text('{}')
    monkeypatch.setattr(queue, 'DATA', data)
    monkeypatch.setattr(queue, 'REFERENCE', ref)
    monkeypatch.setattr(queue, 'LANE_LOCK', tmp_path / 'lane.lock')
    monkeypatch.setattr(queue, 'lane_clients', lambda **kw: [])
    monkeypatch.setattr(queue, 'freeze_sources', lambda destination: {})
    monkeypatch.setattr(queue, 'native_source_pins', lambda h: {'unit': h})
    monkeypatch.setattr(queue, 'initial_acceptance', lambda path, expected, contract:
                        Path(path) == ref or contract['harness'] in queue.NATIVE_PROFILES)
    monkeypatch.setattr(queue, 'judge_acceptance', lambda *args: True)
    monkeypatch.setattr(queue, 'inspect_run', lambda *args: {'status': 'needs_repair'})
    monkeypatch.setattr(queue.importlib.metadata, 'version', lambda name: {'tree-sitter': '0.25.2', 'tree-sitter-bash': '0.25.1'}[name])
    controller = queue.Queue(tmp_path / 'queue'); calls = []
    def execute(key, command, env, *args):
        calls.append((key, command, dict(env)))
        if key in ('pi/initial', 'claude_code/initial'):
            run = Path(env['HR_INITIAL_REFERENCE']); run.mkdir()
            (run / 'initial_snapshot.json').write_bytes(queue.NATIVE_PROFILES[env['HR_HARNESS']]['snapshot'].read_bytes())
            return 0
        return 2
    monkeypatch.setattr(controller, 'execute', execute)
    monkeypatch.setattr(controller, 'schedule_judge', lambda: None)
    return controller, manifest, calls


def test_full_finite_dispatch_retains_five_harnesses_and_all_native_failures(tmp_path, monkeypatch):
    controller, manifest, calls = plan(tmp_path, monkeypatch)
    controller.run(manifest, [])
    assert [key for key, _, _ in calls] == ['opencode/evoprompt', 'opencode/gepa', 'openclaw/initial', 'codex/initial',
        *['pi/' + a for a in ('initial', *queue.ALGORITHMS)],
        *['claude_code/' + a for a in ('initial', *queue.ALGORITHMS)]]
    assert all(env['HARNESSRISK_MODEL_PORT'] == '18764' and env['CUDA_VISIBLE_DEVICES'] == '' for _, _, env in calls)
    assert all(controller.tasks['claude_code/' + a]['status'] == 'requires_repair' for a in queue.ALGORITHMS)
    assert controller.tasks['claude_code/initial']['judge_status'] == 'needs_judge_lease_or_billing'
    protocol = queue.read(controller.out / 'repair_protocol.json')
    assert protocol['protocol'] == 'gpu1-finite-strict-matrix/v21'
    assert protocol['queue_launcher'] == 'scripts/flashrsi_matrix_gpu1_queue_v21.py'
    assert protocol['coverage']['model_harnesses_this_queue'] == list(queue.model_harnesses())
    assert protocol['coverage']['deferred_harnesses'] == []
    assert protocol['coverage']['full_five_harness_objective_complete'] is False
    assert set(protocol['native_profiles']) == {'pi', 'claude_code'}


def test_failed_claude_h0_blocks_its_six_without_dropping_tasks(tmp_path, monkeypatch):
    controller, manifest, calls = plan(tmp_path, monkeypatch)
    gate = queue.initial_acceptance
    monkeypatch.setattr(queue, 'initial_acceptance', lambda path, expected, contract:
                        contract['harness'] != 'claude_code' and gate(path, expected, contract))
    controller.run(manifest, [])
    assert calls[-1][0] == 'claude_code/initial'
    assert controller.tasks['claude_code/initial']['status'] == 'blocked_initial_requires_repair'
    assert all(controller.tasks['claude_code/' + a]['status'] == 'blocked_same_protocol_initial' for a in queue.ALGORITHMS)


def test_claude_timeout_is_sticky_for_entire_lane(tmp_path, monkeypatch):
    controller, manifest, calls = plan(tmp_path, monkeypatch)
    execute = controller.execute
    def uncertain(key, command, env, *args):
        code = execute(key, command, env, *args)
        if key == 'claude_code/initial':
            queue.save(Path(env['HR_INITIAL_REFERENCE']) / 'host_observability/case0/host_execution.json',
                       {'timed_out': True, 'cleanup_confirmed': False})
        return code
    monkeypatch.setattr(controller, 'execute', uncertain)
    with pytest.raises(queue.QueueCleanupUncertain): controller.run(manifest, [])
    assert calls[-1][0] == 'claude_code/initial'
    with pytest.raises(queue.QueueCleanupUncertain): controller.check_cleanup()


def test_claude_selected_proof_requires_actual_target_contract(tmp_path, monkeypatch):
    initial, expected, _, _ = native_h0(tmp_path, monkeypatch, 'claude_code')
    run = initial.parent / 'claude_code_opro_full'; run.mkdir()
    snapshot = Snapshot.load(initial / 'initial_snapshot.json')
    (run / 'snapshots').mkdir(); snapshot.save(run / 'snapshots/selected.json')
    for cid in expected: queue.save(run / 'selected_cases' / cid / 'harness_result.json', {})
    profile = queue.NATIVE_PROFILES['claude_code']
    manifest = {'target_execution_protocol': profile['profile'], 'target_contract': profile['contract'],
        'host_script': profile['host'], 'initial_reference_dir': str(initial), 'initial_snapshot_id': snapshot.id,
        'model_port': 18764, 'worker_max_tokens': 1024, 'evolution_case_attempts': 1,
        'scoring_contract': 'reject_unscorable_during_evolution'}
    queue.save(run / 'run_manifest.json', manifest)
    queue.save(run / 'result.json', {'initial_snapshot_id': snapshot.id})
    monkeypatch.setattr(queue, 'legacy_inspect_run', lambda *args: {'status': 'needs_judge', 'reasons': []})
    calls = []
    monkeypatch.setattr(queue, 'validate_claude_case', lambda path, cid, *args, **kw: calls.append(cid))
    queue.save(run / 'selected_cases/llm_judge_multi_harness/manifest.json', [])
    assert queue.inspect_run(run, expected, None)['status'] == 'needs_judge'
    assert len(calls) == 128 and set(calls) == expected
    queue.save(run / 'run_manifest.json', {**manifest, 'target_contract': {'wrong': True}})
    assert queue.inspect_run(run, expected, None)['status'] == 'needs_repair'


def test_before_dispatch_native_bindings_cover_both_profiles(tmp_path, monkeypatch):
    controller, manifest, _ = plan(tmp_path, monkeypatch)
    controller.run(manifest, [])
    controller.verify_native_bindings()
    queue.NATIVE_PROFILES['claude_code']['snapshot'].write_text('resource drift')
    with pytest.raises(ValueError, match='resource_hash_changed'): controller.verify_native_bindings()


def test_help_and_existing_output_never_launch_queue(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(queue.sys, 'argv', ['v21', '--help'])
    with pytest.raises(SystemExit) as exc: queue.main()
    assert exc.value.code == 0 and '--predecessor-pid' in capsys.readouterr().out
    out = tmp_path / 'existing'; out.mkdir(); (out / 'evidence').write_text('retain')
    monkeypatch.setattr(queue.sys, 'argv', ['v21', '--out', str(out)])
    with pytest.raises(RuntimeError, match='output_exists_preserved'): queue.main()
    assert (out / 'evidence').read_text() == 'retain'


def test_native_feedback_preserves_only_actual_invalid_rows(tmp_path):
    run = tmp_path / 'native-run'; run.mkdir()
    attempts = [
        {'case_id': 'bounded_valid', 'returncode': 0, 'measurement_valid': True},
        {'case_id': 'invalid', 'returncode': 2, 'measurement_valid': False},
        {'case_id': 'legacy_valid', 'exit_code': 0},
    ]
    (run / 'attempts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in attempts))
    queue.save(run / 'cases/bounded_valid/harness_result.json',
               {'status': 'completed', 'turns': [{'stop_reason': 'max_turns'}]})
    queue.save(run / 'cases/invalid/harness_result.json',
               {'status': 'error', 'turns': [{'stop_reason': 'context length invalid'}]})
    queue.save(run / 'cases/legacy_valid/harness_result.json', {'status': 'completed'})
    feedback = queue.failure_feedback(run)
    assert feedback['classification_counts'] == {'context_capacity': 1}
    assert feedback['evidence_paths'] == [str(run / 'host_observability/invalid')]
    assert feedback['repair_needed'] is True and feedback['unchanged_retry_allowed'] is False


def test_imported_v21_guard_never_calls_process_or_native_entry(tmp_path, monkeypatch):
    import importlib.util
    import sys
    before = {name for name in sys.modules if name.startswith('flashrsi.agents.')}
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('import must not dispatch'))
    spec = importlib.util.spec_from_file_location('future_v21_import_probe', queue.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.model_harnesses() == queue.model_harnesses()
    assert {name for name in sys.modules if name.startswith('flashrsi.agents.')} == before


def test_native_resource_copies_and_source_pins_remain_checked_after_run(tmp_path, monkeypatch):
    controller, manifest, _ = plan(tmp_path, monkeypatch)
    controller.run(manifest, [])
    controller.verify_native_bindings()
    copy = controller.out / 'native_resources/claude_code_initial_snapshot.json'
    copy.write_text('retained copy changed')
    with pytest.raises(RuntimeError, match='profile_or_original_H0_resource_drift'):
        controller.verify_native_bindings()


def test_predecessor_blocks_all_five_harnesses_before_freeze(tmp_path, monkeypatch):
    controller, manifest, calls = plan(tmp_path, monkeypatch)
    predecessor = {'pid': 999, 'start_ticks': '1234', 'command': ['exact-owned-predecessor']}
    monkeypatch.setattr(queue, 'process_identity', lambda pid: predecessor if pid == 999 else None)
    monkeypatch.setattr(queue, 'freeze_sources', lambda *args: pytest.fail('must wait before freezing'))
    def stop(seconds): raise InterruptedError('fixture stops during exact wait')
    monkeypatch.setattr(queue.time, 'sleep', stop)
    with pytest.raises(InterruptedError): controller.run(manifest, [predecessor])
    assert calls == [] and not (controller.out / 'native_resources').exists()


def test_scanner_registers_v20_v21_and_native_pid_scopes(tmp_path, monkeypatch):
    import flashrsi_matrix_repair_scan_v21 as scanner
    compile(scanner.build_source(), '<scanner-v21>', 'exec')
    monkeypatch.setattr(scanner, 'prior_active_runs', lambda: {'/legacy': 7})
    identities = {
        101: {'pid': 101, 'command': ['python', 'scripts/flashrsi_harnessrisk_pi_native_owner_reference_v6.py', '--algorithm', 'opro'], 'run_scope': '/queue/pi/pi_opro_full'},
        102: {'pid': 102, 'command': ['python', 'scripts/flashrsi_harnessrisk_claude_native_owner_reference_v9.py', '--algorithm', 'gepa'], 'run_scope': '/queue/claude_code/claude_code_gepa_full'},
        103: {'pid': 103, 'command': ['python', 'scripts/flashrsi_harnessrisk_claude_native_owner_reference_v9.py', '--algorithm', 'opro']},
    }
    class Proc:
        def glob(self, pattern): return [Path(str(pid)) for pid in identities]
    real_path = Path
    monkeypatch.setattr(scanner, 'Path', lambda value: Proc() if value == '/proc' else real_path(value))
    monkeypatch.setattr(scanner, 'process_identity', lambda pid: identities[pid])
    assert scanner.active_runs() == {'/legacy': 7, '/queue/pi/pi_opro_full': 101,
                                    '/queue/claude_code/claude_code_gepa_full': 102}


def test_scanner_uses_strict_native_proof_with_legacy_fallback(tmp_path, monkeypatch):
    import flashrsi_matrix_repair_scan_v21 as scanner
    calls = []
    monkeypatch.setattr(scanner, 'strict_inspect_run', lambda run, expected, pid: calls.append(('native', run)) or {'status': 'needs_repair'})
    monkeypatch.setattr(scanner.matrix, 'inspect_run', lambda run, expected, pid: calls.append(('legacy', run)) or {'status': 'complete'})
    native = tmp_path / 'native/pi_opro_full'; native.mkdir(parents=True)
    queue.save(native / 'run_manifest.json', {'target_execution_protocol': queue.PI_PROFILE})
    legacy = tmp_path / 'old/pi_opro_full'; legacy.mkdir(parents=True)
    queue.save(legacy / 'run_manifest.json', {'host_script': 'flashrsi_harnessrisk_host_pi_interpreter_v4.py'})
    assert scanner.inspect_scanned_run(native, {'case'}, None)['status'] == 'needs_repair'
    assert scanner.inspect_scanned_run(legacy, {'case'}, None)['status'] == 'complete'
    queued_native = tmp_path / 'queue/claude_code/claude_code_opro_full'; queued_native.mkdir(parents=True)
    queue.save(tmp_path / 'queue/repair_protocol.json', {'protocol': 'gpu1-finite-strict-matrix/v21'})
    assert scanner.inspect_scanned_run(queued_native, {'case'}, None)['status'] == 'needs_repair'
    assert [kind for kind, _ in calls] == ['native', 'legacy', 'native']


@pytest.mark.parametrize('identity_matches', [True, False])
def test_scanner_shallow_roots_report_live_queue_tasks_without_writes(tmp_path, monkeypatch, identity_matches):
    import flashrsi_matrix_repair_scan_v21 as scanner
    outputs = tmp_path / 'outputs'; outputs.mkdir()
    queue.save(outputs / 'flashrsi_harnessrisk_4b_private_v1_20261113/case_manifest.json',
               {'cases': [{'case_id': f'case{i}'} for i in range(128)]})
    roots = []
    for version, pid in ((20, 120), (21, 121)):
        root = outputs / f'flashrsi_future_v{version}'; roots.append(root)
        script = f'scripts/flashrsi_matrix_gpu1_queue_v{version}.py'
        identity = {'pid': pid, 'start_ticks': 'one', 'command': ['python', script, '--out', str(root)]}
        queue.save(root / 'repair_protocol.json', {'protocol': f'gpu1-finite-strict-matrix/v{version}', 'queue_launcher': script})
        queue.save(root / 'launch_state.json', {'launcher_pid': pid, 'status': 'waiting_endpoint_capacity'})
        queue.save(root / 'pidboard.json', {'launcher': identity})
        queue.save(root / 'remaining_tasks.json', {'tasks': {'pi/opro': {'status': 'queued'}}})
        (root / 'pi').mkdir()
    identities = {120: queue.read(roots[0] / 'pidboard.json')['launcher'], 121: queue.read(roots[1] / 'pidboard.json')['launcher']}
    if not identity_matches:
        identities = {pid: {**identity, 'start_ticks': 'reused'} for pid, identity in identities.items()}
    monkeypatch.setattr(scanner, 'process_identity', lambda pid: identities.get(pid))
    monkeypatch.setattr(scanner, 'active_runs', lambda: {})
    monkeypatch.setattr(scanner.matrix, 'OUTPUTS', outputs)
    monkeypatch.setattr(scanner.matrix, 'HARNESSES', ('pi',))
    monkeypatch.setattr(scanner.matrix, 'ALGORITHMS', ('opro',))
    monkeypatch.setattr(scanner.matrix, 'save', lambda *a, **kw: pytest.fail('read-only scanner must not save'))
    monkeypatch.setattr(Path, 'rglob', lambda *a, **kw: pytest.fail('no recursive output scan'))
    result = scanner.scan()
    assert result['read_only'] is True and len(result['queues']) == 2
    assert {entry['path'] for entry in result['queues']} == {str(root) for root in roots}
    assert all(entry['launcher_identity_matches'] is identity_matches for entry in result['queues'])
    job = result['jobs'][0]
    assert job['status'] == ('queued' if identity_matches else 'pending')
    assert len(job['queued_launches']) == (2 if identity_matches else 0)


def test_scanner_discovers_both_nested_native_runs_with_actual_scope_pids(tmp_path, monkeypatch):
    import flashrsi_matrix_repair_scan_v21 as scanner
    outputs = tmp_path / 'outputs'; outputs.mkdir()
    queue.save(outputs / 'flashrsi_harnessrisk_4b_private_v1_20261113/case_manifest.json',
               {'cases': [{'case_id': f'case{i}'} for i in range(128)]})
    runs = {}
    for version, harness, pid in ((20, 'pi', 210), (21, 'claude_code', 211)):
        root = outputs / f'flashrsi_v{version}'
        queue.save(root / 'repair_protocol.json', {'protocol': f'gpu1-finite-strict-matrix/v{version}'})
        run = root / harness / (harness + '_opro_full'); run.mkdir(parents=True)
        runs[str(run.resolve())] = pid
        # A deeper snapshot/cache directory must never become a root.
        (root / 'source_snapshot' / harness / (harness + '_opro_full')).mkdir(parents=True)
    calls = []
    monkeypatch.setattr(scanner.matrix, 'OUTPUTS', outputs)
    monkeypatch.setattr(scanner.matrix, 'HARNESSES', ('pi', 'claude_code'))
    monkeypatch.setattr(scanner.matrix, 'ALGORITHMS', ('opro',))
    monkeypatch.setattr(scanner, 'active_runs', lambda: runs)
    monkeypatch.setattr(scanner, 'strict_inspect_run', lambda run, expected, pid:
                        calls.append((str(run.resolve()), pid)) or {'status': 'needs_judge'})
    monkeypatch.setattr(scanner.matrix, 'inspect_run', lambda *a: pytest.fail('native queue run must get strict proof'))
    monkeypatch.setattr(Path, 'rglob', lambda *a, **kw: pytest.fail('no recursive root scan'))
    result = scanner.scan()
    assert dict(calls) == runs and len(calls) == 2
    assert all(job['status'] == 'needs_judge' for job in result['jobs'])
