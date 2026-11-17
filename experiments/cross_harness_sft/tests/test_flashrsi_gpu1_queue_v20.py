"""Pure future controller tests; all dispatch/host/profile boundaries are stubs."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import flashrsi_matrix_gpu1_queue_v20 as queue
from flashrsi.core.snapshot import Snapshot


def resource(tmp_path, monkeypatch):
    tree = tmp_path / 'resource-tree'
    (tree / 'runtime/pi_native').mkdir(parents=True)
    (tree / 'runtime/pi_native/loop.py').write_text('# inert unit resource\n')
    snapshot = Snapshot.from_directory(tree, agent_type='pi', component_rules={'runtime': ('runtime/**',)})
    path = tmp_path / 'resource.json'
    snapshot.save(path)
    profile = {**queue.NATIVE_PROFILES['pi'], 'snapshot': path, 'snapshot_sha256': queue.sha(path), 'snapshot_id': snapshot.id}
    monkeypatch.setitem(queue.NATIVE_PROFILES, 'pi', profile)
    return path, snapshot


def h0(tmp_path, monkeypatch):
    path, snapshot = resource(tmp_path, monkeypatch)
    base = tmp_path / 'pi'
    run = base / 'pi_initial_full'
    run.mkdir(parents=True)
    manifest = {'model_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'flashrsi_revision': 'unit-revision',
                'cases': [{'case_id': f'case{i}', 'sha256': 'a' * 64} for i in range(128)]}
    queue.save(base / 'case_manifest.json', manifest)
    contract = queue.expected_initial_contract((base / 'case_manifest.json').read_bytes(), 'pi')
    snapshot.save(run / 'initial_snapshot.json')
    queue.save(run / 'run_manifest.json', {**contract, 'snapshot_id': snapshot.id,
        'initial_snapshot_sha256': queue.sha(run / 'initial_snapshot.json'), 'code_sha256': {'unit': 'pin'}})
    queue.save(run / 'completed.json', {'status': 'all_cases_attempted', 'execution_protocol': contract['execution_protocol'],
        'snapshot_id': snapshot.id, 'all128_measurement_valid': True, 'attempts': 128,
        'completed_scored': 128, 'failed_or_unscored': 0})
    rows = [{'case_id': row['case_id'], 'returncode': 0, 'measurement_valid': True} for row in manifest['cases']]
    (run / 'attempts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    for row in rows:
        queue.save(run / 'cases' / row['case_id'] / 'harness_result.json', {})
    calls = []
    def validate(initial):
        calls.append((Path(initial), os.environ.get('HR_OUTPUT_ROOT')))
        return Path(initial)
    monkeypatch.setattr(queue, 'validate_native_reference', validate)
    return run, {r['case_id'] for r in rows}, contract, calls


def test_composition_retains_v19_gates_and_routes_only_reviewed_profiles():
    source = queue.build_source()
    compile(source, '<queue-v20>', 'exec')
    assert queue.HOSTS['pi'] == 'flashrsi_harnessrisk_host_pi_native_owner_v6.py'
    assert queue.HOSTS['codex'] == 'flashrsi_harnessrisk_host_codex_context_v18.py'
    assert queue.HOSTS['openclaw'] == 'flashrsi_harnessrisk_host_openclaw_context_v4.py'
    assert tuple(queue.NATIVE_PROFILES) == ('pi',)
    assert queue.ALGORITHMS == ('opro', 'gepa', 'evoprompt', 'ace', 'skillopt', 'evo_harness')
    assert queue.initial_command('pi', Path('/fresh')) == [
        'scripts/flashrsi_harnessrisk_pi_native_owner_initial_v6.py', '--base', '/fresh',
        '--snapshot', str(queue.NATIVE_PROFILES['pi']['snapshot'])]
    assert queue.algorithm_entry('pi', 'gepa') == 'flashrsi_harnessrisk_pi_native_owner_reference_v6.py'
    assert queue.algorithm_entry('opencode', 'gepa') == 'flashrsi_harnessrisk_algorithm_reference_v18.py'
    assert queue.initial_command('codex', Path('/fresh')) == ['scripts/flashrsi_harnessrisk_initial_v19.py']
    assert 'OPENCLAW_CONTEXT_METADATA' in source and 'CONTEXT_METADATA' in source


def test_resource_is_content_validated_and_type_hash_pinned(tmp_path, monkeypatch):
    path, snapshot = resource(tmp_path, monkeypatch)
    binding = queue.native_resource_binding('pi')
    assert binding == {'path': str(path), 'canonical_path': str(path.resolve()),
                       'sha256': queue.sha(path), 'snapshot_id': snapshot.id, 'agent_type': 'pi'}
    raw = json.loads(path.read_text())
    raw['agent_type'] = 'claude_code'
    queue.save(path, raw)
    monkeypatch.setitem(queue.NATIVE_PROFILES['pi'], 'snapshot_sha256', queue.sha(path))
    with pytest.raises(ValueError):
        queue.native_resource_binding('pi')
    path.write_text('{}')
    with pytest.raises(ValueError):
        queue.native_resource_binding('pi')


@pytest.mark.parametrize('mutation', ['hash', 'id'])
def test_resource_pin_mismatch(tmp_path, monkeypatch, mutation):
    resource(tmp_path, monkeypatch)
    monkeypatch.setitem(queue.NATIVE_PROFILES['pi'], 'snapshot_' + ('sha256' if mutation == 'hash' else 'id'), '0' * 64)
    with pytest.raises(ValueError):
        queue.native_resource_binding('pi')


@pytest.mark.parametrize('existing', [None, '/unrelated/output'])
def test_h0_uses_profile_validator_with_scoped_output_root(tmp_path, monkeypatch, existing):
    run, expected, contract, calls = h0(tmp_path, monkeypatch)
    if existing is None:
        monkeypatch.delenv('HR_OUTPUT_ROOT', raising=False)
    else:
        monkeypatch.setenv('HR_OUTPUT_ROOT', existing)
    assert queue.initial_acceptance(run, expected, contract)
    assert calls == [(run, str(run.parent))]
    assert os.environ.get('HR_OUTPUT_ROOT') == existing
    def invalid(initial):
        raise ValueError('native evidence mismatch')
    monkeypatch.setattr(queue, 'validate_native_reference', invalid)
    assert not queue.initial_acceptance(run, expected, contract)
    assert os.environ.get('HR_OUTPUT_ROOT') == existing


@pytest.mark.parametrize('mutation', ['host', 'contract', 'revision', 'snapshot', 'attempt_duplicate',
    'attempt_failed', 'attempt_invalid', 'attempt_timeout', 'attempt_missing', 'extra_case', 'completed'])
def test_native_h0_rejects_legacy_or_partial_contracts(tmp_path, monkeypatch, mutation):
    run, expected, contract, calls = h0(tmp_path, monkeypatch)
    if mutation in ('host', 'contract', 'revision'):
        row = queue.read(run / 'run_manifest.json')
        row[{'host': 'host_script', 'contract': 'contract', 'revision': 'model_revision'}[mutation]] = 'wrong'
        queue.save(run / 'run_manifest.json', row)
    elif mutation == 'snapshot':
        (run / 'initial_snapshot.json').write_text('{}')
    elif mutation.startswith('attempt'):
        rows = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines()]
        if mutation == 'attempt_duplicate': rows[-1] = rows[0]
        if mutation == 'attempt_failed': rows[-1]['returncode'] = 2
        if mutation == 'attempt_invalid': rows[-1]['measurement_valid'] = False
        if mutation == 'attempt_timeout': rows[-1]['timed_out'] = True
        if mutation == 'attempt_missing': rows.pop()
        (run / 'attempts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    elif mutation == 'extra_case':
        (run / 'cases/extra').mkdir()
    elif mutation == 'completed':
        queue.save(run / 'completed.json', {'completed_scored': 128})
    assert not queue.initial_acceptance(run, expected, contract)


def plan_fixture(tmp_path, monkeypatch):
    resource(tmp_path, monkeypatch)
    source = tmp_path / 'source'
    source.mkdir()
    manifest = {'cases': [], 'model_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'flashrsi_revision': 'runtime'}
    for i in range(128):
        p = source / f'case{i}.json'; p.write_text('{}')
        manifest['cases'].append({'case_id': f'case{i}', 'sha256': queue.sha(p)})
    reference = tmp_path / 'reference'; reference.mkdir()
    (reference / 'initial_snapshot.json').write_text('{}')
    monkeypatch.setattr(queue, 'DATA', source)
    monkeypatch.setattr(queue, 'LANE_LOCK', tmp_path / 'lane.lock')
    monkeypatch.setattr(queue, 'REFERENCE', reference)
    monkeypatch.setattr(queue, 'lane_clients', lambda **kw: [])
    monkeypatch.setattr(queue, 'freeze_sources', lambda destination: {})
    monkeypatch.setattr(queue, 'native_source_pins', lambda harness: {'unit': 'pin'})
    monkeypatch.setattr(queue, 'initial_acceptance', lambda path, expected, contract: Path(path) == reference or contract['harness'] == 'pi')
    monkeypatch.setattr(queue, 'judge_acceptance', lambda *args: True)
    monkeypatch.setattr(queue, 'inspect_run', lambda *args: {'status': 'needs_repair'})
    monkeypatch.setattr(queue.importlib.metadata, 'version', lambda name: {'tree-sitter': '0.25.2', 'tree-sitter-bash': '0.25.1'}[name])
    controller = queue.Queue(tmp_path / 'queue')
    calls = []
    def execute(key, command, env, *args):
        calls.append((key, command, dict(env)))
        if key == 'pi/initial':
            run = Path(env['HR_INITIAL_REFERENCE']); run.mkdir()
            (run / 'initial_snapshot.json').write_bytes(queue.NATIVE_PROFILES['pi']['snapshot'].read_bytes())
        return 0 if key == 'pi/initial' else 2
    monkeypatch.setattr(controller, 'execute', execute)
    return controller, manifest, calls


def test_finite_plan_dispatches_pi_all_six_and_preserves_failures(tmp_path, monkeypatch):
    controller, manifest, calls = plan_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(controller, 'schedule_judge', lambda: None)
    controller.run(manifest, [])
    assert [k for k, _, _ in calls] == ['opencode/evoprompt', 'opencode/gepa', 'openclaw/initial', 'codex/initial',
        'pi/initial', *['pi/' + a for a in queue.ALGORITHMS]]
    assert calls[4][1] == queue.initial_command('pi', controller.out / 'pi')
    assert all(c[1][0].endswith('pi_native_owner_reference_v6.py') for c in calls[5:])
    assert all(e['HARNESSRISK_MODEL_PORT'] == '18764' and e['CUDA_VISIBLE_DEVICES'] == '' for _, _, e in calls)
    assert all(controller.tasks['pi/' + a]['status'] == 'requires_repair' for a in queue.ALGORITHMS)
    assert all(controller.tasks['claude_code/' + a]['status'] == 'blocked_protocol_repair' for a in ('initial', *queue.ALGORITHMS))
    assert controller.tasks['pi/initial']['judge_status'] == 'needs_judge_lease_or_billing'
    protocol = queue.read(controller.out / 'repair_protocol.json')
    assert protocol['coverage']['model_harnesses_this_queue'] == ['opencode', 'openclaw', 'codex', 'pi']
    assert protocol['coverage']['deferred_harnesses'] == ['claude_code']
    assert protocol['native_profiles']['pi']['resource']['sha256'] == queue.NATIVE_PROFILES['pi']['snapshot_sha256']
    assert (controller.out / 'native_resources/pi_initial_snapshot.json').is_file()


def test_failed_pi_h0_blocks_only_its_six_algorithms(tmp_path, monkeypatch):
    controller, manifest, calls = plan_fixture(tmp_path, monkeypatch)
    original = queue.initial_acceptance
    monkeypatch.setattr(queue, 'initial_acceptance', lambda path, expected, contract: contract['harness'] != 'pi' and original(path, expected, contract))
    controller.run(manifest, [])
    assert calls[-1][0] == 'pi/initial'
    assert controller.tasks['pi/initial']['status'] == 'blocked_initial_requires_repair'
    assert all(controller.tasks['pi/' + a]['status'] == 'blocked_same_protocol_initial' for a in queue.ALGORITHMS)


def test_exact_predecessor_waits_before_freeze_or_dispatch(tmp_path, monkeypatch):
    controller, manifest, calls = plan_fixture(tmp_path, monkeypatch)
    predecessor = {'pid': 123, 'start_ticks': '987', 'command': ['exact-predecessor']}
    monkeypatch.setattr(queue, 'process_identity', lambda pid: predecessor if pid == 123 else None)
    monkeypatch.setattr(queue, 'freeze_sources', lambda *a: pytest.fail('must not freeze before predecessor exits'))
    def stop(seconds):
        raise InterruptedError('test stops while predecessor still active')
    monkeypatch.setattr(queue.time, 'sleep', stop)
    with pytest.raises(InterruptedError): controller.run(manifest, [predecessor])
    assert calls == []
    assert not (controller.out / 'source_snapshot').exists()
    assert not (controller.out / 'native_resources').exists()


def test_sticky_timeout_blocks_lane_even_after_empty_scan(tmp_path, monkeypatch):
    controller, manifest, calls = plan_fixture(tmp_path, monkeypatch)
    execute = controller.execute
    def uncertain(key, command, env, *args):
        code = execute(key, command, env, *args)
        if key == 'pi/initial':
            queue.save(Path(env['HR_INITIAL_REFERENCE']) / 'host_observability/case0/host_execution.json', {'cleanup_confirmed': False})
        return code
    monkeypatch.setattr(controller, 'execute', uncertain)
    with pytest.raises(queue.QueueCleanupUncertain): controller.run(manifest, [])
    assert calls[-1][0] == 'pi/initial'
    with pytest.raises(queue.QueueCleanupUncertain): controller.check_cleanup()
    assert queue.read(controller.out / 'ready.json')['ready_for_authorized_finite_queue'] is False


def test_selected_native_proof_is_required_beyond_legacy_inspection(tmp_path, monkeypatch):
    resource(tmp_path, monkeypatch)
    run = tmp_path / 'pi/pi_opro_full'; run.mkdir(parents=True)
    monkeypatch.setattr(queue, 'legacy_inspect_run', lambda *a: {'status': 'needs_judge', 'reasons': []})
    assert queue.inspect_run(run, {'case0'}, None)['status'] == 'needs_repair'


def test_cached_judge_and_billing_are_finite_without_api(tmp_path, monkeypatch):
    out = tmp_path / 'queue'; out.mkdir()
    lock = tmp_path / 'worker.lock'
    monkeypatch.setattr(queue, 'WORKER_LOCK', lock)
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('no API allowed'))
    monkeypatch.setattr(queue, 'judge_acceptance', lambda *a: True)
    controller = queue.Queue(out); controller.expected = {'case0'}
    controller.tasks = {'pi/initial': {'status': 'source_valid_needs_judge'}}
    controller.judges = [('pi/initial', tmp_path / 'cases')]
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    controller.schedule_judge()
    assert controller.tasks['pi/initial']['status'] == 'complete'
    assert controller.tasks['pi/initial']['judge_status'] == 'complete_cached'
    queue.save(tmp_path / '.cache/harnessrisk-judge/billing-blocked.json', {'retry_after': queue.time.time() + 3600})
    controller.judges = [('pi/opro', tmp_path / 'uncached')]
    controller.schedule_judge()
    assert len(controller.judges) == 1 and controller.judge is None


def case_sources(run, tmp_path, monkeypatch):
    data = tmp_path / 'original-cases'; data.mkdir()
    manifest = queue.read(run.parent / 'case_manifest.json')
    for row in manifest['cases']:
        path = data / (row['case_id'] + '.json')
        path.write_text(json.dumps({'task': {'user_messages': ['original owner']}}))
        row['sha256'] = queue.sha(path)
    queue.save(run.parent / 'case_manifest.json', manifest)
    contract = queue.expected_initial_contract((run.parent / 'case_manifest.json').read_bytes(), 'pi')
    queue.save(run / 'run_manifest.json', {**queue.read(run / 'run_manifest.json'), **contract})
    monkeypatch.setattr(queue, 'DATA', data)
    return manifest, contract, data


def test_actual_reference_verifier_walks_all128_under_scoped_base(tmp_path, monkeypatch):
    import flashrsi_harnessrisk_pi_native_owner_initial_v6 as native_initial
    import flashrsi_harnessrisk_pi_native_owner_reference_v6 as native_reference
    run, expected, _, _ = h0(tmp_path, monkeypatch)
    _, contract, data = case_sources(run, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(native_initial, 'DATA', data)
    monkeypatch.setattr(native_initial, 'source_pins', lambda: {'unit': 'pin'})
    monkeypatch.setattr(native_reference, 'validate_case', lambda path, cid, sid, owners, source, **kw:
                        calls.append((cid, owners, source, kw)))
    monkeypatch.setattr(queue, 'validate_native_reference', native_reference.validate_reference)
    monkeypatch.setenv('HR_OUTPUT_ROOT', '/prior/unrelated')
    assert queue.initial_acceptance(run, expected, contract)
    assert len(calls) == 128 and {r[0] for r in calls} == expected
    assert all(r[1] == ['original owner'] and r[3]['native_source'] for r in calls)
    assert os.environ['HR_OUTPUT_ROOT'] == '/prior/unrelated'


def test_complete_selected_profile_revalidation_and_native_failure(tmp_path, monkeypatch):
    initial, expected, _, _ = h0(tmp_path, monkeypatch)
    case_sources(initial, tmp_path, monkeypatch)
    run = initial.parent / 'pi_opro_full'; run.mkdir()
    selected = Snapshot.load(initial / 'initial_snapshot.json')
    (run / 'snapshots').mkdir(); selected.save(run / 'snapshots/selected.json')
    for cid in expected: queue.save(run / 'selected_cases' / cid / 'harness_result.json', {})
    queue.save(run / 'run_manifest.json', {'target_execution_protocol': queue.PI_PROFILE,
        'host_script': queue.HOSTS['pi'], 'initial_reference_dir': str(initial), 'initial_snapshot_id': selected.id,
        'model_port': 18764, 'worker_max_tokens': 1024, 'evolution_case_attempts': 1,
        'scoring_contract': 'reject_unscorable_during_evolution'})
    queue.save(run / 'result.json', {'initial_snapshot_id': selected.id})
    monkeypatch.setattr(queue, 'legacy_inspect_run', lambda *a: {'status': 'needs_judge', 'reasons': []})
    calls = []
    monkeypatch.setattr(queue, 'validate_native_case', lambda path, cid, *args, **kw: calls.append(cid))
    assert queue.inspect_run(run, expected, None)['status'] == 'needs_judge'
    assert len(calls) == 128 and set(calls) == expected
    queue.save(run / 'selected_cases/llm_judge_multi_harness/manifest.json',
               [{'case_id': cid, 'status': 'existing'} for cid in expected])
    assert queue.inspect_run(run, expected, None)['status'] == 'needs_judge'
    foreign = run / 'selected_cases/foreign'
    queue.save(foreign / 'harness_result.json', {})
    assert queue.inspect_run(run, expected, None)['status'] == 'needs_repair'
    (foreign / 'harness_result.json').unlink(); foreign.rmdir()
    def failed(path, cid, *args, **kw):
        if cid == 'case7': raise ValueError('settled native proof missing')
    monkeypatch.setattr(queue, 'validate_native_case', failed)
    result = queue.inspect_run(run, expected, None)
    assert result['status'] == 'needs_repair' and 'settled native proof missing' in result['reasons'][0]


def test_judge_artifact_folder_does_not_invalidate_h0_or_allow_foreign_case(tmp_path, monkeypatch):
    run, expected, contract, _ = h0(tmp_path, monkeypatch)
    queue.save(run / 'cases/llm_judge_multi_harness/manifest.json',
               [{'case_id': cid, 'status': 'existing'} for cid in expected])
    assert queue.initial_acceptance(run, expected, contract)
    queue.save(run / 'cases/foreign/harness_result.json', {})
    assert not queue.initial_acceptance(run, expected, contract)


def test_resource_snapshot_roundtrip_preserves_bytes_and_identity(tmp_path, monkeypatch):
    path, snapshot = resource(tmp_path, monkeypatch)
    copy = tmp_path / 'fresh_initial_snapshot.json'
    Snapshot.load(path).save(copy)
    assert copy.read_bytes() == path.read_bytes()
    assert Snapshot.load(copy).id == snapshot.id


def execute_fixture(tmp_path, monkeypatch):
    run, expected, _, _ = h0(tmp_path, monkeypatch)
    manifest, contract, _ = case_sources(run, tmp_path, monkeypatch)
    controller = queue.Queue(tmp_path / 'controller'); controller.out.mkdir()
    controller.manifest, controller.expected, controller.contracts = manifest, expected, {'pi': contract}
    controller.tasks = {'pi/opro': {'status': 'queued'}}
    monkeypatch.setattr(queue, 'native_source_pins', lambda h: {'unit': 'pin'})
    queue.prepare_native_profiles(controller, {})
    queue.save(controller.out / 'repair_protocol.json', {'unit': 'controller'})
    monkeypatch.setattr(queue, 'lane_clients', lambda **kw: [])
    monkeypatch.setattr(queue, 'process_identity', lambda pid: None)
    env = {**os.environ, 'HR_HARNESS': 'pi', 'HR_INITIAL_REFERENCE': str(run),
           'HR_OUTPUT_ROOT': str(run.parent), 'HR_HOST_SCRIPT': queue.HOSTS['pi']}
    return controller, run, env, queue.sha(run.parent / 'case_manifest.json')


@pytest.mark.parametrize('drift', [None, 'resource', 'h0', 'profile'])
def test_dispatch_receipt_retains_resource_and_h0_before_after(tmp_path, monkeypatch, drift):
    controller, run, env, case_hash = execute_fixture(tmp_path, monkeypatch)
    profile = queue.NATIVE_PROFILES['pi']
    original_hash = profile['snapshot_sha256']
    def boundary(*args, **kwargs):
        if drift == 'resource': profile['snapshot'].write_text('resource changed')
        if drift == 'h0': (run / 'initial_snapshot.json').write_text('H0 changed')
        if drift == 'profile': monkeypatch.setattr(queue, 'native_source_pins', lambda h: {'unit': 'changed'})
        return SimpleNamespace(pid=999, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(queue.subprocess, 'Popen', boundary)
    def execute():
        return controller.execute('pi/opro', ['scripts/' + profile['reference'], '--algorithm', 'opro'], env, {}, case_hash)
    if drift:
        with pytest.raises((RuntimeError, ValueError)): execute()
    else:
        assert execute() == 0
    receipt = queue.read(controller.out / 'stage_receipts/pi_opro.json')
    assert receipt['native_resources_before']['pi']['sha256'] == original_hash
    assert receipt['initial_snapshot_sha256_before'] == original_hash
    assert (receipt['native_resources_after']['pi']['sha256'] != original_hash) == (drift == 'resource')
    assert (receipt['initial_snapshot_sha256_after'] != original_hash) == (drift == 'h0')
    assert 'initial_binding_before' in receipt and 'initial_binding_after' in receipt


def test_resource_drift_during_wait_never_reaches_child(tmp_path, monkeypatch):
    controller, _, env, case_hash = execute_fixture(tmp_path, monkeypatch)
    clients = iter([[{'pid': 321, 'start_ticks': '1'}], []])
    monkeypatch.setattr(queue, 'lane_clients', lambda **kw: next(clients))
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: queue.NATIVE_PROFILES['pi']['snapshot'].write_text('changed while waiting'))
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('drift must block child'))
    with pytest.raises(ValueError, match='resource_hash_changed'):
        controller.execute('pi/opro', ['scripts/reference.py', '--algorithm', 'opro'], env, {}, case_hash)
    assert not (controller.out / 'stage_receipts/pi_opro.json').exists()


def test_source_profile_drift_before_dispatch_is_rejected(tmp_path, monkeypatch):
    controller, _, env, case_hash = execute_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, 'native_source_pins', lambda h: {'unit': 'changed'})
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('profile drift must block child'))
    with pytest.raises(RuntimeError, match='profile_or_original_H0_resource_drift'):
        controller.execute('pi/opro', ['scripts/reference.py', '--algorithm', 'opro'], env, {}, case_hash)


def test_cli_help_and_existing_output_preserve_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(queue.sys, 'argv', ['queue-v20', '--help'])
    with pytest.raises(SystemExit) as exc:
        queue.main()
    assert exc.value.code == 0 and '--predecessor-pid' in capsys.readouterr().out
    out = tmp_path / 'existing'; out.mkdir()
    receipt = out / 'remaining_tasks.json'; receipt.write_text('{"retained":true}')
    monkeypatch.setattr(queue.sys, 'argv', ['queue-v20', '--out', str(out)])
    with pytest.raises(RuntimeError, match='output_exists_preserved'):
        queue.main()
    assert receipt.read_text() == '{"retained":true}'
