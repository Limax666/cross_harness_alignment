"""Pure controller fixtures; no actual host/model/Docker/API/GPU execution."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import flashrsi_matrix_gpu1_queue_v17 as queue
import flashrsi_harnessrisk_initial_v17 as initial
import flashrsi_harnessrisk_algorithm_reference_v18 as gepa


def test_initial_transforms_keep_full_inventory_and_actual_budgets():
    source = initial.build_source()
    assert "len(cases)!=128" in source and "proc=run_host(" in source
    assert "16 if harness=='claude_code' else 12" in source
    assert 'snapshot.id' in source and 'case hash changed' in source and 'resume conditions changed' in source
    assert 'subprocess.run(' not in source
    compile(source, '<initial-v17-shape-check>', 'exec')


def test_gepa_entry_composes_contextfit_and_observability():
    source = gepa.build_source()
    assert 'flashrsi_harnessrisk_benchmark_v17' in source
    assert 'OpenCodeContextfitGEPA' in source
    assert "Path(os.environ['HR_INITIAL_REFERENCE']).resolve()" in source
    compile(source, '<gepa-v18-shape-check>', 'exec')


def test_initial_failed_host_retains_bytes_and_status(tmp_path, monkeypatch):
    run = tmp_path / 'initial';(run / 'cases').mkdir(parents=True)
    def boundary(command, **options):
        options['stderr'].write(b'concrete dependency failure')
        options['stdout'].write(b'partial real-shaped stdout')
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(initial.subprocess, 'run', boundary)
    result = initial.run_host(['python', 'host.py', 'case', 'snapshot', 'sid', str(run / 'cases')], cwd=tmp_path)
    assert result.returncode == 2
    directory = run / 'host_observability/case'
    assert (directory / 'host.stderr.log').read_bytes() == b'concrete dependency failure'
    assert json.loads((directory / 'host_execution.json').read_text())['returncode'] == 2


def test_initial_timeout_blocks_replacement_and_preserves_evidence(tmp_path, monkeypatch):
    import subprocess
    run = tmp_path / 'initial';(run / 'cases').mkdir(parents=True)
    def boundary(command, **options):
        options['stderr'].write(b'blocked startup')
        raise subprocess.TimeoutExpired(command, 2460)
    monkeypatch.setattr(initial.subprocess, 'run', boundary)
    with pytest.raises(RuntimeError, match='cleanup_uncertain'):
        initial.run_host(['python', 'host.py', 'case', 'snapshot', 'sid', str(run / 'cases')], cwd=tmp_path)
    proof = json.loads((run / 'host_observability/case/host_execution.json').read_text())
    assert proof['cleanup_confirmed'] is False and proof['timed_out'] is True


def fixture_control(tmp_path):
    expected = {f'case{i}' for i in range(128)}
    queue.save(tmp_path / 'initial_snapshot.json', {'id': 'sid', 'agent_type': 'openclaw'})
    queue.save(tmp_path / 'run_manifest.json', {'snapshot_id': 'sid', 'harness': 'openclaw', 'host_script': 'corrected.py', 'privacy_protocol': 'harnessrisk-private-case-v1'})
    queue.save(tmp_path / 'completed.json', {'snapshot_id': 'sid', 'completed_scored': 128, 'attempts': 128})
    rows=[]
    for cid in expected:
        rows.append({'case_id':cid,'exit_code':0,'rule_evaluation_present':True})
        case=tmp_path/'cases'/cid
        queue.save(case/'harness_result.json',{'snapshot_id':'sid','status':'completed','process_exit_code':0})
        queue.save(case/'privacy_audit.json',{'protocol':'harnessrisk-private-case-v1'})
        queue.save(case/'official_rule_evaluation.json',{'runs':[{'case_id':cid,'status':'completed'}]})
        queue.save(case/'adapter_metadata.json',{'mock_start':{'reset':{'body':{'case_id':cid,'ok':True}}}})
    (tmp_path/'attempts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return expected


def test_full_initial_gate_checks_identity_not_only_128_files(tmp_path):
    expected=fixture_control(tmp_path)
    assert queue.initial_acceptance(tmp_path,expected,'corrected.py')
    assert not queue.initial_acceptance(tmp_path,expected,'old.py')
    queue.save(tmp_path/'cases/case0/harness_result.json',{'snapshot_id':'sid','status':'error','process_exit_code':1})
    assert not queue.initial_acceptance(tmp_path,expected,'corrected.py')


def test_judge_gate_requires_all_exact_accepted_case_ids(tmp_path):
    expected={f'case{i}' for i in range(128)}
    queue.save(tmp_path/'llm_judge_multi_harness/manifest.json',[{'case_id':cid,'status':'existing'} for cid in expected])
    assert queue.judge_acceptance(tmp_path,expected)
    queue.save(tmp_path/'llm_judge_multi_harness/manifest.json',[{'case_id':'same','status':'judged'}]*128)
    assert not queue.judge_acceptance(tmp_path,expected)


def test_source_drift_is_typed_and_never_retried(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    (tmp_path/'owned.py').write_text('original')
    pins={'owned.py':queue.sha(tmp_path/'owned.py')}
    queue.verify_frozen(pins)
    (tmp_path/'owned.py').write_text('changed')
    with pytest.raises(RuntimeError,match='source_hash_drift_requires_new_protocol'):
        queue.verify_frozen(pins)


def test_finite_plan_skips_unrepaired_profiles_and_blocks_rsi_after_failed_h0(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    manifest={'cases':[]}
    for index in range(128):
        cid=f'case{index}';path=source/(cid+'.json');path.write_text('{}')
        manifest['cases'].append({'case_id':cid,'sha256':queue.sha(path)})
    reference=tmp_path/'reference';reference.mkdir();(reference/'initial_snapshot.json').write_text('{}')
    monkeypatch.setattr(queue,'DATA',source)
    monkeypatch.setattr(queue,'LANE_LOCK',tmp_path/'lane.lock')
    monkeypatch.setattr(queue,'REFERENCE',reference)
    monkeypatch.setattr(queue,'lane_clients',lambda **kwargs:[])
    monkeypatch.setattr(queue,'freeze_sources',lambda destination:{})
    monkeypatch.setattr(queue,'initial_acceptance',lambda path,expected,host:Path(path)==reference)
    monkeypatch.setattr(queue,'judge_acceptance',lambda *args:True)
    monkeypatch.setattr(queue,'inspect_run',lambda *args:{'status':'needs_repair'})
    monkeypatch.setattr(queue.importlib.metadata,'version',lambda name:{'tree-sitter':'0.25.2','tree-sitter-bash':'0.25.1'}[name])
    calls=[]
    controller=queue.Queue(tmp_path/'queue')
    def execute(key,command,env,*args):
        calls.append((key,command,env))
        return 2
    monkeypatch.setattr(controller,'execute',execute)
    controller.run(manifest,[])
    assert [key for key,_,_ in calls]==['opencode/evoprompt','opencode/gepa','openclaw/initial','codex/initial']
    assert calls[1][1][0].endswith('reference_v18.py')
    assert all(env['HARNESSRISK_MODEL_PORT']=='18764' and env['CUDA_VISIBLE_DEVICES']=='' for _,_,env in calls)
    assert all(controller.tasks['openclaw/'+a]['status']=='blocked_same_protocol_initial' for a in queue.ALGORITHMS)
    assert controller.tasks['claude_code/initial']['status']=='blocked_protocol_repair'
    assert controller.tasks['pi/initial']['status']=='blocked_protocol_repair'
    assert queue.read(tmp_path/'queue/ready.json')['full_objective_complete'] is False


def test_billing_guard_does_not_call_api_or_block_model_queue(tmp_path,monkeypatch):
    import time
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    queue.save(tmp_path/'.cache/harnessrisk-judge/billing-blocked.json',{'retry_after':time.time()+3600})
    controller=queue.Queue(tmp_path/'out');controller.out.mkdir()
    controller.judges=[('own/initial',tmp_path/'own_cases')]
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('API subprocess must not launch'))
    controller.schedule_judge()
    assert controller.judge is None and len(controller.judges)==1


def test_initial_gate_rejects_wrong_mock_reset_and_unexpected_extra_case(tmp_path):
    expected=fixture_control(tmp_path)
    queue.save(tmp_path/'cases/case0/adapter_metadata.json',{'mock_start':{'reset':{'body':{'case_id':'other','ok':True}}}})
    assert not queue.initial_acceptance(tmp_path,expected,'corrected.py')
    queue.save(tmp_path/'cases/case0/adapter_metadata.json',{'mock_start':{'reset':{'body':{'case_id':'case0','ok':True}}}})
    queue.save(tmp_path/'cases/extra/harness_result.json',{'status':'completed'})
    assert not queue.initial_acceptance(tmp_path,expected,'corrected.py')


def test_existing_queue_directory_is_never_overwritten_on_relaunch(tmp_path,monkeypatch):
    import sys
    out=tmp_path/'existing';out.mkdir();state=out/'remaining_tasks.json';state.write_text('{"retain":"evidence"}')
    monkeypatch.setattr(sys,'argv',['queue','--out',str(out)])
    with pytest.raises(RuntimeError,match='output_exists_preserved'):
        queue.main()
    assert state.read_text()=='{"retain":"evidence"}'
