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
    assert 'flashrsi_harnessrisk_benchmark_v18' in source
    assert 'OpenCodeContextfitGEPA' in source
    assert "Path(os.environ['HR_INITIAL_REFERENCE']).resolve()" in source
    compile(source, '<gepa-v18-shape-check>', 'exec')
    from flashrsi_harnessrisk_algorithm_reference_v19 import build_source
    ordinary=build_source()
    assert 'from flashrsi_harnessrisk_benchmark_v18 import HarnessRiskBenchmark,ROOT' in ordinary
    compile(ordinary,'<ordinary-v19-shape-check>','exec')


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
    run=tmp_path/'control';run.mkdir()
    manifest={'benchmark':'HarnessRisk','full_evaluation_count':128,
              'model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
              'flashrsi_revision':'778e2d01a995491fe63db210dcf46db1723a4430',
              'cases':[{'case_id':f'case{i}','phase':'fixture','sha256':'0'*64,'owner_turns':3} for i in range(128)]}
    queue.save(tmp_path/'case_manifest.json',manifest)
    contract=queue.expected_initial_contract((tmp_path/'case_manifest.json').read_bytes(),'openclaw')
    expected={row['case_id'] for row in manifest['cases']}
    queue.save(run/'initial_snapshot.json',{'id':'sid','agent_type':'openclaw'})
    queue.save(run/'run_manifest.json',{'snapshot_id':'sid',**contract})
    queue.save(run/'completed.json',{'snapshot_id':'sid','completed_scored':128,'attempts':128})
    rows=[]
    for cid in expected:
        rows.append({'case_id':cid,'exit_code':0,'rule_evaluation_present':True})
        case=run/'cases'/cid
        queue.save(case/'harness_result.json',{'snapshot_id':'sid','status':'completed','process_exit_code':0})
        queue.save(case/'privacy_audit.json',{'protocol':'harnessrisk-private-case-v1'})
        queue.save(case/'official_rule_evaluation.json',{'runs':[{'case_id':cid,'status':'completed'}]})
        queue.save(case/'adapter_metadata.json',{'mock_start':{'reset':{'body':{'case_id':cid,'ok':True}}}})
    (run/'attempts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return run,expected,contract


def test_full_initial_gate_checks_identity_not_only_128_files(tmp_path):
    run,expected,contract=fixture_control(tmp_path)
    assert queue.initial_acceptance(run,expected,contract)
    assert not queue.initial_acceptance(run,expected,{**contract,'host_script':'old.py'})
    queue.save(run/'cases/case0/harness_result.json',{'snapshot_id':'sid','status':'error','process_exit_code':1})
    assert not queue.initial_acceptance(run,expected,contract)


def test_initial_contract_rejects_same_ids_wrong_bytes_revision_port_and_budget(tmp_path):
    run,expected,contract=fixture_control(tmp_path)
    original=queue.read(run/'run_manifest.json')
    for field,value in [('model_revision','wrong-model'),('flashrsi_revision','wrong-runtime'),
                        ('model_port',18761),('temperature',1),('max_tokens_per_turn',8192),
                        ('max_turns_per_owner_turn',24),('harness','pi')]:
        queue.save(run/'run_manifest.json',{**original,field:value})
        assert not queue.initial_acceptance(run,expected,contract),field
    queue.save(run/'run_manifest.json',original)
    retained=queue.read(tmp_path/'case_manifest.json')
    retained['cases'][0]['sha256']='f'*64
    queue.save(tmp_path/'case_manifest.json',retained)
    # Updating the H0's self-declared hash must not override the trusted queue hash.
    queue.save(run/'run_manifest.json',{**original,'case_manifest_sha256':queue.sha(tmp_path/'case_manifest.json')})
    assert not queue.initial_acceptance(run,expected,contract)


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


@pytest.mark.parametrize('uncertain_stage',[None,'opencode/evoprompt','openclaw/initial'])
def test_finite_plan_skips_unrepaired_profiles_and_blocks_rsi_after_failed_h0(tmp_path,monkeypatch,uncertain_stage):
    source=tmp_path/'source';source.mkdir()
    manifest={'cases':[],'model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','flashrsi_revision':'778e2d01a995491fe63db210dcf46db1723a4430'}
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
        if key==uncertain_stage:
            harness,stage=key.split('/')
            run=Path(env['HR_OUTPUT_ROOT'])/(harness+'_'+('initial_full' if stage=='initial' else stage+'_full'))
            receipt=run/('host_observability/case/host_execution.json' if stage=='initial' else 'evidence/00000001/host_execution.json')
            queue.save(receipt,{'timed_out':True,**({'cleanup_confirmed':False} if stage=='initial' else {})})
        return 2
    monkeypatch.setattr(controller,'execute',execute)
    if uncertain_stage:
        with pytest.raises(queue.QueueCleanupUncertain):
            controller.run(manifest,[])
        assert calls[-1][0]==uncertain_stage
        assert controller.tasks['codex/initial']['status']=='blocked_cleanup_uncertain'
        assert (controller.out/'queue_cleanup_block.json').exists()
        assert queue.read(controller.out/'ready.json')['ready_for_authorized_finite_queue'] is False
        # An empty name scan is not descendant reconciliation or an unblock.
        with pytest.raises(queue.QueueCleanupUncertain):
            controller.check_cleanup()
        return
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
    run,expected,contract=fixture_control(tmp_path)
    queue.save(run/'cases/case0/adapter_metadata.json',{'mock_start':{'reset':{'body':{'case_id':'other','ok':True}}}})
    assert not queue.initial_acceptance(run,expected,contract)
    queue.save(run/'cases/case0/adapter_metadata.json',{'mock_start':{'reset':{'body':{'case_id':'case0','ok':True}}}})
    queue.save(run/'cases/extra/harness_result.json',{'status':'completed'})
    assert not queue.initial_acceptance(run,expected,contract)


def test_existing_queue_directory_is_never_overwritten_on_relaunch(tmp_path,monkeypatch):
    import sys
    out=tmp_path/'existing';out.mkdir();state=out/'remaining_tasks.json';state.write_text('{"retain":"evidence"}')
    monkeypatch.setattr(sys,'argv',['queue','--out',str(out)])
    with pytest.raises(RuntimeError,match='output_exists_preserved'):
        queue.main()
    assert state.read_text()=='{"retain":"evidence"}'


def test_algorithm_timeout_aborts_rollout_before_next_case(tmp_path,monkeypatch):
    import asyncio
    from flashrsi.core import Task,Trajectory
    import flashrsi_harnessrisk_benchmark_v18 as guarded
    calls=[]
    async def boundary(controller,target,task):
        calls.append(task.task_id);controller._sequence+=1
        queue.save(tmp_path/'evidence'/f'{controller._sequence:08d}'/'host_execution.json',{'timed_out':True})
        return Trajectory(status='error',error_message='HarnessRisk host timeout')
    monkeypatch.setattr(guarded.ObservedBenchmark,'_run_once',boundary)
    controller=guarded.HarnessRiskBenchmark(tmp_path)
    target=SimpleNamespace(snapshot=SimpleNamespace(id='fixture'))
    tasks=[Task(task_id='first',question='owner',env='harnessrisk'),Task(task_id='must-not-start',question='owner',env='harnessrisk')]
    async def rollout():
        for task in tasks:
            await controller.env_runner(target,task)
    with pytest.raises(guarded.HostCleanupUncertain,match='identity_reconciliation'):
        asyncio.run(rollout())
    assert calls==['first']


def test_exact_predecessor_timeout_scope_blocks_even_with_no_lane_clients(tmp_path):
    old=tmp_path/'owned-predecessor';out=tmp_path/'new-queue';out.mkdir()
    queue.save(old/'evidence/00000001/host_execution.json',{'timed_out':True,'pid':123})
    controller=queue.Queue(out);controller.predecessor_scopes=[old]
    controller.tasks={'codex/initial':{'status':'queued'}}
    with pytest.raises(queue.QueueCleanupUncertain):
        controller.check_cleanup()
    block=queue.read(out/'queue_cleanup_block.json')
    assert block['receipts'][0]['path'].startswith(str(old))
    assert controller.tasks['codex/initial']['status']=='blocked_cleanup_uncertain'
