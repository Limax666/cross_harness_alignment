"""Workspace/reward boundaries; full source admission supplies runtime proof."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def modules():
    assert importlib.util.find_spec('hsrl_v5.workspace'), 'bounded workspace runtime missing'
    from hsrl_v5.workspace import safe_path, numeric_reward, snapshot_entries
    return safe_path, numeric_reward, snapshot_entries


def test_workspace_refuses_absolute_parent_and_symlink_paths(tmp_path):
    scoped, _, _ = modules()
    (tmp_path / 'link').symlink_to('/tmp')
    for name in ('../../outside', '/etc/passwd', 'link/file'):
        with pytest.raises(ValueError):
            scoped(tmp_path, name)
    assert scoped(tmp_path, 'output/report.json') == tmp_path / 'output/report.json'


def test_empty_nan_and_out_of_range_grades_are_invalid_not_zero():
    _, grade, _ = modules()
    for value in ({}, [], None, float('nan'), 2.0, {'x': 'not-a-score'}):
        with pytest.raises(ValueError):
            grade(value)
    assert grade({'a': True, 'b': False, 'c': 0.5}) == .5


def test_snapshot_digest_covers_real_source_bytes_and_rejects_symlinks(tmp_path):
    _, _, entries = modules()
    (tmp_path / 'input.json').write_text('{"real":true}')
    original = entries(tmp_path)
    (tmp_path / 'input.json').write_text('{"real":false}')
    assert entries(tmp_path) != original
    (tmp_path / 'escape').symlink_to('/etc/passwd')
    with pytest.raises(ValueError):
        entries(tmp_path)


def test_component_grades_cannot_hide_out_of_range_values_in_average():
    _, grade, _ = modules()
    for value in ({'negative': -1, 'positive': 1}, {'too_high': 1.5, 'zero': 0}):
        with pytest.raises(ValueError, match='unit_interval'):
            grade(value)


def test_numerical_libraries_fit_frozen_container_pid_budget():
    from hsrl_v5.workspace_worker import command_environment
    environment = command_environment()
    # Retained genuine risk_metrics_calculation checker failed importing pandas:
    # OpenBLAS attempted64threads in a64-PID container already running2processes.
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'BLIS_NUM_THREADS'):
        assert environment[key] == '1'
    assert 'API_KEY' not in environment


def test_pending_infrastructure_record_is_retried_but_completed_record_is_reused(tmp_path, monkeypatch):
    from hsrl_v5 import quality_claw as qc
    from hsrl_v5.io import atomic_json, digest
    record = {'id': 'real-id', 'family': 'family', 'reward_profile': 'profile'}
    previous = {'record_sha256': digest(record), 'ready': False,
                'failure_class': 'isolation_or_runtime_infrastructure'}
    atomic_json(tmp_path / 'quality.json', previous)
    class BoundaryReached(Exception):
        pass
    def reach_real_backend(*args):
        raise BoundaryReached('backend-boundary-reached')
    monkeypatch.setattr(qc, 'ClawBackend', reach_real_backend)
    proof = qc.validate_record(record, tmp_path)
    assert proof['reasons'] == ['BoundaryReached:backend-boundary-reached']
    assert len(list(tmp_path.glob('attempt_*'))) == 1
    proof['ready'] = True
    atomic_json(tmp_path / 'quality.json', proof)
    assert qc.validate_record(record, tmp_path) == proof
    assert len(list(tmp_path.glob('attempt_*'))) == 1


def test_bounded_dispatch_stops_after_infrastructure_failure(tmp_path):
    from hsrl_v5.quality_claw import bounded_admission
    import threading
    import time
    seen, lock = [], threading.Lock()
    def external_boundary(record, directory):
        with lock:
            seen.append(record['id'])
        if record['id'] == '1':
            return {'id': '1', 'ready': False, 'reasons': ['engine unavailable'],
                    'failure_class': 'isolation_or_runtime_infrastructure'}
        time.sleep(.05)
        return {'id': record['id'], 'ready': True, 'reasons': []}
    records = [{'id': str(i)} for i in range(100)]
    proofs, blocked = bounded_admission(records, tmp_path, 2, validate=external_boundary)
    assert blocked is True
    assert len(seen) <= 2
    assert any(p['id'] == '1' for p in proofs)
    assert len(list((tmp_path / 'events.jsonl').open())) == len(proofs)


def test_oversized_worker_count_is_rejected_before_source_access(tmp_path):
    from hsrl_v5.quality_claw import run
    with pytest.raises(ValueError, match='workers'):
        run(tmp_path / 'missing', tmp_path / 'out', 17)


def test_pinned_boundary_cache_rejects_old_policy_and_rechecks_actual_checker_bytes(tmp_path):
    from hsrl_v5.quality_claw import prefilter_checker_boundaries
    from hsrl_v5.workspace import CHECKER_BOUNDARY_VERSION
    from hsrl_v5.io import atomic_json, file_hash
    from hsrl_v5 import workspace
    exp = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')
    prepared = Path(__file__).resolve().parents[1] / 'outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116'
    records = [r for r in map(json.loads, (prepared / 'accepted.jsonl').open()) if r['split'] == 'train']
    old_cache = exp / 'outputs/rl/hsrl_v5_claw_runtime_quality_boundary_v2_20261116T193656'
    # A lifecycle edit changes workspace.py identity. Do not rewrite historical
    # evidence or silently reuse its decisions under the new protocol.
    with pytest.raises(ValueError, match='policy_or_audit_identity_changed'):
        prefilter_checker_boundaries(records, tmp_path / 'old_rejected', cache_dir=old_cache)
    eligible = next(row['id'] for row in json.loads((old_cache / 'private_grader_boundary_audit.json').read_text())
                    if row['private_grader_boundary']['status'] == 'readonly_data_checker_static_boundary_pass')
    records = [next(row for row in records if row['id'] == eligible)]
    cache = tmp_path / 'fresh_cache'
    # Exercise the actual current AST gate on genuine original bytes to create
    # this small fixture; the full production dispatch performs its own sweep.
    prefilter_checker_boundaries(records, cache)
    atomic_json(cache / 'protocol.json', {'private_grader_boundary_gate': CHECKER_BOUNDARY_VERSION,
                'code_sha256': {'workspace.py': file_hash(workspace.__file__)}})
    runnable, pending, summary = prefilter_checker_boundaries(records, tmp_path / 'reused', cache_dir=cache)
    assert len(runnable) == 1 and not pending
    assert summary['verified_static_cache']['all_records_and_actual_checker_bytes_rechecked'] is True
    assert summary['source_execution_performed'] is False
    records[0] = {**records[0], 'exclusion_reasons': ['source_record_changed']}
    with pytest.raises(ValueError, match='source_record_changed'):
        prefilter_checker_boundaries(records, tmp_path / 'rejected', cache_dir=cache)
    checker = Path(records[0]['private_checker_path'])
    original_hash = file_hash(checker)
    # No original source bytes are mutated: changing only the cache's expected
    # hash establishes that it must be checked against the actual source file.
    audit = json.loads((cache / 'private_grader_boundary_audit.json').read_text())
    audit[0]['private_grader_boundary']['checker_sha256'] = '0' * 64
    atomic_json(cache / 'private_grader_boundary_audit.json', audit)
    cached_summary = json.loads((cache / 'private_grader_boundary_summary.json').read_text())
    cached_summary['audit_sha256'] = file_hash(cache / 'private_grader_boundary_audit.json')
    atomic_json(cache / 'private_grader_boundary_summary.json', cached_summary)
    with pytest.raises(ValueError, match='original_checker_bytes_changed'):
        prefilter_checker_boundaries([next(row for row in map(json.loads, (prepared / 'accepted.jsonl').open())
                                          if row['id'] == eligible)], tmp_path / 'bytes_rejected', cache_dir=cache)
    assert file_hash(checker) == original_hash


def test_original_checker_crash_and_malformed_scores_are_invalid_not_zero():
    from hsrl_v5.claw_backend import parse_checker_response
    for response in ({'ok': False, 'error': 'wall_timeout'},
                     {'ok': True, 'result': {'returncode': 1, 'stdout': '0', 'stderr': 'traceback'}},
                     {'ok': True, 'result': {'returncode': 0, 'stdout': '{}', 'stderr': ''}},
                     {'ok': True, 'result': {'returncode': 0, 'stdout': 'not-json', 'stderr': ''}}):
        with pytest.raises(ValueError):
            parse_checker_response(response)
    assert parse_checker_response({'ok': True, 'result': {'returncode': 0, 'stdout': '{"a":0,"b":1}', 'stderr': ''}}) == ({'a': 0, 'b': 1}, .5)


def test_real_child_output_limit_kills_command_and_preserves_workspace(tmp_path, monkeypatch):
    from hsrl_v5 import workspace_worker as worker
    monkeypatch.setattr(worker, 'PUBLIC', tmp_path)
    with pytest.raises(RuntimeError, match='output_limit'):
        worker.bounded_command([sys.executable, '-c', 'import sys;sys.stdout.write("x"*1100000)'])
    assert list(tmp_path.iterdir()) == []


def test_real_hf00719_actor_program_checker_is_rejected_before_container(tmp_path, monkeypatch):
    from hsrl_v5.workspace import checker_boundary_contract
    from hsrl_v5 import claw_backend
    prepared = Path(__file__).resolve().parents[1] / 'outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116'
    if not prepared.exists():
        pytest.skip('fixed genuine HF source corpus is not installed')
    record = next(json.loads(line) for line in (prepared / 'accepted.jsonl').open()
                  if '/hf_00719/' in json.loads(line)['private_checker_path'])
    assert record['id'] == 'odt_filemgr_oc_9559d113ef1e'
    gate = checker_boundary_contract(record['private_checker_path'])
    assert gate['status'] == 'pending_unsupported_private_grader_boundary'
    assert gate['checker_sha256'] == 'deff2255f458ae23544766d25b1fd7f3af88c3e2c5ee48a49a6f16055140ac8e'
    assert any('subprocess' in item['resolved'] for item in gate['evidence'])
    def container_must_not_start(*args, **kwargs):
        raise AssertionError('unsupported checker reached a container')
    monkeypatch.setattr(claw_backend, 'WorkspaceWorker', container_must_not_start)
    with pytest.raises(ValueError, match='private_grader_boundary_pending'):
        claw_backend.ClawBackend(record, tmp_path / 'episode')
    assert not (tmp_path / 'episode').exists()


@pytest.mark.parametrize('source', [
    'import subprocess as sp\nsp.run(["python", "output/a.py"])',
    'from subprocess import Popen as launch\nlaunch([])',
    'import os as o\nf=o.system\nf("a")',
    'from os import spawnv as start\nstart(0,"a",[])',
    'from os import execv as start\nstart("a",[])',
    'f=eval\nf("a")', 'f=exec\nf("a")', 'f=compile\nf("a", "a", "exec")',
    'f=__import__\nf("a")',
    'import importlib as i\ni.import_module("output.a")',
    'import runpy as r\nr.run_path("output/a.py")',
    'import code as c\nc.interact()', 'import codeop as c\nc.compile_command("a")',
    'import multiprocessing as m\nm.Process(target=lambda:None)',
    'import ctypes as c\nc.CDLL("a")',
    'import pickle as p\np.loads(b"a")',
    'from dill import loads as f\nf(b"a")',
    'import cloudpickle as p\np.load(open("a"))',
    'from marshal import loads as f\nf(b"a")',
    'import yaml as y\nf=y.load\nf("a")',
    'from yaml import unsafe_load as read\nread("a")',
    'import numpy as n\nn.load("a", allow_pickle=True)',
    'from numpy import load as read\nread("a", **options)',
    'import pandas as p\nf=p.read_pickle\nf("a")',
    'from pandas import read_pickle as read\nread("a")',
    'import os as o\ngetattr(o,"sys"+"tem")("a")',
    'import output.build_report',
    'from output import build_report',
    'import sys\nsys.path.insert(0,"output")',
])
def test_private_grader_gate_refuses_execution_and_deserialization_aliases(tmp_path, source):
    from hsrl_v5.workspace import checker_boundary_contract
    path=tmp_path / 'test.py';path.write_text(source)
    gate=checker_boundary_contract(path)
    assert gate['status'] == 'pending_unsupported_private_grader_boundary'
    assert gate['source_execution_performed'] is False
    assert gate['reasons'] and gate['evidence']


def test_private_grader_gate_allows_data_parsing_without_dynamic_code(tmp_path):
    from hsrl_v5.workspace import checker_boundary_contract
    path=tmp_path/'test.py'
    path.write_text('import json, re\nimport yaml as y\nimport numpy as np\nfrom pandas import read_csv as read\n'
                    'x=json.loads("{}")\nr=re.compile("a")\ny.safe_load("a")\nnp.load("a", allow_pickle=False)\nread("a")')
    gate=checker_boundary_contract(path)
    assert gate['status']=='readonly_data_checker_static_boundary_pass'
    assert gate['reasons']==[]


@pytest.mark.parametrize('source', [
    'import numpy as np\nf=np.load\nf("a",allow_pickle=True)',
    'import numpy as np\nnp.lib.format.read_array(fp,True)',
    'import jinja2\njinja2.Template(actor_text).render()',
    'import pandas as pd\ndf=pd.DataFrame()\ndf.eval(actor_text)',
    'import operator as op\nop.attrgetter("system")(module)("a")',
    'import os as o\no._execvpe("a",[],{})',
    'import os as o\no=other\no.system("a")',
    'open("output/a","w").write("a")',
    'from pathlib import Path\nPath("output/a").write_text("a")',
    'import os\nos.makedirs("output/a",exist_ok=True)',
])
def test_private_grader_gate_refuses_indirect_code_and_readonly_incompatible_writes(tmp_path, source):
    from hsrl_v5.workspace import checker_boundary_contract
    path=tmp_path/'test.py';path.write_text(source)
    assert checker_boundary_contract(path)['status']=='pending_unsupported_private_grader_boundary'


def test_direct_private_grader_worker_cannot_bypass_checker_boundary(tmp_path, monkeypatch):
    from hsrl_v5.workspace import WorkspaceWorker
    checker=tmp_path/'checker';checker.mkdir();(checker/'test.py').write_text('import subprocess\nsubprocess.run([])')
    def docker_must_not_start(*args, **kwargs):
        raise AssertionError('unsupported grader reached Docker')
    monkeypatch.setattr(WorkspaceWorker,'_docker',docker_must_not_start)
    with pytest.raises(ValueError, match='private_grader_boundary_pending'):
        WorkspaceWorker(tmp_path/'evidence',checker=checker,snapshot=tmp_path)
    assert not (tmp_path/'evidence').exists()


def test_genuine_source_prefilter_retains_pending_checker_evidence_without_execution(tmp_path):
    from hsrl_v5.quality_claw import prefilter_checker_boundaries
    prepared = Path(__file__).resolve().parents[1] / 'outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116'
    if not prepared.exists():
        pytest.skip('fixed genuine HF source corpus is not installed')
    selected = [json.loads(line) for line in (prepared/'accepted.jsonl').open()
                if any('/'+name+'/' in json.loads(line)['private_checker_path'] for name in ('hf_00001','hf_00719'))]
    runnable,pending,audit=prefilter_checker_boundaries(selected,tmp_path)
    assert len(runnable)==1 and len(pending)==1
    assert pending[0]['id']=='odt_filemgr_oc_9559d113ef1e'
    assert pending[0]['source_execution_performed'] is False
    assert pending[0]['utility'] is None
    assert audit['pending_unsupported']==1 and audit['eligible_for_runtime_qc']==1
    assert len(json.loads((tmp_path/'private_grader_boundary_audit.json').read_text()))==2
    assert not list(tmp_path.glob('cases/*/attempt_*'))
    proof=json.loads(next(tmp_path.glob('cases/*/quality.json')).read_text())
    assert proof['failure_class']=='source_private_grader_boundary_pending'
    assert proof['private_grader_boundary']['checker_sha256']=='deff2255f458ae23544766d25b1fd7f3af88c3e2c5ee48a49a6f16055140ac8e'


def test_text_replacement_and_explicit_tmp_scratch_are_data_checker_operations(tmp_path):
    from hsrl_v5.workspace import checker_boundary_contract
    path=tmp_path/'test.py'
    path.write_text('text="a".replace("a","b")\nopen("/tmp/private-grader-scratch", "w").write(text)')
    assert checker_boundary_contract(path)['status']=='readonly_data_checker_static_boundary_pass'


def test_already_satisfied_untouched_task_is_pending_not_useful_rl(tmp_path, monkeypatch):
    from hsrl_v5 import quality_claw as qc
    class UntouchedAlreadySatisfied:
        def __init__(self,*args):
            self.initial_entries={};self.protocol_errors=[]
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def call(self,*args):
            return json.dumps({'returncode':0,'stdout':'{"entries":[],"private_grader_visible":false}'})
        def finish(self):return {'valid':True,'utility':1.0,'metrics':{'official_checker_scores':{'already_satisfied':True}}}
    monkeypatch.setattr(qc,'ClawBackend',UntouchedAlreadySatisfied)
    proof=qc.validate_record({'id':'already-satisfied','family':'family','reward_profile':'code-only'},tmp_path)
    assert proof['ready'] is False
    assert proof['untouched_workspace_grade']==1.0
    assert 'untouched_source_verifier_already_satisfied_pending' in proof['reasons']


@pytest.mark.parametrize('source', [
    'import numpy as np\ndef reader(f):return f("a",allow_pickle=True)\nreader(np.load)',
    'import yaml as y\ny=other\ny.load("a")',
    'from pandas.compat.pickle_compat import load as read\nread(file)',
])
def test_private_grader_gate_refuses_escaped_deserializers_and_namespace_rebinding(tmp_path, source):
    from hsrl_v5.workspace import checker_boundary_contract
    path=tmp_path/'test.py';path.write_text(source)
    assert checker_boundary_contract(path)['status']=='pending_unsupported_private_grader_boundary'


def test_data_variables_named_code_or_query_are_not_execution_capabilities(tmp_path):
    from hsrl_v5.workspace import checker_boundary_contract
    path=tmp_path/'test.py'
    path.write_text('code="report text"\nquery="task text"\nchecks={"code":len(code),"query":len(query)}')
    assert checker_boundary_contract(path)['status']=='readonly_data_checker_static_boundary_pass'


def test_html_parser_initialization_and_url_query_data_are_not_dynamic_execution(tmp_path):
    from hsrl_v5.workspace import checker_boundary_contract
    path=tmp_path/'test.py'
    path.write_text('from html.parser import HTMLParser\nfrom urllib.parse import urlparse\nimport platform\n'
                   'class Reader(HTMLParser):\n def __init__(self):\n  super().__init__()\n  self.name=type(self).__name__\n'
                   'parsed=urlparse("https://example.invalid/?a=1")\nquery=parsed.query\nplatform.system()')
    assert checker_boundary_contract(path)['status']=='readonly_data_checker_static_boundary_pass'


def test_private_projection_replaces_contents_preserves_roots_and_originals(tmp_path):
    from hsrl_v5.grader_pool import replace_projection
    source, target = tmp_path / 'source', tmp_path / 'projection'
    source.mkdir(); target.mkdir()
    (source / 'data.json').write_bytes(b'{"real":true}')
    (target / 'old').write_bytes(b'stale')
    inode = target.stat().st_ino
    expected = {'data.json': {'bytes': 13, 'sha256': None}}
    # Derive the expected digest independently from the fixed bytes below.
    import hashlib
    expected['data.json']['sha256'] = hashlib.sha256(b'{"real":true}').hexdigest()
    assert replace_projection(source, target) == expected
    assert target.stat().st_ino == inode and not (target / 'old').exists()
    assert (source / 'data.json').read_bytes() == b'{"real":true}'
    assert (target / 'data.json').stat().st_ino != (source / 'data.json').stat().st_ino
    (source / 'escape').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='unsupported'):
        replace_projection(source, target)


def test_private_grader_identity_rejects_stale_and_malformed_frames():
    from hsrl_v5.grader_pool import validate_response_identity
    from hsrl_v5.isolation import IsolationError
    good = {'ok': True, 'request_nonce': 'a' * 32,
            'result': {'returncode': 0, 'stdout': '{"a":0}', 'stderr': '', 'child_pid': 12}}
    validate_response_identity(good, 'a' * 32)
    for bad in ({**good, 'request_nonce': 'b' * 32},
                {**good, 'ok': False},
                {**good, 'result': {**good['result'], 'child_pid': True}},
                {**good, 'result': {**good['result'], 'child_pid': 1}}):
        with pytest.raises(IsolationError):
            validate_response_identity(bad, 'a' * 32)


def test_private_scratch_clear_unlinks_symlinks_without_following(tmp_path):
    from hsrl_v5.workspace_worker import clear_grader_scratch
    scratch, outside = tmp_path / 'scratch', tmp_path / 'outside'
    scratch.mkdir(); outside.mkdir(); (outside / 'keep').write_text('keep')
    (scratch / 'nested').mkdir(); (scratch / 'nested' / 'cached').write_text('old')
    (scratch / 'link').symlink_to(outside, target_is_directory=True)
    clear_grader_scratch(scratch)
    assert list(scratch.iterdir()) == [] and (outside / 'keep').read_text() == 'keep'


@pytest.fixture
def private_pool_transport(monkeypatch):
    """Replace only forbidden Docker transport; projection/gates/lifecycle stay real."""
    from hsrl_v5.workspace import WorkspaceWorker
    from hsrl_v5.io import atomic_json
    started, calls, stopped = [], [], []
    def start(worker, evidence_dir, *, snapshot, checker):
        worker.evidence = Path(evidence_dir); worker.evidence.mkdir(parents=True)
        worker.name = 'unit-private-' + str(len(started))
        worker.process = object()
        worker.snapshot, worker.checker = Path(snapshot), Path(checker)
        atomic_json(worker.evidence / 'isolation.json', {'container': worker.name, 'all_host_mounts': 'readonly'})
        started.append(worker)
    def request(worker, value, *, timeout):
        calls.append((worker.name, dict(value)))
        return {'ok': True, 'request_nonce': value['request_nonce'],
                'result': {'returncode': 0, 'stdout': '{"score":0}', 'stderr': '', 'child_pid': len(calls) + 2}}
    def close(worker):
        stopped.append(worker.name)
        atomic_json(worker.evidence / 'cleanup.json', {'container': worker.name, 'removal_confirmed': True})
    monkeypatch.setattr(WorkspaceWorker, '__init__', start)
    monkeypatch.setattr(WorkspaceWorker, 'request', request)
    monkeypatch.setattr(WorkspaceWorker, 'close', close)
    return started, calls, stopped


def test_private_pool_reuses_only_grader_recycles_at_64_cases_and_preserves_evidence(tmp_path, private_pool_transport):
    from hsrl_v5.grader_pool import PrivateGraderPool
    from hsrl_v5.workspace import snapshot_entries
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir(); snapshot.mkdir()
    (source / 'test.py').write_text('print("{\\"score\\":0}")')
    (snapshot / 'data').write_bytes(b'original')
    pool = PrivateGraderPool(tmp_path / 'pool')
    entries = snapshot_entries(snapshot)
    proofs = []
    for index in range(65):
        response, proof = pool.grade(snapshot=snapshot, checker=source, entries=entries,
            evidence_dir=tmp_path / 'cases' / str(index), case_id=str(index))
        proofs.append(proof)
    pool.close()
    started, calls, stopped = private_pool_transport
    assert len(started) == 2 and len(stopped) == 2
    assert len({request['request_nonce'] for _, request in calls}) == 65
    assert proofs[0]['container'] == proofs[63]['container'] != proofs[64]['container']
    assert proofs[0]['bind_root_inodes'] == proofs[63]['bind_root_inodes']
    assert (tmp_path / 'cases/0/source_snapshot/test.py').read_bytes() == (source / 'test.py').read_bytes()
    assert json.loads((tmp_path / 'cases/0/cleanup.json').read_text())['removal_confirmed']
    assert not list((tmp_path / 'pool').glob('generation_*/projection'))
    assert (snapshot / 'data').read_bytes() == b'original'


def test_private_pool_stale_nonce_poison_prevents_next_source_dispatch(tmp_path, private_pool_transport, monkeypatch):
    from hsrl_v5.grader_pool import PrivateGraderPool
    from hsrl_v5.workspace import WorkspaceWorker
    from hsrl_v5.isolation import IsolationError
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir(); snapshot.mkdir(); (source / 'test.py').write_text('print(0)')
    def stale(worker, request, *, timeout):
        return {'ok': True, 'request_nonce': '0' * 32,
                'result': {'returncode': 0, 'stdout': '0', 'stderr': '', 'child_pid': 4}}
    monkeypatch.setattr(WorkspaceWorker, 'request', stale)
    pool = PrivateGraderPool(tmp_path / 'pool')
    kwargs = dict(snapshot=snapshot, checker=source, entries={}, case_id='one')
    with pytest.raises(IsolationError, match='identity_or_contract'):
        pool.grade(evidence_dir=tmp_path / 'case1', **kwargs)
    with pytest.raises(IsolationError, match='poisoned'):
        pool.grade(evidence_dir=tmp_path / 'case2', **kwargs)
    assert not (tmp_path / 'case2').exists()
    assert len(private_pool_transport[2]) == 1
    pool.close()


def test_pool_checker_gate_cannot_be_bypassed_by_reuse(tmp_path, private_pool_transport):
    from hsrl_v5.grader_pool import PrivateGraderPool
    from hsrl_v5.isolation import IsolationError
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir(); snapshot.mkdir(); (source / 'test.py').write_text('import subprocess\nsubprocess.run([])')
    pool = PrivateGraderPool(tmp_path / 'pool')
    with pytest.raises(IsolationError, match='boundary_pending'):
        pool.grade(snapshot=snapshot, checker=source, entries={}, case_id='unsafe', evidence_dir=tmp_path / 'case')
    assert not private_pool_transport[0]
    pool.close()


def test_private_pool_detects_changed_snapshot_after_grade_and_preserves_cleanup(tmp_path, private_pool_transport, monkeypatch):
    from hsrl_v5.grader_pool import PrivateGraderPool
    from hsrl_v5.workspace import WorkspaceWorker, snapshot_entries
    from hsrl_v5.isolation import IsolationError
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir(); snapshot.mkdir(); (source / 'test.py').write_text('print(0)')
    (snapshot / 'data').write_bytes(b'frozen')
    original_request = WorkspaceWorker.request
    def corrupt_projection(worker, request, *, timeout):
        response = original_request(worker, request, timeout=timeout)
        (worker.snapshot / 'data').write_bytes(b'changed')
        return response
    monkeypatch.setattr(WorkspaceWorker, 'request', corrupt_projection)
    pool = PrivateGraderPool(tmp_path / 'pool')
    with pytest.raises(IsolationError, match='original_or_projection_changed'):
        pool.grade(snapshot=snapshot, checker=source, entries=snapshot_entries(snapshot),
                   case_id='one', evidence_dir=tmp_path / 'case')
    assert pool.poisoned and (snapshot / 'data').read_bytes() == b'frozen'
    assert json.loads((tmp_path / 'case/cleanup.json').read_text())['removal_confirmed']
    pool.close()


def test_private_pool_uncertain_cleanup_keeps_projection_and_records_failure(tmp_path, private_pool_transport, monkeypatch):
    from hsrl_v5.grader_pool import LaneGraderPools
    from hsrl_v5.workspace import WorkspaceWorker
    from hsrl_v5.isolation import CleanupUncertain
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir(); snapshot.mkdir(); (source / 'test.py').write_text('print(0)')
    lanes = LaneGraderPools(tmp_path / 'pools')
    pool = lanes.get()
    pool.grade(snapshot=snapshot, checker=source, entries={}, case_id='one', evidence_dir=tmp_path / 'case')
    def uncertain(worker):
        raise CleanupUncertain('unit-removal-unconfirmed')
    monkeypatch.setattr(WorkspaceWorker, 'close', uncertain)
    with pytest.raises(CleanupUncertain, match='lane_cleanup_unconfirmed'):
        lanes.close()
    assert pool.projection.is_dir()
    assert not (tmp_path / 'case/cleanup.json').exists()
    cleanup = json.loads((tmp_path / 'pools/cleanup.json').read_text())
    assert cleanup['all_owned_containers_removal_confirmed'] is False and cleanup['errors']


def test_private_projection_retains_actual_risk_metrics_checker_bytes_without_executing(tmp_path):
    from hsrl_v5.grader_pool import replace_projection
    from hsrl_v5.workspace import checker_boundary_contract
    from hsrl_v5.io import file_hash
    prepared = Path(__file__).resolve().parents[1] / 'outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116'
    records = map(json.loads, (prepared / 'accepted.jsonl').open())
    record = next(record for record in records if 'risk_metrics_calculation' in record['id'])
    checker = Path(record['private_checker_path'])
    projection = tmp_path / 'source'; projection.mkdir()
    manifest = replace_projection(checker.parent, projection)
    assert manifest['test.py']['sha256'] == record['original_asset_sha256']['reward/test.py']
    assert file_hash(projection / 'test.py') == file_hash(checker)
    assert checker_boundary_contract(projection / 'test.py')['status'] == 'readonly_data_checker_static_boundary_pass'


def test_private_grader_child_identity_is_actual_os_pid_and_every_request_is_fresh(tmp_path, monkeypatch):
    from hsrl_v5 import workspace_worker as worker
    monkeypatch.setattr(worker, 'PUBLIC', tmp_path)
    results = [worker.bounded_command([sys.executable, '-c', 'import os;print(os.getpid())'], grader=True)
               for _ in range(2)]
    assert all(result['returncode'] == 0 and int(result['stdout']) == result['child_pid'] for result in results)
    assert results[0]['child_pid'] != results[1]['child_pid']


def test_projection_rejects_oversize_before_mutating_existing_projection(tmp_path, monkeypatch):
    from hsrl_v5.grader_pool import replace_projection
    from hsrl_v5 import workspace_worker as worker
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(); target.mkdir()
    (source / 'large').write_bytes(b'123'); (target / 'keep').write_bytes(b'keep')
    monkeypatch.setattr(worker, 'MAX_FILE', 2)
    with pytest.raises(ValueError, match='snapshot_limit'):
        replace_projection(source, target)
    assert (target / 'keep').read_bytes() == b'keep'


def test_private_pool_projects_new_case_bytes_and_removes_previous_case_files(tmp_path, private_pool_transport, monkeypatch):
    from hsrl_v5.grader_pool import PrivateGraderPool
    from hsrl_v5.workspace import WorkspaceWorker, snapshot_entries
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir(); snapshot.mkdir(); (source / 'test.py').write_text('print(0)')
    (source / 'old_reference').write_bytes(b'private case one')
    (snapshot / 'old_output').write_bytes(b'case one')
    original_request = WorkspaceWorker.request
    def observe_projected_bytes(worker, request, *, timeout):
        response = original_request(worker, request, timeout=timeout)
        response['result']['stdout'] = json.dumps({'workspace': sorted(p.name for p in worker.snapshot.iterdir()),
            'grader': sorted(p.name for p in worker.checker.iterdir())})
        return response
    monkeypatch.setattr(WorkspaceWorker, 'request', observe_projected_bytes)
    pool = PrivateGraderPool(tmp_path / 'pool')
    first, _ = pool.grade(snapshot=snapshot, checker=source, entries=snapshot_entries(snapshot),
                          case_id='one', evidence_dir=tmp_path / 'first')
    (source / 'old_reference').unlink(); (source / 'test.py').write_text('print(1)')
    (snapshot / 'old_output').unlink(); (snapshot / 'new_output').write_bytes(b'case two')
    second, _ = pool.grade(snapshot=snapshot, checker=source, entries=snapshot_entries(snapshot),
                           case_id='two', evidence_dir=tmp_path / 'second')
    assert json.loads(first['result']['stdout']) == {'workspace': ['old_output'], 'grader': ['old_reference', 'test.py']}
    assert json.loads(second['result']['stdout']) == {'workspace': ['new_output'], 'grader': ['test.py']}
    assert (tmp_path / 'first/source_snapshot/old_reference').read_bytes() == b'private case one'
    assert (tmp_path / 'first/source_snapshot/test.py').read_text() == 'print(0)'
    pool.close()


def test_lane_cleanup_attempts_every_owned_pool_after_one_removal_failure(tmp_path, private_pool_transport, monkeypatch):
    from hsrl_v5.grader_pool import LaneGraderPools
    from hsrl_v5.workspace import WorkspaceWorker
    from hsrl_v5.isolation import CleanupUncertain
    import threading
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir(); snapshot.mkdir(); (source / 'test.py').write_text('print(0)')
    lanes = LaneGraderPools(tmp_path / 'pools')
    failures = []
    def create(index):
        try:
            lanes.get().grade(snapshot=snapshot, checker=source, entries={}, case_id=str(index),
                              evidence_dir=tmp_path / f'case{index}')
        except BaseException as exc:
            failures.append(exc)
    for index in range(2):
        thread = threading.Thread(target=create, args=(index,)); thread.start(); thread.join()
    assert not failures and len(lanes.pools) == 2
    original_close = WorkspaceWorker.close
    def one_uncertain(worker):
        if worker.name == 'unit-private-0':
            raise CleanupUncertain('unit-removal-unconfirmed')
        original_close(worker)
    monkeypatch.setattr(WorkspaceWorker, 'close', one_uncertain)
    with pytest.raises(CleanupUncertain):
        lanes.close()
    assert lanes.pools[0].projection.exists() and not lanes.pools[1].projection.exists()
    assert json.loads((tmp_path / 'case1/cleanup.json').read_text())['removal_confirmed']


def test_quality_resume_refuses_unconfirmed_previous_pool_cleanup_before_source_access(tmp_path):
    from hsrl_v5.quality_claw import run
    from hsrl_v5.isolation import IsolationError
    pool_directory = tmp_path / 'out/private_grader_pools/dispatch_old'
    pool_directory.mkdir(parents=True)
    (pool_directory / 'cleanup.json').write_text('{"all_owned_containers_removal_confirmed":false}')
    with pytest.raises(IsolationError, match='previous_private_grader_pool_cleanup_unconfirmed'):
        run(tmp_path / 'missing_source', tmp_path / 'out')


@pytest.fixture
def lifecycle_worker(tmp_path):
    from hsrl_v5.workspace import WorkspaceWorker
    worker = WorkspaceWorker.__new__(WorkspaceWorker)
    worker.name, worker.env = 'unit-owned-uuid', {}
    worker.evidence = tmp_path / 'worker'
    worker.evidence.mkdir()
    worker.closed, worker.buffer = False, bytearray()
    return worker


def test_workspace_create_completes_before_attach_and_retains_commands(lifecycle_worker, monkeypatch):
    import io
    import subprocess
    from hsrl_v5 import workspace
    worker, events = lifecycle_worker, []
    def run(command, **options):
        events.append(('create', command))
        assert command == ['docker', 'create', '--rm', '-i', '--name', worker.name, 'image', 'python3']
        return subprocess.CompletedProcess(command, 0, 'a' * 64 + '\n', '')
    class Client:
        stdin, stdout = io.BytesIO(), io.BytesIO()
    def popen(command, **options):
        events.append(('attach', command))
        assert (worker.evidence / 'create.json').exists()
        return Client()
    monkeypatch.setattr(workspace.subprocess, 'run', run)
    monkeypatch.setattr(workspace.subprocess, 'Popen', popen)
    class Selector:
        def register(self, *args): pass
    monkeypatch.setattr(workspace.selectors, 'DefaultSelector', Selector)
    monkeypatch.setattr(workspace.os, 'set_blocking', lambda *args: None)
    # BytesIO has no fd: only the transport boundary is replaced.
    Client.stdin = type('Input', (), {'fileno': lambda self: 0})()
    worker._launch(['docker', 'run', '--rm', '-i', '--name', worker.name, 'image', 'python3'])
    assert events[1] == ('attach', ['docker', 'start', '--attach', '--interactive', worker.name])
    state = json.loads((worker.evidence / 'lifecycle.json').read_text())
    assert state['creation_state'] == 'confirmed' and state['container_id'] == 'a' * 64
    assert json.loads((worker.evidence / 'create.json').read_text())['duration_seconds'] >= 0
    worker.stderr.close()


def test_workspace_unknown_create_cannot_confirm_absence(lifecycle_worker, monkeypatch):
    import subprocess
    from hsrl_v5 import workspace
    from hsrl_v5.isolation import CleanupUncertain
    worker = lifecycle_worker
    def timeout(command, **options):
        raise subprocess.TimeoutExpired(command, options['timeout'])
    monkeypatch.setattr(workspace.subprocess, 'run', timeout)
    monkeypatch.setattr(worker, '_docker', lambda command: subprocess.CompletedProcess(command, 1, '', 'No such container'))
    with pytest.raises(CleanupUncertain, match='creation_outcome_unknown'):
        worker._launch(['docker', 'run', '--name', worker.name, 'image'])
    with pytest.raises(CleanupUncertain, match='creation_outcome_unknown'):
        worker.close()
    cleanup = json.loads((worker.evidence / 'cleanup.json').read_text())
    assert cleanup['removal_confirmed'] is False
    assert json.loads((worker.evidence / 'create.json').read_text())['outcome'] == 'unknown'
    assert worker.closed is False


def test_workspace_cleanup_settles_attach_before_terminal_absence(lifecycle_worker, monkeypatch):
    import subprocess
    from hsrl_v5.io import atomic_json
    worker, events = lifecycle_worker, []
    worker.creation_state = 'confirmed'
    worker.container_id = 'a' * 64
    class Client:
        stdin = stdout = None
        def poll(self): return None
        def kill(self): events.append('kill')
        def wait(self, timeout): events.append('wait'); return -9
    worker.process = Client()
    worker.stderr = worker.selector = None
    def docker(command):
        events.append(command[0])
        return subprocess.CompletedProcess(command, 1, '', 'No such container')
    monkeypatch.setattr(worker, '_docker', docker)
    worker.close()
    assert events.index('wait') < events.index('container')
    cleanup = json.loads((worker.evidence / 'cleanup.json').read_text())
    assert cleanup['removal_confirmed'] is True and cleanup['creation_state'] == 'confirmed'
    assert cleanup['container'] == worker.name


def test_quality_resume_refuses_unknown_actor_creation_before_source_access(tmp_path):
    from hsrl_v5.quality_claw import run
    from hsrl_v5.isolation import IsolationError
    evidence = tmp_path / 'out/cases/case/attempt_old/actual_episode/tool_worker'
    evidence.mkdir(parents=True)
    (evidence / 'lifecycle.json').write_text('{"container":"owned","creation_state":"unknown"}')
    (evidence / 'cleanup.json').write_text('{"container":"owned","removal_confirmed":false}')
    with pytest.raises(IsolationError, match='previous_workspace_cleanup_unconfirmed'):
        run(tmp_path / 'missing_source', tmp_path / 'out')


def test_workspace_explicit_socket_must_stay_in_owned_runtime(tmp_path, monkeypatch):
    from hsrl_v5.workspace import WorkspaceWorker
    from hsrl_v5.isolation import IsolationError
    monkeypatch.setenv('HSRL_WORKSPACE_DOCKER_SOCKET', str(tmp_path / 'docker.sock'))
    with pytest.raises(IsolationError, match='socket_outside_owned_runtime'):
        WorkspaceWorker(tmp_path / 'worker', assets=tmp_path)


def test_quality_only_retries_deadline_after_confirmed_owned_cleanup(tmp_path, monkeypatch):
    from hsrl_v5 import quality_claw as qc
    from hsrl_v5.io import atomic_json
    calls = []
    def once(record, directory, *, grader_pool=None):
        attempt = directory / f'attempt_{len(calls)}'
        calls.append(attempt)
        worker = attempt / 'actual_episode/tool_worker'
        atomic_json(worker / 'lifecycle.json', {'creation_state': 'confirmed', 'container': str(attempt)})
        atomic_json(worker / 'cleanup.json', {'removal_confirmed': True, 'container': str(attempt)})
        return {'id': record['id'], 'ready': len(calls) == 2, 'attempt_directory': str(attempt),
                'reasons': ['IsolationError:source_worker_deadline'] if len(calls) == 1 else [],
                'failure_class': 'isolation_or_runtime_infrastructure' if len(calls) == 1 else None}
    monkeypatch.setattr(qc, '_validate_record_once', once)
    result = qc.validate_record({'id': 'real'}, tmp_path)
    assert result['ready'] and len(calls) == 2
    assert result['runtime_admission_recovery']['attempts'] == 2
    assert calls[0] != calls[1]


@pytest.mark.parametrize('reason,confirmed,poisoned', [
    ('CleanupUncertain:workspace_creation_outcome_unknown', True, False),
    ('IsolationError:source_worker_deadline', False, False),
    ('IsolationError:source_worker_deadline', True, True),
    ('IsolationError:initial_public_asset_hash_mismatch', True, False),
    ('ValueError:private_original_checker_failed', True, False),
])
def test_quality_never_retries_uncertain_cleanup_source_or_identity_errors(tmp_path, monkeypatch, reason, confirmed, poisoned):
    from hsrl_v5 import quality_claw as qc
    from hsrl_v5.io import atomic_json
    calls = []
    def once(record, directory, *, grader_pool=None):
        calls.append(record)
        worker = directory / 'attempt_0/actual_episode/tool_worker'
        atomic_json(worker / 'lifecycle.json', {'creation_state': 'confirmed', 'container': 'owned'})
        atomic_json(worker / 'cleanup.json', {'removal_confirmed': confirmed, 'container': 'owned'})
        return {'ready': False, 'attempt_directory': str(worker.parents[1]), 'reasons': [reason],
                'failure_class': 'isolation_or_runtime_infrastructure'}
    monkeypatch.setattr(qc, '_validate_record_once', once)
    pool = type('Pool', (), {'poisoned': poisoned, 'closed': False})()
    result = qc.validate_record({'id': 'real'}, tmp_path, grader_pool=pool)
    assert not result['ready'] and len(calls) == 1


def test_workspace_selected_socket_is_recorded_and_host_environment_is_not_forwarded(tmp_path, monkeypatch):
    import os
    import subprocess
    from types import SimpleNamespace
    from hsrl_v5 import workspace
    socket = Path(f'/run/user/{os.getuid()}/hsrl-private-test/docker.sock')
    original_stat, original_socket = Path.stat, Path.is_socket
    monkeypatch.setattr(Path, 'is_socket', lambda path: True if path == socket else original_socket(path))
    monkeypatch.setattr(Path, 'stat', lambda path, *args, **kwargs:
                        SimpleNamespace(st_uid=os.getuid()) if path == socket else original_stat(path, *args, **kwargs))
    monkeypatch.setenv('HSRL_WORKSPACE_DOCKER_SOCKET', str(socket))
    monkeypatch.setenv('PRIVATE_SECRET_TEST', 'never-forward')
    def docker(worker, command):
        assert worker.env['DOCKER_HOST'] == 'unix://' + str(socket)
        assert 'PRIVATE_SECRET_TEST' not in worker.env
        return subprocess.CompletedProcess(command, 0,
            '["name=rootless"]' if command[0] == 'info' else workspace.IMAGE + '\n', '')
    def launch(worker, command):
        worker.creation_state = 'confirmed'
        assert command[1] == 'run' and '--network=none' in command
        assert '--memory=1g' in command and '--pids-limit=64' in command
        assert not any('PRIVATE_SECRET_TEST' in value for value in command)
    monkeypatch.setattr(workspace.WorkspaceWorker, '_docker', docker)
    monkeypatch.setattr(workspace.WorkspaceWorker, '_launch', launch)
    monkeypatch.setattr(workspace.WorkspaceWorker, 'read', lambda *args, **kwargs: {'ready': True, 'mode': 'file_tools'})
    worker = workspace.WorkspaceWorker(tmp_path / 'worker', assets=tmp_path)
    isolation = json.loads((worker.evidence / 'isolation.json').read_text())
    assert isolation['docker_socket'] == str(socket)
    assert isolation['mount_targets'] == ['/runner/worker.py', '/inputs']
    assert json.loads((worker.evidence / 'lifecycle.json').read_text())['ready_wait_seconds'] >= 0
