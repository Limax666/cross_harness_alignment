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
        run(tmp_path / 'missing', tmp_path / 'out', 5)


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
