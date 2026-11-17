"""Regression of the actual retained failed official source call; no model run."""
import ast
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import textwrap

import pytest

EVIDENCE = Path(os.environ.get('HSRL_AGENTDOG_FAILURE_EVIDENCE',
    '/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/'
    'hsrl_hourly_monitor_20261117/audits/20261118_0600_19d7f87f-410c-4fa7-a8a3-40b018420843'))


def actual_payload():
    from hsrl_v5.agentdog_worker import execution_payload
    record = json.loads((EVIDENCE / 'failing_record.json').read_text())
    bundle = json.loads((EVIDENCE / 'failing_source_bundle.json').read_text())
    failed = json.loads((EVIDENCE / 'known_failed_call.json').read_text())
    return execution_payload(bundle, record['scenario']), failed


def selected_worker():
    return importlib.import_module(os.environ.get('HSRL_VALUEERROR_WORKER_MODULE',
        'hsrl_source_repair_v1.agentdog_worker'))


def test_actual_official_patient_lookup_handles_nonnumeric_record_fields():
    payload, failed = actual_payload()
    worker = selected_worker()
    episode = worker.ToolEpisode(payload)
    before = copy.deepcopy(episode.state)
    result = episode.call(failed['tool_name'], failed['arguments'])
    assert isinstance(result['tool_result']['records'], list)
    assert isinstance(result['observation'], str)
    assert result['state'] == before == episode.state


def test_worker_change_is_only_referenced_standard_exception_names():
    from hsrl_v5 import agentdog_worker as original
    repaired = selected_worker()
    assert set(repaired.SAFE_BUILTINS) == set(original.SAFE_BUILTINS) | {'ValueError', 'TypeError', 'Exception'}
    assert repaired.SAFE_BUILTINS['ValueError'] is ValueError
    assert repaired.SAFE_BUILTINS['TypeError'] is TypeError
    assert repaired.SAFE_BUILTINS['Exception'] is Exception
    before = ast.parse(Path(original.__file__).read_text())
    after = ast.parse(Path(repaired.__file__).read_text())
    # The only AST edit is extending the SAFE_BUILTINS tuple.
    for left, right in zip(before.body, after.body):
        if isinstance(left, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SAFE_BUILTINS' for t in left.targets):
            right.value.generators[0].iter = left.value.generators[0].iter
    assert ast.dump(before) == ast.dump(after)
    assert repaired.GRADER_SHA256 == original.GRADER_SHA256


def test_all_exception_names_referenced_by_the_real_admitted_corpus_are_supported():
    scan = json.loads((EVIDENCE / 'source_exception_reference_scan.json').read_text())
    assert set(scan['exception_reference_program_counts']) <= set(selected_worker().SAFE_BUILTINS)


@pytest.mark.parametrize('suffix', ['\nimport os\n', '\ndef evil():\n    open("x")\n',
    '\ndef evil():\n    return ().__class__.__mro__\n'])
def test_original_source_security_exclusions_remain_enforced(suffix):
    worker = selected_worker()
    payload, failed = actual_payload()
    source = payload['programs'][failed['tool_name']]['source']
    with pytest.raises(ValueError):
        worker.validate_source(source + suffix)


def test_worker_and_backend_constructor_bodies_keep_frozen_gates():
    from hsrl_source_repair_v1.adapter import JSONWorker, AgentDogBackend
    from hsrl_v5.isolation import JSONWorker as OldWorker
    from hsrl_v5.agentdog_backend import AgentDogBackend as OldBackend
    for old, new in [(OldWorker, JSONWorker), (OldBackend, AgentDogBackend)]:
        assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(old.__init__)))) == ast.dump(
            ast.parse(textwrap.dedent(inspect.getsource(new.__init__))))
    assert AgentDogBackend.call is OldBackend.call
    assert AgentDogBackend.grade is OldBackend.grade
    assert AgentDogBackend.close is OldBackend.close
    assert JSONWorker.close is OldWorker.close
