"""Standalone JSON worker, run only inside a project-owned rootless container.

Tool programs receive state/context but no grading rules. A separate grader
process loads the exact pinned source evaluator class and never runs tool code.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import os
import subprocess
import selectors
import time

SAFE_BUILTINS = {name: getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
                 for name in ('abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'float', 'isinstance',
                              'int', 'len', 'list', 'max', 'min', 'next', 'range', 'round', 'set',
                              'sorted', 'str', 'sum', 'tuple', 'zip', 'ValueError', 'TypeError', 'Exception')}
FORBIDDEN_CALLS = {'open', 'exec', 'eval', '__import__', 'compile', 'input'}
FORBIDDEN_ATTRS = {'system', 'popen', 'run', 'unlink', 'rmdir', 'mkdir', 'makedirs'}
GRADER_SHA256 = '0f08a890ab7aaadc88f526ab997fd438c07c6dc10c5e30f4b73c8549af1f56e2'


class ToolContractError(RuntimeError):
    """A broken source/runtime boundary, never an ordinary invalid action."""


def validate_source(source, entrypoint='execute'):
    if not isinstance(source, str) or len(source) > 12000:
        raise ValueError('tool_source_size_contract')
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    has_metadata = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError('source_import_forbidden')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError('source_unsafe_call')
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRS:
                raise ValueError('source_unsafe_attribute')
        if isinstance(node, ast.Attribute) and node.attr.startswith('__') and node.attr not in {'__class__', '__name__'}:
            raise ValueError('source_dunder_forbidden')
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'TOOL_METADATA' for target in node.targets):
            has_metadata = True
    if entrypoint not in names or not has_metadata:
        raise ValueError('missing_source_entrypoint_or_metadata')


def materialize_scenario(bundle, scenario):
    base = copy.deepcopy(bundle['execution_bundle']['initial_state_template'])
    selected = copy.deepcopy(bundle['execution_bundle']['scenarios'][scenario])
    if scenario == 'clean':
        selected['initial_state_template'] = base
    else:
        def merge(left, right):
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(left.get(key), dict):
                    merge(left[key], value)
                else:
                    left[key] = copy.deepcopy(value)
            return left
        selected['initial_state_template'] = merge(base, selected.get('initial_state_template') or {})
    return selected


def execution_payload(bundle, scenario):
    material = materialize_scenario(bundle, scenario)
    execution = bundle['execution_bundle']
    programs = {}
    for name, dispatch in bundle['server_adapter_manifest']['tool_dispatch_table'].items():
        code = execution['tool_impl_sources'][dispatch['source_key']]
        validate_source(code, dispatch['entrypoint'])
        programs[name] = {'source': code, 'entrypoint': dispatch['entrypoint']}
    risk_config = copy.deepcopy(bundle['task_spec'].get('risk_config', {}))
    risk_spec = copy.deepcopy(bundle['task_spec'].get('risk_spec', {}))
    if not material.get('risk_enabled'):
        risk_config['enabled'] = False
        risk_spec['enabled'] = False
    return {'programs': programs, 'state': material['initial_state_template'],
            'context': {'task_id': bundle['task_spec']['task_id'], 'scenario': material['scenario'],
                        'task_metadata': copy.deepcopy(bundle['task_spec']['task_metadata']),
                        'risk_config': risk_config, 'risk_spec': risk_spec,
                        'planner_trace': copy.deepcopy(bundle['task_spec']['planner_trace']),
                        'attack_materialization': {}}}


class ToolEpisode:
    def __init__(self, payload):
        self.state = copy.deepcopy(payload['state'])
        self.context = copy.deepcopy(payload['context'])
        self.functions = {}
        for name, program in payload['programs'].items():
            validate_source(program['source'], program['entrypoint'])
            namespace = {'__builtins__': SAFE_BUILTINS}
            exec(compile(program['source'], '<frozen-dataset-tool>', 'exec'), namespace, namespace)
            self.functions[name] = namespace[program['entrypoint']]

    def call(self, name, arguments):
        if name not in self.functions:
            raise ToolContractError('unknown_canonical_tool')
        context = copy.deepcopy(self.context)
        context['tool_name'] = name
        # Like the pinned source server, an exception cannot commit a partially
        # modified copy into the authoritative episode state.
        result = self.functions[name](copy.deepcopy(arguments), copy.deepcopy(self.state), context)
        if not isinstance(result, dict) or not isinstance(result.get('tool_result'), dict) or not isinstance(result.get('observation'), str) or not isinstance(result.get('state'), dict):
            raise ToolContractError('source_return_contract_invalid')
        # Validate JSON before committing; nonfinite/unserializable state is not a
        # successful source execution and cannot become a numerical reward zero.
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolContractError('source_return_not_finite_json') from exc
        self.state = copy.deepcopy(result['state'])
        return result


def load_grader(path):
    with open(path, 'rb') as stream:
        code = stream.read()
    if hashlib.sha256(code).hexdigest() != GRADER_SHA256:
        raise ValueError('pinned_source_grader_hash_mismatch')
    tree = ast.parse(code)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RuntimeRuleEvaluator']
    if len(classes) != 1:
        raise ValueError('pinned_source_grader_class_mismatch')
    # Only the exact trusted evaluator class is compiled; source reward_health,
    # server endpoints, fallback rewards and generated tool implementations are
    # not imported or executed in this private grader process.
    namespace = {'copy': copy, 'ast': ast, 'json': json}
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<pinned-source-rule-evaluator>', 'exec'), namespace)
    return namespace['RuntimeRuleEvaluator']()


def _write(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')), flush=True)


def supervisor():
    """Fresh source OS process per episode, without per-episode Docker churn.

    Only trusted framing runs here. Tool code never executes in this process;
    it receives no grader/rule mounts, and cannot reinitialize its own child.
    """
    child = None
    def close_child():
        nonlocal child
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()
            child = None
    def read_child(timeout=5):
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        buffer = bytearray()
        deadline = time.monotonic() + timeout
        try:
            while b'\n' not in buffer:
                if time.monotonic() >= deadline:
                    raise RuntimeError('source_child_deadline')
                for key, _ in selector.select(.1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        raise RuntimeError('source_child_closed')
                    buffer.extend(block)
                    if len(buffer) > 8 * 1024 * 1024:
                        raise RuntimeError('source_child_response_limit')
            if buffer.count(b'\n') != 1:
                raise RuntimeError('source_child_frame_contract')
            return json.loads(buffer)
        finally:
            selector.close()
    _write({'ready': True, 'mode': 'tool_supervisor'})
    try:
        for line in sys.stdin:
            try:
                if len(line.encode()) > 8 * 1024 * 1024:
                    raise ValueError('supervisor_request_limit')
                request = json.loads(line)
                if request['op'] == 'end_episode':
                    close_child()
                    _write({'ok': True})
                    continue
                if request['op'] == 'init':
                    if child is not None:
                        raise RuntimeError('previous_source_process_not_closed')
                    child = subprocess.Popen([sys.executable, '-u', __file__, '--child'], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=sys.stderr,
                                             env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
                                                  'PYTHONHASHSEED': '0', 'PYTHONUNBUFFERED': '1'},
                                             start_new_session=True)
                    if read_child() != {'ready': True, 'mode': 'tools'}:
                        raise RuntimeError('source_child_start_contract')
                if child is None:
                    raise RuntimeError('source_episode_not_initialized')
                child.stdin.write(line.encode())
                child.stdin.flush()
                response = read_child()
                if request['op'] == 'init':
                    response['source_process_id'] = child.pid
                _write(response)
            except Exception as exc:
                close_child()
                _write({'ok': False, 'error_type': 'ToolContractError', 'error': str(exc)[:1000]})
    finally:
        close_child()


def main():
    if '--supervisor' in sys.argv:
        supervisor()
        return
    grader = load_grader('/runner/source_evaluator.py') if '--grader' in sys.argv else None
    episode = None
    _write({'ready': True, 'mode': 'grader' if grader else 'tools'})
    for line in sys.stdin:
        try:
            if len(line.encode()) > 8 * 1024 * 1024:
                raise ValueError('worker_request_size_limit')
            request = json.loads(line)
            if grader:
                value = request['episode']
                utility = grader.evaluate_rule(value, value['success_rule'])
                registered_risk = grader.evaluate_rule(value, value['risk_success_rule'])
                _write({'ok': True, 'source_utility': bool(utility),
                        'registered_call_goal': bool(registered_risk),
                        'grader_sha256': GRADER_SHA256})
            elif request['op'] == 'init':
                if episode is not None:
                    raise ValueError('worker_episode_cannot_be_reinitialized')
                episode = ToolEpisode(request['payload'])
                _write({'ok': True, 'state': episode.state})
            elif request['op'] == 'call':
                if episode is None:
                    raise ValueError('worker_not_initialized')
                result = episode.call(request['name'], request['arguments'])
                _write({'ok': True, 'result': result})
            elif request['op'] == 'snapshot':
                _write({'ok': True, 'state': episode.state})
            else:
                raise ValueError('unknown_worker_operation')
        except Exception as exc:
            _write({'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)[:2000]})


if __name__ == '__main__':
    main()
