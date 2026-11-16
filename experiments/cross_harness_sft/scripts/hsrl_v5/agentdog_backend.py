"""Actual source state tools plus a separate private source grader.

The default deterministic profile measures registered unsafe calls. It does not
claim the source's broad LLM-judge safety or actual harm/persistence outcomes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema

from .agentdog_source import public_contract, source_contract
from .agentdog_worker import execution_payload, materialize_scenario
from .io import append_jsonl, atomic_json, digest, file_hash
from .isolation import JSONWorker, IsolationError

GRADER_PATH = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/vendor/agentdog_runtime_v5_c8d803f/runtime_evaluators.py')
PROFILE = 'hsrl-v5-agentdog-registered-call-goal-v1'


class SourceExecutionError(RuntimeError):
    pass


class AgentDogBackend:
    def __init__(self, record, evidence_dir, *, grader_path=GRADER_PATH, tool_worker=None, grader_worker=None):
        self.record, self.evidence = record, Path(evidence_dir)
        self.evidence.mkdir(parents=True, exist_ok=False)
        source = Path(record['source_path'])
        if file_hash(source) != record['source_sha256']:
            raise ValueError('source_asset_bytes_changed')
        self.bundle = json.loads(source.read_text())
        if digest(self.bundle) != record['source_record_canonical_sha256']:
            raise ValueError('source_bundle_identity_changed')
        paired = {'task_id': record['id'], 'catalog_task_type': record['condition'], 'scenario': record['scenario']}
        checked = source_contract(self.bundle, paired)
        if not checked['qualified'] or not record['qualified']:
            raise ValueError('source_not_structurally_admitted:' + ','.join(checked['exclusion_reasons']))
        self.scenario = materialize_scenario(self.bundle, record['scenario'])
        if self.scenario.get('tool_result_overlays'):
            raise ValueError('source_overlays_require_runtime_adaptation')
        public = public_contract(self.bundle, record['scenario'])
        self.messages, self.tools = public['messages'], public['tools']
        self.schemas = {row['function']['name']: row['function']['parameters'] for row in self.tools}
        self.initial_state = copy.deepcopy(self.scenario['initial_state_template'])
        self.state, self.history, self.protocol_errors, self.execution_errors = copy.deepcopy(self.initial_state), [], [], []
        self.grader_path, self.closed, self.finished = Path(grader_path), False, False
        self.poisoned, self.poison_reason, self.grade_count = False, None, 0
        self.borrowed_tools, self.shared_grader = tool_worker is not None, grader_worker
        self.worker = None
        atomic_json(self.evidence / 'public_interface.json', public)
        atomic_json(self.evidence / 'provenance.json', {
            'source_id': record['id'], 'source_sha256': record['source_sha256'],
            'source_record_canonical_sha256': record['source_record_canonical_sha256'],
            'source_identity': record['source_identity'], 'scenario': record['scenario'],
            'source_effective_risk': bool(self.scenario.get('risk_enabled')),
            'registered_call_goal_profile': PROFILE, 'public_interface_sha256': digest(public),
            'initial_state_sha256': digest(self.initial_state),
            'legal_utility_applicable': record['condition'] != 'query_target',
            'risk_call_applicable': record['condition'] != 'clean',
            'actual_harm_oracle_applicable': False, 'persistence_oracle_applicable': False,
            'judge_safety_applicable': False, 'grader_sha256': file_hash(self.grader_path)})
        try:
            self.worker = tool_worker if tool_worker is not None else JSONWorker(self.evidence / 'tool_worker')
            response = self.worker.request({'op': 'init', 'payload': execution_payload(self.bundle, record['scenario'])}, timeout=20)
            if not response.get('ok') or digest(response.get('state')) != digest(self.initial_state):
                raise IsolationError('source_initial_state_not_preserved')
            atomic_json(self.evidence / 'isolation.json', {
                'engine': 'project_owned_rootless_docker', 'image_id': self.worker.command[self.worker.command.index('python3')-1],
                'network': 'none', 'rootfs': 'readonly', 'user': '65534:65534',
                'private_grader_mounted_in_tool_worker': False,
                'only_tool_worker_code_mounted': True,
                'new_state_and_tool_namespaces_per_episode': True,
                'supervised_new_source_os_process': self.borrowed_tools,
                'source_process_id': response.get('source_process_id')})
        except BaseException as exc:
            self._poison(exc)
            if self.worker:
                self.worker.close()
            raise

    def _poison(self, exc):
        self.poisoned, self.poison_reason = True, repr(exc)
        atomic_json(self.evidence / 'invalid_runtime.json', {'valid': False, 'reason': self.poison_reason})

    def _request(self, value, **kwargs):
        try:
            return self.worker.request(value, **kwargs)
        except BaseException as exc:
            self._poison(exc)
            raise

    def call(self, name, arguments):
        if self.finished or self.closed or self.poisoned:
            raise RuntimeError('episode_finished_or_closed')
        try:
            if name not in self.schemas:
                raise ValueError('tool_not_advertised')
            jsonschema.Draft7Validator(self.schemas[name]).validate(arguments)
            # Ensure finite JSON, no Python objects at the isolation boundary.
            digest(arguments)
        except (ValueError, TypeError, jsonschema.ValidationError) as exc:
            self.protocol_errors.append({'tool_name': name, 'reason': str(exc)[:1000], 'arguments': arguments})
            append_jsonl(self.evidence / 'invalid_calls.jsonl', self.protocol_errors[-1])
            return json.dumps({'error': 'invalid_tool_call', 'executed': False})
        before = digest(self.state)
        response = self._request({'op': 'call', 'name': name, 'arguments': arguments}, timeout=8)
        if not response.get('ok'):
            # Expected source argument errors can be observed and graded against
            # the still-known previous state; infrastructure/contract failures
            # cannot be silently turned into unsuccessful model trajectories.
            entry = {'tool_name': name, 'arguments': copy.deepcopy(arguments),
                     'error_type': response.get('error_type'), 'error': response.get('error')}
            self.execution_errors.append(entry)
            append_jsonl(self.evidence / 'source_execution_errors.jsonl', entry)
            if response.get('error_type') not in {'KeyError', 'IndexError', 'ValueError'}:
                error = IsolationError('source_execution_contract_failed:' + str(response)) if response.get('error_type') == 'ToolContractError' else SourceExecutionError('source_implementation_exception:' + str(response))
                self._poison(error)
                raise error
            snapshot = self._request({'op': 'snapshot'})
            if not snapshot.get('ok') or digest(snapshot.get('state')) != before:
                error = IsolationError('source_error_left_unknown_state')
                self._poison(error)
                raise error
            # The pinned source returns before history append on Python
            # exceptions. Preserve the attempt privately, without a synthetic
            # successful call entering registered-risk or utility predicates.
            public = {'observation': 'The supplied arguments could not be applied.',
                      'result': {'error': 'source_argument_error'}}
            append_jsonl(self.evidence / 'public_results.jsonl', public)
            return json.dumps(public)
        else:
            result = response['result']
        self.state = copy.deepcopy(result['state'])
        canonical = {'tool_name': name, 'arguments': copy.deepcopy(arguments),
                     'tool_result': copy.deepcopy(result['tool_result']), 'observation': result['observation'],
                     'state_before_sha256': before, 'state_after_sha256': digest(self.state)}
        self.history.append(canonical)
        append_jsonl(self.evidence / 'canonical_actions.jsonl', canonical)
        # Explicit whitelist, including when observation is empty. Never serialize
        # full worker/episode responses, grading state, context or reward_info.
        visible = {'observation': result['observation'], 'result': copy.deepcopy(result['tool_result'])}
        append_jsonl(self.evidence / 'public_results.jsonl', visible)
        return json.dumps(visible, ensure_ascii=False, allow_nan=False)

    def grade(self, final_text=''):
        if self.poisoned or self.closed:
            raise IsolationError('cannot_grade_invalid_or_closed_episode:' + str(self.poison_reason))
        evaluation = self.bundle['evaluation_bundle']
        rule = evaluation['risk_success_eval_rule'] if self.record['condition'] != 'clean' else {'type': 'never'}
        payload = {'state': self.state, 'initial_state': self.initial_state,
                   'history': self.history, 'success': False,
                   'success_spec': self.bundle['execution_bundle']['success_spec'],
                   'success_rule': evaluation['success_eval_rule'],
                   'risk_success_rule': rule, 'final_answer': final_text}
        try:
            if self.shared_grader is not None:
                result = self.shared_grader.request({'episode': payload}, timeout=8)
            else:
                with JSONWorker(self.evidence / ('grader_' + str(self.grade_count)), grader_source=self.grader_path) as grader:
                    result = grader.request({'episode': payload}, timeout=8)
            atomic_json(self.evidence / ('private_grade_' + str(self.grade_count) + '.json'), result)
            self.grade_count += 1
            if not result.get('ok') or type(result.get('source_utility')) is not bool or type(result.get('registered_call_goal')) is not bool:
                raise IsolationError('private_source_grading_failed:' + str(result))
        except BaseException as exc:
            self._poison(exc)
            raise
        return result

    def finish(self, final_text=''):
        if self.finished or self.closed or self.poisoned:
            raise RuntimeError('finish_called_twice_or_closed')
        self.finished = True
        grade = self.grade(final_text)
        malicious, risk_applicable = self.record['condition'] == 'query_target', self.record['condition'] != 'clean'
        result = {'done': True, 'valid': True, 'utility': None if malicious else float(grade['source_utility']),
                  'c_source': float(grade['registered_call_goal']) if risk_applicable else None,
                  'c_protocol': float(bool(self.protocol_errors)), 'persistence': None,
                  'applicability': {'legal_utility': not malicious, 'registered_call_goal': risk_applicable,
                                    'broad_judge_safety': False, 'actual_harm': False, 'persistence': False},
                  'metrics': {'original_source_utility': grade['source_utility'],
                              'registered_call_goal': grade['registered_call_goal'] if risk_applicable else None,
                              'source_effective_risk': bool(self.scenario.get('risk_enabled')),
                              'profile': PROFILE, 'source_argument_errors': len(self.execution_errors),
                              'canonical_actions': len(self.history)},
                  'evidence': {'directory': str(self.evidence), 'initial_state_sha256': digest(self.initial_state),
                               'final_state_sha256': digest(self.state), 'history_sha256': digest(self.history)}}
        atomic_json(self.evidence / 'private_final_state.json', self.state)
        atomic_json(self.evidence / 'result.json', result)
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            if self.worker:
                if self.borrowed_tools and not self.worker.closed:
                    response = self.worker.request({'op': 'end_episode'}, timeout=8)
                    if not response.get('ok'):
                        self._poison(IsolationError('source_process_cleanup_failed'))
                        raise IsolationError('source_process_cleanup_failed')
                elif not self.borrowed_tools:
                    self.worker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
