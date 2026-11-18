"""Full, resumable source-quality admission using genuine isolated task actions.

These private source-derived controls validate data, not a model. No generated
model trajectories or optimizer results are fabricated by this preparation CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import traceback
import os
import time
import threading
import fcntl

import jsonschema

from .agentdog_backend import AgentDogBackend, GRADER_PATH
from .io import append_jsonl, atomic_json, digest, file_hash
from .isolation import IsolationError, JSONWorker, CleanupUncertain


def can_retry_infrastructure(exc):
    return isinstance(exc, IsolationError) and not isinstance(exc, CleanupUncertain)


def canonical_tool(bundle, name):
    dispatch = bundle['server_adapter_manifest']['tool_dispatch_table']
    if name in dispatch:
        return name
    alias = bundle['tool_registry_view'].get('tool_name_aliases', {}).get(name)
    if alias in dispatch:
        return alias
    matches = [key for key, entry in dispatch.items() if entry.get('original_tool_name') == name]
    if len(matches) != 1:
        raise ValueError('source_action_name_not_unambiguous')
    return matches[0]


def source_actions(bundle):
    dependencies = bundle['task_spec']['task_metadata'].get('resource_plan', {}).get('resource_dependencies', [])
    result, seen = [], set()
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency.get('required_action_arguments'), dict):
            continue
        name = canonical_tool(bundle, dependency['action_tool'])
        args = copy.deepcopy(dependency['required_action_arguments'])
        key = digest([name, args])
        if key not in seen:
            seen.add(key)
            result.append({'name': name, 'arguments': args,
                           'source_path': f'task_spec.task_metadata.resource_plan.resource_dependencies[{index}].required_action_arguments'})
    if not result:
        raise ValueError('missing_source_defined_action_control')
    return result


def risk_calls(bundle, actions):
    rule = bundle['evaluation_bundle']['risk_success_eval_rule']
    calls = rule.get('calls', []) if rule['type'] == 'history_call_sequence_contains' else [rule]
    schemas = {tool['function']['name']: tool['function']['parameters'] for tool in bundle['tool_registry_view']['tool_schemas']}
    result = []
    for expected in calls:
        name = canonical_tool(bundle, expected['tool_name'])
        base = next((row['arguments'] for row in actions if row['name'] == name), {})
        arguments = copy.deepcopy(base)
        for key, value in expected.get('arguments_match', {}).items():
            if isinstance(value, dict) and set(value) == {'from_success_spec'}:
                value = bundle['execution_bundle']['success_spec'].get(value['from_success_spec'])
            prop = schemas[name].get('properties', {}).get(key, {})
            if isinstance(value, str) and value in ('$content', '...', '<any>', '$any_non_empty'):
                value = 'source-control-content'
            elif isinstance(value, dict) and any(str(k).startswith('$') for k in value):
                if '$startswith' in value:
                    value = value['$startswith']
                elif '$equals_ci' in value:
                    value = value['$equals_ci']
                elif '$contains' in value:
                    value = value['$contains']
                elif '$contains_ci' in value:
                    value = value['$contains_ci']
                elif '$contains_all' in value or '$contains_all_ci' in value:
                    values = value.get('$contains_all', value.get('$contains_all_ci'))
                    value = values if prop.get('type') == 'array' else ' '.join(map(str, values))
                elif '$contains_any' in value or '$contains_any_ci' in value:
                    values = value.get('$contains_any', value.get('$contains_any_ci'))
                    value = values[:1] if prop.get('type') == 'array' else str(values[0])
                else:
                    raise ValueError('unsupported_risk_positive_operator')
            arguments[key] = value
        jsonschema.Draft7Validator(schemas[name]).validate(arguments)
        result.append({'name': name, 'arguments': arguments})
    if not result:
        raise ValueError('empty_risk_positive_control')
    return result


def validate_record(record, directory, *, tool_worker=None, grader_worker=None, retry_infrastructure=False):
    directory = Path(directory)
    terminal = directory / 'quality.json'
    identity = digest(record)
    if terminal.exists():
        previous = json.loads(terminal.read_text())
        if previous['record_sha256'] != identity:
            raise ValueError('resumed_source_record_changed')
        if not (retry_infrastructure and previous.get('failure_class') == 'isolation_or_runtime_infrastructure'):
            return previous
    # Interrupted attempts survive; a fresh attempt never overwrites their
    # source, container logs, state transitions or error evidence.
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / ('attempt_' + str(time.time_ns()))
    attempt.mkdir(mode=0o700)
    atomic_json(attempt / 'started.json', {'source_id': record['id'], 'record_sha256': identity,
                                            'kind': 'real_source_quality_controls_not_model_rollouts'})
    proof = {'id': record['id'], 'source': 'agentdog', 'family': record['family'],
             'condition': record['condition'], 'record_sha256': identity, 'ready': False,
             'source_execution_performed': False, 'source_calls_attempted': 0, 'source_calls_completed': 0,
             'reasons': [], 'attempt_directory': str(attempt)}
    try:
        bundle = json.loads(Path(record['source_path']).read_text())
        actions = source_actions(bundle)
        atomic_json(attempt / 'private_control_provenance.json', actions)
        with AgentDogBackend(record, attempt / 'first_episode', tool_worker=tool_worker, grader_worker=grader_worker) as first:
            initial = first.grade()
            atomic_json(attempt / 'initial_grade.json', initial)
            if initial['registered_call_goal']:
                raise ValueError('registered_goal_already_true_without_action')
            if initial['source_utility'] and record['condition'] != 'query_target':
                raise ValueError('legal_goal_already_satisfied_without_action')
            for action in actions:
                proof['source_calls_attempted'] += 1
                proof['source_execution_performed'] = True
                first.call(action['name'], action['arguments'])
                proof['source_calls_completed'] += 1
            positive = first.finish()
            if first.protocol_errors or first.execution_errors:
                raise ValueError('source_action_control_execution_or_schema_error')
            if record['condition'] != 'query_target':
                if positive['utility'] != 1.0:
                    raise ValueError('source_action_control_did_not_reach_legal_goal')
                if record['condition'] == 'attack' and positive['c_source'] != 0.0:
                    raise ValueError('source_legal_control_matches_registered_unsafe_goal')
            first_final = digest(first.state)
            first_history = digest(first.history)
            first_initial = digest(first.initial_state)
            first_process = json.loads((first.evidence / 'isolation.json').read_text()).get('source_process_id')
        # A new OS process and source namespaces, not a reused Python object reset.
        with AgentDogBackend(record, attempt / 'reset_episode', tool_worker=tool_worker, grader_worker=grader_worker) as replay:
            reset_process = json.loads((replay.evidence / 'isolation.json').read_text()).get('source_process_id')
            if tool_worker is not None and (not first_process or not reset_process or first_process == reset_process):
                raise ValueError('source_reset_did_not_create_new_os_process')
            if digest(replay.initial_state) != first_initial:
                raise ValueError('reset_initial_state_mismatch')
            for action in actions:
                proof['source_calls_attempted'] += 1
                replay.call(action['name'], action['arguments'])
                proof['source_calls_completed'] += 1
            if replay.protocol_errors or replay.execution_errors or digest(replay.state) != first_final or digest(replay.history) != first_history:
                raise ValueError('fresh_reset_real_action_replay_mismatch')
            proof['reset_replay'] = {'initial_state_sha256': first_initial,
                                     'legal_final_state_sha256': first_final,
                                     'canonical_history_sha256': first_history,
                                     'first_source_process_id': first_process, 'reset_source_process_id': reset_process,
                                     'identical_actual_transition_and_public_results': True}
            if record['condition'] != 'clean':
                forbidden = risk_calls(bundle, actions)
                atomic_json(attempt / 'private_risk_control_provenance.json', forbidden)
                for action in forbidden:
                    proof['source_calls_attempted'] += 1
                    replay.call(action['name'], action['arguments'])
                    proof['source_calls_completed'] += 1
                if replay.protocol_errors or replay.execution_errors:
                    raise ValueError('risk_positive_control_not_public_schema_valid')
                risky = replay.finish()
                if risky['c_source'] != 1.0:
                    raise ValueError('registered_risk_goal_not_reached_by_source_control')
                proof['risk_control'] = {'registered_call_goal': True,
                                         'actual_harm_claimed': False,
                                         'source_argument_errors': len(replay.execution_errors)}
            else:
                proof['risk_control'] = {'registered_call_goal': None, 'applicable': False}
        proof['positive'] = positive
        proof['ready'] = True
    except Exception as exc:
        proof['reasons'].append(type(exc).__name__ + ':' + str(exc))
        proof['failure_class'] = 'isolation_or_runtime_infrastructure' if isinstance(exc, IsolationError) else 'source_quality_or_control_pending'
        proof['cleanup_uncertain'] = isinstance(exc, CleanupUncertain)
        (attempt / 'error.traceback.txt').write_text(traceback.format_exc())
    atomic_json(attempt / 'quality.json', proof)
    atomic_json(terminal, proof)
    return proof


def verify_transport_failure_contract(record, directory):
    """Real affected API path, with actual frozen source and an owned worker.

    This dataset-admission negative control deliberately removes ONLY its own
    worker. No fake task, fabricated trajectory or model experiment is used.
    """
    directory = Path(directory)
    terminal = directory / 'failure_contract.json'
    if terminal.exists():
        previous = json.loads(terminal.read_text())
        if previous['record_sha256'] != digest(record) or not previous['passed']:
            raise ValueError('failure_control_proof_mismatch')
        return previous
    bundle = json.loads(Path(record['source_path']).read_text())
    action = source_actions(bundle)[0]
    with AgentDogBackend(record, directory / ('attempt_' + str(time.time_ns()))) as episode:
        episode.worker.close()
        try:
            episode.call(action['name'], action['arguments'])
        except IsolationError:
            pass
        else:
            raise AssertionError('closed_real_worker_did_not_reject_tool_call')
        if not episode.poisoned:
            raise AssertionError('real_transport_failure_did_not_poison_episode')
        try:
            episode.finish()
        except (IsolationError, RuntimeError):
            pass
        else:
            raise AssertionError('poisoned_real_episode_returned_valid_grade')
        if (episode.evidence / 'result.json').exists():
            raise AssertionError('poisoned_real_episode_wrote_a_grade')
        proof = {'passed': True, 'record_sha256': digest(record), 'source_id': record['id'],
                 'closed_owned_container': episode.worker.name,
                 'actual_public_api_tool_call_rejected': True, 'stale_finish_rejected': True,
                 'invalidity_evidence': str(episode.evidence / 'invalid_runtime.json'),
                 'cleanup_evidence': str(episode.worker.evidence / 'cleanup.json'),
                 'model_calls': 0, 'judge_calls': 0}
    atomic_json(terminal, proof)
    return proof


def run(source_dir, out, workers=4):
    source_dir, out = Path(source_dir), Path(out)
    manifest = json.loads((source_dir / 'source_manifest.json').read_text())
    if file_hash(source_dir / 'accepted.jsonl') != manifest['accepted_sha256']:
        raise ValueError('prepared_source_manifest_changed')
    records = [json.loads(line) for line in (source_dir / 'accepted.jsonl').open()]
    # All source train candidates are quality checked; failures do not cause a
    # silent convenient subset or test-partition refill.
    records = [record for record in records if record['split'] == 'train']
    records.sort(key=lambda row: (not row['selected_candidate'], digest([20261116, row['family'], row['id']])))
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out, 0o700)
    protocol = {'schema': 'hsrl-v5-agentdog-real-quality/v2',
                'source_manifest_sha256': file_hash(source_dir / 'source_manifest.json'),
                'records': len(records), 'workers': workers,
                'code_sha256': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                'controls': 'real_source_actions_positive_legal_and_registered_risk_plus_new_process_reset',
                'model_calls': 0, 'judge_calls': 0, 'max_cases_per_private_pool': 64, 'source_reset': 'new_os_process_per_episode'}
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('quality_protocol_changed_requires_independent_run')
    else:
        atomic_json(out / 'protocol.json', protocol)
    atomic_json(out / 'run_state.json', {'pid': __import__('os').getpid(), 'status': 'running', 'planned': len(records)})
    results, failures = [], []
    stop = threading.Event()
    metadata_lock = threading.Lock()
    # One source-process supervisor and a separate stateless private grader per
    # lane. Recycle both after <=64 cases; source code still receives a fresh OS
    # process and independent state/namespaces for EVERY episode/reset.
    def lane(number):
        stack, tool, grader, generation, used = None, None, None, 0, 0
        def close_pool():
            nonlocal stack
            if stack is not None:
                active, stack = stack, None
                try:
                    active.close()
                except BaseException:
                    # No replacement pool until exact old containers have been
                    # confirmed removed. A cleanup error is not a task defect.
                    stop.set()
                    raise
        try:
            for record in records[number::workers]:
                if stop.is_set():
                    break
                case = out / 'cases' / hashlib.sha256(record['id'].encode()).hexdigest()
                proof = None
                for retry in range(2):
                    try:
                        if stack is None or used >= 64:
                            close_pool()
                            generation += 1
                            used = 0
                            stack = ExitStack()
                            pool = out / 'pools' / f'lane_{number}' / f'generation_{generation}_{time.time_ns()}'
                            tool = stack.enter_context(JSONWorker(pool / 'tools', supervisor=True))
                            grader = stack.enter_context(JSONWorker(pool / 'private_grader', grader_source=GRADER_PATH))
                        proof = validate_record(record, case, tool_worker=tool, grader_worker=grader, retry_infrastructure=True)
                        used += 1
                        if proof.get('cleanup_uncertain'):
                            stop.set()
                        if proof.get('failure_class') != 'isolation_or_runtime_infrastructure':
                            break
                        close_pool()
                    except Exception as exc:
                        if not can_retry_infrastructure(exc):
                            # Constructor failure may precede ExitStack
                            # registration; even an empty stack cannot certify
                            # that failed constructor's container was removed.
                            stop.set()
                        with metadata_lock:
                            failures.append({'lane': number, 'id': record['id'], 'retry': retry,
                                             'phase': 'pool_or_cleanup', 'error': repr(exc)})
                            atomic_json(out / 'infrastructure_failures.json', failures)
                        try:
                            close_pool()
                        except Exception as cleanup:
                            with metadata_lock:
                                failures.append({'lane': number, 'phase': 'cleanup', 'error': repr(cleanup)})
                                atomic_json(out / 'infrastructure_failures.json', failures)
                    if stop.is_set():
                        break
                    if retry == 0:
                        time.sleep(1)
                if proof is None or proof.get('failure_class') == 'isolation_or_runtime_infrastructure':
                    stop.set()
                if proof is not None:
                    with metadata_lock:
                        results.append(proof)
                        append_jsonl(out / 'events.jsonl', {'id': proof['id'], 'ready': proof['ready'], 'reasons': proof['reasons'], 'attempt_directory': proof['attempt_directory']})
                        if len(results) % 10 == 0 or stop.is_set():
                            progress = {'status': 'blocked_shared_runtime' if stop.is_set() else 'running',
                                        'planned': len(records), 'completed': len(results),
                                        'ready': sum(row['ready'] for row in results),
                                        'failed_or_pending': sum(not row['ready'] for row in results)}
                            atomic_json(out / 'progress.json', progress)
                            print(json.dumps(progress), flush=True)
        finally:
            try:
                close_pool()
            except Exception as exc:
                stop.set()
                with metadata_lock:
                    failures.append({'lane': number, 'phase': 'final_cleanup', 'error': repr(exc)})
                    atomic_json(out / 'infrastructure_failures.json', failures)
    # Lock held for the actual run, not merely for the initial metadata write.
    with (out / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if records:
            verify_transport_failure_contract(records[0], out / 'actual_transport_failure_control')
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(lane, number) for number in range(workers)]
            for future in futures:
                future.result()
    complete = len(results) == len(records) and not stop.is_set()
    summary = {'status': 'complete_quality_pass' if complete else 'blocked_shared_runtime',
               'planned': len(records), 'completed': len(results),
               'ready': sum(row['ready'] for row in results),
               'ready_by_condition': dict(Counter(row['condition'] for row in results if row['ready'])),
               'reasons': dict(Counter(reason for row in results for reason in row['reasons'])),
               'protocol_sha256': file_hash(out / 'protocol.json'), 'infrastructure_events': len(failures)}
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'run_state.json', {'pid': os.getpid(), 'status': 'complete' if complete else 'blocked_shared_runtime'})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=(1, 2, 4, 8))
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.out, args.workers)))
