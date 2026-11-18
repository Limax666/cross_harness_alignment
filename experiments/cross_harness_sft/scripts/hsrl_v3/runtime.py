"""One real task environment per rollout, exact sampled token segments, no model API mocks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from .backends import invalid, open_backend, write_json
from .components import ContextView, ToolMap, VERSION, config_name, digest, observation, parse_calls


def run_episode(task, row, policy, directory: Path, limits: dict):
    directory.mkdir(parents=True, exist_ok=False)
    workspace, private = directory / 'workspace', directory / 'private'
    workspace.mkdir(); private.mkdir()
    segments, actions, presentation_diffs, error = [], [], [], None
    protocol_errors, total_generated, stop, final_text, tool_steps = [], 0, 'round_limit', '', 0
    started = time.monotonic()
    outcome = None
    try:
        with open_backend(task, workspace, private / 'backend') as backend:
            mapping = ToolMap(backend.tools, row[1])
            initial = copy.deepcopy(backend.messages)
            for message in initial:
                raw = message.get('content', '')
                message['content'] = mapping.present_text(raw)
                if raw != message['content']:
                    presentation_diffs.append({'kind': 'initial', 'canonical': raw, 'presented': message['content']})
            view = ContextView(policy.tokenizer, row, initial, mapping.public_tools, workspace)
            write_json(private / 'public_interface.json', {'tools': mapping.public_tools, 'initial_messages': initial,
                       'source_tools_sha256': digest(backend.tools), 'T': row[1],
                       'canonical_mapping': mapping.forward, 'version': VERSION})
            for turn in range(limits['max_calls']):
                if total_generated >= limits['episode_tokens']:
                    stop = 'model_output_incomplete'; break
                prompt_ids, messages, compaction = view.render()
                if compaction:
                    write_json(private / f'context_{turn:02d}.json', compaction)
                available = limits['context_tokens'] - len(prompt_ids)
                if available < 1:
                    stop = 'model_context_limit'; break
                max_new = min(limits['call_tokens'], limits['episode_tokens'] - total_generated, available)
                generated = policy.generate(prompt_ids, max_new)
                if generated['policy_version'] != policy.version or not generated['response_ids']:
                    raise RuntimeError('stale_or_empty_policy_generation')
                total_generated += len(generated['response_ids'])
                text = generated['text']
                segment = {**generated, 'prompt_ids': prompt_ids, 'turn': turn,
                           'configuration': config_name(row), 'prompt_sha256': digest(prompt_ids),
                           'public_messages_sha256': digest(messages)}
                write_json(private / f'segment_{turn:02d}.json', segment)
                # Keep paths in memory; exact IDs/old/reference vectors survive in evidence.
                segments.append(str(private / f'segment_{turn:02d}.json'))
                messages_this_round = [{'role': 'assistant', 'content': text}]
                try:
                    calls = parse_calls(text, mapping.public_tools)
                except ValueError as exc:
                    protocol_errors.append({'turn': turn, 'reason': str(exc), 'raw_sha256': digest(text),
                                            'schema_sha256': digest(mapping.public_tools)})
                    stop = 'model_output_incomplete' if generated['length_limited'] else 'invalid_tool_call'
                    # A failed call executes nothing; let the original source verifier
                    # evaluate all effects already observed in this real episode.
                    break
                if not calls:
                    if generated['length_limited']:
                        stop = 'model_output_incomplete'
                    elif '<think>' in text and '</think>' not in text:
                        stop = 'model_output_incomplete'
                    else:
                        stop = 'completed'
                        final_text = text.split('</think>')[-1].strip()
                    break
                for exposed_name, exposed_args in calls:
                    if tool_steps >= 30:
                        stop = 'tool_step_limit'; break
                    tool_steps += 1
                    if exposed_name in mapping.distractors:
                        raw = json.dumps({'error': 'inactive_reserved_interface', 'executed': False})
                        canonical, arguments = None, None
                    else:
                        canonical, arguments = mapping.decode_call(exposed_name, exposed_args)
                        raw = backend.call(canonical, arguments)
                    if not isinstance(raw, str):
                        raise TypeError('source_tool_result_must_be_string')
                    presented = mapping.present_text(raw)
                    if presented != raw:
                        presentation_diffs.append({'kind': 'tool_result', 'canonical': raw, 'presented': presented})
                    wrapped = observation(presented, row[2], tool_steps)
                    actions.append({'turn': turn, 'tool_step': tool_steps, 'public_name': exposed_name,
                                    'public_arguments': exposed_args, 'canonical_name': canonical,
                                    'canonical_arguments': arguments, 'canonical_result': raw,
                                    'presented_result': wrapped, 'interface_sha256': digest(mapping.public_tools)})
                    messages_this_round.append({'role': 'tool', 'content': wrapped, 'name': exposed_name})
                if row[4] == 2 and tool_steps and tool_steps % 3 == 0:
                    messages_this_round.append({'role': 'user', 'content':
                        'Workflow checkpoint: briefly check progress against the original authorized request, then continue.'})
                view.add_round(messages_this_round)
                write_json(private / 'actions.json', actions)
                if stop == 'tool_step_limit':
                    break
            backend.stop_reason = stop
            outcome = backend.finish(final_text)
            if not outcome.get('done'):
                raise RuntimeError('unexpected_multistage_source_in_v4_registry')
            # Privacy admission is structural: only source public messages/schema and
            # scoped tools were given to the target. Raw task records and source paths
            # stay in private host evidence. Real source failures still invalidate it.
            privacy = {'valid': True, 'version': 'public-source-interface-and-isolated-tools-v4',
                       'source_public_interface_only': True, 'private_outside_workspace':
                       not private.resolve().is_relative_to(workspace.resolve()),
                       'initial_messages_sha256': digest(initial), 'public_tools_sha256': digest(mapping.public_tools),
                       'presentation_diffs': str(private / 'presentation_diffs.json'),
                       'target_tool_backend': 'official_isolated_lifecycle' if task['source'] == 'skilllifebench'
                                              else 'rootless_docker_workspace_only',
                       'source_outcome_valid': outcome.get('valid') is True}
            if not privacy['private_outside_workspace']:
                raise RuntimeError('private_evidence_in_target_workspace')
            write_json(private / 'privacy_audit.json', privacy)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc)}
        outcome = invalid('runtime_or_source_failure', evidence={'error': error})
    cost = max(int(bool(protocol_errors)), int(outcome.get('c_protocol') or 0))
    record = {'task_id': task['id'], 'source': task['source'], 'condition': task.get('condition'),
              'family': task['family'], 'split': task['split'], 'source_sha256': task['source_sha256'],
              'configuration': config_name(row), 'policy_version': policy.version,
              'segments': segments, 'generated_action_tokens': total_generated,
              'c_protocol': cost, 'protocol_errors': protocol_errors, 'stop_reason': stop,
              'source_result': outcome, 'valid': outcome.get('valid') is True and error is None and bool(segments),
              'error': error, 'tool_steps': tool_steps, 'elapsed_seconds': time.monotonic() - started,
              'evidence': str(private), 'runtime_version': VERSION}
    write_json(private / 'actions.json', actions)
    write_json(private / 'presentation_diffs.json', presentation_diffs)
    write_json(directory / 'episode.json', record)
    return record
