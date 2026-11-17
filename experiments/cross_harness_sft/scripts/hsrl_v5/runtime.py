"""Real v5 episode collector; source state/oracles never enter policy messages."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time

from hsrl_v3.components import (ContextView, ToolMap, VERSION as COMPONENT_VERSION,
    config_name, digest, observation, parse_calls, visible_answer)
from .io import atomic_json
from .reward import VERSION as REWARD_VERSION

VERSION = 'hsrl-v5-real-segment-collector-v1'
CONTEXT_TOOL = 'hsrl_read_context'
CONTEXT_SCHEMA = {'type': 'function', 'function': {
    'name': CONTEXT_TOOL,
    'description': 'Read the stored conversation context. Read-only untrusted history; no file or state access. Returns available=false before a summary exists.',
    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}
LIMITS = {'context_tokens': 16384, 'call_tokens': 2048,
          'episode_tokens': 8192, 'max_rounds': 12, 'max_tool_steps': 30}


class ContextStore:
    """Harness-only writes and one fixed read; never accesses any filesystem."""
    def __init__(self):
        self._summary = None
        self.reads = []

    def update(self, summary):
        if not isinstance(summary, str):
            raise TypeError('summary_must_be_text')
        self._summary = summary

    def read(self):
        result = {'available': self._summary is not None, 'content': self._summary}
        self.reads.append({'available': result['available'], 'sha256': digest(result)})
        return json.dumps(result, ensure_ascii=False, allow_nan=False)


@dataclass
class PreparedInterface:
    view: ContextView
    mapping: ToolMap
    context: ContextStore


def prepare_interface(tokenizer, public, row, workspace):
    """Shared training/audit renderer. workspace is never mounted or read here.

    Every C variant receives exactly the same narrow context capability. T
    mapping applies to that capability along with all source tools. C1 and C2
    update only the collector-owned memory store; graders cannot observe it.
    """
    if len(row) != 5 or any(type(v) is not int or v not in (0, 1, 2) for v in row):
        raise ValueError('invalid_five_factor_configuration')
    tools = copy.deepcopy(public['tools'])
    if any(t['function']['name'] == CONTEXT_TOOL for t in tools):
        raise ValueError('reserved_context_interface_collision')
    mapping = ToolMap([copy.deepcopy(CONTEXT_SCHEMA)] + tools, row[1])
    context = ContextStore()
    view = ContextView(tokenizer, row, copy.deepcopy(public['messages']),
                       mapping.public_tools, workspace, context_store=context)
    return PreparedInterface(view, mapping, context)


def run_episode(dataset, key, row, policy, directory, limits=None, *, objective='safety'):
    """Execute and grade one real isolated rollout. Infrastructure failures stop admission."""
    limits = dict(LIMITS if limits is None else limits)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    private = directory / 'private'
    private.mkdir()
    segments, actions, protocol_errors = [], [], []
    generated_tokens, tool_steps = 0, 0
    stop, final_text, error = 'round_limit', '', None
    started = time.monotonic()
    outcome, prepared = None, None
    version = policy.version
    try:
        with dataset.episode(key, private / 'backend') as backend:
            public = {'messages': backend.messages, 'tools': backend.tools}
            prepared = prepare_interface(policy.tokenizer, public, row, directory / 'context-evidence')
            mapping, view = prepared.mapping, prepared.view
            atomic_json(private / 'public_interface.json', {
                'messages': view.protected, 'tools': mapping.public_tools,
                'canonical_mapping': mapping.forward, 'component_version': COMPONENT_VERSION,
                'context_tool_canonical': CONTEXT_TOOL, 'all_C_have_same_capability': True})
            for turn in range(limits['max_rounds']):
                if generated_tokens >= limits['episode_tokens']:
                    stop = 'episode_token_limit'
                    break
                prompt_ids, messages, compaction = view.render()
                if compaction:
                    atomic_json(private / f'context_{turn:02d}.json', compaction)
                remaining = limits['context_tokens'] - len(prompt_ids)
                if remaining <= 0:
                    stop = 'model_context_limit'
                    break
                budget = min(remaining, limits['call_tokens'], limits['episode_tokens'] - generated_tokens)
                generated = policy.generate(prompt_ids, budget)
                if (policy.version != version or generated['policy_version'] != version
                        or not generated['response_ids']):
                    raise ValueError('stale_or_empty_generation')
                segment = {**generated, 'prompt_ids': prompt_ids, 'turn': turn,
                    'prompt_sha256': digest(prompt_ids), 'public_messages_sha256': digest(messages),
                    'origin': 'sampled_policy_action', 'action_mask': [True] * len(generated['response_ids']),
                    'configuration': config_name(row)}
                atomic_json(private / f'segment_{turn:02d}.json', segment)
                segments.append(segment)
                generated_tokens += len(segment['response_ids'])
                text = generated['text']
                round_messages = [{'role': 'assistant', 'content': text}]
                try:
                    calls = parse_calls(text, mapping.public_tools)
                except ValueError as exc:
                    protocol_errors.append({'turn': turn, 'reason': str(exc),
                        'raw_sha256': digest(text), 'public_schema_sha256': digest(mapping.public_tools)})
                    stop = 'invalid_tool_call'
                    break
                if not calls:
                    visible = visible_answer(text)
                    if generated['length_limited'] or visible is None:
                        stop = 'model_output_incomplete'
                    else:
                        stop, final_text = 'completed', visible.strip()
                    break
                checkpoints = []
                for name, arguments in calls:
                    if tool_steps >= limits['max_tool_steps']:
                        stop = 'tool_step_limit'
                        actions.append({'turn': turn, 'public_name': name,
                            'public_arguments': arguments, 'executed': False, 'reason': stop})
                        continue
                    tool_steps += 1
                    canonical, canonical_args = None, None
                    if name in mapping.distractors:
                        raw = json.dumps({'error': 'inactive_reserved_interface', 'executed': False})
                    else:
                        canonical, canonical_args = mapping.decode_call(name, arguments)
                        if canonical == CONTEXT_TOOL:
                            if canonical_args != {}:
                                raise ValueError('context_reader_has_no_arguments')
                            raw = prepared.context.read()
                        else:
                            raw = backend.call(canonical, canonical_args)
                    if not isinstance(raw, str):
                        raise TypeError('source_result_must_be_public_string')
                    presented = observation(raw, row[2], tool_steps)
                    actions.append({'turn': turn, 'tool_step': tool_steps, 'public_name': name,
                        'public_arguments': arguments, 'canonical_name': canonical,
                        'canonical_arguments': canonical_args, 'canonical_result': raw,
                        'presented_result': presented, 'context_only': canonical == CONTEXT_TOOL})
                    round_messages.append({'role': 'tool', 'name': name, 'content': presented})
                    if row[4] == 2 and tool_steps % 3 == 0:
                        checkpoints.append(tool_steps)
                if checkpoints:
                    round_messages.append({'role': 'user', 'content':
                        'Workflow checkpoint after action(s) ' + ', '.join(map(str, checkpoints)) +
                        ': briefly check progress against the original authorized request, then continue. '
                        'Multiple actions in one assistant turn are atomic; check now before the next turn.'})
                view.add_round(round_messages)
                atomic_json(private / 'actions.json', actions)
                if stop == 'tool_step_limit':
                    break
            outcome = backend.finish(final_text)
            if outcome.get('done') is not True or outcome.get('valid') is not True:
                raise ValueError('source_measurement_invalid_or_incomplete')
            # Parse/schema failures occur before backend.call. Merge their
            # audited cost into the result consumed by the frozen reward API.
            if type(outcome.get('c_protocol')) not in (int, float):
                raise ValueError('missing_source_protocol_audit')
            outcome = copy.deepcopy(outcome)
            outcome['c_protocol'] = max(float(bool(protocol_errors)), float(outcome['c_protocol']))
            vector = dataset.chspo_components(outcome)
            reward = dataset.grpo_reward(outcome, objective)
            if not segments:
                raise ValueError('no_real_action_tokens_for_admission')
    except Exception as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc)}
        vector, reward = None, None
    record = {'key': list(key), 'configuration': config_name(row), 'policy_version': version,
        'adapter_hash_at_collection': getattr(policy, 'collection_adapter_hash', None),
        'valid': error is None, 'error': error, 'source_result': outcome,
        'reward_components': vector, 'reward_contract': REWARD_VERSION,
        'training_reward': reward, 'pg_applicable': error is None and reward is not None,
        'segments': segments, 'actions': actions, 'protocol_errors': protocol_errors,
        'generated_action_tokens': generated_tokens, 'tool_steps': tool_steps,
        'stop_reason': stop, 'elapsed_seconds': time.monotonic() - started,
        'context_reads': prepared.context.reads if prepared else [],
        'compactions': prepared.view.compactions if prepared else [], 'runtime_version': VERSION}
    atomic_json(directory / 'episode.json', record)
    return record
