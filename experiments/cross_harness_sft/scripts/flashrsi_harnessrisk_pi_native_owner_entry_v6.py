"""Run the actual selected Snapshot Pi loop with a persistent native session."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'vendor/FlashRSI')]
from flashrsi_pi_owner_profile_v6 import PROFILE, ProfileError, byte_hash, digest, read_native_session, projection_receipt


def plain(value):
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ProfileError('unserializable_native_field:' + type(value).__name__)


def provider_observer(original, loop, prepared, boundaries, session_reader, source_identity, native_projection_reader):
    """Observation-only wrapper: await the original callable once with exact inputs."""
    signature = inspect.signature(original)
    if not {'client', 'config', 'provider', 'model', 'messages', 'tools', 'context_messages',
            'on_event', 'cancel_event'}.issubset(signature.parameters):
        raise ProfileError('Snapshot_native_provider_observation_API_unsupported')
    async def observed(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        config, context = values['config'], values['context_messages']
        if context is None or loop._api_name(values['provider'], config) != 'openai-completions':
            raise ProfileError('native_provider_context_or_API_unobservable')
        projection_receipt(plain(context))
        kind = 'agent' if values['on_event'] is not None else 'summary'
        if kind == 'agent':
            if not prepared:
                raise ProfileError('native_agent_call_missing_prepare_boundary')
            canonical_context = prepared.pop(0)
            if digest(canonical_context) != digest(plain(native_projection_reader())):
                raise ProfileError('agent_context_differs_from_original_native_session_projection')
        else:
            canonical_context = plain(context)
        payload, _effort = loop.build_openai_payload(config, context, 'openai-completions',
            loop._reasoning_effort(config, config['llm']), values['tools'])
        extra = config['llm'].get('extra_body', config.get('extra_body'))
        if isinstance(extra, Mapping):
            payload.update(extra)
        boundaries.append({'kind': kind, 'canonical_context': canonical_context,
            'native_session_projection_sha256': digest(canonical_context) if kind == 'agent' else None,
            'native_projected_context': plain(context), 'wire_messages': plain(payload['messages']),
            'session_sha256_at_request': byte_hash(session_reader()),
            'native_provider_source': source_identity})
        # Do not change/copy args, kwargs, callbacks, cancellation, result or exception.
        return await original(*args, **kwargs)
    return observed


async def execute(request):
    if request.get('profile') != PROFILE or request.get('harness') != 'pi':
        raise ProfileError('Pi_native_profile_only')
    config = copy.deepcopy(request['config'])
    if (config.get('max_turns') != 12 or config.get('timeout') != 360
            or config.get('context_window') != 32768 or config.get('model_max_tokens') != 4096
            or config['llm'].get('max_tokens') != 4096 or config['llm'].get('temperature') != 0
            or config['llm'].get('provider_name') != 'openai' or config['llm'].get('stream') is not False):
        raise ProfileError('registered_native_limits_or_provider_changed')
    scratch, work = Path(os.environ['HOME']).resolve(strict=True), Path(request['working_dir']).resolve(strict=True)
    snapshot, session_file = Path(request['workspace_dir']).resolve(strict=True), Path(request['session_file'])
    if (not session_file.is_absolute() or not session_file.resolve().is_relative_to(scratch)
            or session_file.resolve().is_relative_to(work)):
        raise ProfileError('session_must_be_case_scratch_only')
    current = scratch
    for part in session_file.relative_to(scratch).parts:
        current = current / part
        if current.is_symlink():
            raise ProfileError('native_session_symlink')
    before = read_native_session(session_file, scratch, missing_ok=request['owner_index'] == 1)
    if byte_hash(before) != request['expected_session_sha256'] or (request['owner_index'] > 1 and not before):
        raise ProfileError('native_session_missing_or_changed_before_owner')
    dependencies = {name: shutil.which(name) for name in ('python', 'python3')}
    if not all(dependencies.values()):
        raise ProfileError('declared_python_dependency_missing_from_corrected_interpreter_PATH')
    from flashrsi.agents import pi_agent as adapter
    original_is_file = Path.is_file
    def scoped_is_file(path):
        if path.name in {'AGENTS.md', 'CLAUDE.md', 'CONTEXT.md'}:
            resolved = path.resolve()
            if not (resolved.is_relative_to(work) or resolved.is_relative_to(snapshot)):
                return False
        return original_is_file(path)
    Path.is_file = scoped_is_file
    config['session_file'] = str(session_file)
    config.pop('session_leaf_id', None)
    config.pop('session', None)
    boundaries, prepared = [], []
    try:
        with adapter._native_runtime(snapshot) as native:
            if not {'hooks', 'on_event', 'cancel_event'}.issubset(inspect.signature(native.run).parameters):
                raise ProfileError('Snapshot_native_observation_API_unsupported')
            loop = sys.modules[native.run.__module__]
            if loop.model_limits(config) != (32768, 4096):
                raise ProfileError('native_model_limits_do_not_match_actual_backend')
            compaction_settings = plain(loop.compaction_settings(config, snapshot))
            def observe_request(value, _signal=None):
                prepared.append(plain(value['context']['messages']))
            original_provider = loop._provider_call
            source = Path(inspect.getsourcefile(original_provider)).resolve(strict=True)
            if (not source.is_relative_to(snapshot) or not inspect.iscoroutinefunction(original_provider)
                    or original_provider.__qualname__ != '_provider_call'):
                raise ProfileError('native_provider_callable_not_in_selected_Snapshot')
            source_identity = {'path': str(source.relative_to(snapshot)), 'sha256': byte_hash(source.read_bytes()),
                'qualname': original_provider.__qualname__, 'line': original_provider.__code__.co_firstlineno,
                'signature': str(inspect.signature(original_provider))}
            session_source = Path(inspect.getsourcefile(loop.Session)).resolve(strict=True)
            if not session_source.is_relative_to(snapshot):
                raise ProfileError('native_session_projection_not_from_selected_Snapshot')
            loop._provider_call = provider_observer(original_provider, loop, prepared, boundaries,
                lambda: read_native_session(session_file, scratch), source_identity,
                lambda: loop.Session(work, session_file).messages())
            try:
                result = await asyncio.wait_for(native.run(config, request['question'], work, snapshot,
                    hooks={'prepare_request': observe_request}), timeout=360)
            finally:
                loop._provider_call = original_provider
            if prepared:
                raise ProfileError('native_prepared_request_never_reached_original_provider')
            native_result = plain(result)
    finally:
        Path.is_file = original_is_file
    return {'profile': PROFILE, 'native': native_result, 'boundaries': boundaries,
        'session_before_sha256': byte_hash(before), 'session_after_sha256': byte_hash(read_native_session(session_file, scratch)),
        'native_model_limits': {'context_window': 32768, 'model_max_tokens': 4096},
        'native_compaction_settings': compaction_settings,
        'dependencies': dependencies, 'owner_index': request['owner_index'],
        'owner_sha256': digest(request['question'])}


def main():
    request = json.load(sys.stdin)
    result = asyncio.run(execute(request))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
