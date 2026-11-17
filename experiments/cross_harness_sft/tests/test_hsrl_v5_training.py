"""Pure framing/interface/accounting checks; no model, source or environment execution."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_json_business_markers_are_not_protocol_framing():
    from hsrl_v3.components import parse_calls
    tool = {'type': 'function', 'function': {'name': 'write_file', 'parameters': {
        'type': 'object', 'properties': {'content': {'type': 'string'}},
        'required': ['content'], 'additionalProperties': False}}}
    content = 'literal </think> <tool_call> </tool_call> <think> in a document'
    block = '<tool_call>' + json.dumps({'name': 'write_file', 'arguments': {'content': content}}) + '</tool_call>'
    assert parse_calls('<think>reasoning</think>' + block + block, [tool]) == [
        ('write_file', {'content': content}), ('write_file', {'content': content})]
    with pytest.raises(ValueError):
        parse_calls(block.removesuffix('</tool_call>'), [tool])
    with pytest.raises(ValueError):
        parse_calls('<tool_call>{"name":"write_file","name":"write_file","arguments":{"content":"x"}}</tool_call>', [tool])
    for bad in ('{"name":[],"arguments":{}}',
                '{"name":"write_file","arguments":{"content":1e999}}'):
        with pytest.raises(ValueError):
            parse_calls('<tool_call>' + bad + '</tool_call>', [tool])


class Characters:
    """Deterministic text accounting double, not an experimental model/tokenizer."""
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['return_dict'] is False
        return list(range(sum(len(m['content']) for m in messages)))

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


@pytest.mark.parametrize('variant', [0, 1, 2])
def test_shared_readonly_context_interface_reaches_compacted_content(tmp_path, variant):
    from hsrl_v5.runtime import prepare_interface, CONTEXT_TOOL
    public = {'messages': [{'role': 'user', 'content': 'authorized request'}], 'tools': []}
    prepared = prepare_interface(Characters(), public, (0, variant, 0, 2, 0), tmp_path)
    name, args = prepared.mapping.encode_call(CONTEXT_TOOL, {})
    assert prepared.mapping.decode_call(name, args) == (CONTEXT_TOOL, {})
    assert json.loads(prepared.context.read()) == {'available': False, 'content': None}
    for i in range(7):
        prepared.view.add_round([{'role': 'assistant', 'content': str(i) + 'x' * 1500}])
    _, messages, audit = prepared.view.render()
    assert audit['variant'] == 2 and audit['retained_rounds'] == 3
    assert name in messages[2]['content']
    assert json.loads(prepared.context.read())['content'] == audit['summary']
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        prepare_interface(Characters(), {'messages': [], 'tools': [
            {'type': 'function', 'function': {'name': CONTEXT_TOOL}}]}, (0, 0, 0, 0, 0), tmp_path)


def test_complete_groups_and_distinct_pg_kl_denominators():
    from train_hsrl_v5 import prepare_update
    from hsrl_v5.reward import VERSION
    def episode(reward, pg):
        return {'valid': True, 'reward_contract': VERSION, 'training_reward': reward,
                'pg_applicable': pg, 'segments': [{'prompt_ids': [1], 'response_ids': [2, 3],
                'old_logprobs': [-1., -1.], 'reference_logprobs': [-1., -1.], 'policy_version': 0}]}
    groups = [[episode(float(i % 2), True) for i in range(8)],
              [episode(None, False) for _ in range(8)]]
    prepared = prepare_update(groups, 0)
    assert prepared['pg_tokens'] == 16 and prepared['kl_tokens'] == 32
    assert prepared['groups'][1]['advantages'] == [0.] * 8
    assert prepared['groups'][0]['advantages'][0] < 0 < prepared['groups'][0]['advantages'][1]
    with pytest.raises(ValueError, match='complete_G8'):
        prepare_update([groups[0][:7]], 0)
    groups[0][0]['segments'][0]['policy_version'] = 1
    with pytest.raises(ValueError, match='stale'):
        prepare_update(groups, 0)


def test_attempt_journal_refuses_uncertain_reexecution(tmp_path):
    from train_hsrl_v5 import reserve_attempt, complete_attempt
    state = {'attempts': 0, 'inflight': None, 'episodes': []}
    reserve_attempt(state, tmp_path / 'ep0', max_attempts=128)
    assert state['attempts'] == 1 and state['inflight']
    with pytest.raises(ValueError, match='inflight'):
        reserve_attempt(state, tmp_path / 'ep1', max_attempts=128)
    complete_attempt(state, {'episode_path': 'ep0/episode.json', 'sha256': 'a'})
    assert state['attempts'] == 1 and state['inflight'] is None
    state['attempts'] = 128
    with pytest.raises(ValueError, match='budget'):
        reserve_attempt(state, tmp_path / 'ep2', max_attempts=128)


def test_published_checkpoint_metadata_preserves_tool_framing_and_model_shape():
    root = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/checkpoints/qwen35-4b-instruct-official-851bf6e')
    tokenizer = json.loads((root / 'tokenizer_config.json').read_text())
    tokens = {value['content']: value for value in tokenizer['added_tokens_decoder'].values()}
    assert all(tokens[marker]['special'] is False for marker in
               ('<tool_call>', '</tool_call>', '<think>', '</think>'))
    config = json.loads((root / 'config.json').read_text())
    assert config['text_config']['hidden_size'] == 2560
    assert config['text_config']['num_hidden_layers'] == 32
    assert config['architectures'] == ['Qwen3_5ForConditionalGeneration']


def test_official_identity_rejects_same_basename_and_pins_download_authority(tmp_path):
    import hashlib
    import train_hsrl_v5 as trainer
    with pytest.raises(ValueError, match='canonical'):
        trainer.verify_official_checkpoint(tmp_path / 'qwen35-4b-instruct-official-851bf6e')
    approved = json.loads(trainer.OFFICIAL_IDENTITY_PATH.read_text())
    root = Path(approved['canonical_path'])
    assert approved['source_repo'] == 'Qwen/Qwen3.5-4B'
    assert approved['source_revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    for name, pin in approved['artifacts'].items():
        metadata = root / '.cache/huggingface/download' / (name + '.metadata')
        assert metadata.read_text().splitlines()[:2] == [approved['source_revision'], pin['digest']]
        assert hashlib.sha256(metadata.read_bytes()).hexdigest() == pin['download_metadata_sha256']
    # Actual small official artifact bytes, without reading model shards.
    sha = trainer.verify_pinned_artifact(root / 'config.json', approved['artifacts']['config.json'])
    assert len(sha) == 64


def test_official_content_guard_rejects_changed_small_artifact_and_weight_bytes(tmp_path, monkeypatch):
    import train_hsrl_v5 as trainer
    approved = json.loads(trainer.OFFICIAL_IDENTITY_PATH.read_text())
    root = Path(approved['canonical_path'])
    config = tmp_path / 'config.json'
    config.write_bytes((root / 'config.json').read_bytes() + b'\n')
    with pytest.raises(ValueError, match='official_artifact'):
        trainer.verify_pinned_artifact(config, approved['artifacts']['config.json'])
    # Drive the complete verifier on a private approved-map fixture retaining
    # the actual official config object ID. No real checkpoint file is edited.
    canonical = tmp_path / 'canonical'
    canonical.mkdir()
    (canonical / 'config.json').write_bytes((root / 'config.json').read_bytes())
    fixture = {**approved, 'canonical_path': str(canonical),
               'artifacts': {'config.json': approved['artifacts']['config.json']}}
    fixture_path = tmp_path / 'approved.json'
    fixture_path.write_text(json.dumps(fixture))
    monkeypatch.setattr(trainer, 'OFFICIAL_IDENTITY_PATH', fixture_path)
    assert trainer.verify_official_checkpoint(canonical)['approved_identity'] == fixture
    (canonical / 'config.json').write_bytes(config.read_bytes())
    with pytest.raises(ValueError, match='official_artifact'):
        trainer.verify_official_checkpoint(canonical)
    # Small byte fixture exercises the same LFS hash guard as the real shards.
    shard = tmp_path / 'model.safetensors'
    pin = {'algorithm': 'sha256', 'digest': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}
    shard.write_bytes(b'abc')
    assert trainer.verify_pinned_artifact(shard, pin) == pin['digest']
    shard.write_bytes(b'abd')
    with pytest.raises(ValueError, match='official_artifact'):
        trainer.verify_pinned_artifact(shard, pin)


@pytest.mark.parametrize('failed_barrier', ['none', 'adapter', 'installed_parent'])
def test_checkpoint_directory_barriers_precede_authoritative_journal(tmp_path, monkeypatch, failed_barrier):
    import os
    from hsrl_v3 import policy as policy_module
    import train_hsrl_v5 as trainer

    class ArtifactWriter:
        def save_pretrained(self, directory, **kwargs):
            (directory / 'nested').mkdir(parents=True)
            (directory / 'adapter_model.safetensors').write_bytes(b'adapter-serialization-boundary')
            (directory / 'nested/metadata.json').write_text('{}')

    class Serializer:
        def save(self, value, destination):
            data = json.dumps(value).encode()
            if hasattr(destination, 'write'):
                destination.write(data)
            else:
                Path(destination).write_bytes(data)

    policy = object.__new__(policy_module.Policy)
    policy.model, policy.torch = ArtifactWriter(), Serializer()
    policy.version, policy.frozen_signature = 1, 'metadata-guard'
    policy.adapter_hash, policy.rng_state = lambda: 'adapter-hash', lambda: {}
    optimizer = type('OptimizerState', (), {'state_dict': lambda self: {}})()
    checkpoint = tmp_path / 'checkpoints/update1'
    previous = tmp_path / 'checkpoints/previous-complete'
    policy.save(previous, optimizer, {'checkpoint': str(previous)})
    old = {'state': {'checkpoint': str(previous)}, 'rng': {}}
    (tmp_path / 'journal.pt').write_text(json.dumps(old))
    state = {'checkpoint': str(checkpoint)}
    events = []
    real_fsync, real_rename = os.fsync, os.rename

    def fsync(fd):
        path = Path(os.readlink(f'/proc/self/fd/{fd}'))
        events.append(('fsync', path))
        if failed_barrier == 'adapter' and path.is_dir() and path.name == 'adapter':
            raise OSError('injected_adapter_directory_barrier_failure')
        if (failed_barrier == 'installed_parent' and path == checkpoint.parent
                and any(kind == 'rename' for kind, _ in events)):
            raise OSError('injected_installed_parent_directory_barrier_failure')
        real_fsync(fd)

    def rename(source, destination):
        events.append(('rename', Path(source)))
        real_rename(source, destination)

    monkeypatch.setattr(policy_module.os, 'fsync', fsync)
    monkeypatch.setattr(policy_module.os, 'rename', rename)
    if failed_barrier != 'none':
        with pytest.raises(OSError, match='directory_barrier_failure'):
            trainer.commit_checkpoint(tmp_path, policy, optimizer, None, state, current_state=old['state'])
        assert json.loads((tmp_path / 'journal.pt').read_text()) == old
        policy_module.validate_checkpoint_files(Path(json.loads((tmp_path / 'journal.pt').read_text())['state']['checkpoint']))
        if failed_barrier == 'adapter':
            assert not checkpoint.exists()
            assert any((p / 'COMPLETE.json').is_file() for p in checkpoint.parent.glob('.update1-*'))
            assert not any(kind == 'rename' for kind, _ in events)
        else:
            # Installed but unreferenced recovery point is retained, and it is
            # never selected over the last authoritative journal checkpoint.
            policy_module.validate_checkpoint_files(checkpoint)
    else:
        trainer.commit_checkpoint(tmp_path, policy, optimizer, None, state)
        install = next(i for i, event in enumerate(events) if event[0] == 'rename')
        staging = events[install][1]
        # Paths moved during rename no longer exist; compare their recorded names.
        all_barriers = [path for kind, path in events[:install] if kind == 'fsync']
        assert all_barriers.index(staging / 'adapter/nested') < all_barriers.index(staging / 'adapter')
        assert all_barriers.index(staging / 'adapter') < all_barriers.index(staging)
        assert staging.parent in all_barriers
        journal_sync = next(i for i, (kind, path) in enumerate(events)
                            if kind == 'fsync' and path.name == '.journal.pt.tmp')
        assert any(kind == 'fsync' and path == checkpoint.parent for kind, path in events[install+1:journal_sync])
        assert json.loads((tmp_path / 'journal.pt').read_text())['state'] == state
        policy_module.validate_checkpoint_files(checkpoint)
