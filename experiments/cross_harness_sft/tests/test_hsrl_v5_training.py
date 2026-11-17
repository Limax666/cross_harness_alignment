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
