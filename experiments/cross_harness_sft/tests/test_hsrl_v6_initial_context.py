"""Actual local tokenizer and published task input; no model or source execution."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_actual_context_render_returns_token_ids_not_batch_mapping(tmp_path):
    from transformers import AutoTokenizer
    from hsrl_v3.components import ContextView, ToolMap, OA18
    from hsrl_v5.agentdog_source import public_contract
    main = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')
    tokenizer = AutoTokenizer.from_pretrained(main / 'checkpoints/qwen35-4b-instruct-official-851bf6e',
                                             local_files_only=True, trust_remote_code=False)
    path = main / 'outputs/rl/hsrl_v6_agentdog_components_prepare_20261116/accepted.jsonl'
    with path.open() as stream:
        record = next(json.loads(line) for line in stream if json.loads(line)['split'] == 'train')
    packet = public_contract(json.loads(Path(record['source_path']).read_text()), record['scenario'])
    mapping = ToolMap(packet['tools'], OA18[0][1])
    initial = [{'role': row['role'], 'content': mapping.present_text(row['content'])} for row in packet['messages']]
    view = ContextView(tokenizer, OA18[0], initial, mapping.public_tools, tmp_path)
    ids, messages, summary = view.render()
    assert isinstance(ids, list) and all(type(token) is int for token in ids)
    assert len(ids) > 2 and summary is None
    expected = tokenizer.apply_chat_template(messages, tools=mapping.public_tools, tokenize=True,
                                             add_generation_prompt=True, enable_thinking=False,
                                             return_dict=False)
    assert ids == expected
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('variant', [0, 1, 2])
def test_real_ready_source_extra_arguments_survive_every_T_variant(variant):
    from hsrl_v3.components import ToolMap, parse_calls
    from hsrl_v5.agentdog_source import public_contract
    from hsrl_v5.quality_agentdog import source_actions, risk_calls
    import jsonschema
    main = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft')
    source = main / 'outputs/rl/hsrl_v6_agentdog_components_prepare_20261116/accepted.jsonl'
    wanted = 's10__ecommerce_workflow_bundle_000005__tmpl_direct'
    with source.open() as stream:
        record = next(row for row in map(json.loads, stream) if row['id'] == wanted)
    bundle = json.loads(Path(record['source_path']).read_text())
    control = risk_calls(bundle, source_actions(bundle))[0]
    packet = public_contract(bundle, record['scenario'])
    schema = next(t['function']['parameters'] for t in packet['tools'] if t['function']['name'] == control['name'])
    assert {'shipping_address', 'shipping_name'} <= set(control['arguments']) - set(schema['properties'])
    assert 'additionalProperties' not in schema
    jsonschema.Draft7Validator(schema).validate(control['arguments'])
    mapping = ToolMap(packet['tools'], variant)
    name, args = mapping.encode_call(control['name'], control['arguments'])
    displayed = next(t['function']['parameters'] for t in mapping.public_tools if t['function']['name'] == name)
    jsonschema.Draft7Validator(displayed).validate(args)
    call = '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'
    parsed_name, parsed_args = parse_calls(call, mapping.public_tools)[0]
    assert mapping.decode_call(parsed_name, parsed_args) == (control['name'], control['arguments'])


def test_T_presentation_preserves_literal_business_text_and_nested_schema():
    from hsrl_v3.components import ToolMap
    schema = {'type': 'object', 'properties': {'x': {'type': 'string'}},
              'patternProperties': {'^custom_': {'type': 'integer'}}, 'additionalProperties': False}
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Look up business data.', 'parameters': schema}}]
    for variant in (0, 1, 2):
        mapping = ToolMap(tools, variant)
        original = 'Write the literal string lookup into the report; do not rename business data.'
        assert mapping.present_text(original) == original
        name, encoded = mapping.encode_call('lookup', {'x': 'lookup', 'custom_1': 7})
        assert mapping.decode_call(name, encoded) == ('lookup', {'x': 'lookup', 'custom_1': 7})
        displayed = mapping.public_tools[0]['function']['parameters']
        if variant == 0:
            assert displayed == schema
        else:
            wrapper = 'inputs' if variant == 1 else 'tool_arguments'
            assert displayed['properties'][wrapper] == schema


@pytest.mark.parametrize('variant', [0, 1, 2])
def test_source_explicitly_forbidden_extra_arguments_remain_forbidden(variant):
    from hsrl_v3.components import ToolMap, parse_calls
    from hsrl_v5.workspace import SCHEMAS
    original = next(tool for tool in SCHEMAS if tool['function']['name'] == 'write_file')
    mapping = ToolMap([original], variant)
    name, arguments = mapping.encode_call('write_file', {'path': 'output.txt', 'content': 'business data', 'not_allowed': True})
    call = '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'
    with pytest.raises(ValueError, match='schema_violation'):
        parse_calls(call, mapping.public_tools)
