"""Contract checks on the real public SkillLife tool schemas; no episodes/models."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
SOURCE = Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/vendor/skilllifebench_security_research_20261116')


def component():
    assert importlib.util.find_spec('hsrl_v3.components'), 'five-dimensional runtime is missing'
    from hsrl_v3 import components
    return components


def tools():
    record = json.loads((SOURCE / 'data/lifecycle/protocol/task-manifest.json').read_text())['tasks'][0]
    return [{'type': 'function', 'function': {key: row[key] for key in ('name', 'description', 'parameters')}}
            for row in record['condition_tools']['clean']]


def test_oa18_covers_every_pair_twice_and_peer_is_nearest():
    c = component()
    rows = c.OA18
    assert len(set(rows)) == 18
    for left, right in itertools.combinations(range(5), 2):
        pairs = [(row[left], row[right]) for row in rows]
        assert all(pairs.count(pair) == 2 for pair in itertools.product(range(3), repeat=2))
    for row in rows:
        peer = c.nearest_peer(row)
        distance = lambda x: sum(a != b for a, b in zip(row, x))
        assert distance(peer) == min(distance(other) for other in rows if other != row)


@pytest.mark.parametrize('variant', [0, 1, 2])
def test_real_source_argument_and_tool_names_roundtrip(variant):
    c = component()
    mapping = c.ToolMap(tools(), variant)
    for original in tools():
        fn = original['function']
        # Source public schema has workspace_id etc. No action is executed here.
        args = {key: 'a-real-public-argument-value' for key in fn['parameters'].get('properties', {})}
        name, displayed = mapping.encode_call(fn['name'], args)
        assert mapping.decode_call(name, displayed) == (fn['name'], args)
    assert len(mapping.public_tools) == len(tools()) + (0 if variant == 0 else 2)
    with pytest.raises(ValueError):
        mapping.decode_call('not_advertised', {})


def test_qwen_native_xml_and_json_calls_preserve_multiline_string():
    c = component()
    schema = {'type': 'function', 'function': {'name': 'write_file', 'parameters': {
        'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
        'required': ['path', 'content'], 'additionalProperties': False}}}
    raw = '<think>private reasoning</think><tool_call>\n<function=write_file>\n<parameter=path>\nx.txt\n</parameter>\n<parameter=content>\na\nb\n</parameter>\n</function>\n</tool_call>'
    parsed = c.parse_calls(raw, [schema])
    assert parsed == [('write_file', {'path': 'x.txt', 'content': 'a\nb'})]
    parsed = c.parse_calls('<tool_call>{"name":"write_file","arguments":{"path":"x.txt","content":"a\\nb"}}</tool_call>', [schema])
    assert parsed[0][1]['content'] == 'a\nb'
    with pytest.raises(ValueError):
        c.parse_calls('<tool_call><function=write_file><parameter=path>x.txt</parameter></function></tool_call>', [schema])
    with pytest.raises(ValueError):
        c.parse_calls('<tool_call><function=unknown></function></tool_call>', [schema])


def test_workspace_path_cannot_reach_evidence_or_source(tmp_path):
    assert importlib.util.find_spec('hsrl_v3.sandbox'), 'private tool sandbox is missing'
    from hsrl_v3.sandbox import scoped_path
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    outside = tmp_path / 'private'
    outside.mkdir()
    (workspace / 'link').symlink_to(outside, target_is_directory=True)
    assert scoped_path(workspace, '/root/.openclaw/workspace/output.json') == workspace / 'output.json'
    for path in ('../private', '/etc/passwd', 'link/oracle.json', '/root/.openclaw/workspace/../private'):
        with pytest.raises(ValueError):
            scoped_path(workspace, path)
