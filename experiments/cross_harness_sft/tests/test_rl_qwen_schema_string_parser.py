import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from rl_tool_attempt_audit import parse_strict_tool_turn, attest_native_calls

SCHEMAS=[{'type':'function','function':{'name':'search','parameters':{'type':'object',
 'properties':{'filename':{'type':'string'},'limit':{'type':'integer'},'recipients':{'type':'array','items':{'type':'string'}}},
 'required':['filename'],'additionalProperties':False}}}]


def xml(body):
 return f'<tool_call><function=search>{body}</function></tool_call>'


def test_schema_string_raw_and_quoted_share_exact_audit(monkeypatch):
 monkeypatch.setenv('VERL_QWEN_SCHEMA_STRING_PARAMS','1')
 for raw in [xml('<parameter=filename>feedback.xlsx</parameter>'),xml('<parameter=filename>"feedback.xlsx"</parameter>')]:
  _,calls=parse_strict_tool_turn(raw,SCHEMAS)
  assert calls==[{'name':'search','arguments':{'filename':'feedback.xlsx'}}]
  assert attest_native_calls([raw],[SimpleNamespace(name='search',arguments={'filename':'feedback.xlsx'})],SCHEMAS)
  assert not attest_native_calls([raw],[SimpleNamespace(name='search',arguments={'filename':'other.xlsx'})],SCHEMAS)


def test_qwen_newline_envelope_preserves_meaningful_spaces(monkeypatch):
 monkeypatch.setenv('VERL_QWEN_SCHEMA_STRING_PARAMS','1')
 raw=xml('<parameter=filename>\n  report.txt  \n</parameter>')
 _,calls=parse_strict_tool_turn(raw,SCHEMAS)
 assert calls[0]['arguments']['filename']=='  report.txt  '
 assert attest_native_calls([raw],[SimpleNamespace(name='search',arguments={'filename':'  report.txt  '})],SCHEMAS)


def test_legacy_strict_default(monkeypatch):
 monkeypatch.delenv('VERL_QWEN_SCHEMA_STRING_PARAMS',raising=False)
 with pytest.raises(ValueError,match='not a JSON value'):
  parse_strict_tool_turn(xml('<parameter=filename>feedback.xlsx</parameter>'),SCHEMAS)


@pytest.mark.parametrize('body',[
 '<parameter=filename>"unterminated</parameter>',
 '<parameter=filename></parameter>',
 '<parameter=filename>{malformed}</parameter>',
 '<parameter=filename>ok</parameter><parameter=filename>dup</parameter>',
 '<parameter=filename>ok</parameter><parameter=recipients>[oops]</parameter>',
 '<parameter=filename>ok</parameter><parameter=limit>abc</parameter>',
 '<parameter=unknown>abc</parameter>',
 '<parameter=filename>ok<tool_call></parameter>',
])
def test_ambiguous_invalid_or_unregistered_values_still_fail_closed(monkeypatch,body):
 monkeypatch.setenv('VERL_QWEN_SCHEMA_STRING_PARAMS','1')
 with pytest.raises(ValueError): parse_strict_tool_turn(xml(body),SCHEMAS)
 assert not attest_native_calls([xml(body)],[],SCHEMAS)


def test_unknown_tool_never_uses_schema_string_fallback(monkeypatch):
 monkeypatch.setenv('VERL_QWEN_SCHEMA_STRING_PARAMS','1')
 with pytest.raises(ValueError):
  parse_strict_tool_turn('<tool_call><function=not_declared><parameter=filename>x</parameter></function></tool_call>',SCHEMAS,allow_schema_errors=True)


def test_prompt_contract_tracks_parser_mode(monkeypatch):
 from rl_multiharness_chat_template import benchmark_mcp_chat_template
 monkeypatch.delenv('VERL_QWEN_SCHEMA_STRING_PARAMS',raising=False)
 assert 'String parameters may be plain text' not in benchmark_mcp_chat_template()
 monkeypatch.setenv('VERL_QWEN_SCHEMA_STRING_PARAMS','1')
 assert 'String parameters may be plain text or JSON-quoted' in benchmark_mcp_chat_template()


def test_duplicate_json_keys_and_nonfinite_numbers_fail_closed(monkeypatch):
 monkeypatch.setenv('VERL_QWEN_SCHEMA_STRING_PARAMS','1')
 schema=[{'type':'function','function':{'name':'search','parameters':{'type':'object',
   'properties':{'options':{'type':'object'},'limit':{'type':'number'}},'additionalProperties':False}}}]
 for body in ['<parameter=options>{"x":1,"x":2}</parameter>', '<parameter=limit>NaN</parameter>']:
  with pytest.raises(ValueError): parse_strict_tool_turn(xml(body),schema)


def test_rollout_batch_rejects_mixed_parser_contracts():
 from rl_whitebox_objective import validate_parser_versions
 validate_parser_versions([{'tool_parser_version':'qwen_schema_strings_v1'}], 'qwen_schema_strings_v1')
 with pytest.raises(ValueError,match='parser'):
  validate_parser_versions([{'tool_parser_version':'strict_json_v1'}], 'qwen_schema_strings_v1')
 with pytest.raises(ValueError,match='parser'):
  validate_parser_versions([{}], 'qwen_schema_strings_v1')
