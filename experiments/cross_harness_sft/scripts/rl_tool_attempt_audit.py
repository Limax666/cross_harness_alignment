"""Fail-closed check of Qwen3.5 XML tool attempts against native execution.

Requires *every* generated assistant turn, including turns with no parsed tools.
It does not decode assistant tokens or reconstruct VeRL's rollout history; the
caller must prove those turns are complete before relying on its verdict.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

_BLOCK = re.compile(r"<tool_call>\s*<function=([a-zA-Z_][\w.-]*)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
_EMPTY_ELEMENT_BLOCK = re.compile(
    r"<tool_call>\s*<([a-zA-Z_][\w.-]*)>\s*</\1>\s*</tool_call>", re.S
)
_NAMED_ELEMENT_BLOCK = re.compile(
    r"<tool_call>\s*<([a-zA-Z_][\w.-]*)>\s*(.*?)\s*</\1>\s*</tool_call>", re.S
)
_PARAM = re.compile(r"<parameter=([a-zA-Z_][\w.-]*)>\s*(.*?)\s*</parameter>", re.S)
_UNCLOSED_FUNCTION_BLOCK = re.compile(
    r"<tool_call>\s*<function=([a-zA-Z_][\w.-]*)>\s*</tool_call>", re.S
)
_MARKER = re.compile(r"</?tool_call>|</?function\b|<function=|</?parameter\b|<parameter=")


def _json_object_no_duplicates(source: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(source, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise ValueError("tool arguments must be a JSON object")
    return result


def _parse_body_arguments(body: str) -> dict:
    residue = _PARAM.sub("", body).strip()
    if residue:
        if _PARAM.search(body):
            raise ValueError("mixed JSON and parameter tags")
        try:
            return _json_object_no_duplicates(residue)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("malformed tool arguments") from exc
    arguments = {}
    for param in _PARAM.finditer(body):
        key = param.group(1)
        if key in arguments:
            raise ValueError(f"duplicate tool argument: {key}")
        try:
            arguments[key] = json.loads(param.group(2))
        except json.JSONDecodeError as exc:
            raise ValueError(f"argument {key} is not a JSON value") from exc
    return arguments


def _canonicalize_empty_element_calls(raw: str, schemas: Sequence[dict]) -> str:
    """Accept only an exact, unambiguous no-argument XML element for a known tool.

    This is a model-output adapter, not a Hermes fuzzy-name repair. A tool with
    required arguments stays invalid; mismatched tags or body content are never
    rewritten. The raw generated text remains unchanged in the audit record.
    """
    declared = {item["function"]["name"]: item["function"]["parameters"] for item in schemas}

    def replace(match: re.Match) -> str:
        name = match.group(1)
        schema = declared.get(name)
        if schema is None or schema.get("required"):
            return match.group(0)
        return f"<tool_call><function={name}></function></tool_call>"

    return _EMPTY_ELEMENT_BLOCK.sub(replace, raw)


def _canonicalize_named_element_calls(raw: str, schemas: Sequence[dict]) -> str:
    """Accept an exact declared tool element with fully parsed JSON arguments.

    This translates an alternate *unambiguous* envelope only. It never repairs
    names, guesses parameter values, or changes the retained raw assistant text.
    """
    declared = {item["function"]["name"] for item in schemas}

    def replace(match: re.Match) -> str:
        name, body = match.groups()
        if name not in declared or not body.strip():
            return match.group(0)
        try:
            _parse_body_arguments(body)
        except ValueError:
            return match.group(0)
        return f"<tool_call><function={name}>{body}</function></tool_call>"

    return _NAMED_ELEMENT_BLOCK.sub(replace, raw)


def _canonicalize_unclosed_no_body_function_calls(raw: str, schemas: Sequence[dict]) -> str:
    """Accept a ``function=`` envelope whose block ends without ``</function>``.

    This is a model-output adapter for the empty-body case only: the rewrite
    closes the envelope and never invents, moves, or repairs arguments. All
    semantic gating (unknown tool names, missing required arguments) stays in
    parse_strict_tool_turn, where these calls surface as blocked attempts
    instead of terminations. Any body content keeps the turn malformed.
    """
    def replace(match: re.Match) -> str:
        return f"<tool_call><function={match.group(1)}></function></tool_call>"

    return _UNCLOSED_FUNCTION_BLOCK.sub(replace, raw)


def _canonicalize_tool_calls(raw: str, schemas: Sequence[dict]) -> str:
    return _canonicalize_named_element_calls(
        _canonicalize_empty_element_calls(
            _canonicalize_unclosed_no_body_function_calls(raw, schemas), schemas),
        schemas)


def parse_strict_tool_turn(raw_assistant: str, schemas: Sequence[dict], *,
                           allow_schema_errors: bool = False) -> tuple[str, list[dict]]:
    """Parse SFT's JSON_VALUE XML syntax without repairing names or arguments.

    Raises before tool execution for malformed markup, unknown tools, duplicate
    arguments, absent required arguments, and values incompatible with the
    declared JSON schema. The returned content is the assistant text outside
    tool blocks; the raw generation must still be retained for the path audit.
    """
    if not isinstance(raw_assistant, str) or not raw_assistant:
        raise ValueError("empty assistant generation")
    raw_assistant = _canonicalize_tool_calls(raw_assistant, schemas)
    declared = {item["function"]["name"]: item["function"]["parameters"] for item in schemas}
    outside = _BLOCK.sub("", raw_assistant)
    if _MARKER.search(outside):
        raise ValueError("malformed or unclosed tool call")
    calls = []
    for match in _BLOCK.finditer(raw_assistant):
        name, body = match.groups()
        if name not in declared and not allow_schema_errors:
            raise ValueError(f"unknown tool: {name}")
        arguments = _parse_body_arguments(body)
        if name not in declared:
            calls.append({"name": name, "arguments": arguments,
                          "blocked_reason": f"Unknown tool '{name}'. Use an exact available tool name."})
            continue
        schema = declared[name]
        missing = [key for key in schema.get("required", []) if key not in arguments or arguments[key] in (None, "")]
        if missing:
            if allow_schema_errors:
                calls.append({"name": name, "arguments": arguments,
                              "blocked_reason": f"Missing required arguments for {name}: {missing}"})
                continue
            raise ValueError(f"missing required arguments for {name}: {missing}")
        properties = schema.get("properties", {})
        for key, value in arguments.items():
            spec = properties.get(key)
            if spec is None:
                if schema.get("additionalProperties") is False:
                    if allow_schema_errors:
                        calls.append({"name": name, "arguments": arguments,
                                      "blocked_reason": f"Undeclared argument for {name}: {key}"})
                        break
                    raise ValueError(f"undeclared argument for {name}: {key}")
                continue
            expected = spec.get("type")
            valid = {
                "string": lambda v: isinstance(v, str),
                "integer": lambda v: type(v) is int,
                "number": lambda v: type(v) in (int, float),
                "boolean": lambda v: type(v) is bool,
                "object": lambda v: isinstance(v, dict),
                "array": lambda v: isinstance(v, list),
            }
            if expected in valid and not valid[expected](value):
                if allow_schema_errors:
                    calls.append({"name": name, "arguments": arguments,
                                  "blocked_reason": f"Argument {key} has wrong type for {name}"})
                    break
                raise ValueError(f"argument {key} has wrong type for {name}")
        else:
            calls.append({"name": name, "arguments": arguments})
    return outside.strip(), calls


def attest_native_calls(raw_assistant_turns: Sequence[str], native_calls: Sequence[Any],
                        schemas: Sequence[dict] | None = None) -> bool:
    """True only for well-formed calls in execution order with identical args.

    No raw turns, leftover call markup, duplicate keys, missing/invalid JSON or
    extra native calls return False. Unknown tool names remain visible and must
    also be rejected by the native path auditor.
    """
    if not raw_assistant_turns or any(not isinstance(t, str) for t in raw_assistant_turns):
        return False
    parsed: list[tuple[str, dict]] = []
    for text in raw_assistant_turns:
        if schemas is not None:
            text = _canonicalize_tool_calls(text, schemas)
        outside = _BLOCK.sub("", text)
        if _MARKER.search(outside):
            return False
        for match in _BLOCK.finditer(text):
            name, body = match.groups()
            try:
                params = _parse_body_arguments(body)
            except ValueError:
                return False
            parsed.append((name, params))
    return len(parsed) == len(native_calls) and all(
        name == call.name and args == call.arguments
        for (name, args), call in zip(parsed, native_calls)
    )
