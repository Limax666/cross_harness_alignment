#!/usr/bin/env python3
"""Minimal local OpenAI-compatible server for HarnessRisk harness adapters."""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
import uuid
import traceback
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
BARE_FUNC_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>(.*?)(?:</function>|</\1>)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)
TOOL_NAME_ALIASES = {"skill_search": "tool_search"}


def _function_name_and_inline_args(raw_name):
    """Accept a literal pseudo-call, but never execute model-generated Python."""
    if "(" not in raw_name:
        return raw_name, {}
    try:
        expression = ast.parse(raw_name, mode="eval").body
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.args:
            raise ValueError("expected function(keyword=value, ...) with literal values")
        arguments = {}
        for keyword in expression.keywords:
            if keyword.arg is None or keyword.arg in arguments:
                raise ValueError("duplicate or expanded tool argument")
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        return expression.func.id, arguments
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(f"invalid tool-call signature: {raw_name}") from exc


def _tool_schemas(tools):
    schemas = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        name = function.get("name")
        if name:
            schemas[name] = function.get("parameters") or {}
    return schemas


def parse_tool_calls(text, tools=None):
    calls = []
    errors = []
    schemas = _tool_schemas(tools)
    for block in CALL_RE.findall(text):
        m = FUNC_RE.search(block)
        if m:
            raw_name = m.group(1).strip()
            body = m.group(2)
        else:
            m = BARE_FUNC_RE.search(block)
            if not m:
                errors.append("malformed <function> block")
                continue
            raw_name = m.group(1).strip()
            body = m.group(2)
        try:
            name, args = _function_name_and_inline_args(raw_name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        name = TOOL_NAME_ALIASES.get(name, name)
        if name not in schemas:
            qualified = [candidate for candidate in schemas if candidate.rsplit("__", 1)[-1] == name]
            if len(qualified) == 1:
                name = qualified[0]
        for key, value in PARAM_RE.findall(body):
            if key in args:
                errors.append(f"{name}: duplicate argument {key}")
                break
            try:
                args[key] = json.loads(value.strip())
            except json.JSONDecodeError:
                args[key] = value.strip()
        else:
            # Hermes' deferred-tool bridge accepts a legacy single-call shape,
            # while its advertised schema requires a calls[] array. Normalize
            # only that documented shape before validating required fields.
            if name == "tool_call" and "calls" not in args and isinstance(args.get("name"), str):
                deferred_args = args.get("arguments", {})
                if isinstance(deferred_args, str):
                    try:
                        deferred_args = json.loads(deferred_args)
                    except json.JSONDecodeError:
                        pass
                if isinstance(deferred_args, dict):
                    args = {"calls": [{"name": args["name"], "arguments": deferred_args}]}
            if name not in schemas:
                errors.append(f"unknown tool {name!r}; available tools: {', '.join(sorted(schemas))}")
                continue
            required = schemas[name].get("required") or []
            missing = [key for key in required if key not in args or args[key] is None or args[key] == ""]
            if missing:
                errors.append(f"{name}: missing required argument(s): {', '.join(missing)}")
                continue
            calls.append((name, args))
    return calls, errors


def visible(text):
    text = CALL_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    # Base Qwen instruct prompts end with control tokens. When no tool call is
    # emitted, the generated assistant turn may include a close marker followed
    # by parts of the serialized prompt; never feed those back as assistant text.
    for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.split(marker, 1)[0]
    return text.strip()


def normalize_messages(messages):
    normalized = []
    for original in messages:
        message = dict(original)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    function["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    function["arguments"] = {}
            call["function"] = function
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        normalized.append(message)
    return normalized


def build_server(args):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = None if args.full_precision else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype,
        use_kernels=False, quantization_config=quant, device_map={"": 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return tokenizer, model


def segment_prompt(messages, tools):
    """Match the role-segment format used by the full-SFT data release."""
    parts = []
    if tools:
        parts.append("<|im_start|>system\nAvailable tools (call only these functions):\n"
                     + json.dumps(tools, ensure_ascii=False)
                     + "\nTool-call syntax: <tool_call><function=TOOL_NAME>"
                       "<parameter=ARG_NAME>JSON_VALUE</parameter></function></tool_call>. "
                       "Keep TOOL_NAME separate from arguments; include every required argument "
                       "listed in its schema. For example, skill_view requires "
                       "<function=skill_view><parameter=name>\"hermes-agent\"</parameter></function>."
                       "<|im_end|>\n")
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""
        if role == "assistant" and message.get("tool_calls"):
            blocks = []
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                name = function.get("name") or ""
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                params = "\n".join(f"<parameter={key}>\n{json.dumps(value, ensure_ascii=False)}\n</parameter>"
                                   for key, value in arguments.items())
                blocks.append(f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
            content = content.rstrip() + ("\n\n" if content.strip() else "") + "\n".join(blocks)
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def make_handler(tokenizer, model, model_id, max_new_tokens, max_input_tokens, prompt_format, temperature, max_requests_per_episode):
    generation_lock = threading.Lock()
    episode = {"requests": 0}
    tool_end_ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    turn_end_ids = [
        tokenizer.encode(marker, add_special_tokens=False)
        for marker in ("<|im_end|>", "<|endoftext|>")
    ]

    class StopAtTurnBoundary(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[0].tolist()
            if tool_end_ids and generated[-len(tool_end_ids):] == tool_end_ids:
                return True
            return any(ids and generated[-len(ids):] == ids for ids in turn_end_ids)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print("[server]", fmt % values, flush=True)

        def send_json(self, value, status=200):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def send_stream(self, response):
            choice = response["choices"][0]
            message = choice["message"]
            delta = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("tool_calls"):
                delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
            for item in (
                {"id": response["id"], "object": "chat.completion.chunk", "created": response["created"],
                 "model": response["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"id": response["id"], "object": "chat.completion.chunk", "created": response["created"],
                 "model": response["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
                 "usage": response["usage"]},
            ):
                self.wfile.write(("data: " + json.dumps(item, ensure_ascii=False) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def heartbeat(self, stopped):
            while not stopped.wait(10):
                try:
                    self.wfile.write(b": generating\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

        def do_GET(self):
            if self.path == "/health":
                return self.send_json({"status": "ok", "model": model_id,
                                       "max_new_tokens": max_new_tokens,
                                       "max_input_tokens": max_input_tokens,
                                       "prompt_format": prompt_format,
                                       "temperature": temperature})
            if self.path == "/v1/models": return self.send_json({"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]})
            self.send_json({"error": "not_found"}, 404)

        def do_POST(self):
            if self.path == "/admin/seed":
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    seed = int(body["seed"])
                    with generation_lock:
                        torch.manual_seed(seed)
                        episode["requests"] = 0
                    print("[episode-seed]", seed, flush=True)
                    return self.send_json({"seed": seed})
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    return self.send_json({"error": str(exc)}, 400)
            if self.path == "/api/show":
                return self.send_json({"name": model_id, "model": model_id, "modified_at": time.time()})
            if self.path != "/v1/chat/completions": return self.send_json({"error": "not_found"}, 404)
            try:
                n = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(n))
                messages = normalize_messages(body.get("messages") or []); tools = body.get("tools") or None
                with generation_lock:
                    episode["requests"] += 1
                    episode_request = episode["requests"]
                if max_requests_per_episode and episode_request > max_requests_per_episode:
                    print("[episode-request-cap]", episode_request, flush=True)
                    response = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
                        "created": int(time.time()), "model": model_id,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "I cannot complete this task within the allowed turns."}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        return self.send_stream(response)
                    return self.send_json(response)
                print("[request]", "stream=" + str(body.get("stream", False)),
                      "messages=" + str(len(messages)), "tools=" + str(len(tools or [])), flush=True)
                streaming = bool(body.get("stream"))
                stopped = threading.Event()
                if streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                    threading.Thread(target=self.heartbeat, args=(stopped,), daemon=True).start()
                if prompt_format == "segment":
                    encoded = tokenizer(segment_prompt(messages, tools), return_tensors="pt", add_special_tokens=False)
                else:
                    encoded = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                        add_generation_prompt=True, return_tensors="pt")
                if isinstance(encoded, Mapping):
                    ids = encoded["input_ids"]; mask = encoded.get("attention_mask")
                else: ids, mask = encoded, None
                print("[prompt-tokens]", int(ids.shape[-1]), flush=True)
                # HarnessRisk multi-turn transcripts can contain very large tool outputs.
                # Keep the system prompt and the most recent context within the 4090 budget.
                if ids.shape[-1] > max_input_tokens:
                    keep = max_input_tokens
                    ids = torch.cat([ids[:, : min(2048, keep // 2)], ids[:, -(keep - min(2048, keep // 2)):]], dim=-1)
                    if mask is not None:
                        mask = torch.ones_like(ids)
                    print("[context-truncated]", "tokens=" + str(int(ids.shape[-1])), flush=True)
                device = next(model.parameters()).device
                inputs = {"input_ids": ids.to(device)}
                if mask is not None: inputs["attention_mask"] = mask.to(device)
                try:
                    with generation_lock, torch.inference_mode():
                        requested = body.get("max_completion_tokens") or body.get("max_tokens") or max_new_tokens
                        out = model.generate(**inputs, do_sample=temperature > 0,
                            **({"temperature": temperature} if temperature > 0 else {}),
                            max_new_tokens=min(max_new_tokens, int(requested)),
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True,
                            stopping_criteria=StoppingCriteriaList([StopAtTurnBoundary()]))
                finally:
                    stopped.set()
                raw = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=False)
                print("[raw-completion]", repr(raw[:500]), flush=True)
                for stop in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"): raw = raw.split(stop, 1)[0]
                calls, call_errors = parse_tool_calls(raw, tools)
                content = visible(raw)
                if call_errors:
                    # Do not send malformed calls to a harness that may fuzzy-repair
                    # the tool name while silently dropping required arguments.
                    print("[invalid-tool-call]", json.dumps(call_errors, ensure_ascii=False), flush=True)
                    calls = []
                    content = (content + "\n" if content else "") + "Tool call rejected: " + "; ".join(call_errors)
                message = {"role": "assistant", "content": content}
                if calls:
                    message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(params, ensure_ascii=False)}} for name, params in calls]
                response = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
                    "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": int(ids.shape[-1]), "completion_tokens": int(out.shape[-1] - ids.shape[-1]), "total_tokens": int(out.shape[-1])}}
                if body.get("stream"):
                    self.send_stream(response)
                else:
                    self.send_json(response)
                print("[completion]", "tokens=" + str(int(out.shape[-1] - ids.shape[-1])),
                      "tool_calls=" + str(len(calls)), flush=True)
            except (BrokenPipeError, ConnectionResetError):
                print("[client-disconnected]", flush=True)
            except Exception as exc:
                print("[server-error]", repr(exc), traceback.format_exc(), flush=True)
                self.send_json({"error": {"message": str(exc), "type": type(exc).__name__}}, 500)
    return Handler


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--role", choices=("base", "sft"), required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base"); ap.add_argument("--adapter")
    ap.add_argument("--model-id", default="qwen35-9b-local"); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9000); ap.add_argument("--max-new-tokens", type=int, default=384); ap.add_argument("--max-input-tokens", type=int, default=8192)
    ap.add_argument("--full-precision", action="store_true", help="Load BF16 weights without 4-bit quantization")
    ap.add_argument("--prompt-format", choices=("native", "segment"), default="native")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-requests-per-episode", type=int, default=0)
    args = ap.parse_args()
    if args.temperature < 0:
        ap.error("--temperature must be non-negative")
    torch.manual_seed(args.seed)
    tokenizer, model = build_server(args)
    print(json.dumps({"role": args.role, "model": args.model_id, "port": args.port, "cuda": torch.cuda.get_device_name(0)}), flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(tokenizer, model, args.model_id, args.max_new_tokens, args.max_input_tokens, args.prompt_format, args.temperature, args.max_requests_per_episode)).serve_forever()


if __name__ == "__main__": main()
