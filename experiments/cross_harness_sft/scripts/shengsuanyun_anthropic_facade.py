"""Anthropic Messages → OpenAI chat/completions facade for Claude Code + shengsuanyun.

Claude Code only speaks Anthropic Messages protocol. shengsuanyun (router.shengsuanyun.com)
only supports OpenAI-compatible chat/completions. This facade:

1. Accepts Anthropic Messages API requests from Claude Code on localhost
2. Converts them to OpenAI chat/completions format
3. Forwards to shengsuanyun
4. Converts responses back to Anthropic Messages format (including streaming SSE)

This preserves tool_use/tool_result blocks, thinking/reasoning content, and multi-turn
conversation structure so the collected trajectories contain complete tool call details.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Config:
    host = "127.0.0.1"
    port = 8411
    upstream = "https://router.shengsuanyun.com/api/v1"
    upstream_key_env = "SHENGSUANYUN_API_KEY"
    local_token = ""
    timeout_read = 900.0
    max_attempts = 3
    upstream_model = ""


STATS: dict[str, Any] = {"requests": 0, "ok": 0, "errors": 0, "first_seen": "", "last_request_at": ""}
LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def anthropic_to_openai_messages(anthropic_messages: list[dict], system: str | list | None = None) -> list[dict]:
    """Convert Anthropic Messages format to OpenAI chat/completions messages."""
    openai_msgs: list[dict] = []

    # System prompt
    if system:
        if isinstance(system, str):
            openai_msgs.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                openai_msgs.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                openai_msgs.append({"role": "user", "content": content})
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            # tool_result → OpenAI tool message
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                tool_content = "\n".join(
                                    b.get("text", "") for b in tool_content if isinstance(b, dict) and b.get("type") == "text"
                                )
                            openai_msgs.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(tool_content),
                            })
                    elif isinstance(block, str):
                        text_parts.append(block)
                if text_parts:
                    openai_msgs.append({"role": "user", "content": "\n".join(text_parts)})

        elif role == "assistant":
            if isinstance(content, str):
                openai_msgs.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "thinking":
                            # Preserve thinking as text prefix
                            thinking = block.get("thinking", "")
                            if thinking:
                                text_parts.append(f"<thinking>{thinking}</thinking>")
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                                },
                            })
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                else:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                openai_msgs.append(assistant_msg)

    return openai_msgs


def openai_to_anthropic_tool_definitions(tools: list[dict]) -> list[dict]:
    """Pass through tool definitions - both formats use similar JSON schema structure."""
    # Anthropic and OpenAI tool schemas are similar enough that we can pass through
    # with minor field name adjustments
    result = []
    for tool in tools:
        if "function" in tool:
            # OpenAI format → Anthropic format
            fn = tool["function"]
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        else:
            # Already in Anthropic format or close enough
            result.append(tool)
    return result


def convert_openai_response_to_anthropic(openai_resp: dict, model: str) -> dict:
    """Convert a non-streaming OpenAI response to Anthropic Messages format."""
    choice = (openai_resp.get("choices") or [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "end_turn")

    content_blocks: list[dict] = []

    # Text content
    text = message.get("content")
    if text:
        # Check for thinking tags
        if "<thinking>" in text and "</thinking>" in text:
            start = text.index("<thinking>")
            end = text.index("</thinking>") + len("</thinking>")
            thinking_text = text[start + len("<thinking>"):end - len("</thinking>")]
            remaining = (text[:start] + text[end:]).strip()
            content_blocks.append({"type": "thinking", "thinking": thinking_text})
            if remaining:
                content_blocks.append({"type": "text", "text": remaining})
        else:
            content_blocks.append({"type": "text", "text": text})

    # Tool calls
    for tc in message.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
        except (json.JSONDecodeError, TypeError):
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{int(time.time()*1000)}"),
            "name": tc["function"]["name"],
            "input": args,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason = "tool_use" if message.get("tool_calls") else ("end_turn" if finish_reason == "stop" else finish_reason)

    usage = openai_resp.get("usage", {})
    return {
        "id": openai_resp.get("id", f"msg_{int(time.time()*1000)}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def stream_openai_to_anthropic(response_stream, model: str, wfile, send_header_fn, end_headers_fn):
    """Convert streaming OpenAI SSE to Anthropic Messages SSE format."""
    msg_id = f"msg_{int(time.time()*1000)}"

    # Send message_start
    start_event = {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "usage": {"input_tokens": 0, "output_tokens": 0},
    }}
    wfile.write(f"event: message_start\ndata: {json.dumps(start_event)}\n\n".encode())
    wfile.flush()

    content_block_index = 0
    current_block_type = None
    total_input = 0
    total_output = 0

    def close_block():
        nonlocal content_block_index, current_block_type
        if current_block_type is not None:
            wfile.write(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n".encode())
            wfile.flush()
            content_block_index += 1
            current_block_type = None

    for raw_line in response_stream:
        line = raw_line.decode("utf-8", "replace").strip() if isinstance(raw_line, bytes) else raw_line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        usage = chunk.get("usage", {})
        if usage:
            total_input = usage.get("prompt_tokens", total_input)
            total_output = usage.get("completion_tokens", total_output)

        # Thinking / reasoning content
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            if current_block_type != "thinking":
                close_block()
                current_block_type = "thinking"
                wfile.write(f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n".encode())
                wfile.flush()
            wfile.write(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'thinking_delta', 'thinking': reasoning}})}\n\n".encode())
            wfile.flush()

        # Text content
        text = delta.get("content")
        if text:
            if current_block_type != "text":
                close_block()
                current_block_type = "text"
                wfile.write(f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode())
                wfile.flush()
            wfile.write(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'text_delta', 'text': text}})}\n\n".encode())
            wfile.flush()

        # Tool calls
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                idx = tc.get("index", 0)
                if current_block_type != "tool_use" or idx != content_block_index:
                    close_block()
                    current_block_type = "tool_use"
                    fn_name = tc.get("function", {}).get("name", "")
                    tc_id = tc.get("id", f"toolu_{int(time.time()*1000)}")
                    wfile.write(f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': fn_name, 'input': {}}})}\n\n".encode())
                    wfile.flush()
                args_delta = tc.get("function", {}).get("arguments", "")
                if args_delta:
                    wfile.write(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_delta}})}\n\n".encode())
                    wfile.flush()

    close_block()

    # message_delta with stop_reason
    finish = (chunk.get("choices") or [{}])[0].get("finish_reason", "stop") if chunk else "stop"
    stop_reason = "tool_use" if current_block_type == "tool_use" else "end_turn"
    wfile.write(f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': {'output_tokens': total_output}})}\n\n".encode())
    wfile.write(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode())
    wfile.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "shengsuanyun-anthropic-facade/1.0"

    def log_message(self, fmt, *args):
        print(f"[shengsuanyun-facade] {fmt % args}", flush=True)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in {"/health", "/healthz"}:
            data = json.dumps({"ok": True, **STATS}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
        self.close_connection = True

    def do_HEAD(self):
        self.send_response(200 if self.path.split("?")[0] in {"/health", "/healthz"} else 404)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_POST(self):
        length = int(self.headers.get("content-length", "0") or 0)
        body = self.rfile.read(length) if length else b""

        # Auth check
        if Config.local_token:
            presented = (self.headers.get("x-api-key") or "").strip()
            auth = (self.headers.get("authorization") or "").strip()
            if auth.lower().startswith("bearer "):
                presented = presented or auth[7:].strip()
            if presented != Config.local_token:
                self._send_error(401, "authentication_error", "Invalid API key")
                return

        api_key = os.environ.get(Config.upstream_key_env, "").strip()
        if not api_key:
            self._send_error(500, "server_error", f"{Config.upstream_key_env} not set on facade")
            return

        try:
            request_body = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error(400, "invalid_request_error", "Invalid JSON body")
            return

        model = request_body.get("model", "")
        is_stream = bool(request_body.get("stream", False))
        system = request_body.get("system")
        messages = request_body.get("messages", [])
        tools = request_body.get("tools", [])
        max_tokens = request_body.get("max_tokens", 4096)

        # Convert to OpenAI format
        openai_messages = anthropic_to_openai_messages(messages, system)
        openai_body: dict[str, Any] = {
            "model": Config.upstream_model or model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "stream": is_stream,
        }
        if tools:
            openai_body["tools"] = openai_to_anthropic_tool_definitions(tools)
            openai_body["tool_choice"] = "auto"

        # Extract temperature if present
        if "temperature" in request_body:
            openai_body["temperature"] = request_body["temperature"]

        upstream_url = Config.upstream.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept-Encoding": "identity",
        }

        with LOCK:
            STATS["requests"] += 1
            STATS["last_request_at"] = now_iso()
            if not STATS["first_seen"]:
                STATS["first_seen"] = now_iso()

        attempt = 0
        while True:
            attempt += 1
            try:
                req_data = json.dumps(openai_body, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(upstream_url, data=req_data, headers=headers, method="POST")
                response = urllib.request.urlopen(req, timeout=Config.timeout_read)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code in {429, 500, 502, 503} and attempt < Config.max_attempts:
                    time.sleep(min(10.0, 2.0 * attempt))
                    continue
                self._send_error(e.code, "upstream_error", detail)
                return
            except Exception as e:
                if attempt < Config.max_attempts:
                    time.sleep(min(10.0, 2.0 * attempt))
                    continue
                self._send_error(502, "proxy_error", f"{type(e).__name__}: {e}")
                return

        with LOCK:
            STATS["ok"] += 1

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                stream_openai_to_anthropic(response, model, self.wfile, self.send_header, self.end_headers)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                response.close()
        else:
            resp_data = response.read().decode("utf-8")
            response.close()
            try:
                openai_resp = json.loads(resp_data)
            except json.JSONDecodeError:
                self._send_error(502, "upstream_error", "Invalid JSON from upstream")
                return
            anthropic_resp = convert_openai_response_to_anthropic(openai_resp, model)
            data = json.dumps(anthropic_resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        self.close_connection = True
        print(f"[shengsuanyun-facade] POST cli_model={model} upstream_model={openai_body['model']} "
              f"tools={len(tools)} stream={is_stream} -> 200", flush=True)

    def _send_error(self, status: int, error_type: str, message: str):
        with LOCK:
            STATS["errors"] += 1
        body = json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description="Anthropic→OpenAI facade for shengsuanyun")
    parser.add_argument("--host", default=Config.host)
    parser.add_argument("--port", type=int, default=Config.port)
    parser.add_argument("--upstream", default=Config.upstream)
    parser.add_argument("--upstream-model", default="", help="Model ID sent to the OpenAI upstream, independent of the Claude CLI transport alias")
    parser.add_argument("--local-token", default=os.environ.get("FACADE_LOCAL_TOKEN", ""))
    parser.add_argument("--upstream-key-env", default=Config.upstream_key_env)
    parser.add_argument("--timeout-read", type=float, default=900.0)
    args = parser.parse_args()

    Config.host = args.host
    Config.port = args.port
    Config.upstream = args.upstream
    Config.upstream_model = args.upstream_model
    Config.local_token = args.local_token
    Config.upstream_key_env = args.upstream_key_env
    Config.timeout_read = args.timeout_read

    server = ThreadingHTTPServer((Config.host, Config.port), Handler)
    print(f"[shengsuanyun-facade] listening on http://{Config.host}:{Config.port} "
          f"upstream={Config.upstream} auth={'local-token' if Config.local_token else 'passthrough'}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
