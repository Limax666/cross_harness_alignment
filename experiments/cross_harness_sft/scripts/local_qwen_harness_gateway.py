#!/usr/bin/env python3
"""Translate Codex Responses and Claude Messages requests to local Qwen chat API.

This adapter intentionally has a small surface: it preserves text, function calls,
and tool results used by the two native HarnessAudit harnesses.  It keeps the
underlying model server unchanged and logs every translated request for audit.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def text_of(value):
    if isinstance(value, str): return value
    if not isinstance(value, list): return "" if value is None else str(value)
    chunks = []
    for item in value:
        if isinstance(item, str): chunks.append(item)
        elif isinstance(item, dict): chunks.append(str(item.get("text") or item.get("content") or ""))
    return "\n".join(x for x in chunks if x)


def post_json(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors="replace")) from exc


def responses_function_calls(chat_calls, advertised_tools):
    """Emit only Responses function calls declared in the current request.

    A fabricated custom ``exec`` call is not a Codex CLI tool, even if its
    JavaScript happens to mention a real MCP tool. Fail closed on ambiguity.
    """
    declared = {}
    for tool in advertised_tools or []:
        if tool.get("type") == "function" and tool.get("name"):
            declared[tool["name"]] = (None, tool["name"])
        elif tool.get("type") == "namespace":
            for nested in tool.get("tools", []):
                if nested.get("type") == "function" and nested.get("name"):
                    flat = f"{tool['name']}__{nested['name']}"
                    declared[flat] = (tool["name"], nested["name"])
                    # Chat Completions models usually return the leaf MCP name.
                    # Keep aliases only when unique; ambiguity fails closed below.
                    declared.setdefault(nested["name"], (tool["name"], nested["name"]))
    leaf_counts = {}
    for tool in advertised_tools or []:
        if tool.get("type") == "namespace":
            for nested in tool.get("tools", []):
                if nested.get("type") == "function" and nested.get("name"):
                    leaf_counts[nested["name"]] = leaf_counts.get(nested["name"], 0) + 1
    output = []
    for call in chat_calls or []:
        fn = call["function"]
        raw_name = fn["name"]
        if raw_name not in declared or ("__" not in raw_name and leaf_counts.get(raw_name, 0) > 1):
            raise ValueError(f"model called undeclared or ambiguous Responses tool: {raw_name!r}")
        namespace, name = declared[raw_name]
        if namespace is None and "__" in raw_name:
            raise ValueError(f"model called undeclared or ambiguous Responses tool: {raw_name!r}")
        arguments = fn.get("arguments") or "{}"
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError(f"tool arguments must be a JSON object: {name}")
        item = {"id":"fc_"+uuid.uuid4().hex, "type":"function_call",
                "status":"completed", "call_id":call.get("id") or "call_"+uuid.uuid4().hex,
                "name":name, "arguments":json.dumps(parsed, ensure_ascii=False)}
        # Codex registers MCP tools as ToolName(namespace, leaf_name). A
        # flattened name in name with no namespace routes to the wrong registry
        # entry and fails with unsupported call.
        if namespace is not None:
            item["namespace"] = namespace
        output.append(item)
    return output


def anthropic_messages(body):
    messages = []
    system = text_of(body.get("system"))
    if system: messages.append({"role": "system", "content": system})
    for message in body.get("messages", []):
        role, content = message.get("role", "user"), message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content}); continue
        text, calls = [], []
        for block in content or []:
            kind = block.get("type") if isinstance(block, dict) else "text"
            if kind in ("text", "thinking"):
                text.append(str(block.get("text") or block.get("thinking") or ""))
            elif kind == "tool_use":
                calls.append({"id": block.get("id", "call_" + uuid.uuid4().hex[:12]), "type": "function", "function": {"name": block.get("name", "tool"), "arguments": json.dumps(block.get("input") or {})}})
            elif kind == "tool_result":
                messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": text_of(block.get("content"))})
        if calls or text:
            entry = {"role": "assistant" if role == "assistant" else role, "content": "\n".join(text)}
            if calls: entry["tool_calls"] = calls
            messages.append(entry)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {"type": "object", "properties": {}})}} for t in body.get("tools", [])]
    return messages, tools


def responses_messages(body):
    messages = []
    instructions = text_of(body.get("instructions"))
    if instructions: messages.append({"role": "system", "content": instructions})
    raw = body.get("input", [])
    if isinstance(raw, str): raw = [{"role": "user", "content": raw}]
    for item in raw:
        if isinstance(item, str): messages.append({"role": "user", "content": item}); continue
        # Responses uses ``developer`` while Qwen's template only accepts
        # system/user/assistant/tool.  A developer instruction has the same
        # precedence as a system instruction for this local, single-policy run.
        role = item.get("role", "user")
        if role == "developer": role = "system"
        if role not in ("system", "user", "assistant", "tool"): role = "user"
        content = item.get("content", "")
        if item.get("type") == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""), "content": text_of(item.get("output"))}); continue
        if item.get("type") == "function_call":
            messages.append({"role":"assistant", "content":"", "tool_calls":[{"id":item.get("call_id", ""),
                "type":"function", "function":{"name":item.get("name", ""), "arguments":item.get("arguments", "{}")}}]}); continue
        messages.append({"role": role, "content": text_of(content)})
    tools = []
    for t in body.get("tools", []):
        if t.get("type") == "function":
            tools.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {"type":"object", "properties":{}})}})
        elif t.get("type") == "namespace":
            # The Responses API groups MCP methods under namespaces. Flatten
            # those methods for Qwen's Chat Completions API; the returned short
            # name is qualified again in canonical_function_name().
            for nested in t.get("tools", []):
                if nested.get("type") != "function" or not nested.get("name"):
                    continue
                tools.append({"type": "function", "function": {"name": nested["name"], "description": nested.get("description", ""), "parameters": nested.get("parameters", {"type":"object", "properties":{}})}})
    # Qwen's template requires system instructions to be at the beginning.
    system_messages = [m for m in messages if m["role"] == "system"]
    # The Qwen 3.5 template permits a single leading system message. Preserve
    # all Codex developer/system instructions by joining them in order.
    prefix = [{"role": "system", "content": "\n\n".join(str(m.get("content", "")) for m in system_messages)}] if system_messages else []
    messages = prefix + [m for m in messages if m["role"] != "system"]
    return messages, tools


def make_handler(upstream, model_id):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, fmt, *args): print("[gateway]", fmt % args, flush=True)
        def reply(self, value, status=200):
            raw = json.dumps(value, ensure_ascii=False).encode(); self.send_response(status)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                # The client cancelled the request.  There is no response to
                # recover here, and re-raising would turn a normal disconnect
                # into a server-side 500 traceback.
                return
        def sse(self, events):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            self.close_connection = True
            try:
                for name, value in events:
                    self.wfile.write((f"event: {name}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n").encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Codex may abandon an SSE request when a turn times out.
                # Treat that as a client cancellation rather than a gateway
                # failure so later tasks can continue.
                return
        def do_GET(self):
            if self.path in ("/health", "/v1/models"):
                return self.reply({"status":"ok"} if self.path == "/health" else {"object":"list","data":[{"id":model_id,"object":"model"}]})
            self.reply({"error":"not_found"}, 404)
        def do_HEAD(self):
            # Claude Code probes the provider before its first Messages call.
            self.send_response(200 if urlsplit(self.path).path == "/api/hello" else 404)
            self.send_header("Content-Length", "0")
            self.end_headers()
        def chat(self, messages, tools, limit):
            print(f"[gateway] request roles={[m.get('role') for m in messages]} tools={len(tools)}", flush=True)
            return post_json(upstream + "/v1/chat/completions", {"model":model_id,"messages":messages,"tools":tools or None,"max_tokens":limit,"stream":False})
        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(size))
                if urlsplit(self.path).path == "/v1/messages":
                    messages, tools = anthropic_messages(body); raw = self.chat(messages, tools, int(body.get("max_tokens", 768)))
                    msg = raw["choices"][0]["message"]; blocks = []
                    for call in msg.get("tool_calls") or []:
                        fn = call["function"]
                        try: args = json.loads(fn.get("arguments", "{}"))
                        except json.JSONDecodeError: args = {}
                        blocks.append({"type":"tool_use","id":call["id"],"name":fn["name"],"input":args})
                    if msg.get("content"): blocks.insert(0, {"type":"text","text":msg["content"]})
                    result = {"id":"msg_"+uuid.uuid4().hex,"type":"message","role":"assistant","model":model_id,"content":blocks,"stop_reason":"tool_use" if blocks and blocks[-1]["type"] == "tool_use" else "end_turn","stop_sequence":None,"usage":{"input_tokens":raw.get("usage",{}).get("prompt_tokens",0),"output_tokens":raw.get("usage",{}).get("completion_tokens",0)}}
                    if body.get("stream"):
                        events = [("message_start", {"type":"message_start","message":{**result,"content":[]}})]
                        for index, block in enumerate(blocks):
                            events.append(("content_block_start", {"type":"content_block_start","index":index,"content_block":block if block["type"] == "tool_use" else {"type":"text","text":""}}))
                            if block["type"] == "text": events.append(("content_block_delta", {"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":block["text"]}}))
                            events.append(("content_block_stop", {"type":"content_block_stop","index":index}))
                        events += [("message_delta", {"type":"message_delta","delta":{"stop_reason":result["stop_reason"],"stop_sequence":None},"usage":result["usage"]}), ("message_stop", {"type":"message_stop"})]
                        return self.sse(events)
                    return self.reply(result)
                if self.path == "/v1/responses":
                    print("[gateway] advertised Responses tools=" + json.dumps([
                        {"type": t.get("type"), "name": t.get("name"),
                         "nested": [x.get("name") for x in t.get("tools", [])][:8]}
                        for t in body.get("tools", [])], ensure_ascii=False), flush=True)
                    messages, tools = responses_messages(body); raw = self.chat(messages, tools, int(body.get("max_output_tokens", 768)))
                    msg = raw["choices"][0]["message"]; content = []
                    if msg.get("content"): content.append({"type":"output_text","text":msg["content"],"annotations":[]})
                    output = [{"id":"msg_"+uuid.uuid4().hex,"type":"message","role":"assistant","status":"completed","content":content}]
                    output.extend(responses_function_calls(msg.get("tool_calls"), body.get("tools", [])))
                    usage = raw.get("usage", {})
                    result = {
                        "id":"resp_"+uuid.uuid4().hex, "object":"response", "created_at":int(time.time()),
                        "status":"completed", "error":None, "incomplete_details":None,
                        "instructions":body.get("instructions"), "max_output_tokens":body.get("max_output_tokens"),
                        "model":model_id, "output":output, "output_text":msg.get("content") or "",
                        "parallel_tool_calls":True, "previous_response_id":body.get("previous_response_id"),
                        "reasoning":{"effort":None,"summary":None}, "store":False,
                        "temperature":0, "top_p":1, "truncation":"disabled",
                        "usage":{"input_tokens":usage.get("prompt_tokens",0), "output_tokens":usage.get("completion_tokens",0), "total_tokens":usage.get("total_tokens",0)},
                    }
                    if body.get("stream"):
                        events = [("response.created", {"type":"response.created","response":{**result,"status":"in_progress","output":[]}})]
                        for index, item in enumerate(output):
                            events.append(("response.output_item.added", {"type":"response.output_item.added","output_index":index,"item":item}))
                            if item["type"] == "message":
                                for part_index, part in enumerate(item["content"]):
                                    events.append(("response.content_part.added", {"type":"response.content_part.added","output_index":index,"content_index":part_index,"part":{"type":"output_text","text":"","annotations":[]}}))
                                    events.append(("response.output_text.delta", {"type":"response.output_text.delta","output_index":index,"content_index":part_index,"delta":part["text"]}))
                                    events.append(("response.output_text.done", {"type":"response.output_text.done","output_index":index,"content_index":part_index,"text":part["text"]}))
                                    events.append(("response.content_part.done", {"type":"response.content_part.done","output_index":index,"content_index":part_index,"part":part}))
                            events.append(("response.output_item.done", {"type":"response.output_item.done","output_index":index,"item":item}))
                        events.append(("response.completed", {"type":"response.completed","response":result}))
                        return self.sse(events)
                    return self.reply(result)
                self.reply({"error":"not_found"}, 404)
            except Exception as exc:
                print("[gateway-error]", repr(exc), flush=True); self.reply({"error":{"message":str(exc)}}, 500)
    return Handler


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--upstream", required=True); ap.add_argument("--port", type=int, required=True); ap.add_argument("--model-id", default="qwen35-9b-local")
    args=ap.parse_args(); print(json.dumps({"gateway":args.port,"upstream":args.upstream}), flush=True)
    ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(args.upstream.rstrip("/"),args.model_id)).serve_forever()
if __name__ == "__main__": main()
