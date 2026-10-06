"""Anthropic Messages forwarder for Claude Code with a pinned egress route.

Why this process exists
-----------------------
Claude Code only speaks Anthropic's Messages protocol, and the teacher slug used on
OpenRouter (`openai/gpt-5.6-sol`) is **region-gated by OpenRouter on the egress IP**
that performs the request: from this workstation a direct request returns

    HTTP 403 {"type":"permission_error","message":"This model is not available in your region."}

while the byte-identical request that leaves through the local egress proxy returns
HTTP 200 with native `tool_use` blocks.  Claude Code's own HTTP client does not honour
`HTTPS_PROXY`, so a small local forwarder is the only place where that route can be
pinned.  (This is the same "Anthropic-compatible facade" the experiment README assumes.)

What this forwarder deliberately does NOT do
-------------------------------------------
* No protocol translation.  The Anthropic Messages body is forwarded verbatim, including
  `thinking`, `context_management`, `output_config`, `metadata`, `cache_control` blocks
  and the `?beta=true` query, because OpenRouter accepts all of them for this model.
* No max_tokens or tool rewriting, so the teacher model sees exactly what a real
  Claude Code session would send.
* No fallback to a different model, ever: a failed upstream status is returned as-is.

The only mutations are transport-level: hop-by-hop headers, forced `identity` encoding
towards upstream, and server-side authentication when a local token is configured.
Diagnostics exist so a failed episode can never be reported as an empty error string.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length", "content-encoding", "host",
}
# Claude Code's own client must never be told to talk to the internet directly for the
# teacher route; these are local-only hops.
NEVER_FORWARDED = HOP_BY_HOP | {"accept-encoding", "anthropic-dangerous-direct-browser-access"}
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
REGION_HINT = "not available in your region"


class Config:
    host = "127.0.0.1"
    port = 8311
    upstream = "https://openrouter.ai/api"
    egress_mode = "auto"          # auto | direct | explicit proxy url
    egress_proxy = ""             # resolved proxy url when egress_mode is a url
    local_token = ""              # when set, callers must present it; upstream key is injected
    upstream_key_env = "OPENROUTER_API_KEY"
    timeout_connect = 30.0
    timeout_read = 900.0
    max_attempts = 3
    error_log = ""
    probe_ttl = 300.0


STATS: dict[str, Any] = {"requests": 0, "ok": 0, "by_status": {}, "retries": 0, "errors": 0,
                         "first_seen": "", "last_request_at": "", "last_status": 0}
LAST_ERROR: dict[str, Any] = {}
PROBE: dict[str, Any] = {"checked_at": 0.0}
LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_upstream(url: str, headers: dict[str, str], body: bytes) -> Any:
    """Open a streaming upstream response using the pinned egress route."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    if Config.egress_mode == "direct":
        # ProxyHandler({}) makes urllib ignore every ambient proxy setting, including
        # the Windows registry entry, which is exactly what 'direct' must mean here.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=Config.timeout_read)
    if Config.egress_mode == "auto":
        return urllib.request.urlopen(request, timeout=Config.timeout_read)  # noqa: S310
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": Config.egress_proxy, "https": Config.egress_proxy}))
    return opener.open(request, timeout=Config.timeout_read)


def egress_label() -> str:
    if Config.egress_mode == "auto":
        # getproxies() merges the environment *and* the Windows registry, which is the
        # distinction that decides whether OpenRouter sees a blocked region or not.
        try:
            found = dict(urllib.request.getproxies())
        except Exception:  # noqa: BLE001
            found = {}
        return "auto:" + (json.dumps(found, ensure_ascii=False) if found else "direct-no-proxy-resolved")
    if Config.egress_mode == "direct":
        return "direct(no-proxy)"
    return "proxy:" + Config.egress_proxy


def record_error(status: int, detail: str, path: str, model: str, tools: int, attempt: int) -> None:
    row = {"timestamp": now_iso(), "status": int(status), "path": path, "model": model, "tools": tools,
           "attempt": attempt, "egress": egress_label(), "detail": detail[:1500]}
    global LAST_ERROR
    with LOCK:
        LAST_ERROR = row
        STATS["errors"] += 1
        if Config.error_log:
            try:
                os.makedirs(os.path.dirname(Config.error_log), exist_ok=True)
                with open(Config.error_log, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                pass


def upstream_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "cross-harness-anthropic-forwarder/2.0"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        print("[anthropic-forward] " + (fmt % args), flush=True)

    # ---- client-facing helpers -------------------------------------------------
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path.split("?")[0] in {"/health", "/healthz"}:
            self.send_response(200)
            self.send_header("Content-Length", "0")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path not in {"/health", "/healthz"}:
            self._send_json(404, {"error": "not found"})
            return
        with LOCK:
            payload = dict(STATS)
            payload.update({"ok": True, "upstream": Config.upstream, "egress": egress_label(),
                            "auth_mode": "local-token+injected-key" if Config.local_token else "passthrough",
                            "last_error": LAST_ERROR or None})
        age = time.monotonic() - float(PROBE.get("checked_at") or 0.0)
        if "ok" not in PROBE or age > Config.probe_ttl:
            PROBE.update(run_probe())
            PROBE["checked_at"] = time.monotonic()
            age = 0.0
        payload["route_probe"] = {k: v for k, v in PROBE.items() if k != "checked_at"}
        payload["route_probe_age_seconds"] = round(age, 1)
        self._send_json(200, payload)

    # ---- the forwarder --------------------------------------------------------
    def _client_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        # Normalize header keys to lowercase for consistent lookup
        for key, value in self.headers.items():
            if key.lower() in NEVER_FORWARDED:
                continue
            headers[key.lower()] = value
        headers["accept-encoding"] = "identity"
        if Config.local_token:
            presented = (headers.get("x-api-key") or "").strip()
            if not presented.startswith("Bearer "):
                authed = (headers.get("authorization") or "").strip()
                presented = authed[7:].strip() if authed.lower().startswith("bearer ") else presented
            if presented != Config.local_token:
                return {}  # caller is misconfigured: fail closed, never touch upstream
            headers = {k: v for k, v in headers.items()
                       if k.lower() not in {"x-api-key", "authorization"}}
            api_key = os.environ.get(Config.upstream_key_env, "").strip()
            if not api_key:
                raise RuntimeError(f"{Config.upstream_key_env} is not available to the forwarder")
            headers["x-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        model, tools, stream = "", 0, False
        try:
            parsed = json.loads(body.decode("utf-8")) if body else {}
            if isinstance(parsed, dict):
                model = str(parsed.get("model") or "")
                tools = len(parsed.get("tools") or [])
                stream = bool(parsed.get("stream"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        path = self.path.split("?")[0]
        url = Config.upstream.rstrip("/") + self.path
        with LOCK:
            STATS["requests"] += 1
            STATS["last_request_at"] = now_iso()
            if not STATS["first_seen"]:
                STATS["first_seen"] = now_iso()

        headers = self._client_headers()
        if not headers:
            detail = '{"type":"error","error":{"type":"authentication_error","message":"local forwarder token mismatch"}}'
            record_error(401, detail, path, model, tools, 1)
            self._respond_text(401, "application/json", detail.encode("utf-8"))
            return

        attempt = 0
        while True:
            attempt += 1
            try:
                response = open_upstream(url, headers, body)
            except urllib.error.HTTPError as error:
                detail = upstream_body(error)
                status = int(error.code)
                retryable = status in RETRY_STATUSES or (status == 403 and REGION_HINT in detail)
                record_error(status, detail, path, model, tools, attempt)
                if retryable and attempt < Config.max_attempts:
                    with LOCK:
                        STATS["retries"] += 1
                    time.sleep(min(20.0, 2.0 * attempt))
                    print(f"[anthropic-forward] upstream HTTP {status} attempt={attempt} "
                          f"model={model or '?'} tools={tools} -> retrying", flush=True)
                    continue
                with LOCK:
                    STATS["by_status"][str(status)] = STATS["by_status"].get(str(status), 0) + 1
                    STATS["last_status"] = status
                self._respond_text(status, error.headers.get("Content-Type") or "application/json",
                                   (detail or json.dumps({"error": "upstream_error", "status": status})).encode("utf-8"))
                return
            except (urllib.error.URLError, socket.timeout, OSError) as error:
                reason = getattr(error, "reason", error)
                detail = f"{type(error).__name__}: {reason}"
                record_error(502, detail, path, model, tools, attempt)
                if attempt < Config.max_attempts:
                    with LOCK:
                        STATS["retries"] += 1
                    time.sleep(min(20.0, 2.0 * attempt))
                    print(f"[anthropic-forward] upstream {detail} attempt={attempt} -> retrying", flush=True)
                    continue
                self._respond_text(502, "application/json", json.dumps(
                    {"type": "error", "error": {"type": "proxy_error", "message": detail[:600]}}).encode())
                return
            break

        content_type = response.headers.get("Content-Type") or "application/json"
        try:
            self.send_response(response.status)
            for key, value in response.headers.items():
                if key.lower() not in HOP_BY_HOP:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            response.close()
            self.close_connection = True
            return
        written = 0
        try:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                written += len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            response.close()
        with LOCK:
            key = str(response.status)
            STATS["by_status"][key] = STATS["by_status"].get(key, 0) + 1
            STATS["last_status"] = response.status
            if response.status < 400:
                STATS["ok"] += 1
        self.close_connection = True
        note = " stream" if stream else ""
        print(f"[anthropic-forward] POST {self.path} model={model or '?'} tools={tools}{note} "
              f"-> {response.status} bytes={written} via {egress_label()}", flush=True)

    def _respond_text(self, status: int, content_type: str, payload: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


def run_probe() -> dict[str, Any]:
    """One tiny authenticated request that proves the pinned egress can serve the model."""
    api_key = os.environ.get(Config.upstream_key_env, "").strip()
    if Config.local_token and not api_key:
        return {"ok": False, "detail": f"{Config.upstream_key_env} unavailable for probe"}
    body = json.dumps({"model": os.environ.get("TEACHER_ROUTE_PROBE_MODEL", "openai/gpt-5.6-sol"),
                       "max_tokens": 1,
                       "messages": [{"role": "user", "content": "Reply OK"}]}).encode()
    headers = {"content-type": "application/json", "accept-encoding": "identity", "anthropic-version": "2023-06-01"}
    if Config.local_token:
        headers["x-api-key"] = api_key
    url = Config.upstream.rstrip("/") + "/v1/messages"
    started = time.monotonic()
    try:
        response = open_upstream(url, headers, body)
        detail = response.read(400).decode("utf-8", "replace")
        status, ok = int(response.status), 200 <= int(response.status) < 300
        response.close()
    except urllib.error.HTTPError as error:
        detail = error.read(400).decode("utf-8", "replace")
        status, ok = int(error.code), False
    except Exception as exc:  # noqa: BLE001
        detail, status, ok = f"{type(exc).__name__}: {exc}", 0, False
    return {"ok": ok, "status": status, "latency_ms": int((time.monotonic() - started) * 1000),
            "egress": egress_label(), "detail": detail[:300]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=Config.host)
    parser.add_argument("--port", type=int, default=Config.port)
    parser.add_argument("--upstream", default=Config.upstream)
    parser.add_argument("--egress", default=os.environ.get("ANTHROPIC_EGRESS", "auto"),
                        help="'auto' (ambient env), 'direct' (bypass proxies), or a proxy url such as http://127.0.0.1:7897")
    parser.add_argument("--local-token", default=os.environ.get("FACADE_LOCAL_TOKEN", ""),
                        help="value Claude Code presents as ANTHROPIC_API_KEY; the real key is injected here")
    parser.add_argument("--upstream-key-env", default=Config.upstream_key_env)
    parser.add_argument("--error-log", default=os.environ.get("FACADE_ERROR_LOG", ""))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-read", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    Config.host, Config.port, Config.upstream = cli.host, cli.port, cli.upstream
    Config.egress_mode = "auto" if cli.egress == "auto" else ("direct" if cli.egress == "direct" else "proxy")
    Config.egress_proxy = "" if Config.egress_mode != "proxy" else cli.egress
    Config.local_token = cli.local_token or ""
    Config.upstream_key_env = cli.upstream_key_env
    Config.error_log = cli.error_log
    Config.max_attempts = max(1, cli.max_attempts)
    Config.timeout_read = cli.timeout_read
    server = ThreadingHTTPServer((Config.host, Config.port), Handler)
    print(f"[anthropic-forward] listening on http://{Config.host}:{Config.port} "
          f"upstream={Config.upstream} egress={egress_label()} auth={'local-token' if Config.local_token else 'passthrough'}",
          flush=True)
    print(f"[anthropic-forward] route probe: {json.dumps(run_probe(), ensure_ascii=False)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
