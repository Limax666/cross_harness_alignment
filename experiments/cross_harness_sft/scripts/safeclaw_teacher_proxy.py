"""Loopback-only fixed-route teacher proxy; the task sandbox never receives the upstream key."""

from __future__ import annotations

import http.client
import secrets
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator


@contextmanager
def teacher_proxy(upstream_key: str) -> Iterator[tuple[str, str]]:
    proxy_token = "sk-safeclaw-" + secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            if self.path != "/api/v1/chat/completions" or self.headers.get("Authorization") != f"Bearer {proxy_token}":
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= 10_000_000:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            connection = http.client.HTTPSConnection("router.shengsuanyun.com", timeout=180)
            try:
                connection.request("POST", "/api/v1/chat/completions", body=body, headers={
                    "Authorization": f"Bearer {upstream_key}",
                    "Content-Type": "application/json",
                    "Accept": self.headers.get("Accept", "text/event-stream"),
                })
                response = connection.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.end_headers()
                while chunk := response.read(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, http.client.HTTPException):
                # Do not echo the upstream key or request body into agent-visible errors.
                try:
                    self.send_error(502, "teacher upstream unavailable")
                except OSError:
                    pass
            finally:
                connection.close()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/v1", proxy_token
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
