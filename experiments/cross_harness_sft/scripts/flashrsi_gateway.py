"""Loopback-only transport for one guarded FlashRSI development episode.

The HTTP token is a local capability, not a sandbox: do not expose this server
to an unisolated adversarial agent or treat it as a B0 release gate.
"""

import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flashrsi_tool_attestation import _invalid_constant, _unique_object

_MAX_BODY = 2 * 1024 * 1024


def create_gateway(proxy, dispatcher, gate, token):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("gateway needs a strong ASCII capability token")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Prompts and credentials must not enter default access logs.

        def _reply(self, status, value):
            body = json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            headers = self.headers.get_all("Authorization", [])
            try:
                authorized = len(headers) == 1 and hmac.compare_digest(headers[0], "Bearer " + token)
            except TypeError:
                authorized = False
            if not authorized:
                self._reply(401, {"error": "unauthorized"})
                return
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if (len(lengths) != 1 or self.headers.get("Transfer-Encoding") is not None
                        or not lengths[0].isdigit() or not 0 < int(lengths[0]) <= _MAX_BODY):
                    raise ValueError("invalid request length")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("invalid content type")
                body = self.rfile.read(int(lengths[0]))
                if len(body) != int(lengths[0]):
                    raise ValueError("truncated request")
                data = json.loads(body, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                if not isinstance(data, dict):
                    raise ValueError("request must be an object")
                if self.path == "/v1/chat/completions":
                    result = proxy.forward(data)
                elif self.path == "/tool":
                    if set(data) != {"name", "arguments"} or not isinstance(data["name"], str) or not isinstance(data["arguments"], dict):
                        raise ValueError("invalid tool request")
                    native_result = dispatcher.dispatch(data["name"], data["arguments"])
                    if not isinstance(native_result, dict) or not isinstance(native_result.get("observation"), str):
                        raise ValueError("invalid tool observation")
                    # Native audit metadata is evaluator-only; expose just the
                    # observation that the model is permitted to see.
                    result = {"observation": native_result["observation"]}
                else:
                    raise ValueError("unknown route")
                self._reply(200, result)
            except Exception:
                gate.abort()
                self._reply(400, {"error": "request rejected; episode invalid"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    return server
