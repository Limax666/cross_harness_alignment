"""Authenticated Unix-socket relay to a fixed trusted model callable.

The socket directory and token must be kept outside the untrusted child. This
transport does not by itself isolate the child, verify schemas, or prove B0.
"""

import hmac
import json
import os
from pathlib import Path
import socket
import socketserver
import stat

from flashrsi_tool_attestation import _invalid_constant, _unique_object

_MAX_MESSAGE = 4 * 1024 * 1024


def _decode(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _token_ok(token):
    return isinstance(token, str) and len(token) >= 32 and token.isascii() and not any(c.isspace() for c in token)


def create_model_relay(path, token, model, upstream):
    if not _token_ok(token) or not isinstance(model, str) or not model or not callable(upstream):
        raise ValueError("invalid relay configuration")
    if not isinstance(path, str) or not os.path.isabs(path) or len(os.fsencode(path)) > 100:
        raise ValueError("invalid relay socket path")
    directory = Path(path).parent
    if os.path.realpath(directory) != str(directory) or os.path.lexists(path):
        raise ValueError("socket path must be new in a canonical directory")
    info = directory.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("socket directory must be private and owned by this user")

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(180)
            answer = {"ok": False}
            try:
                raw = self.rfile.readline(_MAX_MESSAGE + 1)
                if len(raw) > _MAX_MESSAGE or not raw.endswith(b"\n"):
                    raise ValueError("invalid frame")
                payload = _decode(raw)
                supplied = payload.get("token")
                if (set(payload) != {"token", "request"} or not isinstance(supplied, str)
                        or not supplied.isascii() or not hmac.compare_digest(supplied, token)):
                    raise ValueError("invalid capability")
                request = payload["request"]
                if (not isinstance(request, dict) or request.get("model") != model
                        or request.get("stream", False) is not False
                        or type(request.get("n", 1)) is not int or request.get("n", 1) != 1):
                    raise ValueError("unsupported model request")
                response = upstream(request)
                if not isinstance(response, dict):
                    raise ValueError("invalid model response")
                answer = {"ok": True, "response": response}
                encoded = json.dumps(answer, ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
                if len(encoded) > _MAX_MESSAGE:
                    raise ValueError("response too large")
            except Exception:
                encoded = b'{"ok":false}\n'
            try:
                self.wfile.write(encoded)
            except OSError:
                pass

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    server = Server(path, Handler)
    os.chmod(path, 0o600)
    return server


def relay_request(path, token, request):
    if not _token_ok(token) or not isinstance(request, dict):
        raise ValueError("invalid relay request")
    raw = json.dumps({"token": token, "request": request}, ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
    if len(raw) > _MAX_MESSAGE:
        raise ValueError("request too large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(180)
        conn.connect(path)
        conn.sendall(raw)
        with conn.makefile("rb") as stream:
            reply = stream.readline(_MAX_MESSAGE + 1)
    if len(reply) > _MAX_MESSAGE or not reply.endswith(b"\n"):
        raise ValueError("invalid relay response")
    value = _decode(reply)
    if value.get("ok") is not True or set(value) != {"ok", "response"} or not isinstance(value["response"], dict):
        raise ValueError("model relay rejected request")
    return value["response"]
