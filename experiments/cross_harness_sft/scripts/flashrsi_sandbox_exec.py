#!/usr/bin/python3
"""Experimental child launcher for MiniAgent's fixed Python entry point.

The parent must run in a separately prepared network namespace, protect the
trusted config and close unrelated FDs. This is not an independently verified
sandbox; do not use it to release adversarial evaluation yet.
"""
import ctypes
import json
import os
import resource
import sys
import urllib.parse

from flashrsi_landlock import restrict_filesystem
from flashrsi_seccomp import restrict_syscalls


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32)]


def _drop_capabilities():
    libc = ctypes.CDLL(None, use_errno=True)
    header = _CapHeader(0x20080522, 0)
    data = (_CapData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Prevent a prior ambient capability from surviving exec.
    if libc.prctl(47, 4, 0, 0, 0) != 0:  # PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _limit_resources():
    for kind, requested in ((resource.RLIMIT_AS, 4 * 1024**3), (resource.RLIMIT_FSIZE, 32 * 1024**2),
                            (resource.RLIMIT_NOFILE, 64), (resource.RLIMIT_CPU, 120), (resource.RLIMIT_CORE, 0)):
        _, hard = resource.getrlimit(kind)
        limit = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(kind, (limit, limit))


def main():
    config_path = os.environ.get("FLASHRSI_SANDBOX_CONFIG", "")
    if not os.path.isabs(config_path):
        raise ValueError("trusted sandbox config path must be absolute")
    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != {"entry", "python", "gateway_port", "readonly_roots", "writable_root"}:
        raise ValueError("invalid sandbox configuration")
    entry, python = config["entry"], config["python"]
    if (not isinstance(entry, str) or not os.path.isabs(entry) or not isinstance(python, str)
            or not os.path.isabs(python) or sys.argv[1:] != ["-I", "-B", entry]):
        raise ValueError("only the configured MiniAgent entry is allowed")
    writable = config["writable_root"]
    url = os.environ.get("FLASHRSI_GATEWAY_URL", "")
    token = os.environ.get("FLASHRSI_GATEWAY_TOKEN", "")
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != config["gateway_port"]
            or parsed.username is not None or parsed.password is not None or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("invalid gateway URL")
    if not isinstance(token, str) or len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("invalid gateway token")
    child_env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": writable, "TMPDIR": writable,
                 "FLASHRSI_GATEWAY_URL": url, "FLASHRSI_GATEWAY_TOKEN": token}
    _limit_resources()
    _drop_capabilities()
    restrict_syscalls()
    restrict_filesystem(config["readonly_roots"], [writable], allowed_tcp_ports=[config["gateway_port"]],
                        allowed_devices=["/dev/urandom", "/dev/null"])
    os.execve(python, [python, "-I", "-B", entry], child_env)


if __name__ == "__main__":
    main()
