import json
import os
from pathlib import Path
import subprocess
import socket
import sys


def test_landlock_denies_unlisted_files_and_readonly_writes_and_is_inherited(tmp_path):
    readonly = tmp_path / "readonly"
    writable = tmp_path / "writable"
    hidden = tmp_path / "hidden"
    for path in (readonly, writable, hidden):
        path.mkdir()
    (readonly / "public").write_text("visible")
    (hidden / "secret").write_text("private")
    script = r'''
import json, pathlib, subprocess, sys
from flashrsi_landlock import restrict_filesystem
readonly, writable, hidden = map(pathlib.Path, sys.argv[1:])
restrict_filesystem(["/usr", "/lib", "/lib64", str(readonly)], [str(writable)])
def read(path):
    try: return path.read_text()
    except PermissionError: return "DENIED"
def write(path):
    try: path.write_text("changed"); return "ALLOWED"
    except PermissionError: return "DENIED"
child = subprocess.run(["/usr/bin/python3", "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).read_text()", str(hidden / "secret")], capture_output=True)
print(json.dumps({"public": read(readonly / "public"), "secret": read(hidden / "secret"),
                  "readonly_write": write(readonly / "public"), "workspace_write": write(writable / "new"),
                  "child_denied": child.returncode != 0 and b"PermissionError" in child.stderr}))
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "scripts")}
    result = subprocess.run(["/usr/bin/python3", "-c", script, str(readonly), str(writable), str(hidden)],
                            capture_output=True, text=True, env=env, check=True, timeout=10)
    assert json.loads(result.stdout) == {"public": "visible", "secret": "DENIED", "readonly_write": "DENIED",
                                        "workspace_write": "ALLOWED", "child_denied": True}
    assert (readonly / "public").read_text() == "visible"
    assert (hidden / "secret").read_text() == "private"


def test_rejects_relative_or_missing_allowed_paths(tmp_path):
    script = r'''
from flashrsi_landlock import restrict_filesystem
for path in ("relative", "/definitely/nonexistent/flashrsi"):
    try: restrict_filesystem([path], [])
    except (ValueError, OSError): continue
    raise AssertionError("invalid path accepted")
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "scripts")}
    subprocess.run(["/usr/bin/python3", "-c", script], env=env, check=True, timeout=10)


def test_landlock_limits_tcp_connections_and_denies_binding(tmp_path):
    first, second = socket.socket(), socket.socket()
    first.bind(("127.0.0.1", 0)); first.listen()
    second.bind(("127.0.0.1", 0)); second.listen()
    try:
        script = r'''
import json, socket, sys
from flashrsi_landlock import restrict_filesystem
restrict_filesystem(["/usr", "/lib", "/lib64"], [], allowed_tcp_ports=[int(sys.argv[1])])
def connect(port):
    s = socket.socket(); s.settimeout(2)
    try: return s.connect_ex(("127.0.0.1", port))
    finally: s.close()
s = socket.socket()
try: s.bind(("127.0.0.1", 0)); bound = True
except PermissionError: bound = False
print(json.dumps({"allowed": connect(int(sys.argv[1])), "blocked": connect(int(sys.argv[2])), "bound": bound}))
'''
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "scripts")}
        result = subprocess.run(["/usr/bin/python3", "-c", script, str(first.getsockname()[1]), str(second.getsockname()[1])],
                                env=env, capture_output=True, text=True, check=True, timeout=10)
        assert json.loads(result.stdout) == {"allowed": 0, "blocked": 13, "bound": False}
    finally:
        first.close(); second.close()


def test_landlock_allows_only_selected_random_device(tmp_path):
    script = r'''
import json
from flashrsi_landlock import restrict_filesystem
restrict_filesystem(["/usr", "/lib", "/lib64"], [], allowed_devices=["/dev/urandom"])
with open("/dev/urandom", "rb") as f: count = len(f.read(8))
try:
    open("/dev/null", "wb")
    denied = False
except PermissionError:
    denied = True
print(json.dumps({"bytes":count,"other_device_denied":denied}))
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "scripts")}
    result = subprocess.run(["/usr/bin/python3", "-c", script], env=env,
                            capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(result.stdout) == {"bytes": 8, "other_device_denied": True}
