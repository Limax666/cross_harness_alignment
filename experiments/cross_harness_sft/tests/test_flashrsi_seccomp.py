import json
import os
from pathlib import Path
import subprocess


def test_restricted_child_blocks_metadata_and_unix_sockets_but_keeps_loopback_tcp(tmp_path):
    target = tmp_path / "sample"
    target.write_text("unchanged")
    script = r'''
import errno, json, os, socket, sys, threading
from flashrsi_seccomp import restrict_syscalls
restrict_syscalls()
def attempt(action):
    try: action(); return "ALLOWED"
    except OSError as exc: return errno.errorcode.get(exc.errno, "OTHER")
chmod = attempt(lambda: os.chmod(sys.argv[1], 0o600))
unix = attempt(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
netlink = attempt(lambda: socket.socket(socket.AF_NETLINK, socket.SOCK_RAW))
tcp = attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).close())
udp = attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close())
fork = attempt(lambda: os.fork())
done=[]; t=threading.Thread(target=lambda:done.append(True)); t.start(); t.join()
print(json.dumps({"chmod": chmod, "unix": unix, "netlink": netlink, "tcp": tcp,
                  "udp": udp, "fork": fork, "thread_ok": done == [True]}))
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "scripts")}
    result = subprocess.run(["/usr/bin/python3", "-c", script, str(target)], env=env,
                            text=True, capture_output=True, check=True, timeout=10)
    assert json.loads(result.stdout) == {"chmod": "EPERM", "unix": "EPERM", "netlink": "EPERM",
                                        "tcp": "ALLOWED", "udp": "EPERM", "fork": "EPERM", "thread_ok": True}
    assert target.read_text() == "unchanged"
