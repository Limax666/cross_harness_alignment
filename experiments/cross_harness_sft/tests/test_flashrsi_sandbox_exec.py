import json
import os
from pathlib import Path
import subprocess


def test_sandbox_child_filters_environment_and_limits_fs_network_and_syscalls(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scriptdir, writable, private = (tmp_path / name for name in ("script", "work", "private"))
    for path in (scriptdir, writable, private):
        path.mkdir()
    secret = private / "secret"
    secret.write_text("hidden")
    entry = scriptdir / "entry.py"
    entry.write_text('''import errno,json,os,pathlib,resource,socket,sys
secret=pathlib.Path(SECRET_PATH); work=pathlib.Path(os.environ["TMPDIR"])
def check(f):
 try: f(); return "ALLOWED"
 except OSError as e: return errno.errorcode.get(e.errno,"OTHER")
print(json.dumps({"read_secret":check(lambda:secret.read_text()),"chmod":check(lambda:os.chmod(__file__,0o600)),
 "unix":check(lambda:socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)),
 "udp":check(lambda:socket.socket(socket.AF_INET,socket.SOCK_DGRAM)),
 "write_work":check(lambda:(work/"ok").write_text("ok")),"secret_env":os.environ.get("TEST_PRIVATE_SECRET"),
 "as_limit":resource.getrlimit(resource.RLIMIT_AS)[0],"file_limit":resource.getrlimit(resource.RLIMIT_FSIZE)[0],
 "fd_limit":resource.getrlimit(resource.RLIMIT_NOFILE)[0],"core_limit":resource.getrlimit(resource.RLIMIT_CORE)[0]}))
'''.replace("SECRET_PATH", repr(str(secret))))
    config = tmp_path / "trusted.json"
    config.write_text(json.dumps({"entry": str(entry), "python": "/usr/bin/python3", "gateway_port": 12345,
                                  "readonly_roots": ["/usr", "/lib", "/lib64", str(scriptdir)],
                                  "writable_root": str(writable)}))
    env = {**os.environ, "FLASHRSI_SANDBOX_CONFIG": str(config), "FLASHRSI_GATEWAY_URL": "http://127.0.0.1:12345",
           "FLASHRSI_GATEWAY_TOKEN": "test-token-long-enough-32-characters", "TEST_PRIVATE_SECRET": "private"}
    result = subprocess.run(["/usr/bin/python3", str(root / "scripts/flashrsi_sandbox_exec.py"),
                             "-I", "-B", str(entry)], env=env, capture_output=True,
                            text=True, check=True, timeout=10)
    assert json.loads(result.stdout) == {"read_secret": "EACCES", "chmod": "EPERM", "unix": "EPERM",
                                        "udp": "EPERM", "write_work": "ALLOWED", "secret_env": None,
                                        "as_limit": 4 * 1024**3, "file_limit": 32 * 1024**2,
                                        "fd_limit": 64, "core_limit": 0}
    assert secret.read_text() == "hidden"


def test_child_restrictions_with_private_network_namespace(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scriptdir, writable, private = (tmp_path / name for name in ("script", "work", "private"))
    for path in (scriptdir, writable, private):
        path.mkdir()
    secret = private / "secret"
    secret.write_text("hidden")
    entry = scriptdir / "probe.py"
    entry.write_text('''import ctypes,errno,json,os,pathlib,socket
secret=pathlib.Path(SECRET_PATH)
def attempt(f):
 try: f(); return "ALLOWED"
 except OSError as e: return errno.errorcode.get(e.errno,"OTHER")
def connect(host,port):
 s=socket.socket(); s.settimeout(2)
 try: return s.connect_ex((host,port))
 finally: s.close()
libc=ctypes.CDLL(None,use_errno=True); result=libc.unshare(0x10000000)
print(json.dumps({"secret":attempt(lambda:secret.read_text()),"chmod":attempt(lambda:os.chmod(__file__,0o600)),
 "unix":attempt(lambda:socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)),"udp":attempt(lambda:socket.socket(socket.AF_INET,socket.SOCK_DGRAM)),
 "allowed":connect("127.0.0.1",ALLOWED_PORT),"other_port":connect("127.0.0.1",BLOCKED_PORT),
 "outside":connect("1.1.1.1",ALLOWED_PORT),"unshare_denied":result == -1 and ctypes.get_errno() == errno.EPERM,
 "relay_token":os.environ.get("DEV_RELAY_TOKEN")}))
'''.replace("SECRET_PATH", repr(str(secret))))
    supervisor = tmp_path / "supervisor.py"
    supervisor.write_text('''import json,os,pathlib,socket,subprocess,sys
root,entry,work,base=sys.argv[1:]
first=socket.socket(); first.bind(("127.0.0.1",0)); first.listen()
second=socket.socket(); second.bind(("127.0.0.1",0)); second.listen()
port=first.getsockname()[1]; other=second.getsockname()[1]
p=pathlib.Path(entry); p.write_text(p.read_text().replace("ALLOWED_PORT",str(port)).replace("BLOCKED_PORT",str(other)))
config=pathlib.Path(base)/"config.json"
config.write_text(json.dumps({"entry":entry,"python":"/usr/bin/python3","gateway_port":port,"readonly_roots":["/usr","/lib","/lib64",str(p.parent)],"writable_root":work}))
env={**os.environ,"FLASHRSI_SANDBOX_CONFIG":str(config),"FLASHRSI_GATEWAY_URL":f"http://127.0.0.1:{port}","FLASHRSI_GATEWAY_TOKEN":"test-token-long-enough-32-characters","DEV_RELAY_TOKEN":"private"}
r=subprocess.run(["/usr/bin/python3",root+"/scripts/flashrsi_sandbox_exec.py","-I","-B",entry],env=env,capture_output=True,text=True,timeout=10)
print(json.dumps({"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr}))
''')
    cmd = ["unshare", "-Urn", "sh", "-c", 'ip link set lo up && exec /usr/bin/python3 "$@"',
           "rsi", str(supervisor), str(root), str(entry), str(writable), str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
    outer = json.loads(result.stdout)
    assert outer["exit"] == 0, outer["stderr"]
    evidence = json.loads(outer["stdout"])
    assert evidence == {"secret": "EACCES", "chmod": "EPERM", "unix": "EPERM", "udp": "EPERM",
                        "allowed": 0, "other_port": 13, "outside": 101, "unshare_denied": True,
                        "relay_token": None}
