import json
import os
from pathlib import Path
import subprocess


def test_native_process_profile_allows_child_but_blocks_escape(tmp_path):
    script=r'''
import ctypes, errno, json, os, socket
from flashrsi_native_seccomp import restrict_syscalls
restrict_syscalls()
pid=os.fork()
if pid==0:
    os.execv('/bin/true',['true'])
_,status=os.waitpid(pid,0)
def attempt(fn):
    try: fn(); return 'allowed'
    except OSError as e: return errno.errorcode.get(e.errno,'other')
lib=ctypes.CDLL(None,use_errno=True)
def unshare():
    if lib.unshare(0x40000000)!=0:
        error=ctypes.get_errno(); raise OSError(error,'unshare')
print(json.dumps({'child_exit':os.waitstatus_to_exitcode(status),'setsid':attempt(os.setsid),
                  'unix':attempt(lambda:socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)),
                  'unshare':attempt(unshare)}))
'''
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    result=subprocess.run(['/usr/bin/python3','-c',script],env={**os.environ,'PYTHONPATH':str(scripts)},
                          text=True,capture_output=True,check=True,timeout=10)
    assert json.loads(result.stdout)=={'child_exit':0,'setsid':'EPERM','unix':'EPERM','unshare':'EPERM'}


def test_pid_namespace_profile_allows_child_session_control():
    script=r'''
import json, os
from flashrsi_native_seccomp import restrict_syscalls
assert os.getpid()==1
restrict_syscalls(allow_child_control=True)
pid=os.fork()
if pid==0:
    os.setsid()
    os.execv('/bin/true',['true'])
_,status=os.waitpid(pid,0)
print(json.dumps({'pid':os.getpid(),'child_exit':os.waitstatus_to_exitcode(status)}))
'''
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    result=subprocess.run(['unshare','-Urnpf','--mount-proc','/usr/bin/python3','-c',script],
                          env={**os.environ,'PYTHONPATH':str(scripts)},text=True,capture_output=True,check=True,timeout=10)
    assert json.loads(result.stdout)=={'pid':1,'child_exit':0}


def test_native_launcher_refuses_without_pid_namespace_supervisor():
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    result=subprocess.run(['/usr/bin/python3',str(scripts/'flashrsi_native_sandbox_exec.py')],
                          env={**os.environ,'PYTHONPATH':str(scripts)},text=True,capture_output=True,timeout=10)
    assert result.returncode!=0
    assert 'PID-namespace supervisor' in result.stderr
