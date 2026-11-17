"""Authorized CPU-only owned sleeper/FD fixtures; no namespace/native/network."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module():
    import flashrsi_owned_lifetime_v1
    return flashrsi_owned_lifetime_v1

def read_when(path,timeout=5):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:return json.loads(path.read_text())
        except (FileNotFoundError,ValueError):time.sleep(.01)
    raise AssertionError('fixture receipt deadline:'+str(path))

def test_shared_guard_surface_exists():
    assert (ROOT/'scripts/flashrsi_owned_lifetime_v1.py').is_file(), 'host-to-owned-launcher lease guard missing'

@pytest.mark.parametrize('cause',['host_kill','explicit_close','outer_timeout'])
def test_real_owned_guard_reaps_sleeper_after_lease_holder_death(tmp_path,cause):
    m=module();receipt=tmp_path/'guard.json';ready=tmp_path/'ready.json'
    fixture=("import json,os,sys,time\nfrom pathlib import Path\nfrom flashrsi_owned_lifetime_v1 import OwnedLifetime\n"
             "guard=OwnedLifetime([sys.executable,'-c','import time;time.sleep(30)'],Path(sys.argv[1]),namespace_required=False)\n"
             "Path(sys.argv[2]).write_text(json.dumps({'host':os.getpid(),'guardian':guard.process.pid}))\n"
             "time.sleep(30)\n")
    host=subprocess.Popen([sys.executable,'-c',fixture,str(receipt),str(ready)],env={**os.environ,'PYTHONPATH':str(ROOT/'scripts')},close_fds=True)
    try:
        observed=read_when(ready);running=read_when(receipt.with_suffix('.running.json'))
        if cause=='outer_timeout':
            with pytest.raises(subprocess.TimeoutExpired):host.wait(timeout=.05)
        host.kill();host.wait(timeout=5)
        proof=m.await_cleanup(receipt,host.pid,timeout=5)
        assert proof['cleanup_confirmed'] is True and proof['reason']=='lease_EOF'
        assert proof['launcher']['pid']==running['launcher']['pid']
        assert proof['launcher_reaped'] is True and proof['owned_children_reaped'] is True
        with pytest.raises(ProcessLookupError):os.kill(running['launcher']['pid'],0)
    finally:
        if host.poll() is None:host.kill();host.wait(timeout=5)

def test_guard_fd_writer_never_reaches_child_or_grandchild(tmp_path):
    m=module();proof=tmp_path/'guard.json';child_out=tmp_path/'child.json'
    descriptors="import json,os\nv=[]\nfor f in os.listdir('/proc/self/fd'):\n try:v.append(os.readlink('/proc/self/fd/'+f))\n except FileNotFoundError:pass\nprint(json.dumps(v))"
    code=("import json,os,subprocess,sys,time\nfrom pathlib import Path\n"
          "targets=[]\nfor f in os.listdir('/proc/self/fd'):\n"
          " try:targets.append(os.readlink('/proc/self/fd/'+f))\n except FileNotFoundError:pass\n"
          "grandchild=json.loads(subprocess.check_output([sys.executable,'-c',sys.argv[2]],close_fds=True))\n"
          "Path(sys.argv[1]).write_text(json.dumps({'fds':targets,'grandchild_fds':grandchild}))\ntime.sleep(30)\n")
    guard=m.OwnedLifetime([sys.executable,'-c',code,str(child_out),descriptors],proof,namespace_required=False)
    try:
        observed=read_when(child_out)
        lease='pipe:['+str(os.fstat(guard.writer).st_ino)+']'
        assert lease not in observed['fds']+observed['grandchild_fds']
        result=guard.close(timeout=5)
        assert result['cleanup_confirmed'] is True
        assert result['child_pass_fds']==[] and result['child_close_fds'] is True
    finally:guard.close(timeout=5)

def test_async_cancellation_waits_for_guardian_cleanup(tmp_path):
    m=module();receipt=tmp_path/'guard.json';ready=tmp_path/'ready.json'
    fixture=("import sys,time;from pathlib import Path;from flashrsi_owned_lifetime_v1 import OwnedLifetime;"
             "guard=OwnedLifetime([sys.executable,'-c','import time;time.sleep(30)'],Path(sys.argv[1]),namespace_required=False);"
             "Path(sys.argv[2]).write_text('{}');time.sleep(30)")
    async def exercise():
        proc=await asyncio.create_subprocess_exec(sys.executable,'-c',fixture,str(receipt),str(ready),env={**os.environ,'PYTHONPATH':str(ROOT/'scripts')})
        await asyncio.to_thread(read_when,ready)
        task=asyncio.create_task(m.wait_host(proc,receipt,timeout=30))
        await asyncio.sleep(.02)
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
        return proc
    proc=asyncio.run(exercise())
    assert proc.returncode is not None
    assert m.await_cleanup(receipt,proc.pid,timeout=5)['cleanup_confirmed'] is True

@pytest.mark.parametrize('corruption',['host','pending','namespace','binding'])
def test_lifetime_receipt_rejects_unknown_cleanup_and_wrong_namespace_binding(corruption):
    m=module();value={'version':1,'host':{'pid':10,'start':1},'cleanup_confirmed':True,'launcher_reaped':True,
        'owned_children_reaped':True,'namespace_required':True,'namespace':{'nspid':[12,1],'namespace_inode':42},
        'namespace_exited':True,'namespace_pid1_gone':True,'binding':{'case_id':'case'}}
    if corruption=='host':value['host']['start']=None
    if corruption=='pending':value['owned_children_reaped']=False
    if corruption=='namespace':value['namespace']['nspid']=[12,2]
    if corruption=='binding':value['binding']['case_id']='other'
    with pytest.raises(ValueError):m.validate_lifetime_receipt(value,binding={'case_id':'case'},require_namespace=True)

def test_actual_unshare_host_paths_and_both_source_maps_bind_shared_guard():
    import ast
    for name in ('flashrsi_harnessrisk_host_claude_native_owner_v9.py','flashrsi_harnessrisk_host_pi_native_owner_v6.py'):
        source=(ROOT/'scripts'/name).read_text();tree=ast.parse(source)
        assert 'OwnedLifetime' in source and '--kill-child=SIGKILL' in source
        assert 'namespace_parent_death_signal' not in source
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen' for n in ast.walk(tree))
    from flashrsi_harnessrisk_claude_native_owner_initial_v9 import source_pins as claude_pins
    from flashrsi_harnessrisk_pi_native_owner_initial_v6 import source_pins as pi_pins
    assert any(k.endswith('/flashrsi_owned_lifetime_v1.py') for k in claude_pins())
    assert any(k.endswith('/flashrsi_owned_lifetime_v1.py') for k in pi_pins())

def test_guard_checks_lease_eof_before_spawning_any_child(tmp_path):
    m=module();receipt=tmp_path/'guard.json';read_fd,writer=os.pipe()
    config={'host':m.identity(os.getpid()),'command':[sys.executable,'-c','raise SystemExit(99)'],
        'environment':dict(os.environ),'receipt':str(receipt),'timeout':5,'namespace_required':False,'binding':{}}
    guardian=subprocess.Popen([sys.executable,str(ROOT/'scripts/flashrsi_owned_lifetime_v1.py'),'--guardian',str(read_fd)],
        stdin=subprocess.PIPE,close_fds=True,pass_fds=(read_fd,),start_new_session=True)
    os.close(read_fd);os.close(writer)
    guardian.communicate(json.dumps(config).encode(),timeout=5)
    value=m.await_cleanup(receipt,os.getpid(),timeout=2)
    assert value['reason']=='lease_EOF_before_spawn' and value['launcher_spawned'] is False
    assert value['launcher_reaped'] is False and 'launcher' not in value

def test_guard_reaps_its_own_adopted_grandchild(tmp_path):
    m=module();receipt=tmp_path/'guard.json';ready=tmp_path/'ready.json'
    code=("import json,subprocess,sys,time;from pathlib import Path;"
          "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],close_fds=True);"
          "Path(sys.argv[1]).write_text(json.dumps({'grandchild':child.pid}));time.sleep(30)")
    guard=m.OwnedLifetime([sys.executable,'-c',code,str(ready)],receipt,namespace_required=False)
    try:
        observed=read_when(ready);grandchild=m.identity(observed['grandchild'])
        proof=guard.close(timeout=5)
        assert proof['owned_children_reaped'] is True and not m.same_process(grandchild)
    finally:guard.close(timeout=5)

def test_cancellation_during_creation_keeps_child_handle_until_verified_cleanup(monkeypatch,tmp_path):
    m=module();receipt=tmp_path/'guard.json';ready=tmp_path/'ready.json'
    fixture=("import sys,time;from pathlib import Path;from flashrsi_owned_lifetime_v1 import OwnedLifetime;"
             "g=OwnedLifetime([sys.executable,'-c','import time;time.sleep(30)'],Path(sys.argv[1]),namespace_required=False);"
             "Path(sys.argv[2]).write_text('{}');time.sleep(30)")
    async def exercise():
        proc=await asyncio.create_subprocess_exec(sys.executable,'-c',fixture,str(receipt),str(ready),env={**os.environ,'PYTHONPATH':str(ROOT/'scripts')})
        await asyncio.to_thread(read_when,ready)
        async def delayed_creation(*_args,**_kwargs):await asyncio.sleep(.03);return proc
        monkeypatch.setattr(asyncio,'create_subprocess_exec',delayed_creation)
        proof={};task=asyncio.create_task(m.spawn_host('owned-fixture',receipt=receipt,proof=proof))
        await asyncio.sleep(.005);task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert proc.returncode is not None and proof['cleanup_confirmed'] is True
    asyncio.run(exercise())
