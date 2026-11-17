"""Trusted host lease -> guardian -> exact owned launcher/namespace lifetime.

Only the host owns the lease writer. Guardian and all launched descendants use
close_fds and receive no writer. Host SIGKILL therefore causes EOF even when
Python finally blocks cannot execute. No target sandbox permission changes.
"""
from __future__ import annotations
import asyncio
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from flashrsi_pi_owner_profile_v6 import ProfileError,save_json


def identity(pid):
    raw=Path('/proc',str(pid),'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
    return {'pid':pid,'start':int(parts[19]),'ppid':int(parts[1]),'pgid':int(parts[2]),'state':parts[0]}


def same_process(expected):
    try:return identity(expected['pid'])['start']==expected['start']
    except FileNotFoundError:return False


def _children(pid):
    try:return [int(v) for v in Path('/proc',str(pid),'task',str(pid),'children').read_text().split()]
    except FileNotFoundError:return []


def namespace_child(launcher):
    for pid in _children(launcher['pid']):
        try:
            value=identity(pid)
            if value['ppid']!=launcher['pid'] or value['start']<launcher['start']:continue
            fields=Path('/proc',str(pid),'status').read_text().splitlines()
            nspid=next(([int(v) for v in row.split()[1:]] for row in fields if row.startswith('NSpid:')),[])
            inode=os.stat('/proc/'+str(pid)+'/ns/pid').st_ino
            if len(nspid)>1 and nspid[-1]==1:
                return dict(value,nspid=nspid,namespace_inode=inode)
        except (FileNotFoundError,ProcessLookupError):continue
    return None


def guardian(config,lease_fd):
    os.set_inheritable(lease_fd,False)
    # This property belongs to the trusted guardian, not unshare/setuid. It is
    # used only to reap its own adopted descendants after PID1 exits.
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'subreaper setup failed')
    process=None;proof={'version':1,'host':config['host'],'guardian':identity(os.getpid()),
        'command':config['command'],'child_close_fds':True,'child_pass_fds':[],
        'namespace_required':config['namespace_required'],'namespace':None,'binding':config.get('binding',{}),
        'cleanup_confirmed':False,'launcher_spawned':False,'launcher_reaped':False,'owned_children_reaped':False,'signals':[]}
    try:
        # Parent-death setup race: check EOF before spawn, then monitor the same
        # non-inherited pipe. EOF during spawn is handled on the next iteration.
        ready,_,_=select.select([lease_fd],[],[],0)
        if ready and not os.read(lease_fd,1):
            proof.update(reason='lease_EOF_before_spawn',cleanup_confirmed=True,launcher_reaped=False,owned_children_reaped=True)
            return 125
        process=subprocess.Popen(config['command'],env=config['environment'],close_fds=True,pass_fds=(),start_new_session=True)
        launch=identity(process.pid)
        if launch['pgid']!=process.pid:raise ProfileError('owned_launcher_not_new_session_leader')
        proof['launcher']=launch;proof['launcher_spawned']=True
        save_json(Path(config['receipt']).with_suffix('.running.json'),proof)
        deadline=time.monotonic()+config['timeout'];reason='launcher_exit'
        while True:
            if proof['namespace'] is None:proof['namespace']=namespace_child(launch)
            if process.poll() is not None:break
            ready,_,_=select.select([lease_fd],[],[],.005)
            if ready and not os.read(lease_fd,1):reason='lease_EOF';break
            if time.monotonic()>=deadline:reason='whole_host_timeout';break
        proof['reason']=reason
        if process.poll() is None:
            # PID remains our unreaped direct child during verification+kill;
            # reuse cannot redirect this exact owned process-group operation.
            if not same_process(launch) or os.getpgid(process.pid)!=launch['pgid']:
                raise ProfileError('owned_launcher_identity_changed_before_cleanup')
            os.killpg(launch['pgid'],signal.SIGKILL);proof['signals'].append({'pgid':launch['pgid'],'signal':'SIGKILL'})
        process.wait(timeout=10);proof.update(returncode=process.returncode,launcher_reaped=True)
        # Reap only guardian-owned/adopted children. Never enumerate/kill other
        # users' processes. Namespace PID1 death is the kernel's namespace-wide
        # kill boundary even if descendants created separate process groups.
        reap_deadline=time.monotonic()+10;children_gone=False
        while time.monotonic()<reap_deadline:
            try:
                pid,_status=os.waitpid(-1,os.WNOHANG)
                if pid:continue
            except ChildProcessError:children_gone=True;break
            time.sleep(.005)
        proof['owned_children_reaped']=children_gone
        ns=proof['namespace'];proof['namespace_pid1_gone']=not same_process(ns) if ns else None
        proof['namespace_exited']=(proof['namespace_pid1_gone'] is True if config['namespace_required'] else children_gone)
        proof['cleanup_confirmed']=bool(children_gone and (not ns or proof['namespace_pid1_gone']))
        # A successful actual namespace run must have an observed PID1 witness.
        if config['namespace_required'] and process.returncode==0 and ns is None:
            proof['cleanup_confirmed']=False
        return process.returncode if proof['cleanup_confirmed'] else 125
    except BaseException as exc:
        proof['error']={'type':type(exc).__name__,'detail':str(exc)}
        if process is not None and process.poll() is None:
            launch=proof.get('launcher')
            if launch and same_process(launch) and os.getpgid(process.pid)==launch['pgid']:
                os.killpg(launch['pgid'],signal.SIGKILL)
            process.wait(timeout=10)
        raise
    finally:
        os.close(lease_fd);save_json(config['receipt'],proof)


class OwnedLifetime:
    def __init__(self,command,receipt,*,environment=None,timeout=2400,namespace_required=True,binding=None):
        self.receipt=Path(receipt)
        if self.receipt.exists() or self.receipt.with_suffix('.running.json').exists():
            raise ProfileError('owned_lifetime_evidence_already_exists')
        self.host=identity(os.getpid());read_fd,self.writer=os.pipe()
        os.set_inheritable(self.writer,False)
        config={'host':self.host,'command':list(command),'environment':dict(os.environ if environment is None else environment),
                'receipt':str(self.receipt),'timeout':timeout,'namespace_required':namespace_required,'binding':dict(binding or {})}
        try:
            self.process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--guardian',str(read_fd)],
                stdin=subprocess.PIPE,pass_fds=(read_fd,),close_fds=True,start_new_session=True)
            self.process.stdin.write(json.dumps(config).encode());self.process.stdin.close()
        except BaseException:
            os.close(self.writer);self.writer=None;raise
        finally:os.close(read_fd)
    def wait(self,timeout=None):
        code=self.process.wait(timeout=timeout)
        proof=await_cleanup(self.receipt,self.host['pid'],timeout=2,expected_start=self.host['start'])
        return code,proof
    def close(self,timeout=30):
        if self.writer is not None:os.close(self.writer);self.writer=None
        self.process.wait(timeout=timeout)
        return await_cleanup(self.receipt,self.host['pid'],timeout=2,expected_start=self.host['start'])


def await_cleanup(path,host_pid,*,timeout=30,expected_start=None):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            value=json.loads(Path(path).read_text())
            if (value.get('version')!=1 or value.get('host',{}).get('pid')!=host_pid
                    or expected_start is not None and value['host'].get('start')!=expected_start):
                raise ProfileError('owned_lifetime_host_identity_mismatch')
            validate_lifetime_receipt(value)
            return value
        except FileNotFoundError:time.sleep(.01)
    raise ProfileError('owned_lifetime_cleanup_receipt_deadline')


def validate_lifetime_receipt(value,*,binding=None,require_namespace=None):
    no_spawn=(value.get('reason')=='lease_EOF_before_spawn' and value.get('launcher_spawned') is False and 'launcher' not in value)
    if (value.get('version')!=1 or value.get('cleanup_confirmed') is not True
            or not no_spawn and value.get('launcher_reaped') is not True or value.get('owned_children_reaped') is not True
            or type(value.get('host',{}).get('start')) is not int):
        raise ProfileError('owned_lifetime_cleanup_unconfirmed')
    if binding is not None and value.get('binding')!=binding:raise ProfileError('owned_lifetime_case_binding_changed')
    if no_spawn and require_namespace is None:return value
    required=value.get('namespace_required') if require_namespace is None else require_namespace
    if required and (value.get('namespace_required') is not True or not value.get('namespace')
            or value.get('namespace_exited') is not True or value.get('namespace_pid1_gone') is not True
            or value['namespace'].get('nspid',[])[-1:]!=[1]
            or type(value['namespace'].get('namespace_inode')) is not int):
        raise ProfileError('owned_namespace_PID1_exit_unconfirmed')
    return value


async def spawn_host(*command,receipt,proof=None,**kwargs):
    """Keep the actual child handle even if cancellation races subprocess creation."""
    proof=proof if proof is not None else {}
    creation=asyncio.create_task(asyncio.create_subprocess_exec(*command,**kwargs))
    cancelled=False
    while not creation.done():
        try:await asyncio.shield(creation)
        except asyncio.CancelledError:cancelled=True
    process=creation.result()
    if cancelled:
        proof.update(cancelled=True,cleanup_confirmed=False)
        if process.returncode is None:process.kill()
        await process.wait()
        cleanup=asyncio.create_task(asyncio.to_thread(await_cleanup,receipt,process.pid,timeout=30))
        while not cleanup.done():
            try:await asyncio.shield(cleanup)
            except asyncio.CancelledError:pass
        proof.update(lifetime_cleanup=cleanup.result(),cleanup_confirmed=True)
        raise asyncio.CancelledError
    return process


async def wait_host(process,receipt,*,timeout=2460,proof=None):
    """Outer timeout/cancellation always kills only this host, then awaits lease cleanup."""
    proof=proof if proof is not None else {}
    cancelled=False
    try:
        await asyncio.wait_for(process.wait(),timeout=timeout)
    except (TimeoutError,asyncio.CancelledError) as exc:
        cancelled=isinstance(exc,asyncio.CancelledError)
        proof['cancelled' if cancelled else 'timed_out']=True
        proof['cleanup_confirmed']=False
        if process.returncode is None:process.kill()
        await process.wait()
        cleanup=asyncio.create_task(asyncio.to_thread(await_cleanup,receipt,process.pid,timeout=30))
        while not cleanup.done():
            try:await asyncio.shield(cleanup)
            except asyncio.CancelledError:cancelled=True
        proof['lifetime_cleanup']=cleanup.result();proof['cleanup_confirmed']=True
        if cancelled:raise asyncio.CancelledError
        raise ProfileError('host_outer_timeout_after_verified_owned_cleanup')
    return process.returncode


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--guardian':raise SystemExit('internal guardian only')
    raise SystemExit(guardian(json.load(sys.stdin),int(sys.argv[2])))
