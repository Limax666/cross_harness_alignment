"""Trusted-supervisor proc observation over existing scratch-file IPC only.

Entry/native tools gain neither proc access nor a new descriptor. The private
receipt, not the scratch response, is authoritative at final admission.
"""
from __future__ import annotations
import asyncio
import json
import os
import secrets
from pathlib import Path
import threading
import time
from flashrsi_pi_owner_profile_v6 import ProfileError,save_json,read_native_session


def proc_identity(pid):
    raw=Path('/proc',str(pid),'stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    if fields[0] in {'Z','X'}:raise ProfileError('identity_process_not_live')
    return {'pid':pid,'ppid':int(fields[1]),'start':int(fields[19])}


def observe_process_identity(transport_pid,entry_pid,*,expected_executable=None,binding=None,expected_entry_start=None):
    if type(transport_pid) is not int or transport_pid<=1 or type(entry_pid) is not int:
        raise ProfileError('identity_PID_invalid')
    entry=proc_identity(entry_pid);child=proc_identity(transport_pid)
    if expected_entry_start is not None and entry['start']!=expected_entry_start:
        raise ProfileError('entry_start_identity_changed')
    expected=Path(expected_executable or __import__('sys').executable).resolve(strict=True)
    expected_stat=expected.stat();actual_stat=Path('/proc',str(transport_pid),'exe').stat()
    if (expected_stat.st_dev,expected_stat.st_ino)!=(actual_stat.st_dev,actual_stat.st_ino):
        raise ProfileError('native_CLI_executable_identity_changed')
    if child['ppid']!=entry_pid or child['start']<entry['start']:
        raise ProfileError('native_CLI_not_direct_live_entry_child')
    # Re-read both identities to close exit/reuse/reparent races across reads.
    if proc_identity(entry_pid)!=entry or proc_identity(transport_pid)!=child:
        raise ProfileError('native_identity_changed_during_observation')
    return {'observer':'trusted_namespace_supervisor_v9','entry_pid':entry_pid,'entry_start':entry['start'],
            'transport_pid':transport_pid,'transport_ppid':child['ppid'],'transport_start':child['start'],
            'observed_alive':True,'observed_monotonic_ns':time.monotonic_ns(),
            'binding':dict(binding or {}),'executable':{'path':str(expected),'dev':expected_stat.st_dev,'ino':expected_stat.st_ino}}


def validate_identity(receipt,close,*,binding=None,expected_executable=None):
    if (not isinstance(receipt,dict) or receipt.get('observer')!='trusted_namespace_supervisor_v9'
            or receipt.get('observed_alive') is not True or receipt.get('transport_pid')!=close.get('pid')
            or receipt.get('transport_start')!=close.get('start')
            or receipt.get('entry_pid')!=receipt.get('transport_ppid')
            or type(receipt.get('entry_start')) is not int or type(receipt.get('transport_start')) is not int
            or receipt['transport_start']<receipt['entry_start'] or close.get('identity')!=receipt
            or binding is not None and receipt.get('binding')!=binding
            or expected_executable is not None and receipt.get('executable',{}).get('path')!=str(Path(expected_executable).resolve())):
        raise ProfileError('trusted_native_CLI_identity_missing_or_mismatched')
    return receipt


class IdentityObserver:
    def __init__(self,request,response,scratch,entry_pid,deadline,*,expected_executable,binding):
        self.request,self.response,self.scratch=Path(request),Path(response),Path(scratch)
        self.entry_pid,self.deadline=entry_pid,deadline
        self.expected_entry_start=proc_identity(entry_pid)['start']
        self.expected_executable,self.binding=expected_executable,dict(binding)
        self.directory_fd=os.open(self.scratch,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        self.receipt=None;self.error=None;self.stop=threading.Event()
        self.thread=threading.Thread(target=self._serve,daemon=True);self.thread.start()
    def _serve(self):
        try:
            while not self.stop.is_set() and time.monotonic()<self.deadline:
                try:raw=read_native_session(self.request,self.scratch,max_bytes=1024)
                except ProfileError:
                    if self.request.exists():raise
                    self.stop.wait(.005);continue
                value=json.loads(raw)
                if set(value)!={'transport_pid','binding'} or value['binding']!=self.binding:
                    raise ProfileError('identity_request_binding_invalid')
                self.receipt=observe_process_identity(value['transport_pid'],self.entry_pid,
                    expected_executable=self.expected_executable,binding=self.binding,expected_entry_start=self.expected_entry_start)
                atomic_response(self.directory_fd,self.response.name,self.receipt)
                return
            raise ProfileError('native_identity_request_missing_or_deadline')
        except Exception as exc:self.error={'type':type(exc).__name__,'detail':str(exc)}
    def finish(self):
        self.stop.set();self.thread.join(timeout=2)
        if self.directory_fd is not None:
            os.close(self.directory_fd);self.directory_fd=None
        if self.thread.is_alive():raise ProfileError('identity_observer_not_joined')
        if self.error or self.receipt is None:raise ProfileError('trusted_native_identity_unavailable:'+str(self.error))
        return self.receipt


async def request_identity(pid,request,response,scratch,binding):
    """No proc read, socket or inherited FD in this entry-side operation."""
    save_json(request,{'transport_pid':pid,'binding':binding})
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        try:
            receipt=json.loads(read_native_session(Path(response),scratch,max_bytes=4096))
            if receipt.get('transport_pid')!=pid or receipt.get('binding')!=binding:
                raise ProfileError('identity_IPC_wrong_transport_or_replay')
            return receipt
        except ProfileError:
            if Path(response).exists():raise
        await asyncio.sleep(.005)
    raise ProfileError('trusted_native_identity_observation_timeout')


def atomic_response(directory_fd,name,value):
    """Single-directory-FD atomic replace; never follow an actor-controlled leaf/parent."""
    if Path(name).name!=name or name in {'.','..'}:raise ProfileError('identity_response_name_invalid')
    temporary='.identity-'+secrets.token_hex(16)
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory_fd)
    try:
        raw=(json.dumps(value,sort_keys=True)+'\n').encode();offset=0
        while offset<len(raw):offset+=os.write(fd,raw[offset:])
        os.fsync(fd);os.close(fd);fd=None
        os.rename(temporary,name,src_dir_fd=directory_fd,dst_dir_fd=directory_fd);os.fsync(directory_fd)
    finally:
        if fd is not None:os.close(fd)
        try:os.unlink(temporary,dir_fd=directory_fd)
        except FileNotFoundError:pass
