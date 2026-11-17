"""Accepted/unparsed native-model HTTP connections hold the owner boundary."""
from __future__ import annotations
import threading
from flashrsi_claude_owner_profile_v9 import ProfileError


class ModelBarrier:
    def __init__(self,factory,*,thread_factory=threading.Thread):
        self.factory,self.thread_factory=factory,thread_factory
        self.server=factory();self.thread=None;self.owner=0
        self.lock=threading.Lock();self.active=set();self.connections=0;self.failures=[]
        self._observe_server()

    def _observe_server(self):
        server=self.server
        original_accept=server.get_request
        original_dispatch=server.process_request_thread
        def accept():
            connection,address=original_accept()
            with self.lock:
                self.connections+=1;self.active.add(id(connection))
            return connection,address
        def dispatch(connection,address):
            try:return original_dispatch(connection,address)
            except BaseException as exc:
                with self.lock:self.failures.append(type(exc).__name__)
                raise
            finally:
                with self.lock:self.active.discard(id(connection))
        server.get_request=accept
        server.process_request_thread=dispatch

    def open_owner(self,index):
        with self.lock:
            if self.thread is not None or self.active or self.failures or index!=self.owner+1:
                raise ProfileError('native_model_owner_boundary_unsettled')
            self.owner=index;self.connections=0
            self.thread=self.thread_factory(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
            self.thread.start()

    def freeze(self):
        if self.thread is None:raise ProfileError('native_model_accept_loop_unknown')
        self.server.shutdown();self.thread.join();self.thread=None
        # Census has proved no target process survives. Stop accepting first,
        # then account for connections that reached the listen backlog only.
        self.server.socket.setblocking(False)
        try:
            while True:
                try:connection,_address=self.server.socket.accept()
                except BlockingIOError:break
                connection.close()
                self.failures.append('native_model_unparsed_connection_in_listen_backlog')
        finally:self.server.socket.setblocking(True)
        with self.lock:
            if self.active or self.failures or self.server.errors:
                raise ProfileError('native_model_active_unparsed_or_failed_transport')
            return {'owner_index':self.owner,'connections':self.connections,'active':0,'settled':True,
                    'accept_loop_stopped':True,'backlog_drained':True,'listener_retained_for_same_case':True,'failures':[]}

    def close(self):
        if self.thread is not None:
            self.server.shutdown();self.thread.join();self.thread=None
        self.server.server_close()
