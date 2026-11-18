#!/usr/bin/env python3
"""Forward one loopback endpoint across a private Unix socket.

The host side never accepts a destination from the sandbox.  The guest side
is intended to run in a network namespace with only loopback enabled.
"""
import argparse
import ctypes
import os
import selectors
import socket
import socketserver
import threading


def relay(a, b):
    sel = selectors.DefaultSelector()
    for source, target in ((a, b), (b, a)):
        sel.register(source, selectors.EVENT_READ, target)
    try:
        while sel.get_map():
            for key, _ in sel.select():
                data = key.fileobj.recv(65536)
                if not data:
                    return
                key.data.sendall(data)
    except (OSError, ConnectionError):
        pass
    finally:
        sel.close()


class ThreadedUnix(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class ThreadedTCP(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('host', 'guest'))
    p.add_argument('--socket', required=True)
    p.add_argument('--socket-dir')
    p.add_argument('--port', required=True, type=int)
    a = p.parse_args()
    if a.socket_dir:
        os.chdir(a.socket_dir)
    if a.mode == 'host':
        # Do not leave bridge processes behind when the worker exits.
        parent = os.getppid()
        if ctypes.CDLL(None).prctl(1, 15, 0, 0, 0) != 0 or os.getppid() != parent:
            raise RuntimeError('cannot bind bridge lifetime to parent')
    if not 1024 <= a.port <= 65535:
        p.error('invalid port')
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            target = socket.socket(socket.AF_INET if a.mode == 'host' else socket.AF_UNIX)
            try:
                target.connect(('127.0.0.1', a.port) if a.mode == 'host' else a.socket)
                relay(self.request, target)
            finally:
                target.close()
    if a.mode == 'host':
        if os.path.lexists(a.socket):
            raise RuntimeError('refusing to replace existing socket')
        server = ThreadedUnix(a.socket, Handler)
        os.chmod(a.socket, 0o600)
    else:
        server = ThreadedTCP(('127.0.0.1', a.port), Handler)
    with server:
        server.serve_forever(poll_interval=0.1)


if __name__ == '__main__':
    main()
