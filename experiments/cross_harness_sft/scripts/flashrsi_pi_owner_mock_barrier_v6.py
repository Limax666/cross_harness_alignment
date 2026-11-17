"""Trusted transport-only mock gate; no candidate/service code is executed here."""
from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from urllib.parse import unquote, urlparse, urlsplit

from flashrsi_pi_owner_profile_v6 import ProfileError, byte_hash, digest

MAX_HTTP_BYTES = 64 * 1024 * 1024


def forward_http(upstream, method, path, headers, body, *, timeout, connection_factory=http.client.HTTPConnection):
    """Forward the original method, target, headers and payload without JSON rewriting."""
    address = urlsplit(upstream)
    if address.scheme != 'http' or address.hostname != '127.0.0.1' or not address.port:
        raise ProfileError('mock_upstream_must_be_original_case_loopback')
    connection = connection_factory(address.hostname, address.port, timeout=timeout)
    try:
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for name, value in headers:
            connection.putheader(name, value)
        connection.endheaders(body)
        response = connection.getresponse()
        raw = response.read(MAX_HTTP_BYTES + 1)
        if len(raw) > MAX_HTTP_BYTES:
            raise ProfileError('mock_response_archive_size_exceeded')
        return response.status, response.reason, response.getheaders(), raw
    finally:
        connection.close()


def response_settled(path, status, raw):
    """A gateway599 can hide a still-running downstream service after its5s timeout."""
    if status == 599:
        return False
    # Match original MockHandler._dispatch -> path_parts exactly: URL parse,
    # split/filter, then decode each segment once (not before splitting).
    parts = [unquote(part) for part in urlparse(path).path.strip('/').split('/') if part]
    proxy = len(parts) >= 2 and parts[0] == 'proxy'
    admin = parts in (['admin', 'reset_case'], ['admin', 'destroy_case'])
    if proxy or admin:
        try: value = json.loads(raw)
        except (ValueError, UnicodeError): return False
        if isinstance(value, dict) and (value.get('status') == 599 or
                any(isinstance(v, dict) and v.get('status') == 599 for v in value.get('results', {}).values())):
            return False
    return True


class RequestLedger:
    """A connection is pending from accept until full upstream response or failure."""
    def __init__(self, case_id, snapshot_id):
        self.case_id, self.snapshot_id = case_id, snapshot_id
        self.condition = threading.Condition()
        self.owner = 0
        self.sealed = True
        self.active, self.requests, self.failures = set(), [], []

    def open_owner(self, index):
        with self.condition:
            if self.active or self.failures or not self.sealed or index != self.owner + 1:
                raise ProfileError('mock_owner_boundary_not_settled')
            self.owner, self.sealed, self.requests = index, False, []

    def accept(self):
        with self.condition:
            identity = len(self.requests)
            self.requests.append({'connection': identity + 1, 'owner_index': self.owner})
            self.active.add(identity)
            if self.sealed:
                self.failures.append('connection_accepted_after_owner_seal')
            return identity

    def request(self, identity, method, path, headers, body):
        with self.condition:
            self.requests[identity].update(method=method, path=path, headers_sha256=digest(headers),
                                           body_sha256=byte_hash(body))

    def finish(self, identity, outcome):
        with self.condition:
            self.requests[identity].update(outcome)
            if outcome.get('failure') or outcome.get('upstream_complete') is not True:
                self.failures.append(outcome.get('failure', 'upstream_completion_unknown'))
            self.active.discard(identity)
            self.condition.notify_all()

    def seal(self):
        with self.condition:
            self.sealed = True

    def receipt(self):
        with self.condition:
            if not self.sealed or self.active:
                raise ProfileError('mock_requests_pending')
            if self.failures:
                raise ProfileError('mock_requests_uncertain:' + ','.join(self.failures))
            return {'case_id': self.case_id, 'snapshot_id': self.snapshot_id, 'owner_index': self.owner,
                    'sealed': True, 'pending_connections': 0, 'settled': True,
                    'request_count': len(self.requests), 'requests': [dict(r) for r in self.requests], 'failures': []}


def validate_mock_receipt(value, case_id, snapshot_id, index):
    if (not isinstance(value, dict) or value.get('case_id') != case_id or value.get('snapshot_id') != snapshot_id
            or value.get('owner_index') != index or value.get('sealed') is not True
            or value.get('settled') is not True or value.get('pending_connections') != 0
            or value.get('accept_loop_stopped') is not True or value.get('backlog_drained') is not True
            or value.get('failures') != [] or not isinstance(value.get('requests'), list)
            or value.get('request_count') != len(value['requests'])):
        raise ProfileError('actual_mock_request_settlement_missing_or_pending')
    for sequence, request in enumerate(value['requests'], 1):
        if (request.get('connection') != sequence or request.get('owner_index') != index
                or request.get('upstream_complete') is not True or request.get('failure')
                or not request.get('response_sha256') or 'method' not in request or 'path' not in request):
            raise ProfileError('actual_mock_upstream_work_unsettled')
    return value


class MockBarrier:
    """Pause acceptance and drain actual work after the target namespace census."""
    def __init__(self, case_id, snapshot_id):
        self.ledger = RequestLedger(case_id, snapshot_id)
        self.upstream, self.deadline, self.thread = None, 0, None
        barrier = self

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False
            def get_request(self):
                connection, address = super().get_request()
                identity = barrier.ledger.accept()
                connection.settimeout(max(.001, barrier.deadline - time.monotonic()))
                return (connection, identity), address
            def shutdown_request(self, request):
                connection, identity = request
                if identity in barrier.ledger.active:
                    barrier.ledger.finish(identity, {'failure': 'accepted_connection_unparsed_or_incomplete'})
                super().shutdown_request(connection)

        class Handler(BaseHTTPRequestHandler):
            def __init__(self, request, address, server):
                connection, self.identity = request
                super().__init__(connection, address, server)
            def log_message(self, *_args): pass
            def handle(self):
                try: super().handle()
                finally:
                    if self.identity in barrier.ledger.active:
                        barrier.ledger.finish(self.identity, {'failure': 'accepted_request_unparsed_or_incomplete'})
            def handle_one_request(self):
                # Original mock services use HTTP/1.0 and one request per connection.
                return super().handle_one_request()
            def forward(self):
                outcome = {}
                try:
                    size = int(self.headers.get('Content-Length', '0') or '0')
                    if not 0 <= size <= MAX_HTTP_BYTES:
                        raise ProfileError('mock_request_size_exceeded')
                    body = self.rfile.read(size)
                    if len(body) != size:
                        raise ProfileError('mock_request_body_incomplete')
                    headers = list(self.headers.raw_items())
                    barrier.ledger.request(self.identity, self.command, self.path, headers, body)
                    remaining = barrier.deadline - time.monotonic()
                    if remaining <= 0:
                        raise ProfileError('owner_deadline_before_mock_forward')
                    status, reason, response_headers, raw = forward_http(barrier.upstream,
                        self.command, self.path, headers, body, timeout=remaining)
                    outcome.update(status=status, response_sha256=byte_hash(raw),
                                   upstream_complete=response_settled(self.path, status, raw))
                    if not outcome['upstream_complete']:
                        outcome['failure'] = 'gateway_downstream_timeout_or_unobservable_response'
                    try:
                        self.send_response_only(status, reason)
                        for name, value in response_headers: self.send_header(name, value)
                        self.end_headers()
                        self.wfile.write(raw)
                        self.wfile.flush()
                        outcome['response_delivered'] = True
                    except (OSError, ValueError):
                        # Actor exit does not undo upstream state already committed.
                        outcome['response_delivered'] = False
                except Exception as exc:
                    outcome['failure'] = type(exc).__name__ + ':' + str(exc)
                finally:
                    barrier.ledger.finish(self.identity, outcome)
                self.close_connection = True
            do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = forward

        self._server_class, self._handler_class = Server, Handler
        self.server = Server(('127.0.0.1', 0), Handler)
        self.port = self.server.server_port
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def open_owner(self, index, deadline):
        if not self.upstream or self.thread is not None:
            raise ProfileError('mock_barrier_upstream_or_accept_loop_invalid')
        if self.server.fileno() < 0:
            self.server = self._server_class(('127.0.0.1', self.port), self._handler_class)
        self.ledger.open_owner(index)
        self.deadline = deadline
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def freeze(self):
        self.ledger.seal()
        if self.thread is not None:
            self.server.shutdown()
            self.thread.join()
            self.thread = None
        # No target process may survive the preceding namespace census. Stop
        # accept before checking the listen backlog, including unparsed clients.
        self.server.socket.setblocking(False)
        try:
            while True:
                try: connection, _address = self.server.socket.accept()
                except BlockingIOError: break
                identity = self.ledger.accept()
                connection.close()
                self.ledger.finish(identity, {'failure': 'unparsed_connection_in_sealed_listen_backlog'})
        finally:
            self.server.socket.setblocking(True)
        with self.ledger.condition:
            while self.ledger.active:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise ProfileError('mock_requests_pending_at_unchanged_owner_deadline')
                self.ledger.condition.wait(remaining)
        receipt = self.ledger.receipt()
        if time.monotonic() > self.deadline:
            raise ProfileError('mock_settlement_exceeded_unchanged_owner_deadline')
        # Close the paused listener as well: no latent previous-owner socket
        # may be accepted when the same public location reopens for next owner.
        self.server.server_close()
        return dict(receipt, accept_loop_stopped=True, backlog_drained=True)

    def close(self):
        if self.thread is not None:
            self.server.shutdown()
            self.thread.join()
            self.thread = None
        self.server.server_close()
