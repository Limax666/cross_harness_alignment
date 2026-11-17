"""SDK0.2.181 transport: raw observations plus exact pinned close branch copy.

Only observation statements are inserted in _close_observed. Installed SDK
source remains unchanged; source pin is checked before constructing transport.
"""
from __future__ import annotations
import asyncio
import copy
from contextlib import suppress
import hashlib
from pathlib import Path
import anyio
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport, _ACTIVE_CHILDREN
from flashrsi_claude_owner_profile_v9 import ProfileError

SDK_TRANSPORT_SHA256 = 'c532d7a6421854cc09f51b64750a998b4542ab08e3195a1fee88c5e669d4318e'

class ObservedBytes:
    """Forward original byte receive/close unchanged; retain pre-parser bytes."""
    def __init__(self, stream, captured):self.stream,self.captured=stream,captured
    async def receive(self,*args,**kwargs):
        value=await self.stream.receive(*args,**kwargs)
        self.captured.append(value)
        return value
    async def aclose(self):return await self.stream.aclose()
    def __getattr__(self,name):return getattr(self.stream,name)


class ObservedTransport(SubprocessCLITransport):
    def __init__(self, prompt, options, frames, changed):
        import inspect
        source = Path(inspect.getsourcefile(SubprocessCLITransport))
        if hashlib.sha256(source.read_bytes()).hexdigest() != SDK_TRANSPORT_SHA256:
            raise ProfileError('pinned_SDK_transport_source_changed')
        super().__init__(prompt=prompt, options=options)
        self.frames, self.changed = frames, changed
        self.raw_writes = []
        self.raw_stdout_bytes = []
        self.close_evidence = {'events': [], 'returncode': None, 'reaped': None, 'exception': None}

    async def connect(self):
        await super().connect()
        self._stdout_stream.transport_stream = ObservedBytes(self._stdout_stream.transport_stream, self.raw_stdout_bytes)
        process = self._process
        self.close_evidence['pid'] = process.pid
        try:
            raw = Path('/proc', str(process.pid), 'stat').read_text()
            self.close_evidence['start'] = int(raw[raw.rfind(')') + 2:].split()[19])
        except (OSError, ValueError, IndexError):
            self.close_evidence['start'] = None

    async def read_messages(self):
        async for frame in super().read_messages():
            self.frames.append(copy.deepcopy(frame))
            self.changed.set()
            yield frame

    async def write(self, data):
        self.raw_writes.append(data)
        return await super().write(data)

    async def close(self):
        try:
            await self._close_observed()
        except BaseException as exc:
            self.close_evidence['exception'] = type(exc).__name__
            if isinstance(exc, asyncio.CancelledError):
                self.close_evidence['events'].append('cancelled')
            raise

    async def _close_observed(self) -> None:
        """Close the transport and clean up resources.

        The whole body runs inside a shielded cancel scope. Cleanup is
        routinely reached while the caller's task is being cancelled (e.g.
        `async with ClaudeSDKClient()` unwinding on cancel), and an
        unshielded close() would abort at the first await — before the
        terminate/kill escalation ran — orphaning the CLI child, which then
        surfaces as `<defunct>` once nothing is left to wait() on it.

        Every await in *this* scope is bounded (~20s worst case), so an anyio
        cancellation is delayed but never blocked: the stream `aclose()`s are a
        non-blocking `close()` plus a checkpoint on both anyio backends (they
        never await `wait_closed()`, so undrained stdin cannot wedge them), the
        stderr task is cancelled before it is awaited, and the lock acquire and
        every process `wait()` carry an explicit deadline.

        Caveat: an anyio shield only defers cancellation that *originates from
        an anyio cancel scope*. A raw asyncio cancellation (`asyncio.wait_for` /
        `asyncio.timeout` firing, a bare `task.cancel()`, loop shutdown) is
        still delivered at the next await in here, and the escalation below only
        catches `TimeoutError` — so it would be skipped. That is a pre-existing
        limitation of the shield on the asyncio backend rather than something
        this scope introduces, and it is still strictly better than before: the
        `finally` keeps a still-running child in `_ACTIVE_CHILDREN` for the
        atexit reaper instead of dropping it. Making the escalation robust to a
        foreign `CancelledError` is a follow-up.
        """
        self.close_evidence["events"].append("begin")
        if not self._process:
            self._ready = False
            return

        with anyio.CancelScope(shield=True):
            # Cancel stderr reader if active
            if self._stderr_task is not None and not self._stderr_task.done():
                self._stderr_task.cancel()
                with suppress(Exception):
                    await self._stderr_task.wait()
            self._stderr_task = None

            # Close stdin stream (hold the write lock to prevent a race with
            # concurrent writes). Bounded: a writer blocked on a full stdin
            # pipe must not pin the shielded scope forever.
            lock_held = False
            with anyio.move_on_after(5):
                await self._write_lock.acquire()
                lock_held = True
                self.close_evidence["events"].append("lock_acquired")
            if not lock_held:
                self.close_evidence["events"].append("lock_timeout")
            try:
                self._ready = False  # Set inside lock to prevent TOCTOU with write()
                if self._stdin_stream:
                    with suppress(Exception):
                        await self._stdin_stream.aclose()
                        self.close_evidence["events"].append("stdin_closed")
                    self._stdin_stream = None
            finally:
                if lock_held:
                    self._write_lock.release()

            if self._stderr_stream:
                with suppress(Exception):
                    await self._stderr_stream.aclose()
                self._stderr_stream = None

            # Wait for graceful shutdown after stdin EOF, then terminate if
            # needed. The subprocess needs time to flush its session file after
            # receiving EOF on stdin. Without this grace period, SIGTERM can
            # interrupt the write and cause the last assistant message to be
            # lost (see #625).
            try:
                if self._process.returncode is not None:
                    self.close_evidence["events"].append("already_exited")
                if self._process.returncode is None:
                    self.close_evidence["events"].append("graceful_wait")
                    try:
                        with anyio.fail_after(5):
                            await self._process.wait()
                        self.close_evidence["events"].append("graceful_reaped")
                    except TimeoutError:
                        # Graceful shutdown timed out — force terminate
                        self.close_evidence["events"].append("terminate")
                        with suppress(ProcessLookupError):
                            self._process.terminate()
                        try:
                            with anyio.fail_after(5):
                                await self._process.wait()
                        except TimeoutError:
                            # SIGTERM handler blocked — force kill (SIGKILL)
                            self.close_evidence["events"].append("kill")
                            with suppress(ProcessLookupError):
                                self._process.kill()
                            with suppress(Exception), anyio.fail_after(5):
                                await self._process.wait()
            finally:
                # Only stop tracking a child we actually reaped. A still-running
                # process (kill raced, or wait timed out) stays in the set so the
                # atexit reaper gets a chance at it — dropping it here is what
                # turned a cancelled close() into a leaked child. The reaper only
                # sends SIGTERM, so this rescues a child that is on its way out,
                # not one that survived SIGKILL.
                self.close_evidence["returncode"] = self._process.returncode
                self.close_evidence["reaped"] = self._process.returncode is not None
                if self._process.returncode is not None:
                    self.close_evidence["events"].append("reaped")
                    _ACTIVE_CHILDREN.discard(self._process)

            self._process = None
            self._stdout_stream = None
            self._stdin_stream = None
            self._stderr_stream = None
            self._exit_error = None
            self.close_evidence["events"].append("finished")
