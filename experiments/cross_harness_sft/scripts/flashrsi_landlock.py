"""Experimental Linux filesystem confinement for a child process.

Optional network rules limit TCP ports only; they do not isolate UDP, inherited
descriptors, process visibility, Unix sockets, devices or resources. Landlock
also does not mediate metadata calls
such as chmod/chown: "readonly" below means the handled content access rights,
not an immutable filesystem. Do not use as a complete sandbox. Apply only in
a disposable child, with a sanitised environment and closed inherited FDs.
"""

import ctypes
import os
import platform
import stat

_CREATE, _ADD, _RESTRICT = 444, 445, 446
_VERSION = 1
_PATH_BENEATH = 1
_NET_PORT = 2
_NO_NEW_PRIVS = 38
_READ = (1 << 0) | (1 << 2) | (1 << 3)
_ALL = (1 << 15) - 1  # filesystem rights through LANDLOCK_ACCESS_FS_TRUNCATE (ABI 3)
_BIND_TCP = 1 << 0
_CONNECT_TCP = 1 << 1


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _NetPort(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


def _checked(result):
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def restrict_filesystem(readonly_roots, writable_roots, *, allowed_tcp_ports=None, allowed_devices=()):
    """Irreversibly restrict this process and descendants to explicit dirs.

    The calling process must exit on any error; do not retry after a partial
    restriction. Rules are additive, so overlapping read/write roots are
    rejected. This function intentionally makes no sandbox-readiness claim.
    """
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise OSError("unsupported Landlock platform")
    if not isinstance(readonly_roots, (list, tuple)) or not isinstance(writable_roots, (list, tuple)):
        raise ValueError("roots must be lists")
    if not readonly_roots and not writable_roots:
        raise ValueError("at least one allowed root is required")
    if allowed_tcp_ports is not None and (not isinstance(allowed_tcp_ports, (list, tuple)) or
            any(type(port) is not int or not 1 <= port <= 65535 for port in allowed_tcp_ports)):
        raise ValueError("allowed TCP ports must be integers from 1 to 65535")
    devices = {"/dev/urandom": 1 << 2, "/dev/null": (1 << 1) | (1 << 2)}
    if not isinstance(allowed_devices, (list, tuple)) or any(path not in devices for path in allowed_devices):
        raise ValueError("only explicitly supported device nodes may be allowed")
    for path in allowed_devices:
        if not stat.S_ISCHR(os.stat(path, follow_symlinks=False).st_mode):
            raise ValueError("allowed device path is not a character device")
    roots = []
    for path in [*readonly_roots, *writable_roots]:
        if not isinstance(path, str) or not os.path.isabs(path):
            raise ValueError("allowed roots must be absolute directory paths")
        real = os.path.realpath(path, strict=True)
        if real == "/" or not os.path.isdir(real):
            raise ValueError("allowed roots must be non-root directories")
        roots.append(real)
    ro = roots[:len(readonly_roots)]
    rw = roots[len(readonly_roots):]
    for a in ro:
        for b in rw:
            if os.path.commonpath((a, b)) in (a, b):
                raise ValueError("read-only and writable roots must not overlap")
    libc = ctypes.CDLL(None, use_errno=True)
    version = _checked(libc.syscall(_CREATE, 0, 0, _VERSION))
    if version < (4 if allowed_tcp_ports is not None else 3):
        raise OSError("required Landlock ABI is unavailable")
    attr = _Ruleset(_ALL, (_BIND_TCP | _CONNECT_TCP) if allowed_tcp_ports is not None else 0)
    attr_size = ctypes.sizeof(attr) if version >= 4 else ctypes.sizeof(ctypes.c_uint64)
    ruleset = _checked(libc.syscall(_CREATE, ctypes.byref(attr), attr_size, 0))
    try:
        for path, access in [(x, _READ) for x in ro] + [(x, _ALL) for x in rw]:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_DIRECTORY)
            try:
                rule = _PathBeneath(access, fd)
                _checked(libc.syscall(_ADD, ruleset, _PATH_BENEATH, ctypes.byref(rule), 0))
            finally:
                os.close(fd)
        for path in allowed_devices:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                rule = _PathBeneath(devices[path], fd)
                _checked(libc.syscall(_ADD, ruleset, _PATH_BENEATH, ctypes.byref(rule), 0))
            finally:
                os.close(fd)
        for port in allowed_tcp_ports or ():
            rule = _NetPort(_CONNECT_TCP, port)
            _checked(libc.syscall(_ADD, ruleset, _NET_PORT, ctypes.byref(rule), 0))
        _checked(libc.prctl(_NO_NEW_PRIVS, 1, 0, 0, 0))
        _checked(libc.syscall(_RESTRICT, ruleset, 0))
    finally:
        os.close(ruleset)
