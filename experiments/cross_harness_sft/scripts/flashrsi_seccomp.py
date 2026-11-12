"""Experimental syscall restrictions; not a complete security sandbox.

Install only inside a disposable child after network namespace setup, closing
inherited descriptors and setting filesystem restrictions. This denylist is
not an exhaustive proof of containment and must be independently reviewed.
"""

import ctypes
import ctypes.util
import errno
import platform
import socket

_ALLOW = 0x7FFF0000
_ERRNO = 0x00050000
_CMP_NE = 1
_CMP_MASKED_EQ = 7

# This complements, rather than replaces, Landlock and a network namespace.
_DENY = (
    "chmod", "fchmod", "fchmodat", "fchmodat2", "chown", "fchown", "lchown", "fchownat",
    "mount", "umount2", "pivot_root", "move_mount", "open_tree", "fsopen", "fsmount", "fspick",
    "setns", "unshare", "ptrace", "process_vm_readv", "process_vm_writev", "pidfd_open",
    "pidfd_getfd", "pidfd_send_signal", "bpf", "perf_event_open", "userfaultfd", "open_by_handle_at",
    "name_to_handle_at", "io_uring_setup", "io_uring_enter", "io_uring_register", "keyctl", "add_key",
    "request_key", "kill", "tkill", "tgkill", "fork", "vfork", "setsid", "setpgid",
    "chroot", "sethostname", "setdomainname", "setxattr", "lsetxattr", "fsetxattr",
    "removexattr", "lremovexattr", "fremovexattr", "utime", "utimes", "futimesat", "utimensat",
)


class _Compare(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def _check(result):
    if result < 0:
        raise OSError(-result, errno.errorcode.get(-result, "seccomp error"))
    return result


def restrict_syscalls():
    """Irreversibly install an experimental denylist on this child and descendants."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise OSError("unsupported seccomp platform")
    path = ctypes.util.find_library("seccomp")
    if not path:
        raise OSError("libseccomp is unavailable")
    lib = ctypes.CDLL(path, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                           ctypes.c_uint, ctypes.POINTER(_Compare)]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    ctx = lib.seccomp_init(_ALLOW)
    if not ctx:
        raise OSError("seccomp initialization failed")
    try:
        for name in _DENY:
            nr = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
            if nr >= 0:
                _check(lib.seccomp_rule_add(ctx, _ERRNO | errno.EPERM, nr, 0))
        nr = lib.seccomp_syscall_resolve_name(b"socket")
        if nr < 0:
            raise OSError("socket syscall unavailable")
        # AF_INET is needed for the local gateway; other socket domains include
        # pathname/abstract Unix sockets, netlink and packet sockets.
        comparison = _Compare(0, _CMP_NE, socket.AF_INET, 0)
        _check(lib.seccomp_rule_add_array(ctx, _ERRNO | errno.EPERM, nr, 1, ctypes.byref(comparison)))
        # Linux socket type flags occupy the low nibble; preserve CLOEXEC and
        # NONBLOCK on stream sockets while refusing datagram/raw variants.
        for sock_type in (2, 3, 4, 5, 6, 10):
            comparison = _Compare(1, _CMP_MASKED_EQ, 0xF, sock_type)
            _check(lib.seccomp_rule_add_array(ctx, _ERRNO | errno.EPERM, nr, 1, ctypes.byref(comparison)))
        clone = lib.seccomp_syscall_resolve_name(b"clone")
        if clone < 0:
            raise OSError("clone syscall unavailable")
        # The isolated MiniAgent child may use threads, but may not fork a
        # separate process that could outlive its supervising process group.
        comparison = _Compare(0, _CMP_MASKED_EQ, 0x00010000, 0)
        _check(lib.seccomp_rule_add_array(ctx, _ERRNO | errno.EPERM, clone, 1, ctypes.byref(comparison)))
        for flag in (0x00000080, 0x00020000, 0x02000000, 0x04000000, 0x08000000,
                     0x10000000, 0x20000000, 0x40000000):
            comparison = _Compare(0, _CMP_MASKED_EQ, flag, flag)
            _check(lib.seccomp_rule_add_array(ctx, _ERRNO | errno.EPERM, clone, 1, ctypes.byref(comparison)))
        clone3 = lib.seccomp_syscall_resolve_name(b"clone3")
        if clone3 >= 0:
            # clone3's flags are behind a pointer and cannot be inspected by
            # seccomp. ENOSYS lets libc use filtered clone for normal threads.
            _check(lib.seccomp_rule_add(ctx, _ERRNO | errno.ENOSYS, clone3, 0))
        _check(lib.seccomp_load(ctx))
    finally:
        lib.seccomp_release(ctx)
