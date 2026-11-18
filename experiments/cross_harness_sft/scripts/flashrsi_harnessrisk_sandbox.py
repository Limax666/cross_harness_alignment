#!/usr/bin/python3
"""Launch a MiniAgent episode with filesystem and network confinement."""
import json, os, sys
from flashrsi_landlock import restrict_filesystem
from flashrsi_native_seccomp import restrict_syscalls
from flashrsi_native_sandbox_exec import _drop_capabilities, _limit_resources

if os.getppid() != 1:
    raise SystemExit('sandbox requires a PID-namespace supervisor')
config=json.loads(os.environ['HARNESSRISK_SANDBOX_CONFIG'])
if set(config)!={'python','entry','readonly','writable','ports','environment'}:
    raise SystemExit('invalid sandbox config')
if sys.argv[1:] != ['-I','-B',config['entry']]:
    raise SystemExit('entry mismatch')
env=config['environment']
_limit_resources();_drop_capabilities();restrict_syscalls(allow_child_control=True)
restrict_filesystem(config['readonly'],config['writable'],allowed_tcp_ports=config['ports'],allowed_devices=['/dev/null','/dev/urandom'])
os.execve(config['python'],[config['python'],'-I','-B',config['entry']],env)
