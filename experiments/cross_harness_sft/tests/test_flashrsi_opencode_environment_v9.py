"""Verify that the real runtime interpreter remains discoverable in agent tools."""
from pathlib import Path
import os
import shutil
import subprocess
import sys


def test_runtime_python_is_discoverable_and_executable_without_host_path_inheritance(tmp_path):
    from flashrsi_harnessrisk_opencode_environment_v9 import runtime_tool_path
    path = runtime_tool_path(tmp_path, sys.executable)
    assert path.split(os.pathsep)[0] == str(tmp_path / 'bin')
    assert Path(shutil.which('python', path=path)).resolve() == Path(sys.executable).resolve()
    run = subprocess.run(['python', '-c', 'import sys;print(sys.executable)'], env={'PATH': path},
                         capture_output=True, text=True, check=True)
    assert Path(run.stdout.strip()).resolve() == Path(sys.executable).resolve()
    assert path.split(os.pathsep)[-2:] == ['/usr/bin', '/bin']
