"""Expose only the existing allowed runtime interpreter directory to tool lookup."""
import os
from pathlib import Path


def runtime_tool_path(workspace, executable):
    return os.pathsep.join([str(Path(workspace) / 'bin'),
                            str(Path(executable).parent), '/usr/bin', '/bin'])
