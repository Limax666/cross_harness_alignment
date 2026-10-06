#!/usr/bin/env python3
"""Compatibility wrapper for the official AgentHarm MCP collector."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("run_agentharm_codex_native.sh")
    os.execv(str(script), [str(script), *sys.argv[1:]])


if __name__ == "__main__":
    main()
