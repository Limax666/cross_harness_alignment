"""The HarnessRisk OpenClaw wrapper must pass the per-case time limit through."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_CASE = ROOT / "HarnessRisk/harness_adapter/scripts/run_case_scripts/run_case.sh"


def _captured_adapter_args(timeout: str | None) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="harnessrisk-timeout-") as directory:
        root = Path(directory)
        fake_bin = root / "bin"
        fake_bin.mkdir()
        state = root / "state"
        state.mkdir()
        (state / "openclaw.json").write_text("{}", encoding="utf-8")
        for name, body in (
            ("node", "#!/bin/sh\nexit 0\n"),
            ("python3", '#!/bin/sh\nprintf "%s\\n" "$@"\n'),
        ):
            executable = fake_bin / name
            executable.write_text(body, encoding="utf-8")
            executable.chmod(0o755)
        command = ["bash", str(RUN_CASE), "--harness", "openclaw", "--state-dir", str(state),
                   "--model", "local-test"]
        if timeout is not None:
            command.extend(("--timeout-seconds", timeout))
        result = subprocess.run(
            command,
            env={**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin", "CUSTOM_API_KEY": "test-only"},
            text=True, capture_output=True, check=False, timeout=10,
        )
        if result.returncode:
            raise AssertionError(result.stderr)
        return result.stdout.splitlines()


def test_explicit_timeout_is_forwarded():
    arguments = _captured_adapter_args("300")
    assert arguments[arguments.index("--timeout-seconds") + 1] == "300"


def test_default_timeout_is_bounded():
    arguments = _captured_adapter_args(None)
    assert arguments[arguments.index("--timeout-seconds") + 1] == "480"
