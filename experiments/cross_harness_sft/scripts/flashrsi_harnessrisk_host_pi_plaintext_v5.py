"""Pi v5 host: v4 relay/sandbox unchanged, with a named v5 supervisor."""
from __future__ import annotations

from pathlib import Path

import flashrsi_harnessrisk_host_pi_interpreter_v4 as v4


_popen = v4.subprocess.Popen


def _popen_v5(argv, *args, **kwargs):
    expected = str(v4.ROOT / "scripts" / "flashrsi_harnessrisk_pi_interpreter_supervisor_v4.py")
    if not isinstance(argv, list) or not argv or argv[-1] != expected:
        raise RuntimeError("unexpected subprocess in Pi plaintext-v5 host")
    updated = [*argv]
    updated[-1] = str(v4.ROOT / "scripts" / "flashrsi_harnessrisk_pi_plaintext_supervisor_v5.py")
    return _popen(updated, *args, **kwargs)


v4.subprocess.Popen = _popen_v5
v4.main()
