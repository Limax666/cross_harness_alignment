"""OpenClaw v3 case host: retain v2 confinement and pass only adapter case env."""
from flashrsi_harnessrisk_host_openclaw_environment_v2 import main, subprocess
import flashrsi_harnessrisk_host_openclaw_environment_v2 as v2


_popen = subprocess.Popen


def _popen_with_case_environment(*args, **kwargs):
    environment = dict(kwargs["env"])
    environment["HR_OPENCLAW_CASE_ENV_V3"] = "1"
    kwargs["env"] = environment
    return _popen(*args, **kwargs)


v2.subprocess.Popen = _popen_with_case_environment

if __name__ == "__main__":
    main()
