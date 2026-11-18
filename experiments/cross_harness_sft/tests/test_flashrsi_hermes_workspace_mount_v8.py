from pathlib import Path


def test_hermes_workspace_supervisor_uses_private_mount_path_without_expanding_sandbox_roots():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/flashrsi_harnessrisk_supervisor_hermes_workspace_v8.py"
    ).read_text()

    assert "'/workspace'" in source
    assert "'/scratch'" in source
    assert "'/workspace'" in source.split("'writable':", 1)[1]
    assert "unshare" in source and "--mount" in source
    assert "'working_dir':'/workspace'" in source
    assert "str(work)" not in source.split("working_dir", 1)[1].split("child=", 1)[0]


def test_hermes_workspace_host_uses_only_the_dedicated_supervisor():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/flashrsi_harnessrisk_host_hermes_workspace_v8.py"
    ).read_text()

    assert "private-v1-hermes-contextfit-v8-workspace-mount" in source
    assert "flashrsi_harnessrisk_supervisor_hermes_workspace_v8.py" in source
