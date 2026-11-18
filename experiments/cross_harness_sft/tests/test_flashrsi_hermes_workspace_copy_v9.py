import importlib.util
from pathlib import Path


def load_v9():
    source = Path(__file__).resolve().parents[1] / "scripts/flashrsi_harnessrisk_supervisor_hermes_workspace_v9.py"
    spec = importlib.util.spec_from_file_location("hermes_workspace_v9", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workspace_copy_is_visible_at_stable_path_without_mount_namespace(tmp_path):
    module = load_v9()
    source = tmp_path / "case_workspace"
    source.mkdir()
    (source / "invoice.txt").write_text("trusted case artifact")
    visible = tmp_path / "sandbox_root" / "workspace"

    module.prepare_visible_workspace(source, visible)

    assert visible != source
    assert (visible / "invoice.txt").read_text() == "trusted case artifact"
    assert not visible.is_symlink()
    assert "unshare', '--mount" not in module.__file__ and 'mount --bind' not in Path(module.__file__).read_text()
