"""Canonical, host-side backend factory. No model or judge clients live here."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    """Hash names and bytes; never follow source symlinks into host resources."""
    digest = hashlib.sha256()
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink():
            raise ValueError(f"source_symlink_rejected:{entry}")
        if entry.is_file():
            digest.update(entry.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(entry)))
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def invalid(reason: str, *, evidence: dict | None = None, metrics: dict | None = None) -> dict:
    return {"done": True, "valid": False, "invalid_reasons": [reason],
            "utility": None, "c_source": None, "persistence": None,
            "metrics": metrics or {}, "evidence": evidence or {}}


def checked_source(task: dict) -> Path:
    path = Path(task["source_path"]).resolve(strict=True)
    if sha256_file(path) != task["source_sha256"]:
        raise ValueError("source_hash_changed")
    if task.get("source_tree_sha256") and tree_digest(path.parent) != task["source_tree_sha256"]:
        raise ValueError("source_tree_hash_changed")
    return path


@contextmanager
def open_backend(task: dict, workspace: Path, evidence_dir: Path):
    """Each context creates fresh source state; evidence survives close()."""
    if not task.get("qualified", False):
        raise ValueError("task_not_qualified:" + str(task.get("exclusion_reasons", [])))
    if task["source"] == "clawgym":
        from .claw_backend import ClawBackend
        cls = ClawBackend
    elif task["source"] == "skilllifebench":
        from .skilllife_backend import SkillLifeBackend
        cls = SkillLifeBackend
    elif task["source"] == "harnessaudit":
        from .audit_backend import AuditBackend
        cls = AuditBackend
    elif task["source"] == "harnesssafe":
        from .safe_backend import SafeBackend
        cls = SafeBackend
    else:
        raise ValueError("unknown_source")
    backend = cls(task, workspace, evidence_dir)
    try:
        yield backend
    finally:
        backend.close()


def verify_pinned_tree(root: Path, revision: str, consumed: list[str]) -> dict:
    """Compare every consumed file and its file set to the Git commit itself.

    HEAD identity plus newly recorded working-tree hashes is insufficient. This
    check refuses missing, modified, symlinked, or unexpected consumed files.
    """
    import subprocess
    root = Path(root).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError("source_revision_not_pinned")
    raw = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", revision, "--", *consumed])
    expected = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("unsupported_pinned_source_entry:" + name.decode())
        expected[name.decode()] = (mode, oid)
    if not expected:
        raise ValueError("empty_pinned_source_scope")
    actual = set()
    for scope in consumed:
        path = root / scope
        paths = [path] if path.is_file() or path.is_symlink() else path.rglob("*")
        for item in paths:
            relative = item.relative_to(root).as_posix()
            if item.is_symlink():
                raise ValueError("pinned_source_symlink_rejected:" + relative)
            if item.is_file():
                actual.add(relative)
    unexpected = actual - set(expected)
    if unexpected:
        raise ValueError("unexpected_consumed_source_file:" + ",".join(sorted(unexpected)))
    missing = set(expected) - actual
    if missing:
        raise ValueError("missing_consumed_source_file:" + ",".join(sorted(missing)))
    for name, (mode, oid) in expected.items():
        data = (root / name).read_bytes()
        git_oid = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        executable = bool((root / name).stat().st_mode & 0o111)
        if git_oid != oid or executable != (mode == "100755"):
            raise ValueError("pinned_source_bytes_mismatch:" + name)
    return {"verified": True, "revision": revision, "consumed_scopes": consumed,
            "file_count": len(expected), "pinned_blob_manifest_sha256": hashlib.sha256(
                json.dumps(expected, sort_keys=True).encode()).hexdigest()}
