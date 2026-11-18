"""Losslessly expose printable text that a fixture labels as a PDF.

This is intentionally narrow: it never opens a real PDF, never follows a
symlink, and never changes the source file.  The caller records every twin in
the case protocol evidence so this presentation change remains auditable.
"""
from __future__ import annotations

import os
from pathlib import Path

from flashrsi_harnessrisk_privacy import assert_no_oracle


def normalize_mislabelled_plaintext_pdfs(workspace: Path) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for source in sorted(workspace.rglob("*.pdf")):
        if source.is_symlink() or not source.is_file():
            continue
        raw = source.read_bytes()
        if len(raw) > 1_048_576 or raw.startswith(b"%PDF-"):
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any(not (character.isprintable() or character in "\n\r\t") for character in text):
            continue
        assert_no_oracle(text)
        twin = source.with_name(source.name + ".txt")
        if twin.exists():
            raise FileExistsError(f"refusing to overwrite fixture text twin: {twin}")
        temporary = twin.with_name(twin.name + ".tmp")
        try:
            temporary.write_bytes(raw)
            os.replace(temporary, twin)
        finally:
            if temporary.exists():
                temporary.unlink()
        changes.append({"source": str(source.relative_to(workspace)), "text_twin": str(twin.relative_to(workspace)), "bytes": len(raw)})
    return changes
