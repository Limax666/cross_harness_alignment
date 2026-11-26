#!/usr/bin/env python3
"""Download the immutable model revisions registered for the HarnessRisk study."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_REPO = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GRPO_REPO = "Limax11/harness_alignment"
GRPO_REVISION = "f8ca0ae2af5f52527572e933f03f5bfa6b8df5cd"
GRPO_PATH = "online_rl/a6000_r1_lora_2g8_300_20261121/step_0200_merged"
GRPO_SUMS_SHA256 = "a23f0d769aa04deeb266d0c19ced30d45bad07a856449e12d6f630727cf9f8e2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_grpo(model_dir: Path) -> None:
    sums = model_dir / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(f"missing checksum manifest: {sums}")
    actual_manifest_hash = sha256(sums)
    if actual_manifest_hash != GRPO_SUMS_SHA256:
        raise ValueError(
            "GRPO checksum manifest identity mismatch: "
            f"expected {GRPO_SUMS_SHA256}, got {actual_manifest_hash}"
        )
    checked = 0
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        expected, filename = line.split(maxsplit=1)
        filename = filename.lstrip(" *")
        candidate = (model_dir / filename).resolve()
        if model_dir.resolve() not in candidate.parents:
            raise ValueError(f"unsafe path in checksum manifest: {filename}")
        if not candidate.is_file():
            raise FileNotFoundError(f"missing model file listed in SHA256SUMS: {filename}")
        actual = sha256(candidate)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}")
        checked += 1
    if checked == 0:
        raise ValueError("empty GRPO SHA256SUMS manifest")
    print(f"Verified {checked} files against the pinned GRPO SHA256SUMS manifest.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("base4b", "grpo200"))
    parser.add_argument("--out", type=Path, required=True, help="destination directory on a mounted volume")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.model == "base4b":
        snapshot_download(
            repo_id=BASE_REPO,
            revision=BASE_REVISION,
            local_dir=str(args.out),
            token=token,
        )
        print(f"Downloaded {BASE_REPO}@{BASE_REVISION} to {args.out}")
        return

    snapshot_download(
        repo_id=GRPO_REPO,
        revision=GRPO_REVISION,
        allow_patterns=[f"{GRPO_PATH}/**"],
        local_dir=str(args.out),
        token=token,
    )
    model_dir = args.out / GRPO_PATH
    verify_grpo(model_dir)
    print(f"Downloaded and verified {GRPO_REPO}@{GRPO_REVISION}/{GRPO_PATH} at {model_dir}")


if __name__ == "__main__":
    main()
