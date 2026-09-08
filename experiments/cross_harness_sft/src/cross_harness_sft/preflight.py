from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from .utils import load_yaml, resolve_path


REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for collection and QLoRA training.")
    parser.add_argument("--config", required=True, type=Path, help="Collection YAML configuration.")
    parser.add_argument("--mode", choices=("collect", "train", "all"), default="all")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def _check(name: str, ok: bool, detail: str, required: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def collection_checks(config: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    harnesses = config.get("harnesses") or []
    checks.append(_check("native_harness_config", bool(harnesses), f"count={len(harnesses)}"))
    allowed = {"codex_cli", "claude_code", "openclaw", "hermes"}
    for harness in harnesses:
        kind = str(harness.get("type") or "")
        binary = str(harness.get("binary") or "")
        model = str(harness.get("model") or "")
        executable = shutil.which(binary) if binary else None
        checks.append(_check(f"native_type:{harness.get('id')}", kind in allowed, kind))
        checks.append(_check(f"native_binary:{harness.get('id')}", bool(executable), executable or binary or "missing"))
        checks.append(_check(f"native_model:{harness.get('id')}", bool(model), model or "missing"))
        if executable:
            try:
                result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=30, check=False)
                checks.append(_check(f"native_version:{harness.get('id')}", result.returncode == 0,
                                     (result.stdout or result.stderr).strip()[:500]))
            except Exception as exc:
                checks.append(_check(f"native_version:{harness.get('id')}", False, f"{type(exc).__name__}: {exc}"))
    for source in config.get("benchmark_sources") or []:
        kind = str(source.get("type") or "unknown")
        if not kind.endswith("_http"):
            continue
        url = str(source.get("base_url") or "").rstrip("/")
        try:
            response = httpx.get(url + "/health", timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            version = str(payload.get("version") or "") if isinstance(payload, dict) else ""
            checks.append(_check(kind + "_health", bool(version and version != "unknown"), "version=" + (version or "missing")))
        except Exception as exc:  # fail closed; concise error is written to the report
            checks.append(_check(kind + "_health", False, "%s: %s" % (type(exc).__name__, exc)))
    checks.append(_check("no_legacy_profiles", not config.get("harness_profiles"),
                         "formal configs must use harnesses, never harness_profiles"))
    safety = dict(config.get("safety") or {})
    for key_name in ("static_prompt_path", "skillbank_path"):
        path = resolve_path(str(safety.get(key_name) or ""), REPO_ROOT)
        checks.append(_check(key_name, path.is_file(), str(path)))
    return checks


def training_checks(dataset_dir: Path | None, min_free_gb: float) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(_check("nvidia_smi", shutil.which("nvidia-smi") is not None, shutil.which("nvidia-smi") or "missing"))
    for package in ("transformers", "accelerate", "peft", "bitsandbytes"):
        try:
            __import__(package); checks.append(_check("python_package:" + package, True, "installed"))
        except Exception as exc:
            checks.append(_check("python_package:" + package, False, f"{type(exc).__name__}: {exc}"))
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        cuda_version = str(torch.version.cuda or "none")
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if cuda else []
        major_minor = tuple(int(part) for part in cuda_version.split(".")[:2]) if cuda_version != "none" else (0, 0)
        checks.append(_check("torch_cuda", cuda, "torch=%s cuda=%s devices=%s" % (torch.__version__, cuda_version, devices)))
        blackwell = any("5090" in name or "Blackwell" in name for name in devices)
        checks.append(_check("blackwell_cuda_12_8", (not blackwell) or major_minor >= (12, 8), "cuda=" + cuda_version))
    except Exception as exc:
        checks.append(_check("torch_cuda", False, "%s: %s" % (type(exc).__name__, exc)))
    target = dataset_dir or REPO_ROOT
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / (1024 ** 3)
    checks.append(_check("free_disk", free_gb >= min_free_gb, "%.2f GiB free; require %.2f GiB" % (free_gb, min_free_gb)))
    if dataset_dir:
        info = dataset_dir / "dataset_info.json"
        train = dataset_dir / "train.jsonl"
        validation = dataset_dir / "validation.jsonl"
        checks.extend([
            _check("dataset_info", info.is_file(), str(info)),
            _check("train_dataset", train.is_file() and train.stat().st_size > 0, str(train)),
            _check("validation_dataset", validation.is_file() and validation.stat().st_size > 0, str(validation)),
        ])
    return checks


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config, expand_env=args.mode in {"collect", "all"})
    checks: list[dict[str, Any]] = []
    if args.mode in {"collect", "all"}:
        checks.extend(collection_checks(config, args.timeout))
    if args.mode in {"train", "all"}:
        checks.extend(training_checks(args.dataset_dir, args.min_free_gb))
    failures = [item for item in checks if item["required"] and not item["ok"]]
    report = {"status": "pass" if not failures else "fail", "checks": checks, "failure_count": len(failures)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
