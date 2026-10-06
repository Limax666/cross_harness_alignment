#!/usr/bin/env python3
"""Collect native SafeClawArena teacher episodes, with resumable raw evidence."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENT / "vendor" / "SafeClawArena"
DEFAULT_OUTPUT = EXPERIMENT / "outputs" / "safeclawarena_glm53flash_native"
PLATFORMS = {
    "openclaw": ("openclaw-env:2026.3.12", "Dockerfile"),
    "nemoclaw": ("nemoclaw-env:2026.3.11", "Dockerfile.nemoclaw"),
    "seclaw": ("seclaw-env:0.1.0", "Dockerfile.seclaw"),
}
MODEL = "bigmodel/glm-5.3-flash"
ENDPOINT = "https://router.shengsuanyun.com/api/v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def docker_image_id(image: str) -> str:
    response = run(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    image_id = response.stdout.strip()
    return image_id if response.returncode == 0 and image_id.startswith("sha256:") else ""


def trace_stats(report: dict) -> dict:
    raw = report.get("session_transcript_raw", "")
    roles = []
    tool_calls = 0
    tool_results = 0
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        roles.append(message.get("role", ""))
        for part in message.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "toolCall":
                tool_calls += 1
            if message.get("role") == "toolResult":
                tool_results += 1
    return {
        "raw_bytes": len(raw.encode("utf-8")),
        "message_count": len(roles),
        "assistant_messages": roles.count("assistant"),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "trace_complete": bool(raw and "assistant" in roles),
    }


def collect_tasks() -> dict[str, list[Path]]:
    tasks = {}
    for dimension in ("ssi", "pse", "cdf", "ipi"):
        found = sorted((SOURCE / "tasks" / dimension).glob("*.json"))
        if not found:
            raise RuntimeError(f"No {dimension} tasks found")
        tasks[dimension] = found
    total = sum(map(len, tasks.values()))
    if total != 406:
        raise RuntimeError(f"Expected 406 official tasks; found {total}")
    return tasks


def ensure_secret_redaction_patch() -> None:
    judge = SOURCE / "scripts" / "judge.py"
    source = judge.read_text(encoding="utf-8")
    if "auth_profiles_captured" in source and "[REDACTED_PROVIDER_KEY]" in source:
        return
    patch = EXPERIMENT / "scripts" / "safeclawarena_secret_redaction.patch"
    applied = run(["git", "apply", str(patch)], cwd=SOURCE)
    if applied.returncode:
        raise RuntimeError("Cannot apply SafeClawArena credential-redaction patch: "
                           + applied.stderr.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=[*PLATFORMS, "all"], default="all")
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--build-images", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-timeout", type=int, default=2400)
    args = parser.parse_args()
    if args.case_timeout < 900:
        parser.error("--case-timeout must be >= 900 seconds")

    key = os.environ.get("SHENGSUANYUN_API_KEY", "")
    if not key:
        parser.error("SHENGSUANYUN_API_KEY is not set")
    tasks = collect_tasks()
    ensure_secret_redaction_patch()
    selected = list(PLATFORMS) if args.platform == "all" else [args.platform]
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output, 0o700)
    manifest = args.output / "manifest.jsonl"
    source_commit = run(["git", "rev-parse", "HEAD"], cwd=SOURCE, check=True).stdout.strip()

    with (args.output / ".collection.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another SafeClawArena collection is active", file=sys.stderr)
            return 2

        docker = run(["docker", "info", "--format", "{{.ServerVersion}}"])
        if docker.returncode or not docker.stdout.strip():
            print("Docker is unavailable: " + docker.stderr.strip(), file=sys.stderr)
            return 2

        # The upstream runner requires a config path. Keep it in a private,
        # temporary directory and clean it on normal exit or SIGTERM.
        signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(143))
        with tempfile.TemporaryDirectory(prefix="safeclaw-model-", dir=args.output) as temp_dir:
            # Upstream appends /v1 itself. Pass the base ending in /api.
            config = Path(temp_dir) / "model-config.json"
            config.write_text(json.dumps({
                "model": MODEL,
                "api_key": key,
                "api_base_url": ENDPOINT.removesuffix("/v1"),
            }), encoding="utf-8")
            config.chmod(0o600)

            unavailable = []
            failed = 0
            for platform in selected:
                image, dockerfile = PLATFORMS[platform]
                if platform == "seclaw" and not (SOURCE / "seclaw" / "package.json").exists():
                    reason = "Upstream Dockerfile.seclaw requires unshipped seclaw/ source"
                    append_jsonl(manifest, {"time": now(), "platform": platform,
                                            "status": "platform_unavailable", "reason": reason})
                    print(f"{platform}: {reason}", file=sys.stderr)
                    unavailable.append(platform)
                    continue
                image_id = docker_image_id(image)
                if not image_id:
                    if not args.build_images:
                        print(f"{platform}: missing Docker image {image}; use --build-images", file=sys.stderr)
                        unavailable.append(platform)
                        continue
                    print(f"Building {platform} image {image}", flush=True)
                    build = subprocess.run(["docker", "build", "-t", image, "-f", dockerfile, "."], cwd=SOURCE)
                    if build.returncode:
                        append_jsonl(manifest, {"time": now(), "platform": platform,
                                                "status": "image_build_failed", "exit_code": build.returncode})
                        unavailable.append(platform)
                        continue
                    image_id = docker_image_id(image)
                    if not image_id:
                        unavailable.append(platform)
                        continue

                platform_dir = args.output / platform
                platform_dir.mkdir(mode=0o700, exist_ok=True)
                smoke = [paths[0] for paths in tasks.values()]
                all_tasks = smoke if args.phase == "smoke" else smoke + [
                    task for paths in tasks.values() for task in paths[1:]
                ]
                smoke_tool_calls = 0
                for position, task in enumerate(all_tasks):
                    task_id = task.stem
                    result_path = platform_dir / f"{task_id}.json"
                    task_sha = hashlib.sha256(task.read_bytes()).hexdigest()
                    if result_path.exists():
                        try:
                            existing = json.loads(result_path.read_text(encoding="utf-8"))
                            provenance = existing.get("_collection") or {}
                            if (trace_stats(existing)["trace_complete"]
                                    and provenance.get("source_commit") == source_commit
                                    and provenance.get("task_sha256") == task_sha
                                    and provenance.get("image_id") == image_id
                                    and provenance.get("teacher_model") == MODEL):
                                if position < 4:
                                    smoke_tool_calls += trace_stats(existing)["tool_calls"]
                                print(f"{platform}/{task_id}: already captured", flush=True)
                                if position == 3 and smoke_tool_calls == 0:
                                    print(f"{platform}: no native tool calls in four smoke cases", file=sys.stderr)
                                    failed += 1
                                    break
                                continue
                        except (OSError, ValueError):
                            pass
                        archive = platform_dir / "previous_attempts"
                        archive.mkdir(mode=0o700, exist_ok=True)
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        shutil.move(str(result_path), str(archive / f"{task_id}-{stamp}.json"))
                    command = [sys.executable, "scripts/judge.py", str(task),
                               "--platform", platform, "--model-config", str(config),
                               "--output", str(platform_dir)]
                    print(f"{platform}/{task_id}: collecting", flush=True)
                    status = "runner_error"
                    details = {}
                    try:
                        completed = run(command, cwd=SOURCE, timeout=args.case_timeout)
                        log = (completed.stdout + "\n" + completed.stderr).replace(key, "[REDACTED_PROVIDER_KEY]")
                        (platform_dir / f"{task_id}.log").write_text(log, encoding="utf-8")
                        details["exit_code"] = completed.returncode
                        if result_path.exists():
                            report = json.loads(result_path.read_text(encoding="utf-8"))
                            details.update(trace_stats(report))
                            route = report.get("model_config_override") or {}
                            details["route_verified_in_report"] = (
                                route.get("model") == MODEL
                                and route.get("api_base_url") == ENDPOINT.removesuffix("/v1")
                            )
                            status = "captured" if (
                                completed.returncode == 0
                                and details["trace_complete"]
                                and details["route_verified_in_report"]
                            ) else "incomplete"
                            report["_collection"] = {
                                "source_commit": source_commit, "task_sha256": task_sha,
                                "image_id": image_id, "platform": platform,
                                "teacher_model": MODEL, "teacher_endpoint": ENDPOINT,
                            }
                            result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
                        else:
                            status = "missing_report"
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                        details["timeout_seconds"] = args.case_timeout
                    except (OSError, ValueError) as exc:
                        details["error"] = str(exc)
                    record = {"time": now(), "platform": platform, "task_id": task_id,
                              "dimension": task.parent.name, "task_sha256": task_sha,
                              "source_commit": source_commit, "image_id": image_id,
                              "teacher_model": MODEL,
                              "teacher_endpoint": ENDPOINT, "status": status, **details}
                    append_jsonl(manifest, record)
                    print(f"{platform}/{task_id}: {status}; tools={details.get('tool_calls', 0)}", flush=True)
                    if position < 4:
                        smoke_tool_calls += details.get("tool_calls", 0)
                    if status != "captured":
                        failed += 1
                    if position < 4 and status != "captured":
                        print("Smoke task failed; stopping this platform before bulk collection", file=sys.stderr)
                        break
                    if position == 3 and smoke_tool_calls == 0:
                        print(f"{platform}: no native tool calls in four smoke cases", file=sys.stderr)
                        failed += 1
                        break
    if unavailable or failed:
        print(f"Collection incomplete: unavailable_platforms={unavailable}, failed_cases={failed}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
