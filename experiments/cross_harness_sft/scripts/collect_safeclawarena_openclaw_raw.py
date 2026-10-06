#!/usr/bin/env python3
"""Collect OpenClaw teacher sessions from SafeClawArena without running its judge.

Each task gets an isolated HOME via bubblewrap. Unsupported fixtures are recorded,
not silently treated as valid training trajectories.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from safeclaw_teacher_proxy import teacher_proxy

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "vendor" / "SafeClawArena"
INSTALL = ROOT / ".safeclaw-openclaw"
OUTPUT = ROOT / "outputs" / "safeclawarena_openclaw_glm53flash_raw"
MODEL = "bigmodel/glm-5.3-flash"
BASE_URL = "https://router.shengsuanyun.com/api/v1"
SUPPORTED_ENV = {"reset_level", "skills", "workspace_files", "memory_files", "pre_injected_memory",
                 "sim_google_data", "gog_available", "expected_exfil_channel", "auth_profiles_file",
                 "config_overrides", "plugins", "requires_gateway_restart", "_canaries", "git_init"}


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())


def nested_merge(base: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            nested_merge(base[key], value)
        else:
            base[key] = value


def fixture_path(home: Path, raw: str) -> Path:
    if raw.startswith("/root/"):
        relative = raw.removeprefix("/root/")
    elif raw.startswith("/tmp/"):
        relative = "tmp/" + raw.removeprefix("/tmp/")
    elif raw.startswith("/"):
        raise ValueError(f"unsupported absolute fixture path: {raw}")
    else:
        relative = ".openclaw/workspace/" + raw
    target = (home / relative).resolve()
    if not target.is_relative_to(home.resolve()):
        raise ValueError(f"fixture escapes isolated home: {raw}")
    return target


def write_fixture(home: Path, path: str, content: str) -> None:
    target = fixture_path(home, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def provision(task: dict, home: Path, proxy_url: str, proxy_token: str) -> None:
    env = task["environment"]
    workspace = home / ".openclaw" / "workspace"
    shutil.copytree(BENCH / "configs" / "platforms" / "workspace", workspace, dirs_exist_ok=True)
    config = json.loads((BENCH / "configs" / "platforms" / "openclaw.json").read_text())
    config["gateway"]["bind"] = "loopback"
    config["agents"]["defaults"]["workspace"] = "/root/.openclaw/workspace"
    config["agents"]["defaults"]["model"]["primary"] = f"openai/{MODEL}"
    config["agents"]["defaults"]["models"] = {f"openai/{MODEL}": {}}
    config["models"] = {"mode": "merge", "providers": {"openai": {
        "baseUrl": proxy_url, "api": "openai-completions", "apiKey": proxy_token,
        "models": [{"id": MODEL, "name": MODEL, "contextWindow": 200000,
                    "maxTokens": 16384, "input": ["text"], "reasoning": False,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
    }}}
    nested_merge(config, env.get("config_overrides", {}))
    # Dataset-supplied config is an attack surface; it cannot redirect the
    # teacher model or replace the collection credential.
    config["agents"]["defaults"]["model"]["primary"] = f"openai/{MODEL}"
    config["models"]["providers"]["openai"]["baseUrl"] = proxy_url
    config["models"]["providers"]["openai"]["apiKey"] = proxy_token
    config_path = home / ".openclaw" / "openclaw.json"
    write_json(config_path, config)
    config_path.chmod(0o600)
    profiles = {"version": 1, "profiles": {"openai:default": {
        "type": "api_key", "provider": "openai", "key": proxy_token}}}
    canaries = env.get("auth_profiles_file", {}).get("content")
    if canaries:
        profiles["profiles"].update(json.loads(canaries))
    auth_path = home / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
    write_json(auth_path, profiles)
    auth_path.chmod(0o600)

    for skill in env.get("skills", []):
        if skill.get("location", "workspace") != "workspace":
            raise ValueError("managed skill fixture requires the official container layout")
        name = skill["name"]
        if "/" in name or name in {".", ".."}:
            raise ValueError(f"unsafe skill name: {name}")
        skill_root = workspace / "skills" / name
        content = skill.get("content", "")
        if skill.get("content_file"):
            content = (BENCH / skill["content_file"]).read_text(encoding="utf-8")
        write_fixture(home, f"skills/{name}/SKILL.md", content)
        for rel, body in skill.get("script_files", {}).items():
            write_fixture(home, f"skills/{name}/{rel}", body)
            (skill_root / rel).chmod(0o700)

    for file in env.get("workspace_files", []):
        content = file.get("content", "")
        if file.get("content_file"):
            content = (BENCH / file["content_file"]).read_text(encoding="utf-8")
        write_fixture(home, file["path"], content)
    for file in env.get("memory_files", []):
        write_fixture(home, file["path"], file.get("content", ""))
    memory = env.get("pre_injected_memory", {})
    if memory:
        content = memory.get("content", "")
        if memory.get("needle_position"):
            block = "## Session Notes\\n- Worked on various tasks\\n- No issues found\\n"
            needle = memory.get("needle_content", "")
            position = memory["needle_position"]
            content = ("# Memory\n\n" + needle + "\n\n" + block * 200 if position == "beginning"
                       else "# Memory\n\n" + block * 200 + "\n\n" + needle if position == "end"
                       else "# Memory\n\n" + block * 100 + "\n\n" + needle + "\n\n" + block * 100)
        write_fixture(home, memory.get("target_file", "MEMORY.md"), content)
    for plugin in env.get("plugins", []):
        plugin_id = plugin["plugin_id"]
        if "/" in plugin_id or plugin_id in {".", ".."}:
            raise ValueError(f"unsafe plugin ID: {plugin_id}")
        for relative, content in plugin.get("files", {}).items():
            write_fixture(home, f".openclaw/extensions/{plugin_id}/{relative}", content)
    for file in env.get("sim_google_data", []):
        write_fixture(home, "/tmp/sim_google_data/" + file["path"], file.get("content", ""))
    if env.get("git_init"):
        git_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
                   "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
        commit_base = ["git", "-c", "core.hooksPath=/dev/null", "-c", "user.name=Developer",
                       "-c", "user.email=dev@example.com", "commit", "-q", "-m"]
        def git(*parts: str) -> None:
            subprocess.run(["git", *parts], cwd=workspace, env=git_env, capture_output=True,
                           text=True, check=True, timeout=30)
        def commit(label: str) -> None:
            subprocess.run([*commit_base, label], cwd=workspace, env=git_env,
                           capture_output=True, text=True, check=True, timeout=30)
        git("init", "-q")
        git("add", "-A")
        commit("initial commit")
        for file_name, addition, label in (
            ("README.md", "# v0.1.0\n", "Add README and project docs"),
            ("CHANGELOG.md", "# changelog\n", "Add changelog and update dependencies"),
            (".env", "\n", "Update configuration for staging environment"),
        ):
            with (workspace / file_name).open("a", encoding="utf-8") as out:
                out.write(addition)
            git("add", "-A")
            commit(label)
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    sim = local_bin / "sim-google"
    shutil.copy2(BENCH / "tools" / "sim-google" / "sim-google", sim)
    sim.chmod(0o700)


def sandbox_command(home: Path, install: Path, args: list[str]) -> list[str]:
    return ["bwrap", "--die-with-parent", "--unshare-pid", "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--ro-bind", "/etc", "/etc",
            "--dir", "/run", "--dir", "/run/systemd", "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
            "--proc", "/proc", "--dev-bind", "/dev", "/dev", "--dir", "/opt",
            "--ro-bind", str(install), "/opt/openclaw", "--bind", str(home), "/root",
            "--dir", "/tmp", "--bind", str(home / "tmp"), "/tmp", "--chdir", "/root/.openclaw/workspace",
            "--clearenv",
            "--setenv", "HOME", "/root", "--setenv", "OPENCLAW_STATE_DIR", "/root/.openclaw",
            "--setenv", "OPENCLAW_CONFIG_PATH", "/root/.openclaw/openclaw.json",
            "--setenv", "SIM_GOOGLE_DATA_DIR", "/tmp/sim_google_data",
            "--setenv", "PATH", "/root/.local/bin:/usr/bin:/bin", "--", *args]


def redact(value: str, secret: str) -> str:
    return value.replace(secret, "[REDACTED_PROVIDER_KEY]")


def run_task(task_file: Path, output: Path, install: Path, proxy_url: str,
             proxy_token: str, timeout: int) -> dict:
    raw = task_file.read_bytes()
    task = json.loads(raw)
    task_id = task["metadata"]["task_id"]
    result_path = output / f"{task_id}.json"
    task_sha = hashlib.sha256(raw).hexdigest()
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if (old.get("task_sha256") == task_sha and old.get("status") == "captured"
                and old.get("assistant_text_count", 0) > 0
                and old.get("tool_calls") == old.get("tool_results")
                and old.get("collector_version") == 2):
            return {"task_id": task_id, "status": "already_captured",
                    "tool_calls": old.get("tool_calls", 0), "tool_results": old.get("tool_results", 0),
                    "message_count": old.get("message_count", 0)}
        archive = output / "previous_attempts"
        archive.mkdir(exist_ok=True)
        result_path.replace(archive / f"{task_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json")
    record = {"time": stamp(), "task_id": task_id, "dimension": task_file.parent.name,
              "task_sha256": task_sha, "harness": "openclaw", "teacher_model": MODEL,
              "teacher_endpoint": BASE_URL, "harness_version": "2026.3.12",
              "source_revision": "a11f5cceaba0676be721021f8d232638fd111305",
              "collector_version": 2, "credential_isolation": "external_loopback_proxy",
              "status": "incomplete", "sessions": []}
    env = task.get("environment", {})
    unsupported = sorted(set(env) - SUPPORTED_ENV)
    if unsupported:
        record.update(status="unsupported_fixture", unsupported_fields=unsupported)
        return record
    with tempfile.TemporaryDirectory(prefix=f"safeclaw-{task_id}-", dir=output) as temp:
        home = Path(temp)
        (home / "tmp").mkdir()
        try:
            provision(task, home, proxy_url, proxy_token)
        except (OSError, ValueError, KeyError) as exc:
            record.update(status="fixture_error", error=str(exc))
            return record
        session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"safeclaw:{task_id}"))
        for session in task.get("sessions", []):
            setup = session.get("pre_session_setup") or {}
            if setup.get("wait_seconds"):
                time.sleep(min(int(setup["wait_seconds"]), 30))
            if setup.get("restart_gateway"):
                # A local CLI invocation is a fresh process. Give it a fresh
                # conversation ID as the upstream gateway runner does.
                session_id = str(uuid.uuid4())
            for index, message in enumerate([{"content": session["user_instruction"]},
                                             *(session.get("follow_up_messages") or [])]):
                if message.get("delay_seconds"):
                    time.sleep(min(int(message["delay_seconds"]), 30))
                instruction = message.get("content") or message.get("message") or message.get("text")
                if not isinstance(instruction, str):
                    record.update(status="unsupported_fixture", unsupported_fields=["follow_up_message_shape"])
                    break
                command = sandbox_command(home, install, [
                    "node", "/opt/openclaw/node_modules/openclaw/openclaw.mjs", "agent", "--local",
                    "--session-id", session_id, "--message", instruction, "--json"])
                print(f"{task_id}/{session['session_id']}/{index}: OpenClaw started", flush=True)
                started = time.monotonic()
                try:
                    process = subprocess.run(command, capture_output=True, text=True,
                                             timeout=min(timeout, int(session.get("timeout_seconds", timeout))))
                    status = "completed" if process.returncode == 0 else "agent_error"
                    stdout, stderr = process.stdout, process.stderr
                    exit_code = process.returncode
                    if process.returncode == 0:
                        try:
                            payload = json.loads(stdout)
                            if (payload.get("meta") or {}).get("stopReason") == "error":
                                status = "agent_error"
                            elif any("LLM request timed out" in str(p.get("text", ""))
                                     for p in payload.get("payloads", [])):
                                status = "agent_error"
                        except json.JSONDecodeError:
                            status = "agent_error"
                except subprocess.TimeoutExpired as exc:
                    status = "timeout"
                    stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                    stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                    exit_code = None
                record["sessions"].append({"session_id": session["session_id"], "message_index": index,
                    "role": session.get("role"), "user_instruction": instruction, "status": status,
                    "duration_seconds": round(time.monotonic()-started, 2), "exit_code": exit_code,
                    "stdout": redact(stdout, proxy_token), "stderr": redact(stderr, proxy_token)})
                print(f"{task_id}/{session['session_id']}/{index}: {status} "
                      f"in {time.monotonic()-started:.1f}s", flush=True)
                if status != "completed":
                    break
            if record["status"] == "unsupported_fixture" or record["sessions"][-1]["status"] != "completed":
                break
        transcript_dir = home / ".openclaw" / "agents" / "main" / "sessions"
        raw_transcript = "".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(transcript_dir.glob("*.jsonl"))) if transcript_dir.exists() else ""
        record["session_transcript_raw"] = redact(raw_transcript, proxy_token)
        messages, calls, results, assistant_texts = 0, 0, 0, 0
        for line in raw_transcript.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "message":
                continue
            messages += 1
            message = event.get("message") or {}
            if message.get("role") == "toolResult":
                results += 1
            if message.get("role") == "assistant" and any(
                part.get("type") == "text" and str(part.get("text", "")).strip()
                for part in (message.get("content") or []) if isinstance(part, dict)
            ):
                assistant_texts += 1
            calls += sum(part.get("type") == "toolCall" for part in (message.get("content") or []) if isinstance(part, dict))
        complete = bool(assistant_texts and calls == results
                        and all(s["status"] == "completed" for s in record["sessions"]))
        record.update(message_count=messages, assistant_text_count=assistant_texts,
                      tool_calls=calls, tool_results=results, trace_complete=complete,
                      status="captured" if complete else "incomplete")
        write_json(result_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--case-id", help="Run one task ID for rollout validation")
    args = parser.parse_args()
    key = os.environ.get("SHENGSUANYUN_API_KEY", "")
    if not key:
        parser.error("SHENGSUANYUN_API_KEY is required")
    install = INSTALL.resolve()
    binary = install / "node_modules" / "openclaw" / "openclaw.mjs"
    if not binary.is_file():
        parser.error(f"OpenClaw 2026.3.12 missing: {binary}")
    package = json.loads((binary.parent / "package.json").read_text(encoding="utf-8"))
    if package.get("version") != "2026.3.12":
        parser.error(f"expected OpenClaw 2026.3.12, found {package.get('version')}")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BENCH, capture_output=True,
                              text=True, check=True).stdout.strip()
    if revision != "a11f5cceaba0676be721021f8d232638fd111305":
        parser.error(f"SafeClawArena revision changed: {revision}")
    for program in ("bwrap", "node"):
        if not shutil.which(program):
            parser.error(f"{program} is required")
    os.umask(0o077)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = output / "manifest.jsonl"
    tasks = [sorted((BENCH / "tasks" / dim).glob("*.json")) for dim in ("ssi", "pse", "cdf", "ipi")]
    if sum(map(len, tasks)) != 406:
        parser.error("expected 406 upstream task files")
    selected = [xs[0] for xs in tasks] if args.phase == "smoke" else [p for xs in tasks for p in xs]
    if args.case_id:
        selected = [p for xs in tasks for p in xs if p.stem == args.case_id]
        if len(selected) != 1:
            parser.error(f"unknown or ambiguous case ID: {args.case_id}")
    with (output / ".collection.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another raw collection is active")
        print(f"{stamp()} OpenClaw raw collection start phase={args.phase} cases={len(selected)} "
              f"model={MODEL} judge=disabled", flush=True)
        counts: dict[str, int] = {}
        with teacher_proxy(key) as (proxy_url, proxy_token):
            for task in selected:
                result = run_task(task, output, install, proxy_url, proxy_token, args.timeout)
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                append_jsonl(manifest, {k: v for k, v in result.items() if k not in {"sessions", "session_transcript_raw"}})
                print(f"{result['task_id']}: {result['status']} tools={result.get('tool_calls', 0)}", flush=True)
                if args.phase == "smoke" and result["status"] not in {"captured", "already_captured"}:
                    return 2
        print(f"{stamp()} OpenClaw raw collection end counts={json.dumps(counts, sort_keys=True)}", flush=True)
    return 0 if set(counts) <= {"captured", "already_captured"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
