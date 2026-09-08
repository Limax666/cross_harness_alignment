from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .utils import sha256_json


@dataclass
class HarnessRun:
    final_answer: str
    reasoning: str
    native_events: list[dict[str, Any]]
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    command: list[str]
    usage: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    session_id: str = ""


def _events(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    if not rows:
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                rows.append(value)
        except json.JSONDecodeError:
            pass
    return rows


def _strings(value: Any, keys: set[str]) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in keys and isinstance(child, str):
                yield child
            yield from _strings(child, keys)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child, keys)


def _extract(rows: list[dict[str, Any]], fallback: str = "") -> tuple[str, str, dict[str, Any], str]:
    finals = list(_strings(rows, {"final", "final_answer", "result", "output", "text", "content"}))
    thoughts = list(_strings(rows, {"reasoning", "thinking", "analysis", "summary"}))
    usage: dict[str, Any] = {}
    session_id = ""
    for row in rows:
        if isinstance(row.get("usage"), dict):
            usage.update(row["usage"])
        session_id = str(row.get("session_id") or row.get("sessionId") or row.get("thread_id") or session_id)
    return (finals[-1].strip() if finals else fallback.strip(), "\n".join(dict.fromkeys(thoughts)), usage, session_id)


def _mcp(mcp: dict[str, Any]) -> tuple[str, list[str], dict[str, str]]:
    return str(mcp["command"]), [str(x) for x in mcp.get("args", [])], {str(k): str(v) for k, v in (mcp.get("env") or {}).items()}


class NativeHarness(ABC):
    """Real installed harness. A direct model-API or mock fallback is forbidden."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.harness_id = str(config["id"])
        self.binary = str(config["binary"])
        self.model = str(config["model"])
        self.timeout = int(config.get("timeout_seconds", 900))
        self.runtime_env = {str(k): str(v) for k, v in (config.get("env") or {}).items()}
        if not self.model:
            raise ValueError(f"{self.harness_id}: model is required")

    def executable(self) -> str:
        value = shutil.which(self.binary)
        if not value:
            raise RuntimeError(f"real harness binary is unavailable: {self.binary}")
        return value

    def version(self) -> str:
        process = subprocess.run([self.executable(), "--version"], capture_output=True, text=True,
                                 timeout=30, check=False, encoding="utf-8", errors="replace")
        lines = (process.stdout or process.stderr).strip().splitlines()
        if process.returncode or not lines:
            raise RuntimeError(f"cannot identify {self.harness_id} version")
        return lines[0]

    def manifest(self) -> dict[str, Any]:
        version = self.version()
        return {"harness_id": self.harness_id, "harness_family": self.__class__.__name__,
                "harness_version": version, "harness_binary": self.executable(),
                "harness_sha256": sha256_json({"config": self.config, "version": version}),
                "native_harness": True, "controlled_teacher": False, "teacher_id": self.model}

    def invoke(self, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        started = time.monotonic()
        stream_logs = bool(self.config.get("stream_logs", True))
        label = f"agent model={self.model} harness={self.harness_id}"
        if stream_logs:
            print(f"[{label}] started", flush=True)
        process = subprocess.Popen(command, cwd=cwd, env={**os.environ, **env}, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1)
        stdout_rows: list[str] = []
        stderr_rows: list[str] = []

        def pump(pipe: Any, rows: list[str], target: Any, channel: str) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    rows.append(line)
                    if stream_logs:
                        print(f"[{label}][{channel}] {line}", end="", file=target, flush=True)
            finally:
                pipe.close()

        stdout_thread = threading.Thread(target=pump, args=(process.stdout, stdout_rows, sys.stdout, "stdout"), daemon=True)
        stderr_thread = threading.Thread(target=pump, args=(process.stderr, stderr_rows, sys.stderr, "stderr"), daemon=True)
        stdout_thread.start(); stderr_thread.start()
        try:
            return_code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            stdout_thread.join(); stderr_thread.join()
            raise subprocess.TimeoutExpired(command, self.timeout, output="".join(stdout_rows),
                                            stderr="".join(stderr_rows))
        stdout_thread.join(); stderr_thread.join()
        result = subprocess.CompletedProcess(command, return_code, "".join(stdout_rows), "".join(stderr_rows))
        result.elapsed = time.monotonic() - started  # type: ignore[attr-defined]
        if stream_logs:
            print(f"[{label}] finished exit_code={result.returncode} duration_seconds={result.elapsed:.2f}", flush=True)  # type: ignore[attr-defined]
        return result

    @abstractmethod
    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun: ...


class CodexHarness(NativeHarness):
    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun:
        server, args, server_env = _mcp(mcp)
        home = workdir / "codex-home"; home.mkdir(parents=True, exist_ok=True)
        auth_source_value = str(self.config.get("auth_source") or "")
        if auth_source_value:
            auth_source = Path(auth_source_value).expanduser().resolve()
            if not auth_source.is_file():
                raise FileNotFoundError(f"{self.harness_id}: Codex auth source not found: {auth_source}")
            auth_target = home / "auth.json"
            shutil.copyfile(auth_source, auth_target)
            auth_target.chmod(0o600)
            # ChatGPT-authenticated Codex resolves the account model catalog from
            # this cache. Without it, refresh can time out before MCP startup.
            for state_name in ("models_cache.json", "installation_id"):
                state_source = auth_source.parent / state_name
                if state_source.is_file():
                    shutil.copyfile(state_source, home / state_name)
        env_toml = "\n".join(f'{k} = {json.dumps(v)}' for k, v in server_env.items())
        mcp_approval_mode = str(self.config.get("mcp_approval_mode") or "approve")
        if mcp_approval_mode not in {"auto", "prompt", "writes", "approve"}:
            raise ValueError(f"{self.harness_id}: invalid mcp_approval_mode: {mcp_approval_mode}")
        mcp_startup_timeout = int(self.config.get("mcp_startup_timeout_seconds", 30))
        mcp_tool_timeout = int(self.config.get("mcp_tool_timeout_seconds", 120))
        provider = str(self.config.get("provider") or "openai")
        provider_toml = ""
        if provider != "openai":
            base_url = str(self.config.get("base_url") or "")
            env_key = str(self.config.get("env_key") or "")
            if not base_url or not env_key:
                raise ValueError(f"{self.harness_id}: custom provider requires base_url and env_key")
            wire_api = str(self.config.get("wire_api") or "responses")
            provider_toml = (
                f'model_provider = {json.dumps(provider)}\n'
                f'[model_providers.{provider}]\n'
                f'name = {json.dumps(str(self.config.get("provider_name") or provider))}\n'
                f'base_url = {json.dumps(base_url)}\n'
                f'env_key = {json.dumps(env_key)}\n'
                f'wire_api = {json.dumps(wire_api)}\n'
            )
        (home / "config.toml").write_text(
            f'model = {json.dumps(self.model)}\n'
            f'model_reasoning_effort = {json.dumps(str(self.config.get("reasoning_effort") or "low"))}\n'
            f'approval_policy = "never"\nsandbox_mode = "workspace-write"\nweb_search = "disabled"\n'
            f'[features]\napps = false\nplugins = false\nremote_plugin = false\n'
            f'browser_use = false\ncomputer_use = false\nmulti_agent = false\nshell_tool = false\n'
            f'{provider_toml}'
            f'[mcp_servers.benchmark]\ncommand = {json.dumps(server)}\nargs = {json.dumps(args)}\n'
            f'default_tools_approval_mode = {json.dumps(mcp_approval_mode)}\n'
            f'required = true\nstartup_timeout_sec = {mcp_startup_timeout}\n'
            f'tool_timeout_sec = {mcp_tool_timeout}\n'
            + (f'[mcp_servers.benchmark.env]\n{env_toml}\n' if env_toml else ""), encoding="utf-8")
        cmd = [self.executable(), "exec", "--json", "--ephemeral", "-C", str(workdir), prompt]
        result = self.invoke(cmd, workdir, {**self.runtime_env, "CODEX_HOME": str(home)})
        rows = _events(result.stdout); final, reasoning, usage, sid = _extract(rows)
        return HarnessRun(final, reasoning, rows, result.stdout, result.stderr, result.returncode,
                          result.elapsed, cmd, usage, provider, sid)  # type: ignore[attr-defined]


class ClaudeCodeHarness(NativeHarness):
    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun:
        server, args, server_env = _mcp(mcp)
        cfg = workdir / "claude-mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"benchmark": {"command": server, "args": args, "env": server_env}}}), encoding="utf-8")
        cmd = [self.executable(), "-p", prompt, "--model", self.model, "--output-format", "stream-json", "--verbose",
               "--max-turns", str(self.config.get("max_turns", 20)), "--mcp-config", str(cfg), "--strict-mcp-config",
               "--allowedTools", "mcp__benchmark__*", "--permission-mode", "dontAsk"]
        result = self.invoke(cmd, workdir, self.runtime_env)
        rows = _events(result.stdout); final, reasoning, usage, sid = _extract(rows)
        return HarnessRun(final, reasoning, rows, result.stdout, result.stderr, result.returncode,
                          result.elapsed, cmd, usage, "anthropic", sid)  # type: ignore[attr-defined]


class OpenClawHarness(NativeHarness):
    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun:
        server, args, server_env = _mcp(mcp)
        cfg = workdir / "openclaw.json"
        cfg.write_text(json.dumps({"mcpServers": {"benchmark": {"command": server, "args": args, "env": server_env}},
                                   "tools": {"allow": ["mcp:benchmark"]}}), encoding="utf-8")
        prompt_file = workdir / "prompt.txt"; prompt_file.write_text(prompt, encoding="utf-8")
        cmd = [self.executable(), "agent", "exec", "--json", "--isolated", "--auth-env-only", "--message-file",
               str(prompt_file), "--cwd", str(workdir), "--config", str(cfg), "--model", self.model]
        result = self.invoke(cmd, workdir, {**self.runtime_env, "OPENCLAW_CONFIG_PATH": str(cfg), "OPENCLAW_STATE_DIR": str(workdir / "state")})
        rows = _events(result.stdout); final, reasoning, usage, sid = _extract(rows)
        return HarnessRun(final, reasoning, rows, result.stdout, result.stderr, result.returncode,
                          result.elapsed, cmd, usage, str(self.config.get("provider", "")), sid)  # type: ignore[attr-defined]


class HermesHarness(NativeHarness):
    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun:
        import yaml
        server, args, server_env = _mcp(mcp)
        home = workdir / "hermes-home"; home.mkdir(parents=True, exist_ok=True)
        cfg = home / "config.yaml"
        cfg.write_text(yaml.safe_dump({"model": self.model, "mcp_servers": {"benchmark": {
            "command": server, "args": args, "env": server_env, "timeout": self.timeout, "connect_timeout": 60}}}), encoding="utf-8")
        tail = [str(x) for x in self.config.get("command_tail", ["chat", "--json", "--prompt"])]
        cmd = [self.executable(), *tail, prompt]
        result = self.invoke(cmd, workdir, {**self.runtime_env, "HERMES_HOME": str(home), "HERMES_CONFIG": str(cfg)})
        rows = _events(result.stdout); final, reasoning, usage, sid = _extract(rows, result.stdout if result.returncode == 0 else "")
        return HarnessRun(final, reasoning, rows, result.stdout, result.stderr, result.returncode,
                          result.elapsed, cmd, usage, str(self.config.get("provider", "")), sid)  # type: ignore[attr-defined]


TYPES: dict[str, type[NativeHarness]] = {"codex_cli": CodexHarness, "claude_code": ClaudeCodeHarness,
                                         "openclaw": OpenClawHarness, "hermes": HermesHarness}


def load_harness(config: dict[str, Any]) -> NativeHarness:
    kind = str(config.get("type") or "")
    if kind not in TYPES:
        raise ValueError(f"formal collection requires a native harness type, got {kind!r}")
    return TYPES[kind](config)
