from __future__ import annotations

import json
import os
import shutil
import subprocess
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
        session_id = str(row.get("session_id") or row.get("sessionId") or session_id)
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
        result = subprocess.run(command, cwd=cwd, env={**os.environ, **env}, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=self.timeout, check=False)
        result.elapsed = time.monotonic() - started  # type: ignore[attr-defined]
        return result

    @abstractmethod
    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun: ...


class CodexHarness(NativeHarness):
    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun:
        server, args, server_env = _mcp(mcp)
        home = workdir / "codex-home"; home.mkdir(parents=True, exist_ok=True)
        env_toml = "\n".join(f'{k} = {json.dumps(v)}' for k, v in server_env.items())
        (home / "config.toml").write_text(
            f'model = {json.dumps(self.model)}\napproval_policy = "never"\nsandbox_mode = "workspace-write"\n'
            f'[mcp_servers.benchmark]\ncommand = {json.dumps(server)}\nargs = {json.dumps(args)}\n'
            + (f'[mcp_servers.benchmark.env]\n{env_toml}\n' if env_toml else ""), encoding="utf-8")
        cmd = [self.executable(), "exec", "--json", "--ephemeral", "--ignore-user-config", "-C", str(workdir), prompt]
        result = self.invoke(cmd, workdir, {**self.runtime_env, "CODEX_HOME": str(home)})
        rows = _events(result.stdout); final, reasoning, usage, sid = _extract(rows)
        return HarnessRun(final, reasoning, rows, result.stdout, result.stderr, result.returncode,
                          result.elapsed, cmd, usage, "openai", sid)  # type: ignore[attr-defined]


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
