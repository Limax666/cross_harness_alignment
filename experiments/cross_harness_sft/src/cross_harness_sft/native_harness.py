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
from urllib.parse import urlsplit

from .utils import sha256_json


def _tail_of(text: str, limit: int = 600) -> str:
    """Last non-empty lines of a stream, used when stderr is empty by design."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return " / ".join(lines[-3:])[:limit]


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
    terminal_status: str = ""


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


# Loopback hosts that may serve an Anthropic-compatible facade. The forwarder, not the
# harness child process, owns the upstream credential.
LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _claude_result_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if row.get("type") == "result":
            return row
    return {}


def _extract_claude(rows: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any], str, str]:
    """Claude Code stream-json has an authoritative terminal `result` row.

    Using the generic key scan would let a trailing `tool_result` observation become
    the trajectory's "final answer", so the assistant text and thinking blocks are
    read from the native events instead and the terminal row decides completion.
    """
    terminal = _claude_result_row(rows)
    session_id = str(terminal.get("session_id") or "")
    texts: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    for row in rows:
        if row.get("type") != "assistant":
            continue
        if not session_id:
            session_id = str(row.get("session_id") or "")
        message = row.get("message") if isinstance(row.get("message"), dict) else {}
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type") or "")
            if kind == "text" and str(block.get("text") or "").strip():
                texts.append(str(block["text"]).strip())
            elif kind == "thinking" and str(block.get("thinking") or "").strip():
                reasoning.append(str(block["thinking"]).strip())
        usage_row = message.get("usage")
        if isinstance(usage_row, dict):
            usage.update(usage_row)
    final = str(terminal.get("result") or "").strip() or (texts[-1] if texts else "")
    error = str(terminal.get("subtype") or "")
    return final, "\n".join(dict.fromkeys(reasoning)), usage, session_id, error


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

    def invoke(self, command: list[str], cwd: Path, env: dict[str, str],
               remove_env: Iterable[str] = ()) -> subprocess.CompletedProcess[str]:
        started = time.monotonic()
        merged = {**os.environ, **env}
        for name in remove_env:
            merged.pop(str(name), None)
        stream_logs = bool(self.config.get("stream_logs", True))
        label = f"agent model={self.model} harness={self.harness_id}"
        if stream_logs:
            print(f"[{label}] started", flush=True)
        process = subprocess.Popen(command, cwd=cwd, env=merged, stdout=subprocess.PIPE,
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
            f'{provider_toml}'
            f'[features]\napps = false\nplugins = false\nremote_plugin = false\n'
            f'browser_use = false\ncomputer_use = false\nmulti_agent = false\nshell_tool = false\n'
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
    """Real Claude Code CLI. LLM transport may sit behind a local Anthropic-compatible facade.

    The teacher model itself is still whatever `model` names: Claude Code issues genuine
    Anthropic Messages requests with its own system prompt, tools and tool loop. The facade
    exists only because (a) this Claude Code build validates `--model` locally, so the
    non-Anthropic slug travels in `ANTHROPIC_MODEL`, and (b) the upstream route may be
    region-gated on the egress IP, which Claude Code's HTTP client cannot pin.
    """

    def _facade(self) -> tuple[str, str]:
        facade = dict(self.config.get("facade") or {})
        base_url = str(facade.get("base_url") or self.runtime_env.get("ANTHROPIC_BASE_URL") or "").strip()
        if not base_url:
            raise RuntimeError(f"{self.harness_id}: claude_code needs facade.base_url (a local "
                               f"Anthropic-compatible forwarder); refusing an implicit upstream route")
        host = (urlsplit(base_url).hostname or "").lower()
        if bool(facade.get("require_loopback", True)) and host not in LOOPBACK:
            raise RuntimeError(f"{self.harness_id}: facade.base_url must be loopback, got host {host!r}")
        token_env = str(facade.get("local_token_env") or "").strip()
        if not token_env:
            raise RuntimeError(f"{self.harness_id}: facade.local_token_env is required so the upstream "
                               "credential never enters the harness child process")
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise RuntimeError(f"{self.harness_id}: {token_env} is unset; the local facade token is required")
        return base_url, token

    def manifest(self) -> dict[str, Any]:
        base_url, _ = self._facade()
        value = super().manifest()
        # Provenance must state how the teacher was reached: the harness, its version and the
        # real binary are Claude Code's; only the LLM hop goes through a local forwarder.
        value.update({
            "llm_transport": "anthropic_messages_via_local_forwarder",
            "llm_forwarder": base_url,
            "llm_gateway": str(self.config.get("provider") or "anthropic"),
            "harness_isolation": "bare + isolated CLAUDE_CONFIG_DIR + built-in tools disabled",
        })
        return value

    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun:
        base_url, local_token = self._facade()
        server, args, server_env = _mcp(mcp)
        cfg = workdir / "claude-mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"benchmark": {"command": server, "args": args, "env": server_env}}}),
                       encoding="utf-8")
        # Isolated Claude Code state: an empty config dir plus --bare means no user settings,
        # hooks, plugins, CLAUDE.md or auto-memory can leak into the experiment.
        home = workdir / "claude-home"; home.mkdir(parents=True, exist_ok=True)
        cmd = [self.executable(), "-p", prompt, "--output-format", "stream-json", "--verbose", "--bare",
               "--max-turns", str(self.config.get("max_turns", 20)), "--mcp-config", str(cfg), "--strict-mcp-config",
               "--allowedTools", "mcp__benchmark__*", "--tools", "", "--permission-mode", "dontAsk",
               "--no-session-persistence", "--disable-slash-commands", "--no-chrome"]
        # `--model` is only valid for names Claude Code knows natively; any other slug is routed
        # through ANTHROPIC_MODEL by the facade. No --fallback-model: that would swap teacher.
        cli_model = str(self.config.get("cli_model") or "").strip()
        if cli_model:
            cmd += ["--model", cli_model]
        env = {**self.runtime_env,
               "ANTHROPIC_BASE_URL": base_url,
               "ANTHROPIC_API_KEY": local_token,
               "ANTHROPIC_MODEL": self.model,
               "CLAUDE_CODE_SUBAGENT_MODEL": self.model,
               "CLAUDE_CONFIG_DIR": str(home)}
        # A native CLI alias must remain a supported Claude model name locally.
        # The local facade pins the actual upstream Base/SFT model separately.
        for alias_var in ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                          "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"):
            if cli_model:
                env.pop(alias_var, None)
            else:
                env[alias_var] = self.model
        # Ambient upstream credentials must not reach the harness child process.
        scrub = [str(name) for name in (self.config.get("scrub_env") or
                                        ["OPENROUTER_API_KEY", "ANTHROPIC_AUTH_TOKEN"])]
        result = self.invoke(cmd, workdir, env, remove_env=scrub)
        rows = _events(result.stdout)
        final, reasoning, usage, session_id, terminal = _extract_claude(rows)
        api_error = next((str(block.get("text") or "").strip() for row in rows if row.get("type") == "assistant"
                          for block in ((row.get("message") or {}).get("content") or [])
                          if isinstance(block, dict) and block.get("type") == "text"
                          and str(block.get("text") or "").lstrip().startswith("API Error")), "")
        detail = (result.stderr or "").strip() or _tail_of(result.stdout)
        if api_error:
            raise RuntimeError(f"{self.harness_id}: Claude Code surfaced a transport error via {base_url}: "
                               f"{api_error[:400]}")
        if terminal.startswith("error") and terminal != "error_max_turns":
            raise RuntimeError(f"{self.harness_id}: Claude Code produced no usable terminal result row "
                               f"(subtype={terminal}): {detail[:600]}")
        # Reaching Claude Code's configured turn budget is a controlled stop, not
        # a transport failure. Preserve the native trace so the collector can score
        # it as an incomplete benchmark attempt and continue with the next case.
        if terminal == "error_max_turns":
            final = ""
        elif result.returncode == 0 and not final:
            raise RuntimeError(f"{self.harness_id}: Claude Code produced no usable terminal result row "
                               f"(subtype={terminal or 'missing'}): {detail[:600]}")
        return HarnessRun(final, reasoning, rows, result.stdout, result.stderr, result.returncode,
                          result.elapsed, cmd, usage, str(self.config.get("provider") or "anthropic"),
                          session_id, terminal)  # type: ignore[attr-defined]


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
        # Hermes resolves its API route from its isolated config.  Preserve the
        # collector's explicit model while passing an optional provider so each
        # episode can use a non-default OpenAI-compatible backend.
        model_config: Any = self.model
        provider = str(self.config.get("provider") or "").strip()
        if provider:
            model_config = {"default": self.model, "provider": provider}
        config_payload: dict[str, Any] = {"model": model_config, "mcp_servers": {"benchmark": {
            "command": server, "args": args, "env": server_env, "timeout": self.timeout, "connect_timeout": 60}}}
        # A named custom provider belongs in Hermes' isolated config, while
        # its credential remains an inherited environment variable.
        provider_config = dict(self.config.get("provider_config") or {})
        if provider and provider_config:
            config_payload["providers"] = {provider: provider_config}
        cfg.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
        # Current Hermes releases intentionally expose programmatic one-shot
        # output as text, not a stream-JSON protocol.  Passing the prompt in a
        # file avoids putting arbitrary benchmark text on the process command
        # line and preserves it byte-for-byte.
        prompt_file = workdir / "prompt.txt"; prompt_file.write_text(prompt, encoding="utf-8")
        tail = [str(x) for x in self.config.get("command_tail", ["chat", "--query-file", "{prompt_file}", "--oneshot", "--quiet"])]
        tail = [str(prompt_file) if part == "{prompt_file}" else part for part in tail]
        cmd = [self.executable(), *tail]
        result = self.invoke(cmd, workdir, {**self.runtime_env, "HERMES_HOME": str(home), "HERMES_CONFIG": str(cfg)})
        rows = _events(result.stdout)
        # The captured stdout is the real native CLI result. Tool calls and
        # observations are independently recorded by the MCP audit, which is
        # the authoritative source used by collect_native for SFT messages.
        if not rows and result.returncode == 0 and result.stdout.strip():
            rows = [{"type": "hermes_cli_text_result", "text": result.stdout.strip()}]
        final, reasoning, usage, sid = _extract(rows, result.stdout if result.returncode == 0 else "")
        return HarnessRun(final, reasoning, rows, result.stdout, result.stderr, result.returncode,
                          result.elapsed, cmd, usage, str(self.config.get("provider", "")), sid)  # type: ignore[attr-defined]


class NanoBotHarness(NativeHarness):
    """Real NanoBot one-shot agent with an isolated, MCP-only tool surface."""

    def run(self, prompt: str, workdir: Path, mcp: dict[str, Any]) -> HarnessRun:
        server, args, server_env = _mcp(mcp)
        home = workdir / "nanobot-home"; home.mkdir(parents=True, exist_ok=True)
        # NanoBot refuses a config/session root contained by the agent's
        # workspace.  Keep the two as sibling directories under the episode.
        workspace = workdir / "workspace"; workspace.mkdir(parents=True, exist_ok=True)
        provider = str(self.config.get("provider") or "custom").strip()
        key_env = str(self.config.get("api_key_env") or "SHENGSUANYUN_API_KEY").strip()
        base_url = str(self.config.get("base_url") or "").strip()
        if not base_url:
            raise ValueError(f"{self.harness_id}: NanoBot requires base_url")
        max_turns = int(self.config.get("max_tool_iterations") or 8)
        if max_turns < 1:
            raise ValueError(f"{self.harness_id}: max_tool_iterations must be positive")
        # Keep only the benchmark's stdio MCP server.  This is both the actual
        # NanoBot tool contract and an enforcement layer for the prompt policy.
        config = {
            "providers": {provider: {"apiKey": "${" + key_env + "}", "apiBase": base_url}},
            "modelPresets": {"primary": {
                "label": "AgentHarm teacher", "provider": provider, "model": self.model,
                "maxTokens": int(self.config.get("max_tokens") or 8192),
                "contextWindowTokens": int(self.config.get("context_window") or 131072),
                "temperature": float(self.config.get("temperature") or 0.1),
            }},
            "agents": {"defaults": {"modelPreset": "primary", "max_tool_iterations": max_turns}},
            "tools": {
                "exec": {"enable": False}, "file": {"enable": False}, "web": {"enable": False},
                "cli_apps": {"enable": False}, "my": {"enable": False}, "image_generation": {"enabled": False},
                "restrict_to_workspace": True,
                "mcp_servers": {"benchmark": {
                    "type": "stdio", "command": server, "args": args, "env": server_env,
                    "cwd": str(workspace), "tool_timeout": int(self.config.get("mcp_tool_timeout_seconds") or 120),
                    "enabled_tools": ["*"],
                }},
            },
        }
        cfg = home / "nanobot_config.json"
        cfg.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # --message is NanoBot's supported non-interactive agent entrypoint.
        # A per-episode session name preserves its native session record inside
        # the temporary workspace while avoiding cross-case context leakage.
        session_id = f"agentharm-{workdir.name[:32]}"
        cmd = [self.executable(), "agent", "--config", str(cfg), "--workspace", str(workspace),
               "--session", session_id, "--no-markdown", "--message", prompt]
        result = self.invoke(cmd, workspace, self.runtime_env)
        rows = _events(result.stdout)
        if not rows and result.returncode == 0 and result.stdout.strip():
            rows = [{"type": "nanobot_cli_text_result", "text": result.stdout.strip(), "session_id": session_id}]
        final, reasoning, usage, sid = _extract(rows, result.stdout if result.returncode == 0 else "")
        return HarnessRun(final, reasoning, rows, result.stdout, result.stderr, result.returncode,
                          result.elapsed, cmd, usage, provider, sid or session_id)  # type: ignore[attr-defined]


TYPES: dict[str, type[NativeHarness]] = {"codex_cli": CodexHarness, "claude_code": ClaudeCodeHarness,
                                         "openclaw": OpenClawHarness, "hermes": HermesHarness,
                                         "nanobot": NanoBotHarness}


def load_harness(config: dict[str, Any]) -> NativeHarness:
    kind = str(config.get("type") or "")
    if kind not in TYPES:
        raise ValueError(f"formal collection requires a native harness type, got {kind!r}")
    return TYPES[kind](config)
