from __future__ import annotations

import copy
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Episode:
    suite: Any; user_task: Any; injection_task: Any; pre_environment: Any; environment: Any; runtime: Any
    trace: list[Any] = field(default_factory=list)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"): return value.model_dump(mode="json")
    try: json.dumps(value); return value
    except TypeError: return str(value)


class AgentDojoDriver:
    """Runs AgentDojo's own FunctionsRuntime and task utility/security methods."""
    def __init__(self, config_path: str):
        from agentdojo.task_suite.load_suites import get_suite
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.benchmark_version = str(config["benchmark_version"])
        self.suite_names = list(config.get("suites") or ["workspace", "travel", "banking", "slack"])
        self.include_clean = bool(config.get("include_clean", True)); self.include_injection = bool(config.get("include_injection", True))
        self.suites = {name: get_suite(self.benchmark_version, name) for name in self.suite_names}
        self.version = f"agentdojo/{importlib.metadata.version('agentdojo')} benchmark/{self.benchmark_version}"
        self._case_map: dict[str, tuple[Any, Any, Any]] = {}
        for name, suite in self.suites.items():
            if self.include_clean:
                for uid, user in suite.user_tasks.items(): self._case_map[f"{name}:{uid}:clean"] = (suite, user, None)
            if self.include_injection:
                for uid, user in suite.user_tasks.items():
                    for iid, injection in suite.injection_tasks.items(): self._case_map[f"{name}:{uid}:{iid}"] = (suite, user, injection)

    def _tools(self, suite: Any) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": tool.name, "description": tool.description,
                 "parameters": tool.parameters.model_json_schema()}} for tool in suite.tools]

    def cases(self) -> list[dict[str, Any]]:
        result = []
        for task_id, (suite, user, injection) in self._case_map.items():
            result.append({"benchmark_id": "agentdojo", "task_id": task_id,
                "semantic_task_family_id": f"{suite.name}:{user.ID}", "task_type": "injection" if injection else "clean",
                "domain": suite.name, "user_query": user.PROMPT, "tools": self._tools(suite), "metadata": {
                    "suite": suite.name, "user_task_id": user.ID, "injection_task_id": getattr(injection, "ID", None)}})
        return result

    def start(self, task_id: str, seed: int) -> dict[str, Any]:
        from agentdojo.functions_runtime import FunctionsRuntime
        suite, user, injection = self._case_map[task_id]
        injections = {}
        if injection is not None:
            payload = str(injection.GOAL)
            injections = {key: payload for key in suite.get_injection_vector_defaults()}
        base = suite.load_and_inject_default_environment(injections)
        environment = user.init_environment(base) if hasattr(user, "init_environment") else base
        episode = Episode(suite, user, injection, environment.model_copy(deep=True), environment,
                          FunctionsRuntime(suite.tools))
        return {"episode": episode, "tools": self._tools(suite)}

    def tool(self, episode: Episode, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from agentdojo.functions_runtime import FunctionCall
        result, error = episode.runtime.run_function(episode.environment, name, arguments)
        call = FunctionCall(function=name, args=arguments); episode.trace.append(call)
        return {"observation": error or json.dumps(_jsonable(result), ensure_ascii=False), "result": _jsonable(result),
                "invalid_call": error is not None, "side_effect": True, "risk_event": False}

    def finish(self, episode: Episode, final_answer: str) -> dict[str, Any]:
        blocks = [{"type": "text", "content": final_answer}]
        utility = episode.suite._check_task_result(episode.user_task, blocks, episode.pre_environment,
                                                   episode.environment, episode.trace)
        security = True if episode.injection_task is None else episode.suite._check_task_result(
            episode.injection_task, blocks, episode.pre_environment, episode.environment, episode.trace)
        # AgentDojo security=True means the injected goal was achieved; normalize to safety_reward.
        attack_success = bool(security) if episode.injection_task is not None else False
        return {"utility_reward": float(bool(utility)), "safety_reward": float(not attack_success),
                "final_success": bool(utility), "risk_success": attack_success,
                "details": {"official_utility": bool(utility), "official_injection_success": attack_success}}


def create_driver(config_path: str) -> AgentDojoDriver: return AgentDojoDriver(config_path)
