"""HTTP host for stateful official-benchmark drivers.

Drivers own task loading, environment mutation and scoring. The host never
substitutes a heuristic verifier.
"""
from __future__ import annotations

import argparse
import importlib
import uuid
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Driver(Protocol):
    version: str
    def cases(self) -> list[dict[str, Any]]: ...
    def start(self, task_id: str, seed: int) -> dict[str, Any]: ...
    def tool(self, episode: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def finish(self, episode: Any, final_answer: str) -> dict[str, Any]: ...


class Start(BaseModel): task_id: str; seed: int = 0
class Call(BaseModel): name: str; arguments: dict[str, Any]
class Finish(BaseModel): final_answer: str


def create_app(driver: Driver) -> FastAPI:
    app = FastAPI(title="Cross-harness official benchmark worker")
    episodes: dict[str, Any] = {}

    @app.get("/health")
    def health() -> dict[str, Any]: return {"ok": True, "version": driver.version, "driver": type(driver).__name__}

    @app.get("/cases")
    def cases() -> dict[str, Any]: return {"cases": driver.cases()}

    @app.post("/episodes/start")
    def start(request: Start) -> dict[str, Any]:
        value = driver.start(request.task_id, request.seed); episode_id = uuid.uuid4().hex
        episodes[episode_id] = value["episode"]
        return {"episode_id": episode_id, "tools": value["tools"]}

    @app.post("/episodes/{episode_id}/tool")
    def tool(episode_id: str, request: Call) -> dict[str, Any]:
        if episode_id not in episodes: raise HTTPException(404, "unknown episode")
        return driver.tool(episodes[episode_id], request.name, request.arguments)

    @app.post("/episodes/{episode_id}/finish")
    def finish(episode_id: str, request: Finish) -> dict[str, Any]:
        if episode_id not in episodes: raise HTTPException(404, "unknown episode")
        episode = episodes.pop(episode_id)
        return driver.finish(episode, request.final_answer)
    return app


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--driver", required=True, help="module:factory")
    p.add_argument("--config", required=True); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, required=True)
    args = p.parse_args(); module_name, factory_name = args.driver.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name); driver = factory(args.config)
    import uvicorn
    uvicorn.run(create_app(driver), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__": main()
