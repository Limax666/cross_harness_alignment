"""Run the pinned FlashRSI GEPA method for the qualified AgentDojo/MiniAgent cell."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path

from flashrsi.agents.llm import LLM
from flashrsi.config import prepare_run_dir
from flashrsi.core import Snapshot
from flashrsi._run_summary import _finish_summary

from flashrsi_gepa_security import (
    AgentDojoSecurityBenchmark,
    SafetyGEPA,
    SuiteAwareMiniAgent,
    ROOT,
    MODEL_REVISION,
    FLASHRSI_REVISION,
)
from flashrsi_restricted_snapshot import make_restricted_snapshot
from flashrsi.agents.miniagent import MiniAgent
from cross_harness_sft.backends.agentdojo import AgentDojoDriver


OUT = ROOT / "outputs/b1_gepa_security_20261112"
MANIFEST = ROOT / "outputs/b0_full_agentdojo_20261112_v2/run_manifest.json"
SPLIT_MANIFEST = ROOT / "../../hf_transfer/extracted/data/sft/teacher_release_v7/family_split_manifest.jsonl"
QUALIFICATION = OUT / "agentdojo_miniagent_qualification_retry.json"
SAFETY_POLICY = (ROOT / "configs/flashrsi_b0_safety_policy.txt").read_text(encoding="utf-8")


def _preflight() -> tuple[dict, dict]:
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split_hash = hashlib.sha256(SPLIT_MANIFEST.resolve().read_bytes()).hexdigest()
    if split_hash != frozen["family_split_manifest_sha256"]:
        raise RuntimeError("family split manifest differs from the frozen B0 hash")
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    rows = qualification["families"]
    if len(rows) != 2 or any(
        row.get("manifest_sha256") != split_hash
        or row.get("same_user_task") is not True
        or row.get("same_tool_schemas") is not True
        or row.get("required_facts_visible_both") is not True
        or row.get("reset_verified_both") is not True
        or row.get("attack_oracle_negative") is not False
        or row.get("attack_oracle_positive") is not True
        or row.get("utility_oracle_positive_both") is not True
        for row in rows
    ):
        raise RuntimeError("AgentDojo/MiniAgent controls did not pass")
    if frozen.get("model_revision") != MODEL_REVISION or frozen.get("flashrsi_revision") != FLASHRSI_REVISION:
        raise RuntimeError("frozen model or FlashRSI revision mismatch")
    return frozen, qualification


async def main() -> None:
    frozen, qualification = _preflight()
    driver = AgentDojoDriver(str(ROOT / "configs/agentdojo_worker.yaml"))
    base_config = {
        # The GEPA Target is a subclass defined in this adapter module, so
        # FlashRSI cannot infer MiniAgent's package source directory by MRO.
        "source_dir": str(Path(importlib.import_module(MiniAgent.__module__).__file__).parent),
        "llm": {
            "model_name": "qwen35-4b-preflight",
            "api_key": "environment-episode-runner-only",
            "base_url": "http://127.0.0.1:18761/v1",
            "max_tokens": 512,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0},
        },
        "max_turns": 12,
        "timeout": 330,
    }
    prototype = MiniAgent(base_config)
    suites = {}
    for suite_name in ("workspace", "travel", "banking", "slack"):
        row = next(
            case for case in frozen["cases"]
            if case["family"].startswith(suite_name + ":") and case["arm"] == "clean"
        )
        _, _, _ = driver._case_map[row["task_id"]]
        native_tools = driver.cases()
        tool_schemas = next(item["tools"] for item in native_tools if item["task_id"] == row["task_id"])
        snapshot = make_restricted_snapshot(prototype, tool_schemas, safety_policy=SAFETY_POLICY)
        expected = frozen["snapshot_by_suite"][suite_name]["id"]
        if snapshot.id != expected:
            raise RuntimeError(f"base Snapshot mismatch for {suite_name}")
        suites[suite_name] = snapshot

    target = SuiteAwareMiniAgent(base_config, suites)
    benchmark = AgentDojoSecurityBenchmark({}, run_dir=OUT)
    benchmark.bind_target = target.bind_benchmark
    target.bind_benchmark(benchmark)
    train, dev, test = benchmark.load().values()
    if not train or not dev or not test:
        raise RuntimeError("frozen family-disjoint splits are incomplete")
    if len({task.metadata["family"] for task in train} & {task.metadata["family"] for task in dev}):
        raise RuntimeError("train/dev family leakage")
    if len({task.metadata["family"] for task in train + dev} & {task.metadata["family"] for task in test}):
        raise RuntimeError("test family leakage")

    token_path = ROOT / "outputs/preflight_20261111/vllm_api_token"
    worker_key = token_path.read_text(encoding="utf-8").strip()
    worker = LLM({
        "type": "llm",
        "llm": {
            "model_name": "qwen35-4b-preflight",
            "api_key": worker_key,
            "base_url": "http://127.0.0.1:18761/v1",
            "max_tokens": 2048,
            "temperature": 0.2,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "timeout": 300,
        "max_retries": 1,
    })
    dev_count = len(dev)
    minibatch = min(3, len(train))
    metric_budget = 2 * dev_count + 2 * minibatch
    algorithm = SafetyGEPA(
        {
            "num_epochs": 1,
            "reflection_minibatch_size": 3,
            "max_metric_calls": metric_budget,
            "skip_perfect_score": True,
            "acceptance_criterion": "strict_improvement",
            "candidate_selection_strategy": "pareto",
            "max_evidence_chars": 8000,
            "seed": 42,
            "use_merge": False,
        },
        target=target,
        worker=worker,
        benchmark=benchmark,
        safety_policy=SAFETY_POLICY,
    )
    run_config = {
        "run": {"output_dir": str(OUT / "runs")},
        "benchmark": {"name": "agentdojo_security", "split": "auto"},
        "agent": {"name": "miniagent", **base_config},
        "algorithm": {"name": "gepa", "worker": {"type": "llm", "llm": {"model_name": "qwen35-4b-preflight", "api_key": worker_key, "base_url": "http://127.0.0.1:18761/v1"}}},
    }
    run_dir = prepare_run_dir(run_config)
    benchmark.run_dir = run_dir
    algorithm._run_dir = run_dir
    for suite_name, snapshot in suites.items():
        snapshot.save(run_dir / "snapshots" / f"initial-{suite_name}.json")
    if worker.component_rules:
        worker.snapshot.save(run_dir / "snapshots" / "worker.json")
    invocation = {
        "status": "running",
        "algorithm": "FlashRSI GEPA (pinned upstream implementation + safety admission gate)",
        "target_and_worker": "Qwen/Qwen3.5-4B",
        "model_revision": MODEL_REVISION,
        "flashrsi_revision": FLASHRSI_REVISION,
        "benchmark": "AgentDojo v1.2.2",
        "harness": "FlashRSI MiniAgent",
        "train_cases": len(train),
        "dev_cases": dev_count,
        "test_cases": len(test),
        "reflection_minibatch_size": minibatch,
        "num_epochs": 1,
        "max_metric_calls": metric_budget,
        "budget_definition": "baseline dev + one worst-case candidate screen and dev pass",
        "split_manifest_sha256": frozen["family_split_manifest_sha256"],
        "qualification_artifact": str(QUALIFICATION),
        "qualification_exposure": [row.get("historical_development_exposure") for row in qualification["families"]],
        "run_dir": str(run_dir),
        "candidate_proposals": "recorded by official GEPA Algorithm worker observer",
        "path_audit_coverage_reported_separately": True,
    }
    (run_dir / "b1_invocation.json").write_text(json.dumps(invocation, indent=2) + "\n", encoding="utf-8")
    try:
        selected = await algorithm.evolve()
        selected.save(run_dir / "snapshots" / "selected.json")
        held_out = await algorithm._evaluate(selected, test, label="test")
        result = {
            **invocation,
            "status": "finished",
            "initial_snapshot_id": target.snapshot.id,
            "selected_snapshot_id": selected.id,
            "metric_calls": algorithm._metric_calls,
            "iterations": getattr(algorithm, "_sequence", None),
            "test_metrics_mean": {
                name: sum(float(item.metrics[name]) for item in held_out) / len(held_out)
                for name in held_out[0].metrics
            } if held_out else {},
            "test_task_count": len(held_out),
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("status", "run_dir", "selected_snapshot_id", "metric_calls", "test_task_count", "test_metrics_mean")}, ensure_ascii=False), flush=True)
    except BaseException as error:
        (run_dir / "error.json").write_text(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        _finish_summary(run_dir)


if __name__ == "__main__":
    asyncio.run(main())
