#!/usr/bin/env python3
"""Only authorized VeRL entrypoint for native full-parameter online GRPO pilot.

Never run vendor main_ppo directly: it bypasses GuardedNativePilotTrainer.
This launcher refuses to start Ray unless the complete compatible backend is
already installed. A successful import alone is not evidence of an RL update.
"""
from __future__ import annotations

import importlib.util
import importlib.metadata
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.utils.config import validate_config
from verl.trainer.ppo.utils import need_critic, need_reference_policy

from rl_native_pilot_task_runner import NativePilotTaskRunner
from rl_verl_chat_template import HERMES_RL_CHAT_TEMPLATE

ROOT = Path(__file__).resolve().parents[1]
# Versioned pilot datasets only: the original four-cell pool and the expanded
# sixteen-cell pool (2026-10-05). Cell membership is still enforced per batch
# by NativePilotTaskRunner against rl_online_batch_gate.CELLS.
DATASETS = (
    ROOT / "outputs/rl/native_pilot_train_20260930_hermes_mcp.parquet",
    ROOT / "outputs/rl/native_pilot_train_20261005_hermes_mcp_16cells.parquet",
)
DATASET = DATASETS[-1]
AGENT_CONFIG = ROOT / "configs/rl_native_pilot_agent_loop.yaml"
SFT = ROOT / "checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"


@hydra.main(config_path="../vendor/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("vLLM is not installed; refusing to start an unverified online optimizer job")
    if importlib.metadata.version("vllm") != "0.23.0+cu129":
        raise RuntimeError("pilot requires the locally pinned CUDA-12.9 vLLM 0.23.0 wheel")
    for dataset in (*DATASETS, AGENT_CONFIG, SFT / "config.json"):
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
    visible = [device for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device.strip()]
    if not visible or len(visible) != config.trainer.n_gpus_per_node:
        raise ValueError("explicit CUDA_VISIBLE_DEVICES must equal VeRL trainer GPU count")
    if config.actor_rollout_ref.model.path != str(SFT):
        raise ValueError("actor must start at the immutable merged SFT checkpoint")
    if config.actor_rollout_ref.model.custom_chat_template not in (None, HERMES_RL_CHAT_TEMPLATE):
        raise ValueError("refusing to override a different tool-call template")
    config.actor_rollout_ref.model.custom_chat_template = HERMES_RL_CHAT_TEMPLATE
    if config.trainer.get("pilot_policy_snapshot") != "step:0":
        raise ValueError("pilot trainer admission snapshot must be the initial SFT weight version")
    if config.actor_rollout_ref.rollout.agent.agent_loop_config_path != str(AGENT_CONFIG):
        raise ValueError("pilot custom native loop config is mandatory")
    if config.data.train_files not in [str(path) for path in DATASETS]:
        raise ValueError("pilot cannot train on validation or other data")
    if config.trainer.use_v1:
        raise ValueError("pilot guard currently intercepts only the legacy _update_actor path")
    OmegaConf.resolve(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, NativePilotTaskRunner)


if __name__ == "__main__":
    main()
