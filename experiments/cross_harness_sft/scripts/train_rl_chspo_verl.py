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

from rl_native_chspo_task_runner import NativeMultiharnessTaskRunner
from rl_multiharness_chat_template import BENCHMARK_MCP_CHAT_TEMPLATE

ROOT = Path(__file__).resolve().parents[1]
# Frozen train-family pool; admission checks every sampled cell before update.
DATASET = ROOT / "outputs/rl/multiharness_agentdojo_agentharm_pool_v5.parquet"
AGENT_CONFIG = ROOT / "configs/rl_multiharness_agent_loop.yaml"
SFT = ROOT / "checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"


@hydra.main(config_path="../vendor/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("vLLM is not installed; refusing to start an unverified online optimizer job")
    if importlib.metadata.version("vllm").split("+", 1)[0] != "0.23.0":
        raise RuntimeError("pilot requires vLLM 0.23.0")
    import torch
    if torch.version.cuda != "12.9":
        raise RuntimeError("pilot requires the CUDA-12.9 PyTorch/vLLM stack")
    for dataset in (AGENT_CONFIG, SFT / "config.json", SFT / "model.safetensors"):
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
    visible = [device for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device.strip()]
    if not visible or len(visible) != config.trainer.n_gpus_per_node:
        raise ValueError("explicit CUDA_VISIBLE_DEVICES must equal VeRL trainer GPU count")
    if config.actor_rollout_ref.model.path != str(SFT):
        raise ValueError("actor must start at the immutable merged SFT checkpoint")
    if config.actor_rollout_ref.model.custom_chat_template not in (None, BENCHMARK_MCP_CHAT_TEMPLATE):
        raise ValueError("refusing to override a different tool-call template")
    config.actor_rollout_ref.model.custom_chat_template = BENCHMARK_MCP_CHAT_TEMPLATE
    if config.trainer.get("pilot_policy_snapshot") != "step:0":
        raise ValueError("pilot trainer admission snapshot must be the initial SFT weight version")
    if config.actor_rollout_ref.rollout.agent.agent_loop_config_path != str(AGENT_CONFIG):
        raise ValueError("pilot custom native loop config is mandatory")
    if config.data.train_files != str(DATASET):
        raise ValueError("pilot cannot train on validation or other data")
    if not Path(config.data.train_files).is_file():
        raise FileNotFoundError(config.data.train_files)
    if config.trainer.use_v1:
        raise ValueError("pilot guard currently intercepts only the legacy _update_actor path")
    OmegaConf.resolve(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, NativeMultiharnessTaskRunner)


if __name__ == "__main__":
    main()
