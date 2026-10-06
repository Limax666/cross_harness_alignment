"""Legacy VeRL TaskRunner with the native pilot admission gate actually installed.

Use with verl.trainer.main_ppo.run_ppo(config, NativePilotTaskRunner).
This is deliberately a distinct entrypoint: vendor main_ppo otherwise constructs
an unguarded RayPPOTrainer and must never be used for this pilot.
"""
from __future__ import annotations

import os
import socket

import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config import HFModelConfig
from verl.utils.dataset.rl_dataset import collate_fn

from rl_guarded_verl_trainer import GuardedNativePilotTrainer
from rl_online_batch_gate import CELLS
from rl_verl_chat_template import HERMES_RL_CHAT_TEMPLATE


@ray.remote
class NativePilotTaskRunner(BaseTaskRunner):
    def run(self, config):
        if config.trainer.use_v1 or config.actor_rollout_ref.rollout.n != 4:
            raise ValueError("pilot requires legacy VeRL trainer with rollout.n=4")
        if config.data.train_batch_size != len(CELLS) or not 1 <= config.trainer.total_training_steps <= 12:
            raise ValueError("bounded pilot requires one batch of all registered cells and one to twelve online updates")
        if config.data.max_prompt_length < 4096 or config.actor_rollout_ref.rollout.prompt_length < 4096:
            raise ValueError("full workspace tool schemas require at least 4096 prompt tokens")
        if config.actor_rollout_ref.rollout.response_length < 1024 or config.trainer.save_freq <= 0:
            raise ValueError("pilot requires enough tool response tokens and a saved checkpoint")
        if config.actor_rollout_ref.model.get("lora", {}).get("rank", 0) or config.actor_rollout_ref.model.get("lora_rank", 0):
            raise ValueError("full-parameter GRPO only; LoRA is prohibited")
        if config.actor_rollout_ref.rollout.agent.default_agent_loop != "native_agentdojo_pilot":
            raise ValueError("native pilot loop not selected")
        if config.actor_rollout_ref.rollout.name != "vllm":
            raise ValueError("native pilot XML stop-string contract verified only for vLLM")
        if not config.actor_rollout_ref.rollout.calculate_log_probs:
            raise ValueError("native pilot requires generation-time old-policy log probabilities")
        if config.algorithm.adv_estimator != "grpo" or config.trainer.get("pilot_policy_snapshot") != "step:0":
            raise ValueError("pilot must initialize from the SFT policy snapshot")
        if (config.trainer.resume_mode != "disable" or
                config.trainer.total_epochs != config.trainer.total_training_steps):
            raise ValueError("bounded pilot must start from SFT with one four-cell batch per epoch")
        if config.trainer.critic_warmup != 0 or config.trainer.val_only:
            raise ValueError("pilot must execute the full-parameter actor update")
        if config.reward.reward_model.enable or config.algorithm.use_kl_in_reward:
            raise ValueError("pilot reward must come only from independently verified native result")
        if config.trainer.get("val_before_train", True):
            raise ValueError("validation dataset must not be mixed into the train-family pilot")
        print(f"NativePilotTaskRunner {socket.gethostname()} PID={os.getpid()}", flush=True)
        print(f"pilot_config: model={config.actor_rollout_ref.model.path} "
              f"backend={config.actor_rollout_ref.rollout.name} "
              f"gpus={config.trainer.n_gpus_per_node} "
              f"batch={config.data.train_batch_size}x{config.actor_rollout_ref.rollout.n}", flush=True)
        OmegaConf.resolve(config)
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
        model_cfg: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer, processor = model_cfg.tokenizer, model_cfg.processor
        tokenizer.chat_template = HERMES_RL_CHAT_TEMPLATE
        dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True,
                                    max_samples=config.data.get("train_max_samples", -1))
        if len(dataset) != len(CELLS) or {row["task_id"] for row in dataset.dataframe} != CELLS:
            raise ValueError("VeRL dataset is not exactly the registered train cells")
        if any(row.get("split") != "train" or row.get("policy_snapshot") != "step:0" for row in dataset.dataframe):
            raise ValueError("untrusted training dataset split/snapshot")
        sampler = create_rl_sampler(config.data, dataset)
        trainer = GuardedNativePilotTrainer(
            config=config, tokenizer=tokenizer, processor=processor,
            role_worker_mapping=self.role_worker_mapping, resource_pool_manager=self.init_resource_pool_mgr(config),
            ray_worker_group_cls=ray_worker_group_cls, train_dataset=dataset, val_dataset=dataset,
            collate_fn=collate_fn, train_sampler=sampler,
        )
        trainer.init_workers()
        trainer.fit()
