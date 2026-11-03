"""Legacy VeRL TaskRunner with the native multi-harness admission gate actually installed.

Use with verl.trainer.main_ppo.run_ppo(config, NativeMultiharnessTaskRunner).
This is deliberately a distinct entrypoint: vendor main_ppo otherwise constructs
an unguarded RayPPOTrainer and must never be used for this pilot.
"""
from __future__ import annotations

import os
import socket
import hashlib
import json
from pathlib import Path

import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config import HFModelConfig
from verl.utils.dataset.rl_dataset import collate_fn

from rl_guarded_chspo_trainer import GuardedCHSPOTrainer, require_objective_version
from rl_multiharness_chat_template import BENCHMARK_MCP_CHAT_TEMPLATE
from rl_multiharness_stratified_sampler import StratifiedTaskSampler, EXPECTED_STRATA, GROUPS_PER_STRATUM
from rl_informative_sampler import InformativeTaskSampler



@ray.remote
class NativeMultiharnessTaskRunner(BaseTaskRunner):
    def run(self, config):
        require_objective_version(config.trainer.get("rl_algorithm"))
        if config.trainer.use_v1 or config.actor_rollout_ref.rollout.n != 4:
            raise ValueError("multi-harness requires legacy VeRL trainer with rollout.n=4")
        registered_cells = __import__("pandas").read_parquet(config.data.train_files)
        cell_count = len(registered_cells)
        if (config.data.train_batch_size != len(EXPECTED_STRATA) * GROUPS_PER_STRATUM or
                cell_count < config.data.train_batch_size or
                not 1 <= config.trainer.total_training_steps <= 300):
            raise ValueError("multi-harness requires a balanced sampled batch and 1..300 online updates")
        if config.data.max_prompt_length < 4096 or config.actor_rollout_ref.rollout.prompt_length < 4096:
            raise ValueError("full workspace tool schemas require at least 4096 prompt tokens")
        if config.actor_rollout_ref.rollout.response_length < 1024 or config.trainer.save_freq <= 0:
            raise ValueError("multi-harness requires enough tool response tokens and a saved checkpoint")
        if config.actor_rollout_ref.model.get("lora", {}).get("rank", 0) or config.actor_rollout_ref.model.get("lora_rank", 0):
            raise ValueError("full-parameter GRPO only; LoRA is prohibited")
        if config.actor_rollout_ref.rollout.agent.default_agent_loop != "native_multiharness_v1":
            raise ValueError("native multi-harness loop not selected")
        if config.actor_rollout_ref.rollout.name != "vllm":
            raise ValueError("native multi-harness XML stop-string contract verified only for vLLM")
        if not config.actor_rollout_ref.rollout.calculate_log_probs:
            raise ValueError("native multi-harness requires generation-time old-policy log probabilities")
        if config.algorithm.adv_estimator != "grpo" or config.trainer.get("pilot_policy_snapshot") != "step:0":
            raise ValueError("multi-harness must initialize from the SFT policy snapshot")
        resume_step = 0
        if config.trainer.resume_mode == "resume_path":
            checkpoint = Path(str(config.trainer.resume_from_path)).resolve()
            try:
                resume_step = int(checkpoint.name.removeprefix("global_step_"))
            except ValueError as exc:
                raise ValueError("resume requires an explicit global_step checkpoint") from exc
            if checkpoint.name != f"global_step_{resume_step}" or not 0 < resume_step < config.trainer.total_training_steps:
                raise ValueError("resume checkpoint step is outside the scheduled training interval")
            state_dir = Path(str(config.trainer.pilot_metrics_path)).resolve().parent
            provenance = json.loads((state_dir / "chspo_resume_provenance.json").read_text())
            data_sha = hashlib.sha256(Path(str(config.data.train_files)).read_bytes()).hexdigest()
            if (provenance.get("checkpoint") != str(checkpoint) or
                    provenance.get("step") != resume_step or
                    provenance.get("train_sha256") != data_sha):
                raise ValueError("CHS-PO resume state does not match checkpoint and frozen train pool")
            require_objective_version(provenance.get("objective_version"))
            if config.trainer.get("rl_algorithm") == "CHS-PO-harness-safety-dual-v4":
                priority_path = state_dir / "chspo_priority_snapshot.json"
                priority_bytes = priority_path.read_bytes()
                if hashlib.sha256(priority_bytes).hexdigest() != provenance.get("priority_snapshot_sha256"):
                    raise ValueError("v4 priority snapshot differs from checkpoint provenance")
                priority = json.loads(priority_bytes)
                if priority.get("last_step") != resume_step:
                    raise ValueError("v4 priority snapshot does not match checkpoint step")
            for name, key in (("chspo_duals.json", "duals"),
                              ("chspo_safety_streaks.json", "safety_streaks")):
                if json.loads((state_dir / name).read_text()) != provenance[key]:
                    raise ValueError(f"checkpoint-matched {name} was modified")
        elif config.trainer.resume_mode != "disable":
            raise ValueError("CHS-PO supports a fresh SFT start or an explicit verified checkpoint")
        if not (config.trainer.total_training_steps <= config.trainer.total_epochs <= 600):
            raise ValueError("CHS-PO needs a bounded epoch budget for zero-signal resampling")
        if config.trainer.critic_warmup != 0 or config.trainer.val_only:
            raise ValueError("multi-harness must execute the full-parameter actor update")
        if config.reward.reward_model.enable or config.algorithm.use_kl_in_reward:
            raise ValueError("multi-harness reward must come only from independently verified native result")
        if config.trainer.get("val_before_train", True):
            raise ValueError("validation dataset must not be mixed into the train-family pilot")
        print(f"NativeMultiharnessTaskRunner {socket.gethostname()} PID={os.getpid()}", flush=True)
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
        tokenizer.chat_template = BENCHMARK_MCP_CHAT_TEMPLATE
        dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True,
                                    max_samples=config.data.get("train_max_samples", -1))
        import pandas as pd
        registered = registered_cells
        if len(dataset) != cell_count or set(dataset.dataframe["cell_id"]) != set(registered["cell_id"]):
            raise ValueError("VeRL dataset differs from frozen train cells")
        if any(row.get("split") != "train" or row.get("policy_snapshot") != "step:0" for row in dataset.dataframe):
            raise ValueError("untrusted training dataset split/snapshot")
        if config.trainer.get("rl_algorithm") == "CHS-PO-harness-safety-dual-v4":
            if not resume_step:
                raise ValueError("v4 is registered only as a checkpointed step-140 continuation")
            sampler = InformativeTaskSampler(dataset.dataframe, priority)
        else:
            sampler = StratifiedTaskSampler(dataset.dataframe)
        # One sampled batch is one epoch. VeRL skips restoring an exhausted
        # data.pt at an epoch boundary; the custom sampler must be advanced
        # explicitly so resumed step 11 does not repeat step 1's task cells.
        sampler.epoch = int(provenance.get("sampler_epoch", resume_step)) if resume_step else 0
        trainer = GuardedCHSPOTrainer(
            config=config, tokenizer=tokenizer, processor=processor,
            role_worker_mapping=self.role_worker_mapping, resource_pool_manager=self.init_resource_pool_mgr(config),
            ray_worker_group_cls=ray_worker_group_cls, train_dataset=dataset, val_dataset=dataset,
            collate_fn=collate_fn, train_sampler=sampler,
        )
        trainer.init_workers()
        trainer.fit()
