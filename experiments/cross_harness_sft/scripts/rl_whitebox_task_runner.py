"""Controlled whitebox AgentDojo experiment; no native CLI product claims."""
import json
from pathlib import Path
import pandas as pd
import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config import HFModelConfig
from verl.utils.dataset.rl_dataset import collate_fn
from rl_multiharness_chat_template import benchmark_mcp_chat_template
from rl_whitebox_trainer import WhiteboxTrainer
from rl_whitebox_objective import FixedLedgerSampler
from rl_whitebox_reward_contract_pair import paired_reward_profile
from rl_paired_assignment import AssignmentCursor

@ray.remote
class WhiteboxTaskRunner(BaseTaskRunner):
    def run(self,config):
        if config.trainer.use_v1 or config.actor_rollout_ref.rollout.n!=4 or config.data.train_batch_size!=24:
            raise ValueError('fixed legacy 24 x G4 online batch required')
        if config.actor_rollout_ref.model.lora_rank!=0 or config.algorithm.adv_estimator!='grpo':
            raise ValueError('full-parameter grouped policy update required')
        if not config.actor_rollout_ref.rollout.calculate_log_probs:raise ValueError('old logprobs required')
        if config.trainer.rl_algorithm not in ('GRPO-whitebox-v1','CHS-PO-whitebox-v1','GRPO-progress-v1','CHS-PO-progress-v1'):
            raise ValueError('unknown objective')
        loop = OmegaConf.load(config.actor_rollout_ref.rollout.agent.agent_loop_config_path)
        reward_contract = loop[0].get('reward_contract', 'legacy')
        if config.trainer.rl_algorithm == 'GRPO-whitebox-v1' and reward_contract != 'outcome_only_v1':
            raise ValueError('GRPO baseline requires outcome_only_v1 reward contract')
        if config.trainer.rl_algorithm == 'CHS-PO-whitebox-v1' and reward_contract != 'legacy':
            raise ValueError('CHS-PO requires legacy reward contract')
        if config.trainer.rl_algorithm in ('GRPO-progress-v1','CHS-PO-progress-v1') and reward_contract != 'verified_progress_v1':
            raise ValueError('progress experiment requires verified_progress_v1 reward contract')
        if config.trainer.rl_algorithm in ('GRPO-progress-v1','CHS-PO-progress-v1'):
            paired_reward_profile(config.trainer.rl_algorithm, loop[0])
        if reward_contract == 'verified_progress_v1' and (loop[0].get('path_weight') != 0 or
                loop[0].get('reward_variant') not in ('outcome','progress','progress_protocol')):
            raise ValueError('verified progress requires zero legacy path weight and a defined variant')
        if config.trainer.val_before_train or config.trainer.test_freq!=-1:
            raise ValueError('engineering acceptance only, no validation/test feedback')
        if config.reward.reward_model.enable or config.algorithm.use_kl_in_reward:
            raise ValueError('only native verifier reward allowed')
        if config.trainer.balance_batch:raise ValueError('retain exact task ledger row order')
        registered_cells=pd.read_parquet(config.data.train_files)
        cell_count=len(registered_cells)
        OmegaConf.resolve(config)
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        resource_pool_manager = self.init_resource_pool_mgr(config)
        if config.trainer.get('separate_reference_gpu', False):
            from rl_9b_resource_layout import separate_reference_layout
            resource_pool_manager.resource_pool_spec = separate_reference_layout(
                self.role_worker_mapping, self.mapping,
                actor_gpus=config.trainer.n_gpus_per_node, nodes=config.trainer.nnodes,
            )
        validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
        model_cfg: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer, processor = model_cfg.tokenizer, model_cfg.processor
        tokenizer.chat_template = benchmark_mcp_chat_template()
        dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True,
                                    max_samples=config.data.get("train_max_samples", -1))
        registered = registered_cells
        if len(dataset) != cell_count or set(dataset.dataframe["cell_id"]) != set(registered["cell_id"]):
            raise ValueError("VeRL dataset differs from frozen train cells")
        if any(row.get("split") != "train" or row.get("policy_snapshot") != "step:0" for row in dataset.dataframe):
            raise ValueError("untrusted training dataset split/snapshot")
        ledger=json.loads(Path(config.trainer.assignment_ledger).read_text())
        cursor=AssignmentCursor(ledger)
        sampler=FixedLedgerSampler(list(dataset.dataframe),cursor)
        trainer = WhiteboxTrainer(
            assignment_cursor=cursor,registered_cells=list(dataset.dataframe),
            config=config, tokenizer=tokenizer, processor=processor,
            role_worker_mapping=self.role_worker_mapping, resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls, train_dataset=dataset, val_dataset=dataset,
            collate_fn=collate_fn, train_sampler=sampler,
        )
        trainer.init_workers()
        trainer.fit()
