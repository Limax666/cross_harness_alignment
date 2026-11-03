#!/usr/bin/env python3
import os
from pathlib import Path
import hydra
from verl.trainer.main_ppo import run_ppo
from rl_whitebox_task_runner import WhiteboxTaskRunner
from rl_multiharness_chat_template import benchmark_mcp_chat_template

ROOT=Path(__file__).resolve().parents[1]
SFT=ROOT/'checkpoints/qwen35-9b-base-v7-a6000_20261018_r7/hf_merged_step_291'
SFT_2B=ROOT/'checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267'
CHSPO_2B=ROOT/'outputs/rl/multiharness_chspo_r20g_v4_from130_20261012/hf_merged_step_150'
DATA=ROOT/'outputs/rl/9b_whitebox_pool_v4/train.parquet'
EXPANDED_DATA=ROOT/'outputs/rl/2b_whitebox_expanded_candidate_20261030/train.parquet'
CURRICULUM_DATA=ROOT/'outputs/rl/2b_whitebox_curriculum_candidate_20261030/train.parquet'

@hydra.main(config_path='../vendor/verl/verl/trainer/config',config_name='ppo_trainer',version_base=None)
def main(config):
    model_path=Path(config.actor_rollout_ref.model.path).resolve()
    if model_path not in (SFT.resolve(),SFT_2B.resolve(),CHSPO_2B.resolve()) or not (model_path/'model.safetensors').is_file():
        raise ValueError('model must be an approved frozen initialization')
    if Path(config.data.train_files).resolve() not in (DATA.resolve(), EXPANDED_DATA.resolve(), CURRICULUM_DATA.resolve()):raise ValueError('approved immutable whitebox pool required')
    if Path(config.trainer.assignment_ledger).resolve()!=Path(config.data.train_files).resolve().parent/'assignment_ledger.json':raise ValueError('assignment ledger must match pool')
    if Path(config.data.train_files).resolve()==CURRICULUM_DATA.resolve():
        from rl_curriculum_schedule import verify_frozen_curriculum
        verify_frozen_curriculum(CURRICULUM_DATA.parent)
    visible=[x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x.strip()]
    expected=config.trainer.n_gpus_per_node + int(config.trainer.get('separate_reference_gpu',False))
    if len(visible)!=expected or len(set(visible))!=expected:
        raise ValueError('explicit distinct GPU mask required for actor and reference pools')
    config.actor_rollout_ref.model.custom_chat_template=benchmark_mcp_chat_template()
    run_ppo(config,WhiteboxTaskRunner)

if __name__=='__main__':main()
