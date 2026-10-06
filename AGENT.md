# 跨 Harness 安全对齐：新服务器接手说明

更新：2026-10-06。先读 `experiments/cross_harness_sft/README.md`，尤其是 Phase 1 数据与全参数 SFT、Phase 3 在线 RL 和 Phase 4 评测协议。本文件记录已经实际完成的工作及下一步入口；不能把计划、启动成功或训练集奖励当作 held-out 能力提升。

## 已完成的全参数 SFT

原始候选视图有 4,162 条教师轨迹。按原始任务家族切分，再做完整性、真实工具证据、官方结果和 harness 协议检查；独立 `ali/qwen3.8-flash` judge 对安全、效用、工具正确性、harness 忠实度作四维审核。接受集为 1,603 train、246 validation。`Qwen/Qwen3.5-2B-Base` 的 4,096-token 完整轨迹预检留下 1,421 train、232 validation，隔离 196 条超长轨迹；train 有 316 条良性工具任务及 1,105 条风险相关任务，覆盖 8 个 harness、139 个任务家族。源、切分、分类和哈希见 `experiments/cross_harness_sft/data/sft/teacher_release_v7/` 中随本分支提交的几个 manifest；原始轨迹和完整 SFT Parquet 没有上传 GitHub。

VeRL BF16/FSDP **全参数**训练，`lora_rank=0`，两张 24 GiB GPU，全局 batch 16、每卡 micro-batch 1、梯度累积 8、最大长度 4096；AdamW 峰值 LR `1e-5`、10% warmup、cosine、weight decay 0.05、梯度裁剪 1.0；3 epoch 共 267 次更新，每 8 步验证。训练入口为 `experiments/cross_harness_sft/scripts/train_teacher_release_v7_full_sft_2b.sh`。32 步 train-loss 均值约从 1.3 降至 0.65；独立 validation loss 最低 0.893（step 144），末步 0.900。当前评测及 RL 初始化用最终合并权重 `checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267`，不是 step 144。图与数据在 `experiments/cross_harness_sft/reports/qwen35_2b_full_sft_training_curves_20260925.{png,csv}`。

## SFT 后的实际评测

HarnessRisk：每个 harness、每个模型使用相同 128 case × 3 独立 rollout，384 条均落盘并由 `openai/gpt-5.4-nano` 评分。论文用 GPT-5.4，二者不能作同评分器的严格排名。指标分别为 Utility↑、ASR↓、Persistence↓、Detection↑：

| Harness | Base | SFT | 结论 |
|---|---|---|---|
| Hermes | 1.8% / 0.3% / 0.0% / 4.4% | 1.0% / 1.6% / 0.8% / 4.2% | Utility 下降，ASR 上升 |
| NanoBot | 12.8% / 4.2% / 19.5% / 0.8% | 7.8% / 3.1% / 13.0% / 1.0% | 风险指标下降，Utility 同时下降 |
| OpenClaw | 1.0% / 0.0% / 30.2% / 0.0% | 0.0% / 0.0% / 22.9% / 0.5% | 几乎无法完成任务；初始化文件污染 Persistence 评分 |

详见 `experiments/cross_harness_sft/reports/harnessrisk_qwen35_2b_v7_sft_comparison_20260927.md`。OpenClaw 的 `IDENTITY.md` 等初始化文件被 judge 计为持久化证据；在重评分前不要解释该项 Base→SFT 变化为真实安全收益。Hermes 的 `skill_view` 参数格式问题和 OpenClaw 工具协议仍需排查。低 ASR 与极低 Utility 同时出现，不能声称跨 harness 安全对齐成功。

AgentDojo：固定种子抽 100 个任务（9 clean、91 injection），Base/SFT × Codex/Claude Code 各计划 3 次。官方 verifier 的已评分子集：Codex Base 232/300，Utility 14/232、ASR 0/214；Codex SFT 246/300，Utility 56/246、ASR 3/221。Codex 还有 68/54 条超时未评分，两个模型的评分分母不同。Claude Code 两组都 300/300；Base Utility 51/300、ASR 2/273，SFT Utility 57/300、ASR 5/273，分别有 162/156 条达到 CLI 最大轮次。SFT 的已评分 Utility 有上升，但两个 harness 的攻击成功事件也增加；因此不能将其概括为净安全提升。完整口径在 `experiments/cross_harness_sft/reports/yuque_multiharness_sft_harnessrisk_report_20260920.md` 的第 8 节。

## 在线 RL 已做到哪里

VeRL 当前策略在线生成 + AgentDojo 官方重置环境及 verifier + Hermes MCP 工具名称/参数映射已经跑通。16 个 train task cells 来自 8 个 AgentDojo 家族，各有 clean/injection 两臂；G=4，共 64 个独立 rollout/update。模型是上述 SFT，reference 冻结；全参数更新、无 LoRA。`outputs/rl/hermes_agentdojo_grpo_12step_16cells_20261005` **真实完成 6 次在线 GRPO 更新并保存 step-6 checkpoint**；step 7 的 16 组奖励全部零方差，门禁拒绝继续更新。原 metrics 的组数仍写了旧版 4 组，使用 `metrics.corrected.jsonl` 和对应 corrected 图。训练集上 Utility 约 0.19→0.03，平均回复长度 370→222 token，大量轨迹超时或未完成，呈现“避免违规但也不完成任务”的中性塌缩。

独立的 4 个未入训 workspace 家族上，24 个配对 episode/模型：SFT clean Utility 1/12、injection Utility 1/12；RL 两项均 0/12；ASR 均 0/12。invalid call/incomplete 从 SFT 的 17/24 升到 RL 的 24/24。使用与在线循环一致的 blocked-call 反馈重跑后，结论相同。结果证据随本分支提交 `metrics.corrected.jsonl`、`rl_training_dashboard_corrected.png` 和 `heldout_agentdojo_validation_protocol_parity/comparison.json`。这只是小样本单 harness 诊断，已明确显示当前 checkpoint 不宜宣传为安全能力提升。

2026-10-06 的 2048-token r17 单卡尝试未产生 rollout 或 optimizer step：GPU1 上 actor/reference + vLLM 加载后，**step 0 初始权重同步**需额外 2 GiB，而只剩约 658 MiB，因 OOM 退出。日志留在旧服务器的 `outputs/rl/hermes_agentdojo_grpo_r17_budget2048_16cells_20261005/launch.log`。新服务器用两张实际空闲 GPU 预检；不要用单卡失败日志画新 RL 曲线。代码中的 bounded path reward 虽支持 `path_weight<=0.25`，但 `verified_safe_subgoals` 尚未由真实路径证据填充，调大权重本身不能解决中性塌缩。

## 正式多 harness 在线 RL 尚未交付

GitHub 当前**只有 Hermes 协议映射 + AgentDojo 官方可复位工具环境的 16-cell 在线 pilot**。`configs/rl_native_pilot_agent_loop.yaml`、`scripts/rl_verl_native_agentdojo_loop.py` 和已提交的 `outputs/rl/native_pilot_train_20261005_hermes_mcp_16cells.parquet` 都只服务这条路径；在线 loop 的 harness 字段也是 Hermes。`scripts/rl_group_sampler.py` 可表达多 harness 分层采样，但尚无 NanoBot/OpenClaw 等 harness 的生产级可复位 train-family fixture、真实工具执行适配器、逐步/最终副作用审计和对应 task-cell manifest，因此**不存在可从 GitHub 再拉取的正式多 harness RL 数据包或启动脚本**。旧机 `data/raw/multi_harness_teacher_v1/` 和 `data/sft/teacher_release_v7/` 是离线教师/SFT 数据，不能代替当前策略在线 rollout 的执行环境；HarnessRisk 的保留评测 case 不能直接转成 RL train cell。下一步需逐 harness 建立隔离 train-family 环境、验证器及配对 benign/risk cells，通过反例、reset、工具协议和 family-disjoint 验证门槛后再扩展 VeRL loop 与 CHS-PO 采样，不能仅把 pilot Parquet 复制或改标签宣称多 harness RL。

## 新服务器复现与继续工作的顺序

1. 克隆本仓库的 `handoff/rl-20261006` 分支（或其合并后的 `main`），确认读到本文件、实验 README 和 `experiments/cross_harness_sft/scripts/rl_*`。代码已移除被误纳入旧 Git 历史的虚拟环境；本分支也不包含 checkpoint、原始轨迹和 API key。
2. 在兼容 CUDA 12.9 的服务器上建立 `experiments/cross_harness_sft/.venv-rl-pilot`。旧机实测核心版本：Torch 2.11.0+cu129、vLLM 0.23.0+cu129、Ray 2.58.0、Transformers 5.9.0、PEFT 0.21.1、Hydra 1.3.7。执行 `bash experiments/cross_harness_sft/scripts/bootstrap_rl_sources.sh`，它固定 VeRL `02a318c303a7b60871cb63e4ce2f779b522874d9`、AgentDojo `089ed468cf3ed0322acc66b0211f26d9d90dbf60`，并应用 `patches/verl_project_local_02a318c3.patch`（含 optimizer 状态按需回迁等本地修复）。安装两者到 RL 环境并确认 Hermes `5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04` 的 Python 环境；按需设置 `HERMES_AGENT_ROOT` 和 `HERMES_AGENT_PYTHON`。`rl_native_pilot_agent_loop.yaml` 的 worker 路径已改为相对路径。
3. 从私有 Hugging Face 模型仓库 `Limax11/harness_alignment` 下载 **step 267 的 4.2 GiB SFT 合并 checkpoint**，固定 revision `7b608bf7517bd8b2f08ec8eecedd6dea4bdbbe26`，放到仓库相对目录 `experiments/cross_harness_sft/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267`；新服务器先以有权限的 Hugging Face 账号运行 `hf auth login`，再运行 `hf download Limax11/harness_alignment --revision 7b608bf7517bd8b2f08ec8eecedd6dea4bdbbe26 --local-dir experiments/cross_harness_sft/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267`。GitHub 不托管该权重。16-cell 训练 Parquet 已随本分支上传到 `experiments/cross_harness_sft/outputs/rl/native_pilot_train_20261005_hermes_mcp_16cells.parquet`。若需要复现 SFT 数据构建，还须另行转移忽略的 `data/raw`、`teacher_release_v7` 完整数据和源 benchmark；不能用 Git 中 manifest 伪装成数据本身。
4. 先核查 Hermes MCP 转换、AgentDojo 真工具执行及 reset/path audit；解决 `verified_safe_subgoals` 的真实证据来源，并以固定安全/不安全反例核对奖励边界。然后在两张空卡上用**新的** `RL_RUN_DIR` 做 2048-token、16-cell、G=4 的小批在线更新；入口是 `scripts/train_rl_native_pilot_verl.sh`，设 `CUDA_VISIBLE_DEVICES=<两卡>`、`RL_N_GPUS=2`、`RL_GPU_UTIL=0.45`、`RL_MAX_NUM_SEQS=2`、`RL_RESPONSE_LENGTH=2048`、`RL_TRAIN_BATCH_SIZE=16`、`RL_DATA=<上面的 Parquet>`。不得复用已有 run 目录；若批次无有效组内优势，保留门禁停机。
5. 每次保存 checkpoint 后，用 `scripts/evaluate_rl_sft_pair.sh` 对同一家族隔离验证集配对比较 SFT/RL 的 Utility、ASR、invalid calls、超时及真实副作用；保存 JSONL、metrics 和曲线。只有验证同时守住安全边界与正常任务效用，才扩大任务池或开始 CHS-PO、Balance-GRPO、DAPO 等消融。不要把训练 reward 上升当作能力涨点。

旧机 SFT 和 RL checkpoint 均未上传 GitHub；**只有 SFT step 267** 已上传至上述私有 Hugging Face 仓库，RL checkpoint 尚未上传。若从旧机迁移其他权重，可用 `rsync --partial`；先检查目标磁盘、模型文件大小和新服务器 GPU 驱动，再运行全参数在线更新。所有外部服务密钥应通过新服务器环境变量配置。
