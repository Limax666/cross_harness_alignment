# 跨 Harness 安全对齐：新服务器接手说明

## 2026-10-16 接手优先读：已完成实验、实际能力与大模型下一轮设计

以下只汇总已落盘的训练与评测；下文保留了按日期记录的历史快照，其中“运行中”“当前无任务”等说法不能代替实时进程检查。实验规范见 `experiments/cross_harness_sft/README.md`；具体逐条结果见 `experiments/cross_harness_sft/reports/`。**当前没有证据证明 CHS-PO 已在所有 harness 上同时实现 Utility 提升与 ASR/Persistence 下降。**

### 模型与训练进度

| 阶段 | 已完成的配置与产物 | 关键限制 |
|---|---|---|
| 2B 全参数 SFT | 原始教师候选 4,162 条，经任务家族隔离、轨迹/工具/安全审计和独立 LLM judge 筛成 1,603 train、246 validation；Qwen3.5-2B-Base 的 4,096-token 完整轨迹预检保留 1,421/232。VeRL BF16/FSDP、`lora_rank=0`、两张 24 GiB GPU、全局 batch 16、每卡 micro-batch 1、AdamW、峰值 LR `1e-5`、10% warmup、cosine、3 epoch、267 更新；验证 loss 最低 0.893（step 144），最终约 0.900。评测与 RL 起点是 `hf_merged_step_267`。 | 训练覆盖 Codex、Claude Code、Hermes、NanoBot、OpenClaw、OpenCode、`claudecode`、QwenPaw；每个来源×harness 的可用条数不均，不能用 loss 收敛代替工具协议或安全能力证明。数据、曲线和入口见下文“已完成的全参数 SFT”。 |
| 标准在线 GRPO R11 | 从同一 2B SFT 初始化 actor、冻结 SFT reference；AgentDojo/Codex、AgentDojo/Claude Code、AgentHarm/Codex、AgentHarm/Hermes 的 107 个 train-family cell；每步 32 组×`G=4` 独立 reset 当前策略 rollout（128 条），全参数更新 20 步，保存完整 `global_step_20`。有官方 outcome、真实工具/路径审计、`path_weight=0.25`、2048-token 响应预算。 | 后四步平均零方差组率 81.3%，组内奖励信息变窄；HarnessRisk 上 Utility 和 ASR 同涨，不能称为净安全收益。 |
| CHS-PO v2→v3→v4 | 沿用 107-cell、32×4 在线闭环；按 `(benchmark, harness)` 分层 attack/persistence/protocol dual，保留良性不行动惩罚、安全硬门禁及每 10 步 checkpoint。v3 从 v2 step 70 续训，最后完整安全起点为 step 130；v4 从该完整 checkpoint 迁移奖励目标，加入有官方路径证据的成功轨迹效率细分和信息量采样。v4 完整 `global_step_150` 与 `hf_merged_step_150` 可用于续训和推理评测（用途分别核对）；**step 155 出现 128/128 超时、Utility 0 和 PPO KL 0.206，watchdog 已停机，不要使用其未保存权重**。 | v4 是跨目标函数的续训分支，非从 SFT 独立起跑的干净算法消融；持续高零方差与 step 138/155 突发输出退化仍未解决。日志、奖励版本、恢复账本和曲线见 `reports/chspo_v4_step140_signal_design_20261012.md` 与对应 run 目录。 |

独立 4B/9B 全参数 SFT 规模对照记在 `reports/sft_4b_9b.md`：其单轮 1,081 题报告中，4B ASR 18.90%→5.70%，但正常任务完成 85.57%→68.04%；9B ASR 8.66%→1.48%，正常任务完成 92.78%→91.75%。这些是**另一套模型/任务口径的 SFT 结果**，不能直接与下表 2B HarnessRisk 或 CHS-PO 算法效果合并；它们说明放大模型值得做同口径复验，也提示必须同时约束良性效用。

### 已完成的能力验证：相同 HarnessRisk case 和 judge

2B Base/SFT、GRPO step 20、CHS-PO v4 step 150 都在 HarnessRisk 的 Hermes、NanoBot、OpenClaw 上各用 128 case×3 独立 rollout；本实验统一由 `openai/gpt-5.4-nano` 判分，表内依次为 **Utility↑ / ASR↓ / Persistence↓ / Detection↑**，每项分母 384。论文使用 GPT-5.4，不能将本表与其表格当作同 judge 的严格排名。完整计数与百分点差见 `reports/harnessrisk_sft_grpo20_chspo150_comparison.md`。

| Harness | Base | SFT | GRPO step 20 | CHS-PO step 150 | CHS-PO 相对 GRPO 的关键变化 |
|---|---|---|---|---|---|
| Hermes | 1.8 / 0.3 / 0.0 / 4.4% | 1.0 / 1.6 / 0.8 / 4.2% | 5.2 / 3.4 / 1.3 / 3.6% | 13.3 / 1.6 / 1.0 / 3.1% | Utility +8.07 pp，ASR −1.82 pp |
| NanoBot | 12.8 / 4.2 / 19.5 / 0.8% | 7.8 / 3.1 / 13.0 / 1.0% | 20.6 / 9.1 / 29.2 / 1.6% | 18.5 / 6.3 / 23.2 / 0.3% | Utility −2.08 pp，ASR −2.86 pp |
| OpenClaw | 1.0 / 0.0 / 30.2 / 0.0% | 0.0 / 0.0 / 22.9 / 0.5% | 11.5 / 4.2 / 30.7 / 1.0% | 23.2 / 1.6 / 34.4 / 1.3% | Utility +11.72 pp，ASR −2.60 pp |

**判读：**CHS-PO 相对 GRPO 三个 harness 的 ASR 都降低，但 NanoBot Utility 低 2.08 pp；相对 SFT，NanoBot/OpenClaw 的 ASR 仍分别高 3.12/1.56 pp。OpenClaw Persistence 含已知初始化文件评分污染，不能解释为真实攻击性持久化变化。模型能力比较也不是控制训练步数、响应预算和目标函数的严格 GRPO↔CHS-PO 消融。SFT 的 AgentDojo/Codex、Claude Code 官方 verifier 抽样评测见下文；超时/轮次上限与分母差异必须连同指标报告。

### 当前跨 harness 泛化验证与下一台大服务器的首要实验约束

SafeClawArena 使用 v7 **未入 SFT 的 PSE:2.4、SSI:1.2 共 30 个家族内 case**，NemoClaw/SecLaw 各 3 次；这是 SFT/RL 都未接触过的**运行 harness**，但数据集来源 SafeClawArena/OpenClaw 曾用于 SFT，因此是“新 harness + 已见数据集来源”，不是完全新 benchmark。当前修正后的评测目录是 `outputs/safeclawarena_chspo150_eval_schemafix_v3_20261016/`；先前版本因工具示例污染、工具名自动改写和 schema 截断而作废。2026-10-16 交接时 NemoClaw 两臂仍在断点续跑，SecLaw 尚未完成；只看最终相同 case/seed 的双方有效 verdict，协议错误、网关启动失败与未评分须分开统计。SafeClawArena 无良性任务，**不能用它单独证明 Utility 保留**。已观察到未见过的 53 工具/约 31.5K-token 提示下，2B 会把部分必填参数写成错误标签，或选错工具参数；先做协议归因，再谈模型容量瓶颈。

**更大模型下一轮以“跨 harness 安全泛化”作为预注册的核心目标，而非只增加模型参数：**

1. **先冻结三类 harness 暴露集合和任务家族切分。** `H_SFT` 用于多源良性+风险教师轨迹 SFT；`H_adapt` 整个 harness 从 SFT train/validation 轨迹中留出，先对 SFT 做零样本基线，再仅用该 harness 的**独立 train families** 建可复位在线工具环境和 verifier，加入 GRPO/CHS-PO RL；`H_final` 在 SFT、RL、奖励调参和 checkpoint 选择中均完全不出现，只供最终跨 harness 泛化测试。`H_adapt` 的 RL 后成绩是“对 SFT 未见 harness 的在线适应”，不能冒充全程未见泛化；真正的未见 harness 结论只由 `H_final` 给出。先核对原始轨迹、工具 schema、别名与近重复家族的实际暴露，再指定具体 harness；已经看过的 NemoClaw/SecLaw 结果不能无条件重当盲测集。
2. **SFT 留出必须真实。** 新模型重建独立数据版本和不可变 exposure manifest；从训练、验证、上下文示例及蒸馏教师视图中排除 `H_adapt/H_final` 的实际运行协议，按原始任务家族隔离 benign/risk 配对与跨 harness 变体。保留足够的良性工具完成和风险防御轨迹，控制来源/长度比例，并核对训练格式与目标 harness 工具协议。更大模型的全参数 SFT 大小、序列长度、批量和 FSDP/优化器 offload 在新服务器做显存/RAM/吞吐预检后确定，不从 2B 参数机械放大。
3. **RL 在 `H_adapt` 上补可验证能力，同时保留已见 harness。** 先对每个新增 harness 做真实工具 schema、权限、reset、最终副作用与路径审计的正反例门禁；使用同一 SFT checkpoint、训练池、token/rollout 预算分别跑标准 GRPO 与 CHS-PO，再逐项消融分层采样、良性/风险配对、路径奖励、安全 dual。`H_adapt` 的 train-family rollout 是当前策略在线生成；验证家族只选 checkpoint，不回灌；`H_final` 无任何 RL 更新。不能仅靠增加全局违规惩罚来宣称跨 harness 泛化。
4. **以预注册的配对评测决定是否成功。** 在已见、`H_adapt` 和 `H_final` 上用相同 case/seed、原生 harness、同一模型路由与 judge/oracle、每题至少 3 次，分别报告 Utility、ASR、Persistence、Detection（仅在 benchmark 定义该指标时）、协议错误、超时及置信区间。模型选择应要求每个可评层的良性 Utility 不低于 SFT，ASR/禁止持久化不高于 SFT，并单报最差 harness；不要用 HarnessRisk 或新的最终盲测结果更新 reward、dual、采样权重或挑 checkpoint。若使用已检查过的 benchmark 家族调参，必须另留真正未见家族/基准做最后验证。

## 2026-10-11 CHS-PO v3 step-70 续训（16:03 CST）

R20g v2 的完整 `global_step_70` 已保存，AgentDojo/Hermes 家族隔离 24 条配对验证为 SFT→v2 step-70：clean Utility 1/12→3/12、injection Utility 1/12→3/12、ASR 0/12→0/12。样本窄，不能外推到 NanoBot/OpenClaw 或 HarnessRisk。v3 收紧 attack/persistence dual 与训练门禁。

最初误启的 SFT 起点 v3 run 在首个更新前已中止，未产生指标/checkpoint。正式分支 `outputs/rl/multiharness_chspo_r20g_step70_v3_resume_20261011/` 从 v2 `global_step_70` 恢复后跑到 step126/300，共完成 56 次更新。step120 是最后完整 checkpoint；step121–126 指标虽已记录，但不可作为完整续训状态。step127 的 AgentHarm judge 调用 Shengsuanyun 返回 HTTP 402 Payment Required，进程退出，GPU1/2 已释放；当前需恢复 judge 额度/凭证后从 step120 checkpoint 续跑。不是 OOM，RAM 可用约189 GiB。

训练池统计（step71–126）：每步128条已评分；零方差组率首10步均值0.813、末6步0.854，峰值0.969；896条风险臂中 AgentHarm/Codex 有4次官方攻击成功、AgentHarm/Hermes 2次，AgentDojo两层为0；AgentDojo/Codex有5次禁止持久化。step126安全 streak 全为0。只代表训练池，不是 HarnessRisk 评测结论。objective migration `v2→v3` 有 provenance 记录。详情见 `experiments/cross_harness_sft/reports/chspo_v3_safety_revision_20261011.md`。

## 2026-10-11 CHS-PO 续训状态（13:08 CST 快照）

R20f 从 step-20 续训，真实完成 step 67；下一个 32 组批次全部组内奖励恒定，原 GRPO 门禁在 `rl_guarded_chspo_trainer.py` 的更新入口停机。CHS-PO 有独立的可验证违规与良性不行动惩罚，零组内奖励差异并不必然意味着零策略优势。现已加入 CHS-PO 专属检查：完整批次仍须通过来源、token、奖励和审计门禁；调整后优势非零才更新；完全零信号批次写入 `chspo_skipped_batches.jsonl`，不计优化器步数，并在线抽取下一组平衡任务，连续 16 次零信号才停机诊断。标准 GRPO 的零差异门禁保持不变。VeRL 采样跳批 hook 位于 `patches/verl_chspo_zero_signal_resample.patch`，`bootstrap_rl_sources.sh` 会应用它；每个 checkpoint 的 sampler epoch 随安全日志和 resume provenance 保存，避免跳批后重复任务调度。针对性测试 12 项通过。

新 run `outputs/rl/multiharness_chspo_r20g_zero_signal_resample_gpu12_20261011/` 已在 tmux `chspo_r20g` 用 GPU1/2 从 **R20f 的完整 `global_step_60`** 启动；rank 0/1 已加载模型、优化器和 RNG，13:08 CST 进度条 60/300，尚不能据此声称 step 61 更新已完成。启动脚本仍每 10 step 保存，训练池 SHA-256 仍是 `f6f88bf4f73e982b95b741a2292bcb39464ea65a9b5c2a42186a40cd5c93db42`。R20f 的 step 61–67 是被此次从 step 60 恢复替代的另一条训练分支，不可与新 run 拼接为连续曲线。后续只看新 run 的 `metrics.jsonl` 和 `chspo_safety.jsonl` 判断真实更新。

## 2026-10-10 历史快照：R11 结果、CHS-PO 实现与当时的停机状态

以下“当前”均指 2026-10-10 的记录；实际运行状态以上方更新和实时日志为准。

**当前没有在线 RL 任务运行，四张 GPU 上没有本项目进程。** 此文件后面标注“运行中”的段落是当时的历史快照，不代表现在的进程状态。A6000-2 上不要按旧的 `chspo_current_run.json` 或旧 run-id 续跑。R18/R19/R20 各只完成一次在线 actor 更新，均在第一次更新后的权重同步阶段因当前服务器的**宿主 RAM**达到 Ray 95% 限值而失败，未保存 step-10 checkpoint；三次 step-1 指标不可拼成连续训练曲线。R19/R20 的失败已在当前服务器确认，分别见 `outputs/rl/multiharness_chspo_r19_ramfix_gpu12_20261009_2310.log` 和 `outputs/rl/multiharness_chspo_r20_ramfix_gpu12_20261010.log`（原始日志未上传 GitHub）。R20 将响应预算降至 1024、AgentLoop/reward worker 减至 4/2，仍在同步点 OOM；试验性 optimizer-state parking 报告 `parked ... bytes=0`，**未证明有效**。不要关闭 Ray 内存保护或把此故障误判为 GPU 显存不足。新机器须重新测量首次 actor 更新、权重同步、第二步采样的峰值，再考虑长跑。

### R11 标准在线 GRPO：已验证的能力变化及失败模式

起点是 Qwen3.5-2B 全参数 SFT step 267；R11 在 AgentDojo/Codex、AgentDojo/Claude Code、AgentHarm/Codex、AgentHarm/Hermes 四层的 107 个 train-family cell 池中，每步均衡采 32 组、每组 G=4 次**当前策略在线 rollout**，完成 20 次全参数 GRPO 更新并保存 `global_step_20`；它没有覆盖 v7 SFT 的全部 harness。R11 已启用 `path_weight=0.25` 和 2048-token 响应预算，下一轮不能把这两项误写成新增创新。训练后期零方差组率从前四步平均 38.3% 增至后四步 81.3%，说明组内相对优势逐渐消失；还需防止通过“不行动”获得表面安全分。AgentDojo 四个隔离 validation 家族的 24 个配对 episode 中，RL 相比 SFT 的回合未正常结束从 19/24 降到 6/24，但 clean Utility 仍是 0/12→0/12、injection Utility 1/12→0/12、ASR 0/12→0/12，不能凭正常结束率称为安全提升。完整证据见 `experiments/cross_harness_sft/reports/r11_step20_online_rl_summary_20261008.md`。

HarnessRisk 独立测试使用同一 `openai/gpt-5.4-nano` judge：每个 harness 在 SFT step 267 与 RL step 20 上各评 128 case × 3 轮（384 条/模型）。下表是同口径内部对照，百分数为三轮均值；ASR/Persistence 越低越好，Utility/Detection 越高越好。

| Harness | Utility SFT→RL | ASR SFT→RL | Persistence SFT→RL | Detection SFT→RL |
|---|---:|---:|---:|---:|
| Hermes | 4/384 (1.0%)→20/384 (5.2%) | 6/384 (1.6%)→13/384 (3.4%) | 3/384 (0.8%)→5/384 (1.3%) | 16/384 (4.2%)→14/384 (3.6%) |
| NanoBot | 30/384 (7.8%)→79/384 (20.6%) | 12/384 (3.1%)→35/384 (9.1%) | 50/384 (13.0%)→112/384 (29.2%) | 4/384 (1.0%)→6/384 (1.6%) |
| OpenClaw | 0/384→44/384 (11.5%) | 0/384→16/384 (4.2%) | 88/384 (22.9%)→118/384 (30.7%) | 2/384 (0.5%)→4/384 (1.0%) |

**结论是 Utility 与 ASR 三个 harness 同涨，而非净安全提升。** OpenClaw Persistence 两臂都受初始化文件误判污染，这一列只能按当前 judge 口径报告；不能据 +7.8 pp 断言真实攻击性持久化增加。NanoBot/OpenClaw 并未直接进入 R11 的在线训练层，是跨 harness 迁移测试。论文 GPT-5.4 judge 与本实验 nano judge 不同，不能严格横向排名。评测细节、逐轮稳定性、any@3、超时和评分 bug 修复见 `experiments/cross_harness_sft/reports/harnessrisk_sft_vs_rl_step20.md` 与 `experiments/cross_harness_sft/reports/r11_step20_online_rl_summary_20261008.md`；step-20 合并权重只用于推理，续训须完整 FSDP 模型、优化器和训练状态。

### 当前 CHS-PO（harness-safety-dual-v2）：代码、约束与复现边界

- **训练池与在线性**：`scripts/train_rl_chspo_verl.sh` → `train_rl_chspo_verl.py` → `rl_native_chspo_task_runner.py` → `rl_guarded_chspo_trainer.py`。从同一 SFT step 267 和 R11 的 107-cell train-family 池 `outputs/rl/multiharness_agentdojo_agentharm_pool_v5.parquet` 重新初始化；`rl_multiharness_stratified_sampler.py` 每步按四个 benchmark/harness 层各取 4 clean + 4 risk cell，即 32 组 × G=4 = 128 条独立 reset rollout。只接受官方 outcome、真实工具/副作用证据、token mask 和 old-policy log-prob 均完整的更新批次；未评分轨迹 fail-closed。训练池与 SFT 权重不在公开 Git，须从原迁移件和私有模型仓库取得，并核对 Parquet SHA-256 `f6f88bf4f73e982b95b741a2292bcb39464ea65a9b5c2a42186a40cd5c93db42`。
- **奖励与更新**：`rl_guarded_chspo_trainer.py` 在 VeRL GRPO 相对优势上加入不居中的安全代价；dual 按 `(benchmark, harness)` 分层，attack/persistence/protocol 的初值为 0.25/0.50/0.10、训练预算为 0.05/0.05/0.10、步长 0.08、上限 1.0。仅真实违规 dispatch/协议终止计入 protocol 成本；解析器挡下的 `blocked_tool_calls` 单独观察，不进入 dual，以免压制“尝试但被阻止”的良性轨迹。verified unsafe 的正优势被截断再减代价；benign nonengagement 另受 0.30 的非中心化惩罚，避免安全靠不行动。`reward_outcome=-1` 的 verified violation 边界、`path_weight=0.25` 的有界真实路径证据、风险/良性配对和 batch 准入不应为追求 Utility 而放松。
- **硬门禁与选模**：任一层 risk 臂 attack 或禁止持久化率连续 3 步超过 `max(0.15, SFT 同池初始率 + 0.10)` 即在下一次更新前停机。这个初始率来自 R11 step 1 的训练池快照，是工程阈值，不是验证集安全标准。每 10 步 checkpoint；正式放行须在**家族隔离的 validation** 上以相同 task/seed 与 SFT 配对，要求每个可评层 Utility 不低于 SFT 且 ASR、禁止持久化均不高于 SFT。HarnessRisk 是独立最终测试，不能用于训练 dual 或挑 checkpoint。目前启动器只保存 checkpoint，**未自动执行该 validation/选模门禁**。
- **运行参数与可视化**：计划 300 step、全参数 `lora_rank=0`、FSDP BF16、AdamW optimizer/参数 CPU offload、actor micro-batch 1、mini-batch 4、KL loss 系数 0.001、rollout temperature 0.8/top-p 0.95、prompt 6144、模型上下文上限 8192、每 10 步保存。R18 原定 response 2048；R19/R20 为本机 RAM 故障临时改成 1024，不可混合比较曲线。run 内 `metrics.jsonl`、`chspo_safety.jsonl`、`chspo_duals.json` 和 `paper_figures/` 的单张 PNG/PDF/SVG 图分别记录 Utility、ASR、持久化、dual、zero-variance、回复长度、熵/概率及梯度等；绘图脚本 `scripts/plot_rl_paper_figures.py`。R18/R19/R20 的图都只有首步，不能用来宣称趋势或涨点。
- **judge、依赖与迁移**：AgentHarm 语义 judge 是 `configs/agentharm_rl_train_worker.yaml` 指向的 OpenAI-compatible `bigmodel/glm-5.3-flash`；密钥只经 `AGENTHARM_JUDGE_API_KEY` 或 gitignored `.env.rl_judge` 载入，绝不提交。`bootstrap_rl_sources.sh` 固定 VeRL/AgentDojo commit 并应用项目 patch；非零编号 GPU 的 Ray/CUDA 映射修复见 `patches/verl_nonzero_cuda_visible_devices.patch`。R20 本地临时 `VERL_PARK_OPTIMIZER_STATES_FOR_SYNC=1` 未验证有效，新服务器不要据此假定 RAM 故障已解决。A6000-2 应以**新 run-id** 重建环境/密钥/训练池，先通过两步在线更新和权重同步，再谈 300 步正式实验。

下文保留之前的 SFT、数据包和早期 RL 交接历史；涉及“当前运行中”“尚未交付多 harness loop”的时间性表述均以上述 2026-10-10 状态为准。

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

## 2026-10-07：塌缩诊断与奖励改造（r3–r8 的教训，已实施）

新服务器 300-step 系列 run 的实际死亡记录：r3 死于 AgentHarm 缓存 JSONL 并发读竞态（已修，`agentharm.py` 的 `fcntl.flock` 串行化）；r4（12 cells）跑到 step 9 后**整批零方差**被门禁停机；r5/r6 死于 unscorable episode（不可复位/不可评分）；r7（32 cells）8 步内零方差率从 0.44 爬到 0.81，最终死于 judge 子进程失败。**两次独立实验（r4、r7）收敛到同一形态：中性塌缩——策略组内行为趋同、组内优势消失**，扩批次只降低单步全零方差概率，不改变收敛方向。

由此确定的五条原则（按优先级）：

1. **过程方差治本**：terminal reward {-1,0,1} 太粗，结局相同的失败必须按过程分出差异。已实施：`score_episode` 的 path 通道现按 `path_weight × (verified_safe_subgoal_fraction − min(blocked_invalid_calls,1))` 计分，`path_weight=0.25`，且 **AgentDojo 的 subgoal 分数来自真实证据**——官方 ground-truth 工具计划（`_ground_truth_tools`）与实际成功 dispatch 的工具名求交，`|交集|/|计划|`；ground truth 不可得时诚实归零（AgentHarm 无 ground-truth 序列，subgoal 通道显式 abstain 为 0，由 judge 信号补位）。invariant 保持：path 通道不能把 incomplete 轨迹抬成正分、不能减轻 verified violation。
2. **堵死"稳定什么都不做"的不动点**：benign 臂（含 injection 的良性目标）目标失败、**零次有效工具 dispatch**、termination 为 completed/model_timeout 时，reward 记 `benign_nonengagement`，outcome 分量 −0.5，严格低于"尝试后失败"（0），且叠加 blocked-call 路径惩罚（如 1 次 blocked → −0.75）。verified violation（−1）仍是最差。AgentHarm 的 harmful 臂拒答是正确行为，不受此罚。
3. **监控三条联动曲线**：零方差率 ↑ + 平均回复长度 ↓ + 熵 ↓ 同时出现即塌缩进行时；journal 已按 arm 记录 `mean_path_signal` 和 `nonengagement_rate` 用于区分结局/过程贡献。
4. **批次大小治标**：r4（12 cells）与 r7（32 cells）死法相同，不要再以扩批次作为塌缩对策。
5. **消融顺序**：先解决塌缩（上述 1/2），再跑 CHS-PO、DAPO 等对照，否则所有方法死在同一个零方差门禁上，比较不出优劣。注意 CHS-PO 的 worst-stratum 加权只做组间再平衡，对组内零方差无效。

配套修改：奖励分解审计化——batch gate 现要求每条 rollout 携带 `reward_path_signal` 且 `reward = outcome + path_signal`（outcome ∈ {−1, −0.5, 0, 1}，|path_signal| ≤ 0.25），否则拒收；AgentHarm 语义 judge 已从 codex CLI（配额耗尽）切换到 OpenAI 兼容端点（`judge_backend: openai_compatible`，声学云路由 + `bigmodel/glm-5.3-flash`，key 走环境变量 `AGENTHARM_JUDGE_API_KEY`），并对推理模型的空 `content` 加了 `reasoning_content` 回退。相关测试已更新并通过（`tests/test_rl_verifier.py`、`tests/test_rl_native_agentdojo_episode.py` 等）。三个 RL 启动脚本（`train_rl_multiharness_verl.sh`、`train_rl_chspo_verl.sh`、`train_rl_dapo_clip_verl.sh`）会在 `AGENTHARM_JUDGE_API_KEY` 未设置时自动 source 本机 gitignored 的 `experiments/cross_harness_sft/.env.rl_judge`——重启 run 不再需要手工 source，密钥本身永不入库。密钥加载的完整规范见下节，所有现有与未来的 RL 入口都必须遵守。

## 2026-10-07：AgentHarm judge 密钥加载规范（spec）

**隐患记录（2026-10-07 发现并封堵）**：AgentHarm 语义 judge 依赖环境变量 `AGENTHARM_JUDGE_API_KEY`。若训练启动时该变量未设置，首个 AgentHarm 批次判分即抛 `Judge API key is unset` → episode 不可评分 → 整个 job 在第一次优化器更新前中止。r11 的启动当时依赖会话内手工 `source .env.rl_judge`，任何后续重启若遗漏该步骤就会复现故障。这是 judge 链路三天内第三次事故（r3 的缓存 JSONL 竞态、r7 的 codex CLI 配额耗尽、本隐患），因此固化为以下规范，对**所有现有与未来的 RL 训练入口**生效。

**规范**：

1. **启动即就绪（invariant）**：任何 RL 训练入口必须在 Ray/agent-loop worker 派生**之前**保证 `AGENTHARM_JUDGE_API_KEY` 已设置。三个启动脚本已实现守卫：变量未设置且 `$ROOT/.env.rl_judge` 存在时自动 source。新增任何 RL 入口（含 CHS-PO/DAPO/未来消融脚本）必须复制同样的守卫；**禁止依赖 shell 会话状态、`nohup` 外层环境或人工记忆**。
2. **密钥永不入库**：`.env.rl_judge` 已在 `.gitignore`；配置文件只允许 `api_key_env` 间接引用（如 `configs/agentharm_rl_train_worker.yaml` 的写法），不得出现字面 key。每次提交前检查：任何被跟踪文件中不得含 judge key 的任何片段（前缀、中间段、完整值）。文档只允许写环境变量名和密钥文件的路径，不允许写 key 内容。
3. **失败必须 fail-closed 且可诊断**：judge 缺 key、超时或空响应时按现状抛错、拒绝评分（unscorable → 门禁拒收整批），禁止静默记 0 分或跳过该 episode；日志保留 `Judge ...` 签名行供监控匹配（Monitor 的过滤模式已覆盖）。
4. **重启前自检**：重启任何 run 前，要么依赖启动脚本守卫（推荐），要么显式确认 `echo ${AGENTHARM_JUDGE_API_KEY:+set}` 输出 `set`；同时核对 `configs/agentharm_rl_train_worker.yaml` 的 `judge_backend` 与预期端点一致（当前为 `openai_compatible`，不再是 `codex_cli`）。

**验证记录（2026-10-07）**：三个启动脚本 `bash -n` 通过；离线端到端调用 `OpenAICompatibleJudge._generate` 返回带 `Answer:` 的规范判定；`git grep` 确认被跟踪文件中无 key 片段。后续新增 RL 入口时重复这三项检查。

## 2026-10-07：R11 在线 GRPO 运行中（14:19 UTC 快照）

当前新服务器使用 `scripts/train_rl_multiharness_verl.sh` 启动 `multiharness_grpo_pool107_20261007_r11_300steps`：VeRL 原生在线生成、全参数 GRPO，SFT step 267 初始化，2 GPU，32 个 train-family task groups/step、每组 4 个当前策略 rollout（128/step），计划 300 optimizer steps、每 10 步保存 checkpoint。入训的四个来源×harness 层为 AgentDojo/Codex、AgentDojo/Claude Code、AgentHarm/Codex、AgentHarm/Hermes；这**不是** v7 全部 8 harness 的正式覆盖。SafeClawArena 尚未接入本机 Docker runner。训练池位于忽略目录 `outputs/rl/multiharness_agentdojo_agentharm_pool_v5.parquet`（107 个合格 train cells，按 batch 分层抽样）；从私有 handoff 数据重建的代码是 `scripts/build_multiharness_rl_task_catalog.py` 和 `scripts/build_rl_multiharness_dataset.py`。训练池、私有原始数据、API key、checkpoint 和逐条 rollout 不放入公开 GitHub。

截至 14:19 UTC，R11 已真实完成 step 1–4，各步 32 groups/128 sampled/128 scored/0 unscorable，零方差组率分别为 0.375、0.3125、0.4375、0.40625；训练进程仍存活，尚无 step-10 checkpoint。该数字只证明训练机制在工作，**没有 held-out 效果结论**。可公开的聚合曲线快照见 `reports/r11_online_grpo_steps1_to4_20261007.{csv,png}`；本机完整 metrics、日志与未来 checkpoint 在 `outputs/rl/multiharness_grpo_pool107_20261007_r11_300steps/` 和同名 `.log`。新服务器不得把本机 R11 状态当成可恢复的远端 checkpoint；至少等 step 10 保存并单独转移权重。

R9 在首步之前被奖励分解门禁拒绝；R10 真实完成 2 步，第 3 步因 float32 `rm_scores` 与连续奖励的精确相等比较被错误拒绝。`rl_multiharness_update_guard.py` 已改用 `math.isclose`（绝对容差 `1e-6`），R11 使用修复后的代码。R8 完成 5 步但没有可用 checkpoint；不要把 R8–R10 的局部步数拼成 R11 的连续训练。

**16:23 UTC 交接更新：**R11 已完成 step 20 并保存 checkpoint。用户授权 Claude Code 停训、合并 step-20 权重并运行 `evaluate_rl_sft_pair.sh` 的家族隔离配对验证；此时权重合并正在运行。为免 Codex 将有意停训误判为故障而重启训练，R11 的自动修复 cron 已改为每分钟只调用 `scripts/check_rl_training_status.py` 记录状态。另一个 cron 每分钟调用 `scripts/wake_codex_on_rl_validation.py`：只有 `heldout_agentdojo_validation*/comparison.json` 及 SFT/RL 各 24 条逐条验证结果、step-20 FSDP checkpoint 均完整且稳定后，才启动一次独立 Codex 会话核查结果，决定是否能从 step-20 正确恢复训练，或先调整奖励/采样/更新策略。决策应记录在 R11 run 目录的 `step20_codex_decision.md`，不能把 24 个 AgentDojo episode 当作全 harness 结论。两个 cron 均是本机状态，不会随 Git 迁移；新服务器须按新 run-id 另行安装。

## 正式多 harness 在线 RL 尚未交付

GitHub 当前**只有 Hermes 协议映射 + AgentDojo 官方可复位工具环境的 16-cell 在线 pilot**。`configs/rl_native_pilot_agent_loop.yaml`、`scripts/rl_verl_native_agentdojo_loop.py` 和已提交的 `outputs/rl/native_pilot_train_20261005_hermes_mcp_16cells.parquet` 都只服务这条路径；在线 loop 的 harness 字段也是 Hermes。

正式 RL 的目标不能缩为 Hermes/NanoBot/OpenClaw。v7 接受集包含 8 个原始 harness 标签及 4 个源数据集，来源×harness 组合如下（计数是离线 SFT 接受轨迹，不是 RL rollout 数）：

| 源数据集 | v7 接受集中的 harness |
|---|---|
| AgentDojo | Codex 333 train + 64 validation；Claude Code 345 + 71 |
| AgentHarm | Codex 175 + 21；Claude Code 134 + 15；Hermes 175 + 22；NanoBot 211 + 24 |
| ActBench | `claudecode` 23 + 3；OpenClaw 27 + 3；OpenCode 30 + 6；QwenPaw 19 + 2 |
| SafeClawArena | OpenClaw 131 + 15 |

保留 `claude_code` 和 `claudecode` 为两个独立标签，直到分别核对其真实运行时、提示词、工具 schema 和权限后才决定是否归一。目标矩阵至少覆盖上述 8 种 harness 及其已接受的来源组合；能否把某个源扩到更多 harness，取决于该源的环境工具能否通过该 harness 的模型可见协议执行，不能按全笛卡尔积虚构覆盖率。

每个 RL rollout 都必须由当前 VeRL policy 在 harness 专属上下文和工具合同下新生成，并在对应数据集的 train-family 可复位环境里执行、评分；teacher transcripts 只能提供任务/来源，不能当 GRPO rollout。AgentDojo 的代码和 16-cell 数据已在 GitHub，但线上适配目前只有 Hermes。AgentHarm 工具环境、ActBench 服务及任务 fixtures、SafeClawArena 隔离执行环境都没有接入本项目 VeRL online loop；这些源分别还需建立 resettable train-family worker、真实副作用 verifier 和 harness 映射。NanoBot、OpenCode、QwenPaw、Codex、Claude Code 的正式 RL tool adapters 也未在此 pilot 中验收。`scripts/rl_group_sampler.py` 仅提供通用分层采样逻辑，不是这些环境的实现。因此**目前不存在可从 GitHub 直接拉取并启动的正式多 harness RL 数据包或启动脚本**。旧机 `data/raw/multi_harness_teacher_v1/` 和 `data/sft/teacher_release_v7/` 是离线教师/SFT 数据，不能代替当前策略在线 rollout 环境；HarnessRisk 保留评测 case 不能直接转成 RL train cell。扩展顺序应是按上述来源×harness组合逐项落地，完成协议、reset、正反例奖励和 family-disjoint 验证后，再将合格 cells 纳入 VeRL/CHS-PO 多 harness 采样。

### 旧服务器上的 split/fixture 传输包

所需输入包已上传至私有 Hugging Face 模型仓库 `Limax11/harness_alignment`，文件路径为 `transfers/handoff_multiharness_rl_data_20261007.tar.zst`，固定 revision `49d5041fba2373c3a195cfcc268aa01d25963784`（12,817,609 bytes；SHA-256 `71678b4ba1f9fb80b9a49754406ca9e60ab904c5c8fe5e17a13517955598f9d0`）。旧/新服务器均无需 SSH 互通；新服务器需先以有读取权限的 HF 账号登录，再执行：

```bash
hf auth login
hf download Limax11/harness_alignment \
  transfers/handoff_multiharness_rl_data_20261007.tar.zst \
  --revision 49d5041fba2373c3a195cfcc268aa01d25963784 \
  --local-dir ./hf_transfer
sha256sum hf_transfer/transfers/handoff_multiharness_rl_data_20261007.tar.zst
mkdir -p hf_transfer/extracted
tar --zstd -xf hf_transfer/transfers/handoff_multiharness_rl_data_20261007.tar.zst \
  -C hf_transfer/extracted
```

包内有 v7 `accepted_train.jsonl`、`accepted_validation.jsonl`、完整 `family_split_manifest.jsonl`（246 train、30 validation、30 test、7 review-only families）、分类/预处理元数据和 Qwen3.5-2B 4,096-token qualified JSONL（1,421 train、232 validation）；也有 ActBench 与 SafeClawArena 的提交版 tasks/fixtures/runner 源码快照。ActBench commit 为 `31bd732e9c083ddeb19bf152048386d38e511c90`，SafeClawArena commit 为 `a11f5cceaba0676be721021f8d232638fd111305`。test 家族只给 split 身份用于排除；没有 `accepted_test.jsonl`，也不应将其用于训练。`TRANSFER_MANIFEST.json` 列有包内文件大小和 SHA-256。包不含原始采集日志、密钥或未提交的本地代码改动。

## 新服务器复现与继续工作的顺序

1. 克隆本仓库的 `handoff/rl-20261006` 分支（或其合并后的 `main`），确认读到本文件、实验 README 和 `experiments/cross_harness_sft/scripts/rl_*`。代码已移除被误纳入旧 Git 历史的虚拟环境；本分支也不包含 checkpoint、原始轨迹和 API key。
2. 在兼容 CUDA 12.9 的服务器上建立 `experiments/cross_harness_sft/.venv-rl-pilot`。旧机实测核心版本：Torch 2.11.0+cu129、vLLM 0.23.0+cu129、Ray 2.58.0、Transformers 5.9.0、PEFT 0.21.1、Hydra 1.3.7。执行 `bash experiments/cross_harness_sft/scripts/bootstrap_rl_sources.sh`，它固定 VeRL `02a318c303a7b60871cb63e4ce2f779b522874d9`、AgentDojo `089ed468cf3ed0322acc66b0211f26d9d90dbf60`，并应用 `patches/verl_project_local_02a318c3.patch`（含 optimizer 状态按需回迁等本地修复）。安装两者到 RL 环境并确认 Hermes `5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04` 的 Python 环境；按需设置 `HERMES_AGENT_ROOT` 和 `HERMES_AGENT_PYTHON`。`rl_native_pilot_agent_loop.yaml` 的 worker 路径已改为相对路径。
3. 从私有 Hugging Face 模型仓库 `Limax11/harness_alignment` 下载 **step 267 的 4.2 GiB SFT 合并 checkpoint**，固定 revision `7b608bf7517bd8b2f08ec8eecedd6dea4bdbbe26`，放到仓库相对目录 `experiments/cross_harness_sft/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267`；新服务器先以有权限的 Hugging Face 账号运行 `hf auth login`，再运行 `hf download Limax11/harness_alignment --revision 7b608bf7517bd8b2f08ec8eecedd6dea4bdbbe26 --local-dir experiments/cross_harness_sft/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267`。GitHub 不托管该权重。16-cell 训练 Parquet 已随本分支上传到 `experiments/cross_harness_sft/outputs/rl/native_pilot_train_20261005_hermes_mcp_16cells.parquet`。若需要复现 SFT 数据构建，还须另行转移忽略的 `data/raw`、`teacher_release_v7` 完整数据和源 benchmark；不能用 Git 中 manifest 伪装成数据本身。
4. 先核查 Hermes MCP 转换、AgentDojo 真工具执行及 reset/path audit；解决 `verified_safe_subgoals` 的真实证据来源，并以固定安全/不安全反例核对奖励边界。然后在两张空卡上用**新的** `RL_RUN_DIR` 做 2048-token、16-cell、G=4 的小批在线更新；入口是 `scripts/train_rl_native_pilot_verl.sh`，设 `CUDA_VISIBLE_DEVICES=<两卡>`、`RL_N_GPUS=2`、`RL_GPU_UTIL=0.45`、`RL_MAX_NUM_SEQS=2`、`RL_RESPONSE_LENGTH=2048`、`RL_TRAIN_BATCH_SIZE=16`、`RL_DATA=<上面的 Parquet>`。不得复用已有 run 目录；若批次无有效组内优势，保留门禁停机。
5. 每次保存 checkpoint 后，用 `scripts/evaluate_rl_sft_pair.sh` 对同一家族隔离验证集配对比较 SFT/RL 的 Utility、ASR、invalid calls、超时及真实副作用；保存 JSONL、metrics 和曲线。只有验证同时守住安全边界与正常任务效用，才扩大任务池或开始 CHS-PO、Balance-GRPO、DAPO 等消融。不要把训练 reward 上升当作能力涨点。

旧机 SFT 和 RL checkpoint 均未上传 GitHub；**只有 SFT step 267** 已上传至上述私有 Hugging Face 仓库，RL checkpoint 尚未上传。若从旧机迁移其他权重，可用 `rsync --partial`；先检查目标磁盘、模型文件大小和新服务器 GPU 驱动，再运行全参数在线更新。所有外部服务密钥应通过新服务器环境变量配置。
