# CHSPO (r18) 设计评审 —— 启动前检查

评审人：Claude Code，2026-10-09。评审对象：`scripts/rl_guarded_chspo_trainer.py`、`rl_native_chspo_task_runner.py`、`train_rl_chspo_verl.py`、`train_rl_chspo_verl.sh`（13:30 版本）。目标：吸取 R11 GRPO step-20 教训（三 harness utility 与 ASR 同涨），确保 CHSPO 不会复现中性塌缩或安全越界。

**结论：设计方向正确（per-harness dual、配对臂强制、advantage 级安全代价、fail-closed 门禁、2048 预算、10 步 checkpoint、双卡守卫、与 R11 同池同批可比），但有 1 个高危问题必须在启动前修复，2 个中危问题强烈建议同批修复。**

## P1（高危，已数值复现）：protocol 成本通道会在 advantage 层重造中性塌缩

**问题。** `constrained_advantages` 的 protocol 成本把 `blocked_tool_calls`（双臂都计）与 `invalid_tool_dispatched`/`protocol_error` 一起计入，预算 0.01、`DUAL_STEP=0.25`。R11 实测协议违规率高达 51.6%（AgentDojo/Codex 风险臂），即初始 rate 远超预算 → protocol dual 每 step 上涨 `0.25×(rate−0.01)`，约 10 步内冲到 `DUAL_MAX=1.0` 并饱和。此后**每条带 blocked call 的轨迹在 advantage 层被压到 −1.0 以下**，而组内"干脆不做"的轨迹（outcome −0.5，无 cost）只保留 GRPO 中心化后的负优势（≈−0.3）。

**数值复现**（调用真实的 `constrained_advantages`，良性 cell 组内 G=4）：

| 轨迹 | GRPO adv | 初始 dual | protocol dual=1.0 |
|---|---:|---:|---:|
| 尝试但被 block | +0.19 | −0.28 | **−1.00** |
| 尝试但被 block | −0.06 | −0.34 | **−1.06** |
| 不做（timeout） | −0.31 | −0.31 | −0.31 |
| 尝试且安全 | +0.19 | +0.19 | +0.19 |

奖励层的家规不变式（AGENT.md 原则 2："尝试后失败(0) 严格高于不做(−0.5)"）在 advantage 层被**反转**：梯度明确偏好不参与。这正是 r4/r7 中性塌缩的方向，且机制更强（r4/r7 是组内无差异，这里还主动倒挂）。后果预测：ASR/persistence 确实会降——因为模型不动手——但 utility 同崩，"utility 大幅提分且 ASR/persistence 明显下降"两个目标只能达成一半。

**修复建议（组合采用）：**
1. `blocked_tool_calls` 移出 dual 成本（或仅 risk 臂计入）；保留 `invalid_tool_dispatched` + `termination=="protocol_error"` 作为 protocol 成本。
2. 对 `benign_nonengagement` 在 advantage 层做对称的非中心化惩罚（同样 `clamp(max=0) − 固定 dual`），保证"不做"的相对优势永远 ≤ "尝试后失败"，把原则 2 从奖励层贯彻到更新层。
3. `DUAL_STEP` 降到 0.05–0.1；预算按 SFT 基线实测 rate 校准（protocol budget 从 0.01 提到 ~0.10 量级）。当前每 harness risk 样本仅 32 条，rate 噪声 σ≈0.04，比 attack/persistence 预算 0.02 大一倍——dual 会在噪声驱使下随机漂移，"自适应"名不副实。

## P2（中危）：dual 按 harness 键控，跨 benchmark 混算成本率

`by_harness[item["harness"]]` 使 `codex` 一个 dual 同时混合 AgentDojo 与 AgentHarm 的成本率。家规明确"AgentHarm 有害臂成功判定与 AgentDojo 注入效用/ASR 含义不同，不可跨数据集直接合并或比较"——混合 rate 驱动的 dual 更新正是这种合并。且基类 journal 的 strata 就是 `(benchmark, harness)`。**建议 dual 键改为 `(benchmark, harness)`**（8 个键，与 `EXPECTED_STRATA` 对齐），一行改动。

## P3（中危）：安全越界没有硬门禁

dual 是软约束，越界时训练照常继续。`chspo_safety.jsonl` 已逐步记录 rates/duals，加硬门禁很便宜：任一 `(benchmark, harness)` 的 risk 臂 `attack` 或 `persistence` 率连续 3 步超过硬上限（建议 0.15）→ 抛错停机（与项目 fail-closed 风格一致）。否则最坏情形是训练 100 步后才发现安全层已崩。

## P4–P6（建议与确认项）

- **P4 归因对照**：CHSPO 相对 R11 同时改了两个变量（dual 安全约束 + budget 1024→2048）。若算力允许，同步跑一个 GRPO-2048 对照（同池同批同预算、去掉 dual）到 step 20，与 R11 评测对齐，否则论文里 2048 的贡献无法与 CHSPO 奖励分离。
- **P5 checkpoint 选择流程**：`save_freq=10` ✓，但"只从验证家族选 checkpoint"目前无自动化。启动前应定死选择规则（utility ↑ 且 ASR/persistence 不劣于 SFT 才入选）并把每个 checkpoint 的 `evaluate_rl_sft_pair.sh` 排进计划。
- **P6 行序断言**：`constrained_advantages` 只校验 evidence 数量与 advantage 行数一致，未校验行序。admission gate 已在 R11 验证 reward 对齐，风险低，但加一条 cell_id 顺序断言是一行代码。

## 已确认无问题的项

- 用户指出的"12 cells / 12 steps"限制已在当前版本解除：runner 校验 1..300 步、batch = 8 strata × 4 组 = 32 组 × G=4 = 128 条/步，与 R11 完全可比 ✓
- 2048 response + 6144 prompt = 8192 `max_model_len` 精确匹配；expandable_segments / optimizer offload 按 24GB 卡经验配置 ✓
- `AGENTHARM_JUDGE_API_KEY` 启动守卫 ✓；run 目录 `metrics.jsonl` 防覆盖 ✓；路径奖励 `path_weight=0.25` + nonengagement −0.5 沿用 ✓
- GPU：评测 step-20 占卡已释放（tmux 全空，GPU 0/1 仅 ~20MiB）；GPU 2/3 为他人进程，脚本默认 `CUDA_VISIBLE_DEVICES=0,1` 且有 >256MiB 预占拒启检查 ✓
- 配对臂强制（`unpaired harness batch` 拒收）、LoRA 禁用、SFT step 267 初始化快照校验 ✓

## 放行条件

修复 P1（三条组合建议至少落实 1+2）并处理 P2/P3 后即可启动。启动后前 10 步监控三条塌缩曲线（零方差率↑ + 平均回复长度↓ + 熵↓）与 `chspo_safety.jsonl` 的 dual 饱和情况；step-10 checkpoint 即做一次验证家族配对评测，不要等到 step 20。
