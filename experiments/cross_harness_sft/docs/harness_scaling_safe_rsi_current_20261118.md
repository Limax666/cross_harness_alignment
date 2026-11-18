# Harness Scaling 安全 RSI：当前执行规格

本文件和 [跨服务器交接](new_server_handoff_20261118.md) 是 2026-11-18 的执行入口。较早文档中的 AgentDojo 主训练、小池、SkillLife 主池、75/25、GPU0 固定分工及“尚未启动”均属于历史方案。

## 三阶段与当前边界

1. FlashRSI × HarnessRisk：固定官方 4B 权重，对比每个真实 harness 的初始 Snapshot 和六种 RSI 算法选出的 Snapshot。RSI 主要修改执行资源，不等于训练模型参数。保留同协议初始/演化对照；选回初始须明确写 `H_r=H_0`。OpenCode 没有 skill 组件，SkillOpt/Evo-Harness 记 N/A。
2. Harness Scaling 参数 RL：一个统一白盒执行框架拆成 S 系统指令、T 工具接口、O 观察、C 上下文管理、W 工作流，各三实现，OA18 strength-2 配置；同任务 K=2 配置成员分别用当时策略重新采样。当前实际运行的是 GRPO-S、单卡 LoRA 4B。真实 Codex/OpenCode/Hermes 产品不充当这 18 个白盒配置。
3. 固定同一 FlashRSI Snapshot，替换 RL 前后模型，评估安全、合法效用和未见 harness 泛化。要分离资源演化与参数学习的贡献。

CHS-PO、DAPO、Dr.GRPO、多 seed 和 9B 确认仍属于后续对照，不能把当前 GRPO-S 代码/数据叫作已经完成这些算法。当前参数训练从官方 4B 权重开始，不是此前 SFT 权重；FlashRSI 资源不会被无记录地拼入白盒执行器。

## 当前训练池与采样

| 项目 | 冻结设置 |
|---|---|
| 总池 | 14,000 个 task-condition，真实发布为 READY |
| 通用 | ClawGym-Task 7,000；不是 ClawGym-Bench 测试集 |
| 安全 | AgentDoG 7,000＝attack 5,500＋matched clean 1,000＋query_target 500 |
| 主配比 | 每完整更新 8 个通用组＋8 个安全组，即 50:50；不能用尚未采完的 partial 比例判断偏离 |
| 每更新 | 16 组 × G=8＝128 条真实交互轨迹 |
| K=2 | 相邻两个更新用同 16 任务、不同 OA18 配置、各自当前策略、全新环境；每个成员独立采样，禁止复用旧策略轨迹 |
| manifest SHA256 | `220521662b8c40127929dbe07ad74a631d2d564a700954d1c1382e0615373c99` |

源角色比例由冻结 sampler 在多个配置块中展开，不保证每一个安全半批都含全部三种角色。原始来源 pin 和官方 grader pin 见 `configs/safe_rsi_source_pins_20261118.json` 与 `scripts/hsrl_v5/release.py`。训练/开发/测试按既有 family/component 划分，禁止查看保留集后重新选训练家族。

## 模型、奖励与预算

- 官方模型：`Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；逐文件独立 authority 在 `configs/hsrl_official4b_851bf6e.json`。
- BF16 基座冻结，语言分支 LoRA r16/alpha32/dropout0；32,464,896 个 FP32 可训参数，496 个张量。reference 是禁用 LoRA 的同一不可变官方基座。
- GRPO-S：`U_app - C_source_app - 0.1*C_protocol`。组件是否适用取官方源掩码；不适用不是“评分失败”。适用但缺失、非有限或基础设施失败不能计零。GRPO-U、CHS-PO 向量适配等定义见 `hsrl_v5/reward.py`，不能混用旧奖励契约。
- Claw utility 为官方代码 checker 部分；AgentDoG 的 source cost 是注册工具调用攻击目标，不是实际伤害/persistence/LLM judge ASR。RL 训练不调用付费 LLM judge。HarnessRisk 系统评测的 judge 是另一条流程。
- 16K context、单次生成 2,048、每 episode 8,192、12 rounds、30 tool steps；8 个独立环境，模型 batch 最多 8，总 padded-token budget 32,768。
- AdamW lr `1e-5`，beta `.9/.999`、eps `1e-8`、weight decay0；warmup2 次真实更新，PPO clip `.2`、KL `.001`、梯度 clip1。以当前 `train_hsrl_v5.CONFIG` 和 run manifest 为准。
- seed267，最多 38,400 次累计尝试/300 次更新；失败及隔离开销不清零。当前源修复消耗了 120 次未用于更新的旧 partial，逐更新预算上界因此为299，完整 K2 末端还须单独审计，禁止为凑300暗中加预算。

## 已实测与不能声称的结论

最近完整独立验收的是 update6：128/128 源评分，496 个 adapter 张量变化，loss `-0.0275005330`，preclip norm `19.8418350`，从实际 AdamW moment 重建裁剪后 norm `0.999999903`。update5/6 构成同16任务的完整 K2 对，512 个工具/grader容器独立创建并清理。两个更新各耗时97.35/113.57分钟。

发布时点的状态快照见 [源服务器观察记录](../reports/hsrl_source_handoff_observation_20261118.json)。它记录 live counter，不是停机交接证明；该时点已观察到 update7，而本次交接收录的最近完整小时验收为 update6。后续 v9 在第 10 步采样时中断，已校验第 9 步 checkpoint。恢复策略和最新验收口径见 [v10 在线采样故障恢复](hsrl_online_rollout_recovery_20261118.md)。

旧 v8 因受限 source namespace 缺 `ValueError/TypeError/Exception` 停止；v9 使用 `hsrl_source_repair_v1.adapter.Dataset`，只补这三种普通异常名，原 grader/奖励/沙箱不改。旧112条 valid 未更新＋8条 invalid/aborted共120次尝试保留隔离；从已接受 update4 的权重/AdamW/scheduler及已对账的最终RNG继续，整步重新采128。原失败组正式8/8评分、9次 get_patient_info 与后续参数更新均已验收。

这些证明真实在线 RL、评分隔离和参数更新可运行，不能据训练集分数宣称安全/通用能力提升。原历史梯度尖峰根因尚未证明；保留诊断，不因后续有限梯度就说已修复。旧较宽 QA 回归缺最终回执，未记通过；14项定向 JUnit 和真实 API/正式轨迹/优化器证据保留。

FlashRSI 初始和演化对照、规则与 judge 完整覆盖要分别统计；不能把 attempted、进程存活、部分覆盖、规则评分或选中 Snapshot 本身当成全量验收。跨硬件、路径、harness协议变化另记 cohort，不拼接旧污染结果。
