# 论文研读汇报：Harness Scaling 对跨 Harness Agent 安全对齐的启发

> 研读日期：2026-10-25。论文：Fei Bai 等，*Harness Scaling for Generalizable Agentic Reinforcement Learning*，arXiv:2610.29458v1，2026-10-21；当前为预印本，研究类型为实证方法论文。[论文 PDF](https://arxiv.org/pdf/2610.29458) · [作者项目页](https://github.com/RUCAIBox/Harness-Scaling)。本稿按 [academic-paper-review skill](https://www.skills.sh/bytedance/deer-flow/academic-paper-review) 的“主张—证据—方法—局限”框架复核；以下数值均来自论文原表，与本项目的联系单独标为“我们的判断”。

## 一页结论

这篇论文与我们**同样把 harness 当作 RL 泛化的核心变量**。它证明：在固定训练预算下，有组织地改变执行协议，比只在一个固定 harness 中做 RL，更能提高未见 harness 的**任务完成率**。但它的优化目标和主要指标是 Pass@1，没有训练或报告攻击成功率（ASR）、恶意持久化及良性效用之间的安全权衡。因此它是我们跨 harness RL **采样设计的直接相关工作**，不是 CHS-PO 安全约束奖励的先例，也不能用其 Pass@1 结果替代我们的安全结论。[论文 §3–4、Table 1–2](https://arxiv.org/pdf/2610.29458#page=3)

对论文定位最重要的一点：**“首次提出跨 harness RL”已不适合作为我们的创新表述**。我们应明确贡献为：在真实工具及副作用可审计的跨 harness 在线环境里，同时优化良性任务完成与攻击/持久化约束，并用未见运行 harness 检验安全泛化。这是基于两项工作目标差异的研究定位，不是已经验证的效果。

## 论文究竟做了什么

作者将 harness 拆为五个正交设计因素：系统提示 `S`、工具接口 `T`、工具观测 `O`、上下文管理 `C`、执行工作流 `W`。每个因素有 3 种实现，组合成 **3⁵=243** 个白盒执行配置。变体包括提示表达方式、工具别名/不透明名称及干扰工具、观测元数据、不同程度的对话压缩、先规划或周期性进度核对。作者要求变体保持任务和环境语义不变；具体变体由 Opus-4.7 辅助生成。[论文 §3.1、附录 A/Table 5](https://arxiv.org/pdf/2610.29458#page=15)

全量跑 243 种组合成本高，也可能稀释每个组合的训练次数。作者用 **strength-2 正交设计选 18 个训练配置**：每个因素的三个水平各出现 6 次，任意两个因素的每种水平配对各出现 2 次。另选 18 个完整组合供白盒留出测试。这里的“未见”指**完整因素组合未见**，不表示其每个工具或组件水平都未见。[论文 §3.2、附录 B](https://arxiv.org/pdf/2610.29458#page=16)

训练时，对每个任务随机选一个锚点 harness，再选结构上最近的配置；主实验 `K=2`，让**同一个任务在连续两个训练步骤**分别经过两个相近但不相同的 harness。邻近度是五个因素的 Hamming 距离。全局随机锚点保证覆盖，局部邻近配对控制单任务的变化幅度。策略优化仍是标准 PPO/GRPO 裁剪目标，损失只作用于策略生成 token；**Harness Scaling 改的是任务×harness 的在线 rollout 分布，不是提出新奖励或新优势函数**。[论文 §3.3、式 (6)–(7)](https://arxiv.org/pdf/2610.29458#page=5)

| 实验配置 | 论文设置 |
|---|---|
| 模型 | Qwen3-4B-Instruct、Qwen3-30B-A3B-Instruct |
| 训练任务与算法 | ClawGym-Task；GRPO、PPO；300 次更新 |
| 每次更新 | GRPO：16 任务×8 rollout；PPO：128 任务×1 rollout |
| 训练资源与关键参数 | VeRL + 异步 vLLM；4B 用 16 张 H20、30B-A3B 用 32 张 H20；actor LR `1e-6`，裁剪 0.20/0.28；作者未加 KL/entropy 正则 |
| 泛化评测 | 与训练任务不重叠的 ClawGym-Bench、JobBench-Easy、PinchBench；18 个未见组合，外部 Claude Code，补充 OpenClaw；每任务 3 次独立评测后报告平均 Pass@1 |

以上是该论文的资源与目标设置，**不是我们 CHS-PO 应照抄的参数**；我们曾出现高零方差组和输出退化，不能因为本文不使用 KL/entropy 正则就移除现有训练稳定性检查。[论文 §4.1、附录 C/Table 7](https://arxiv.org/pdf/2610.29458#page=17)

## 核心结果：哪些证据支持了什么

| 比较，均为 Pass@1 | Single-harness RL | Harness Scaling | 增量 |
|---|---:|---:|---:|
| 4B + GRPO，六列总平均（3 benchmark × 白盒/Claude Code） | 29.81 | 39.04 | +9.23 pp |
| 30B-A3B + GRPO，六列总平均 | 37.29 | 42.78 | +5.49 pp |
| 4B + GRPO，ClawGym-Bench 的 18 个未见组合 | 37.33 | 52.78 | +15.45 pp |
| 30B-A3B + GRPO，ClawGym-Bench 的 18 个未见组合 | 48.12 | 58.60 | +10.48 pp |

这是**同模型、同算法的单 harness 与结构化多 harness**对比；原表还显示 PPO 在两个模型尺度上也有提升。[论文 Table 1](https://arxiv.org/pdf/2610.29458#page=7)

对完全外部的 **Claude Code**，另一组基线是在 OpenClaw 内训练的黑盒 SFT/RL。4B 的三个 benchmark 平均 Pass@1 是 OpenClaw 黑盒 SFT **35.08**、黑盒 GRPO **33.38**、白盒 Harness Scaling GRPO **37.22**；30B-A3B 分别为 **38.08 / 36.28 / 41.20**。论文也报告未参与白盒训练的 OpenClaw 上，相比单 harness RL，4B 平均 **32.02→36.07**，30B-A3B **36.66→41.20**。这些是**任务能力的外部迁移**，没有提供攻击安全指标。[论文 Table 2、附录 Table 9](https://arxiv.org/pdf/2610.29458#page=8)

消融说明“组织方式”比只增加数量重要：4B 的 ClawGym-Bench 白盒/Claude Code 成绩，随机选 18 配置为 **47.90/42.78**，随机任务分配为 **50.46/43.17**，正交选集加邻近分配为 **52.78/45.37**。`K=2` 好于 `K=1` 和相同配置重复两次的 Repeat-2；`K=4` 没有进一步收益。243 配置的全量训练结果不能叫留出泛化：那一组训练**包含**白盒测试配置，只是固定预算压力测试。[论文 Table 3、Figure 3、附录 Table 10](https://arxiv.org/pdf/2610.29458#page=8)

作者另用 GPT-5.4 对完整轨迹评分六项执行能力：指令遵循、接口处理、观测读取、状态跟踪、工作流遵循、错误恢复。这有助于解释 Pass@1 变化，但仍是执行能力诊断，**不是安全检测率**。本稿作者 GitHub 项目页在研读时只公开 README 和图片，尚未看到可直接运行的训练实现；复现细节主要依赖论文及附录。[论文 §5、附录 D；作者项目页](https://arxiv.org/pdf/2610.29458#page=20)

## 按论文评审框架核对：主张、证据与限制

| 论文主张 | 直接证据 | 我们对证据强度的判断 |
|---|---|---|
| 结构化多 harness RL 提高未见组合的任务表现 | Table 1：两个模型尺度、GRPO/PPO、三个 benchmark、18 个留出组合；同预算单配置对照 | **较强**，覆盖算法和尺度；但只在作者构建的白盒因素空间内证明组合泛化 |
| 白盒训练能迁移至真实外部 harness | Table 2 的 Claude Code、附录 Table 9 的 OpenClaw；前者还与 OpenClaw 黑盒 SFT/RL 对照 | **中等**，有两种外部 harness，但不能外推到任意权限/工具架构或安全任务 |
| 收益来自有组织的变化，而不只是更多样本 | Table 3 的 Random-18/Random Assign；Table 10 的 `K=1`、Repeat-2、`K=2/4` | **较强**，控制了部分替代解释；训练时间、token 成本及重复实验方差仍需更完整披露 |

**优点。** ① 方法把“换 harness”拆成可操作因素，且用正交选集控制一阶与二阶覆盖（§3、附录 B），比只枚举产品名更适合做归因；② 有随机选集、随机分配、Repeat-2、不同 `K` 和两种 RL 算法的消融（§4.3、附录 C.3），支持“同任务变化”的额外价值；③ 单独测试未见组合和 Claude Code/OpenClaw，明确区分白盒组合泛化与外部 harness 迁移（§4.1–4.2）。这些是对**任务能力**主张的优点。[论文 §3–4、附录 C](https://arxiv.org/pdf/2610.29458)

**局限与建议。** ① 五因素变体由模型辅助生成；论文描述了保持语义不变的约束，但仅凭描述难以独立确认每种工具别名、干扰工具和上下文压缩都严格等价。建议公开配置、生成提示和自动等价检查，以界定“只改变 harness”的构造效度（附录 A）。② 三次评测给出均值，但主结果表未报告置信区间或逐配置方差；在多 benchmark、多个 held-out 配置下，建议提供配对差值及区间，展示最差配置是否也受益（Table 1–2）。③ 作者项目页研读时只有 README/图片，尚无可运行训练代码；相较 16/32 H20 的资源需求，复现门槛仍高。建议公开配置、采样器、版本化任务池及计算量/墙钟成本（附录 C）。这些限制**不是**“未研究 ASR”的同义批评：安全性本来就不是本文声称解决的问题。[论文附录 A/C；作者项目页](https://github.com/RUCAIBox/Harness-Scaling)

| 内部方法学核对项（1–5） | 判断 | 依据 |
|---|---:|---|
| 技术正确性 | 4 | 因素空间、正交覆盖与更新目标自洽；语义等价需外部审计 |
| 新颖性 | 4 | 新在结构化配置选择及同任务邻近分配，而非发明跨 harness RL 或新 PPO/GRPO 目标 |
| 实验设计 | 4 | 同预算、多尺度/算法、随机与重复曝光对照；外部 harness 数量有限 |
| 统计呈现 | 3 | 三次独立评测均值可用，但主表缺区间和逐配置波动 |
| 可复现性 | 3 | 论文给出配置/超参数和 18 个训练配置；研读时作者仓库未见可运行代码 |
| 资源可扩展性 | 3 | 18/243 降低组合数量；但 300 更新用 16/32 H20，未量化 token/墙钟节省 |

此表是**本项目内部研读打分，不是会议审稿意见或论文官方评分**。总体判断是：针对“跨 harness 任务能力”，这是有较完整控制实验的方法工作；针对“跨 harness 安全对齐”，它提供了采样假设，效果仍须我们自己验证。

与其直接相关的前序工作需要讲清：作者团队的 [ClawGym II](https://arxiv.org/abs/2608.16798) 已提出黑盒 harness RL 与 mix-harness 训练；[Polar](https://arxiv.org/abs/2605.24220) 和 [Agent Lightning v1.0](https://arxiv.org/abs/2608.17528) 重点解决任意 harness 的模型调用代理、轨迹重建和训练接线。本论文的新增点是**在同一任务/环境下怎样选择和分配 harness 变化，以提升未见配置的任务泛化**。因此介绍我们的研究时，同样不能把“能在多个 harness 上跑 GRPO”当作独占贡献。

如需向作者追问，最有价值的三个问题是：① 是否可提供全部变体的语义等价测试与工具协议错误率？② `K=2` 中最近邻并列时如何选取、会否改变成对任务的曝光分布？③ 白盒与外部 harness 的配对提升在逐配置最差分位数、token/计算预算下是否仍稳定？这三项分别对应构造效度、采样可复现性和泛化稳健性。

## 与我们项目的关系：重合点、空白和可借鉴项

| 维度 | Harness Scaling | 我们当前 2B SFT→GRPO/CHS-PO |
|---|---|---|
| 优化目标 | 未见执行配置上的任务 Pass@1 | 良性 Utility，同时限制攻击成功、禁止持久化和协议违规 |
| harness 空间 | 同一任务环境上的 `S/T/O/C/W` 可控变体，18 个训练配置 | AgentDojo、AgentHarm 等真实任务环境；Codex、Claude Code、Hermes 等实际协议/工具路径；已见与未见 harness 必须严格标记 |
| RL 改动 | 改训练配置选取及同任务跨配置分配；原 PPO/GRPO 目标不变 | 分层良性/风险配对、可验证结果奖励、按 benchmark×harness 的安全成本与 dual、审计/硬门禁 |
| 证据 | ClawGym/JobBench/PinchBench Pass@1，白盒组合和外部 Claude Code/OpenClaw | HarnessRisk 的 Utility/ASR/Persistence/Detection；SafeClawArena 未见 NemoClaw/SecLaw 的攻击安全迁移诊断 |
| 仍缺 | 没回答能力提高时 ASR/持久化会否恶化 | 缺同预算的结构化 harness 采样消融，未见 harness 安全泛化已有负面结果 |

**我们的判断 1：采样层值得借鉴。** 目前 RL 的“多 harness”主要按来源×harness 分层抽任务；它并未系统保证**同一任务**在两个语义等价但接口不同的 harness 下接受在线更新。可先建立真实可复位的跨 harness 配对任务，再比较“单 harness、随机多 harness、结构化多 harness × 标准 GRPO、CHS-PO”。固定模型起点、训练任务家族、rollout/token 预算和评测 seeds，才能把采样收益与安全约束收益拆开。

**我们的判断 2：不能直接把 18 个白盒变体当 18 个真实 harness。** Hermes 与 NanoBot 的工具权限、mock 服务、状态路径和恢复流程可能实际不同。只有经 schema、权限、任务结果和副作用验证**语义等价**的变体，才适合该论文的同任务邻近配对；否则变的是任务/安全边界，而非纯 harness 表达。特别是工具名/参数扰动不能交给模糊“自动修复”掩盖：我们曾在 Hermes `skill_view` 见到缺必填参数循环。

**我们的判断 3：安全目标必须单独保留。** 我们的 GRPO step-20 在 HarnessRisk 三种 harness 上 Utility 与 ASR 同涨；CHS-PO step-150 对 GRPO 压低了三种 harness 的 ASR，但相对 SFT 仍有部分安全回退，且 SafeClawArena 的未见 NemoClaw/SecLaw 上 ASR 从 SFT 的 **7.8%→23.3% / 0%→8.9%**。因此，不能以本文的 Pass@1 上升推断我们的方法能自动获得安全泛化。下一版应在按 harness 记录良性 Utility、ASR、持久化、协议错误及最差层表现的前提下，测试结构化采样是否扩大有效奖励组，并由独立留出集决定是否有净安全收益。[本项目 HarnessRisk 对照](./harnessrisk_sft_grpo20_chspo150_comparison.md) · [SafeClawArena 对照](./safeclawarena_heldout_sft_vs_chspo150_20261015.md)

## 可向老师汇报的下一步实验

1. **先冻结暴露矩阵。** 明确 `H_SFT`、SFT 未见但 RL 可适应的 `H_adapt`、SFT/RL 都未见的 `H_final`；家族层面隔离 train/validation/test。现有 NemoClaw/SecLaw 结果已经用于诊断，不能再无条件当最终盲测。
2. **做最小的采样消融。** 对已通过语义等价与真实工具审计的任务，比较单配置、随机两配置、邻近两配置；固定 RL 算法和总 rollout 预算。先用 GRPO 测同任务跨 harness 变化是否提升 Utility、减少协议错误并改善奖励组差异，再换 CHS-PO 检查安全约束是否抑制 ASR/Persistence。
3. **预注册安全验收。** 在相同 case/seed 的原生 harness 运行中，每题至少 3 次；分别报良性 Utility、ASR、禁止持久化、Detection（仅 benchmark 提供时）、超时和协议错误，并单列每个未见 harness。若 Utility 涨而 ASR 也涨，判为安全泛化失败；不能以全局平均掩盖某一 harness 的回退。

**汇报时的一句话定位：** Harness Scaling 说明“有结构的 harness 多样性”能提高 Agentic RL 的执行能力泛化；我们的研究要解决更难的追加条件——当模型在新 harness 中更会行动时，如何仍让攻击成功与持久化风险下降。
