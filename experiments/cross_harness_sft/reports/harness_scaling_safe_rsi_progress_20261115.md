# Harness Scaling 安全 RSI：实验设置与评测进展

数据核查截至 2026-11-15 20:50（北京时间）。当前得到的是 **4B 模型在 FlashRSI × HarnessRisk 下的初始及资源演化安全基线**，CHS-PO 权重训练与三阶段闭环尚未形成可报告结果。

## 实验设置

研究目标是检验：模型在多种执行 harness 下递归改进时，通用任务效用和安全性如何变化；再以跨 harness 在线强化学习 CHS-PO 改善安全与效用，并在未见任务家族和 harness 上检验泛化。总体技术路线分为：Phase 1 构建可复位任务和五维 harness 配置；Phase 2 在配置间在线交互并更新模型；Phase 3 将内部诊断反馈用于下一轮任务与配置生成。当前 FlashRSI 基线只评测固定模型下的 Snapshot 演化，并不等同于上述完整闭环。

本轮模型为 Qwen3.5-4B-Instruct，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；FlashRSI revision `778e2d01a995491fe63db210dcf46db1723a4430`。安全评测使用 HarnessRisk 全部 128 个 case；演化使用其中 train/dev，最终 128 条因此包含已暴露 case，不是独立未见测试。规划七种 harness：MiniAgent、OpenCode、OpenClaw、Hermes、Pi、Claude Code、Codex；每个比较初始 Snapshot 和 OPRO、GEPA、EvoPrompt、ACE、SkillOpt、Evo-Harness。OpenCode 上游 Snapshot 无 skill 组件，后两种标为 N/A，共 40 个适用算法单元。

本机使用受限演化预算，并非 FlashRSI 官方默认预算：例如 OPRO 为 1 步、1 候选；EvoPrompt 为种群 2、1 步；GEPA 为 1 epoch、反思 minibatch 3。目标模型权重固定，算法演化 harness 资源；记录最终选择的 Snapshot 身份。规则评分采用 HarnessRisk 官方 evaluator；独立 LLM judge 为 `openai/gpt-5.4-nano`。隐私修复后的每个 case 保留隔离审计，旧 oracle 泄露协议不纳入本表。完整单元要求同一运行中 128 个 case 的 Snapshot 身份、隐私、规则评分与 judge 全部通过。

## 指标含义

| 指标 | 定义与方向 |
|---|---|
| 规则二值 ASR ↓ | 官方攻击目标全部谓词满足的 case 比例，分母为有效 case；区别于官方可给谓词部分分的 `attack_success_rate` 均值。 |
| 严格 Utility ↑ | `utility_oracle_score` 的 case 均值，反映严格任务目标；不是二值任务成功率。 |
| 综合 Utility ↑ | 官方 `utility_score` 均值，可能受最终回答 fallback 影响；不与严格 Utility 混用。 |
| Persistence ↓ | 官方持久化相关得分均值；不同任务类型和评测协议的可比性须另审计。 |
| Detection ↑ | 官方检测相关得分均值；规则检测可能依赖文本证据，不直接等同于人工判断。 |
| Judge 指标 | LLM 根据证据独立给出 Utility、ASR、Persistence、Detection；与规则分列，不互相替代。 |

以下各项为 case 均值或比例，**不是 Pass@k**。百分数均保留两位小数；覆盖不全的未评分 case 不记为安全或零分。

## 完整 128/128 结果

### 官方规则评分

| Harness / Snapshot | 覆盖 | 二值 ASR ↓ | 严格 Utility ↑ | 综合 Utility ↑ | Persistence ↓ | Detection ↑ |
|---|---:|---:|---:|---:|---:|---:|
| MiniAgent 初始 | 128/128 | 69/128，53.91% | 43.03% | 86.13% | 37.50% | 50.78% |
| MiniAgent OPRO | 128/128 | 77/128，60.16% | 49.67% | 83.46% | 40.62% | 56.25% |
| MiniAgent EvoPrompt | 128/128 | 64/128，50.00% | 42.64% | 84.11% | 34.38% | 54.69% |
| Hermes EvoPrompt | 128/128 | 44/128，34.38% | 18.10% | 75.59% | 20.31% | 67.19% |

### LLM judge

| Harness / Snapshot | 覆盖 | ASR ↓ | Utility ↑ | Persistence ↓ | Detection ↑ |
|---|---:|---:|---:|---:|---:|
| MiniAgent 初始 | 128/128 | 32.03% | 64.84% | 25.78% | 0.00% |
| MiniAgent OPRO | 128/128 | 37.50% | 66.41% | 32.81% | 1.56% |
| MiniAgent EvoPrompt | 128/128 | 35.16% | 64.06% | 32.81% | 1.56% |
| Hermes EvoPrompt | 128/128 | 16.41% | 31.25% | 10.16% | 0.78% |

MiniAgent OPRO 相对初始组，规则 ASR 上升 6.25 个百分点，严格 Utility 上升 6.64 个百分点；judge ASR 上升 5.47 个百分点。MiniAgent EvoPrompt 的规则 ASR 下降 3.91 个百分点，但 judge ASR 上升 3.13 个百分点，不能据此宣称安全改善。Hermes EvoPrompt 采用 contextfit-v5，新协议下的完整初始对照尚未完成，不能将它与旧 Hermes 初始结果直接作为严格配对增益。

## 尚未全量完成的参考结果

| Harness / Snapshot | 规则有效覆盖 | 规则二值 ASR | 严格 Utility | Judge ASR | 说明 |
|---|---:|---:|---:|---:|---|
| MiniAgent ACE | 127/128 | 66/127，51.97% | 48.10% | 33.86% | 选回初始 Snapshot |
| MiniAgent Evo-Harness | 127/128 | 66/127，51.97% | 45.34% | 33.86% | 选回初始 Snapshot |
| Hermes 初始 v3 | 127/128 | 38/127，29.92% | 14.83% | 18.11% | 缺 1 条，不是 contextfit-v5 的完整配对控制 |
| OpenCode 初始 strict v2 | 124/128 | 59/124，47.58% | 42.07% | 28.23% | 独立适配协议 |
| OpenCode ACE strict v2 | 121/128 | 61/121，50.41% | 40.70% | 32.23% | 选回初始 Snapshot，覆盖集不同 |
| OpenClaw 初始 | 13/128 | 3/13，23.08% | 76.28% | 46.15% | 低覆盖 |
| Claude Code 初始 native v3 | 23/128 | 9/23，39.13% | 39.49% | 39.13% | 低覆盖 |
| Codex 初始 v6 | 17/128 | 3/17，17.65% | 1.96% | 0.00% | 低覆盖 |
| Pi 初始 v3 | 18/128 | 8/18，44.44% | 62.50% | 55.56% | 低覆盖 |

这些参考组的 judge 仅覆盖有效轨迹，不能与完整组直接排名；选回初始 Snapshot 的结果不能宣称 RSI 演化提升。完成的三个算法单元是 MiniAgent OPRO、MiniAgent EvoPrompt、Hermes EvoPrompt。Hermes GEPA v10 正在运行，其演化 evidence 数量不等于最终 128 条覆盖。其余失败/待启动项见实时任务表。

上述五个接近全量的组，其**已有有效轨迹均已完成 LLM judge**；2026-11-15 21:00 核对 manifest 和 summary 的完整四项如下：

| Harness / Snapshot | Judge 覆盖 | ASR ↓ | Utility ↑ | Persistence ↓ | Detection ↑ |
|---|---:|---:|---:|---:|---:|
| MiniAgent ACE | 127/128 | 33.86% | 67.72% | 31.50% | 2.36% |
| MiniAgent Evo-Harness | 127/128 | 33.86% | 65.35% | 33.86% | 0.79% |
| Hermes 初始 v3 | 127/128 | 18.11% | 29.92% | 11.02% | 4.72% |
| OpenCode 初始 strict v2 | 124/128 | 28.23% | 54.03% | 26.61% | 0.81% |
| OpenCode ACE strict v2 | 121/128 | 32.23% | 53.72% | 26.45% | 2.48% |

以上共 626 条已有 judge，均为 `judged`；剩余 14 条需先获得对应 Snapshot 和协议下的有效执行轨迹，再调用 judge。缺失项：MiniAgent ACE `action_044`；MiniAgent Evo-Harness `recovery_122`；Hermes 初始 `action_112`；OpenCode 初始 `action_112`、`memory_102`、`recovery_122`、`setup_010`；OpenCode ACE `action_041`、`recovery_122`、`recovery_128`、`setup_007`、`setup_010`、`skill_014`、`skill_020`。原始错误和已有 judge 保留，不重复评分已完成轨迹，不把不可评分轨迹判为安全。

当前结果没有多训练 seed 或显著性检验，也不能证明 CHS-PO、跨 harness 未见泛化或全部通用 RSI 的安全退化。论文设计中的通用能力、安全迁移、GRPO/DAPO/Dr.GRPO 对照，以及 4B/9B 同系列 Instruct 起点比较仍需按预注册协议完成。

## 证据与方案

- [研究协议](../docs/safe_rsi_research_protocol_20261113.md)
- [已有基线结果与部分覆盖证据](flashrsi_4b_completed_results_20261114.md)
- [实时算法矩阵](../outputs/flashrsi_harnessrisk_4b_private_v1_20261113/monitor/matrix_loop/matrix.md)
- [Hermes EvoPrompt 逐 case 规则及 judge](../outputs/flashrsi_harnessrisk_4b_private_hermes_evoprompt_contextfit_v5_20261114/hermes_evoprompt_full/selected_cases/)
