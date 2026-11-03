# 2B 跨 harness 安全 RL：r5b 后的实验设计（2026-10-30）

目标：在相同资源和任务预算下，比较单 harness GRPO、Harness Scaling GRPO 和 CHS-PO，测量未见配置及真实 harness 的安全—效用泛化。论文：https://arxiv.org/html/2610.29458 。本文借鉴五因素分解、均衡选择、相邻配置 K=2 的跨步任务分配与留出验证；安全奖励和安全约束是本项目新增设计，不是论文结果。

## 数据与采样

沿用 AgentDojo，按 v7 原始家族拆分 train/validation/test；train 中按 family × clean/injection 配对，状态复位、官方 utility/injection verifier 和副作用路径审计均须可用。先执行冻结 SFT 模型的基线采样和质量门禁，再构建训练池；保留所有失败样本，不按模型输出剔除训练 rollout。后续接入 AgentHarm 必须独立验证工具、重置和证据。测试集不可用于筛选任务、调参或选择 checkpoint。

现行池 `outputs/rl/9b_whitebox_pool_v4` 的 18 个结构化配置按 S/T/O/W 变化，C 固定 0；这不是论文的完整五因素设计。保持同任务 K=2，锚点与最小汉明距离邻居分配到连续两步，保留 assignment ledger、prompt/tool contract、种子和策略版本。固定训练未见的白盒组合及原生 CLI harness 做泛化验证；根据 SFT 实际暴露标记 RL-unseen 与全流程 unseen，不把白盒配置称为原生 CLI。

## 入场门禁与基线

使用 SFT step-267，从固定 train 和 family-disjoint validation 样本运行无参数更新的采样，逐 harness 与任务臂核验 真实调用、完成、verifier、错误和超时；阈值必须在观察该批数据前登记并保存，成功/失败计入全部尝试的分母。r5b 首步的 89/96 model_timeout、24 次有效工具执行暴露严重交互问题，未满足新方案入场条件。诊断工具 `scripts/audit_whitebox_rollouts.py` 将来源 batch 指标和门禁判断落盘；历史 r5b 第一步为初始 SFT 策略，仅可用作 train 诊断，不能替代 family-disjoint validation。

## 奖励与更新

第一阶段为标准 GRPO：仅使用可验证终局奖励，安全完成 +1，验证违规 -1，安全未完成 0；路径信号和非参与额外优势惩罚为独立消融，不改变基线。已加入 `outcome_only_v1` 奖励版本和独立配置，GRPO 入口检查配置契约；历史奖励保持兼容。评分器的基础反例检查通过，完整在线链路及家族隔离验证仍待执行，不能把当前实现称为已完成的正式基线。审计证据缺失必须标记、补采或弃权，不能假装安全。

先用短程运行验证稳定性及模型协议，训练中持续记录相对冻结 SFT 的 KL、相邻策略 KL、clip fraction、熵、梯度、奖励分布、每类/配置成功、有效调用、长度、耗时和无信号组；预先登记基于基线的停止规则。请求超时需在生成请求上真正可取消，区分墙钟、token 截断和正常终止；现有 480 秒仅在请求前检查，不构成硬超时。优化器和采样预算的任何调整需记录，并对基线/CHSPO公平匹配。每 10 次更新保存完整状态及配置。

稳定 GRPO 后，逐项加入按 benchmark × harness 分层的攻击、持久化、协议约束，以 GRPO、单约束和多约束做消融；不把拒收证据等同于安全惩罚。具体阈值先依据独立 validation 的 SFT 基线制定并冻结。

## 评测

同样 case、harness、seed、采样预算比较 SFT、单 harness GRPO、跨 harness GRPO、CHS-PO；checkpoint 只用训练配置的 family-disjoint validation 选择。测试 in-space held-out 配置及真实 CLI，报告每 harness 和最差 harness 的 Utility、ASR、Persistence、Detection（有官方定义时）、完成率、协议错误与超时，以及家族配对置信区间和全部尝试分母；三次 rollout 的 mean 与 any@3 分开报告。安全效用是否提升必须由独立评测判断，不以训练 reward 判断。

## 2026-10-30 step-150 起点诊断与工作流修订

step-150 新实验首批 96 条中，W0 的 24 条均至少一次有效 dispatch，W1/W2 的 72 条均没有有效 dispatch；W1/W2 共有 79 次调用、均被阻断。离线对原始 assistant turn 重放解析，118 次可解析调用，证明格式解析通过不等于工具成功执行。严格工作流会在计划回合前拦截直接工具调用；这与本次 2B 模型行为强烈冲突，是已定位的有效信号缺失因素。乱码与其他截断现象仍须独立定位，不能都归因于它。

增加版本化软工作流 `enforce_plan: false`：计划提示与周期检查点保留，模型输出计划时按原流程继续，直接调用工具时允许进行 schema 与安全审计后 dispatch；严格模式仍保留做消融。每条白盒证据记录 workflow_plan_policy。新旧轨迹执行语义不同，不可合并为同配置对照。此更改不放宽 schema、风险动作或路径审计。训练必须先用独立在线采样验证该修复对不同工作流和失败样本的影响。

固定输入 backend 对照保存于 `outputs/rl/diagnose_step150_backends_20261030/`，包括同 prompt、工具 schema、token IDs、log-probs 和原文；Transformers 贪心首回合 4 条中，SFT 有 2 条参数缺失，step-150 4 条可解析且参数满足 schema。该结果不能证明多轮、随机采样或 VeRL actor-to-vLLM 同步正确。生成请求级 deadline 的 opt-in 原型已接线，超时须按后台实际请求 ID 显式 abort，失败则整批不可评分；模拟客户端测试通过，真实后端测试未完成，不可标记为已验证。
