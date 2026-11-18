# 9B SFT 真实 Harness 多数据集 RL 设计

状态：设计稿，待用户审阅；未开始实现或训练。

## 目标

从 Qwen3.5-9B SFT step-291 策略出发，在真实 Codex CLI、Claude Code 等 harness 中，使用多个可复位、可评分的数据集环境做当前策略在线 RL。检验跨来源、跨任务家族和跨 harness 的安全对齐与正常任务能力保持，并在 RL 未见 harness、独立安全基准上评估迁移。

用户观察到 Harness Scaling 风格的白盒 harness 组件训练在另一台服务器效果弱于此前直接使用真实 Codex、Claude Code 多数据集在线 RL 的 2B 方案。本设计因此把真实 harness 多数据集训练作为主路线；白盒组件不进入首轮主训练。

## 研究问题与成功标准

1. 相比同一 9B SFT 起点的 GRPO、CHS-PO 是否能提升真实 harness 中的安全任务结果，同时保留 benign Utility？
2. 改进能否跨 AgentDojo、HarnessAudit 等来源及家族留出？
3. 策略能否迁移到 RL 未见、且尽可能 SFT/后训练也未见的真实 harness？

只有在有效评分覆盖达到预注册下限、benign Utility 不发生实质退化，且攻击成功率、持久化和真实协议/副作用违规在家族留出评测中没有超过预注册容忍界限时，才支持安全对齐主张。所有阈值、置信区间方法和最小样本量须在测试解封前由冻结的 manifest 指定。训练 reward 上升不作为成功标准。

## 训练数据与 Harness

- **首批数据源候选**：AgentDojo、HarnessAudit。按原始任务家族切分后，筛选能在目标 harness 中真实执行、重置状态、复核任务语义并由可靠 verifier 评分的 train families。
- **首批训练 harness 候选**：Codex CLI、Claude Code。每个来源分别审计两者的原生工具协议、权限、调用/返回映射、上下文及副作用可见性。一个来源未通过某 harness 的执行与 verifier 门禁时，不将其强行填入完整交叉矩阵。
- **在线数据要求**：训练 rollout 必须由当前策略在原生 harness 中实时生成并执行。离线 teacher transcript 只用于任务/环境定义或监督基线，不能冒充 RL rollout。
- **扩展候选**：通过同样门禁后，再逐个考虑 AgentHarm、ActBench、SafeClawArena 等来源或 Hermes、OpenClaw、OpenCode、Kimi 等 harness；每次扩展更新版本化 manifest，不能把未接入的来源/产品写成已覆盖。
- **家族切分**：先按 `(benchmark namespace, original family id)` 冻结 train/validation/test split，再生成变体、跨 harness 映射或 rollout。某个家族的所有 benign/adversarial 臂和 harness 实例必须落在同一 split。不得把 HarnessAudit 已用于评测/开发的 case 悄悄移入训练；需逐 case 做 exposure 和近重复审计。

## 首轮矩阵

| 维度 | 设计 |
|---|---|
| 起点 | 同一冻结 9B SFT step-291 checkpoint；RL 新建 optimizer，step 从 0 计 |
| 主算法 | 标准 GRPO 与 CHS-PO；同一 task ledger、采样数、token/turn/time 预算、optimizer 预算及评分器 |
| 训练来源 | AgentDojo × HarnessAudit 中通过环境/评分门禁的家族 |
| 训练执行器 | Codex CLI、Claude Code；按来源×harness 可执行交集采样，并显式记录缺失 cell |
| 对照 | 未做 RL 的冻结 SFT；若预算允许，保留同配置 Base/SFT 对照而不替代 SFT 主基线 |
| 训练/选模 | 仅使用 train families 更新；仅用训练 harness 的 validation families 选 checkpoint |
| 盲测 | 未参与更新、奖励校准或选模的 family-disjoint test families；目标 harness 与来源固定后揭盲 |

首轮不预设 300 步、固定 cell 数或家族规模。先盘点两来源的 family 交集与独立可执行覆盖，完成算力/内存和有效 rollout 小试后，再按可用的唯一家族数、每步有效组数和预算冻结训练规模。不得以重复少量家族填满“多数据集”配额。

## 泛化评测

### 1. 训练内 Harness、家族留出

在 Codex 和 Claude Code 上，用训练未见但同来源的 test families 比较 SFT、GRPO、CHS-PO。分别报告正常任务 Utility、攻击/注入防护、安全违规、无效工具调用、超时、执行完成率和有效评分覆盖。统计按 benchmark、family、arm、harness 分层，提供配对差异和 family-clustered uncertainty。

### 2. RL 未见 Harness

候选为未参与 RL rollout、奖励调参或选模的真实 harness。按 9B SFT 的准确暴露清单进一步标为 `RL-unseen`、`post-training-unseen` 或 `exposure-unknown`。只有 SFT prompt、工具 schema、轨迹和选模过程也没有该 harness 暴露证据时，才称后训练未见；预训练暴露未知时，不称模型全程未见。为该评测实现 harness adapter 时，仅可用不泄漏目标任务/评分的工程固定样例，不得用目标安全结果迭代训练设计。

### 3. 未见 Harness 与独立安全迁移：HarnessSecurity-Bench

新增首选候选 **HarnessSecurity-Bench (HSB)**，与 HarnessAudit/AgentDojo 训练分工：在它具备真实可执行 harness 的任务上，用 HSB 的独立 security oracle 与 utility verifier 测 9B SFT、RL 后策略的跨 harness 安全能力及效用保持。项目页和论文列出的六种被评测 harness 为 Codex CLI、Claude Code、Gemini CLI、gptme、Qwen Code、GitHub Copilot CLI；基准含 23 个编码任务、9 类安全机制、五类攻击面。官方仓库当前提供任务规格、容器环境、参考解和独立 verifier，并称 harness 数据、rating records、trial artifacts 和分析脚本仍在准备公开发布。因此任务包可用不代表完整复现实验产物/人类 harness inventory 都已开放。

六种 harness 中 Codex CLI 与 Claude Code 属于首批 RL 训练候选，不能在这两个产品上声称 unseen-harness 泛化；Gemini CLI 已有本项目评测和开发暴露，也不能默认视作未见。gptme、Qwen Code、GitHub Copilot CLI 是**候选留出 harness**，但必须逐一审计 9B SFT 的真实轨迹、工具 schema、prompt、checkpoint 选择和本项目此前评测暴露。若确认本项目后训练未见，按 `post-training-unseen` 报告；若只未参与 RL 则为 `RL-unseen`；证据不全记 `exposure-unknown`。预训练产品暴露未知时不得声称模型全程未见。

HSB 本身是安全设置与关闭/开启机制的成对比较设计；其六个执行 harness 不自动意味着每个模型都有六个“未见 harness”。正式运行前需确认目标版本/配置/机制在 HSB runner 中被支持、原生 CLI 能力证据充分、模型 endpoint 真实接通、容器隔离可用、utility 与 attack oracles 均可复现。不要用 benchmark 作者在 GLM-5.2 上的 2,500 次研究运行替代本项目对 9B 的模型评测，也不要把 task utility 当作 Pass@k。

在锁定 RL checkpoint 后，优先跑 HSB 的完整适用任务/机制×目标 harness 矩阵，并用官方配对配置比较安全机制开/关的 attack effect 与 task utility。每个任务的重复采样次数和随机性按公开 runner/original protocol 复现并写入 manifest；逐次报告 task success、attack effect、执行/评分覆盖和成本。若任务定义支持独立随机 rollout，可单独报告能力 **Pass@k**；HSB 的安全结果仍应分别报告 ASR/attack effect，不能把“至少一次任务完成”的 Pass@k 解释成安全率。不同 harness 或 mechanism 的结果不可未加分层就汇成一个总指标。

再用项目历史未评测过的 HSB task families 或其他独立安全基准（候选 HarnessRisk/SafeClawArena 的未用家族）做外部复验。若 HSB 任务已用于适配器调试、奖励设计或 checkpoint 选择，就不能作为盲测；应保留未触碰的 task family/机制组合，或把其标作 development validation。不同来源的指标不合并成未经定义的统一 ASR。若 HarnessAudit 作为训练来源，则只用其冻结 train families；其 test families 仍可作为家族留出，但不能再称 HarnessAudit 为完全独立 benchmark。

## 指标与重复采样

- **Utility**：来源官方 benign task verifier 的通过率/得分，必须明示分母和评分资格。
- **ASR**：有效攻击/注入评测中达成来源定义攻击目标的比例；与 benign Utility 分开。
- **Persistence**：观测到未授权状态跨阶段或重置边界保留的比例，按 case oracle 定义。
- **Detection**：只有存在预注册、可观测的检测标签/事件时报告；不把安全拒绝话术自动算作检测。
- **执行健康**：attempted、completed、scored、invalid、timeout、protocol-invalid 计数与覆盖率；无效/未评分轨迹不当作安全。
- **Pass@k**：只用于有明确任务成功判据的能力评测。每题 k 次独立采样时，Pass@k 指至少一次成功的题目比例；同时报告 k、温度/采样设置、每次 completion 规则及任务数。Pass@1 是单次成功率；Pass@k 不是 ASR，也不能用它替代安全指标。若用温度 0 的重复运行或同一个确定性输出，不声称随机采样意义上的 Pass@k。安全指标可另报 `ASR_any@k` / `Persistence_any@k`，但须给出定义和全部尝试分母，不能叫 Pass@k。

所有比较至少提供每 harness、每数据源、宏平均、最差 harness、有效覆盖和 family-clustered 置信区间。报告每个 harness 的实际有效任务数和共同支持集；覆盖不足时标记 N/A，不通过聚合掩盖。

## Reward 与防止效用塌缩

奖励以官方任务完成和可验证安全副作用为主；verifier 缺失、环境错误和模型失败分开标记。不能把基础设施不可评分当作负奖励或安全成功。训练池必须同时含 benign utility 与安全/攻击防御任务。设置验证门禁监测 benign Utility、拒答/不行动、invalid tool calls、timeout、ASR 与副作用；模型以“全拒绝”降低攻击成功时不得视为合格。

GRPO 与 CHS-PO 共用采样、任务、基础标量 outcome、生成预算和训练预算。CHS-PO 的 safety dual、path reward 或非行动惩罚如要使用，须先由 9B train/validation 家族做奖励反例校准并预注册；不直接搬用 2B 的阈值或 dual 状态。若任一奖励改动影响 Utility 与安全排序，作为单独消融。

## 工程门禁与阶段

1. **暴露/数据审计**：核实 9B SFT step-291 实际训练轨迹、harness、families、checkpoint 选择记录；冻结两来源 family split 与别名/近重复清单。
2. **Harness×Dataset 验收**：对每个候选组合验证真实 CLI 身份、model endpoint、工具调用/结果映射、权限、独立 reset、官方 verifier、正反例和副作用审计；输出可评分 cell 矩阵。
3. **小批在线冒烟**：SFT 起点分别在 Codex/Claude Code 与两数据源做少量当前策略在线 rollouts；证明 episode reset、评分、token/log-prob 对齐、资源回收和必要奖励差异。此阶段不据小样本作能力结论。
4. **冻结主实验**：依据门禁结果确定可执行交集、家族量、采样/更新/生成预算、种子、reward 和停止规则；生成不可变 manifest。标准 GRPO 与 CHS-PO 对齐运行。
5. **验证与选模**：仅使用训练 harness 的 validation families，按预注册的安全—效用联合规则选 checkpoint。
6. **锁定后泛化测试**：先锁定 checkpoint 和评测配置，再运行未见 family、RL 未见 harness 和独立安全迁移评测。

## 非目标与风险

- 首轮不复刻 Harness Scaling 的 18 个白盒配置，不将配置数量当作真实产品 harness 覆盖。
- 不把 2B 的结果外推到 9B，也不因先前观察就预设真实 harness RL 一定更优；通过匹配 SFT 起点、数据 family、采样和更新预算做直接检验。
- 不声称对预训练中未见的 harness 泛化，除非有可审计的模型预训练暴露证据。
- 主要风险是 Codex/Claude 的环境 reset、tool schema 与 verifier 难以跨来源等价；如果某 cell 无法可靠执行，则缩小已验收矩阵或先完成 adapter/verifier，不填入未经验证的数据。
- 9B SFT 的训练暴露、HarnessAudit 在本地具体已用 case、目标 RL-unseen harness 的无泄漏候选均待审计结果确认；任何 unknown 都在最终报告中显式标注。

## 需要在实施计划冻结的参数

当前文档确定研究设计，不指定未经审计的训练规模。工程阶段需给出并验证：可训练 family 数与每层覆盖、SFT exposure manifest、harness×dataset 资格矩阵、模型/CLI/环境版本、算力可承载的 rollout 与 optimizer batch、k 与采样参数、GRPO/CHS-PO reward 及 dual 参数、最小有效覆盖、联合 checkpoint 规则、统计功效/置信区间方案和独立测试集清单。
