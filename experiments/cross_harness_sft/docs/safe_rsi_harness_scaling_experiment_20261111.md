# 跨 Harness 安全 RSI：实验设计与 4B 启动规范

日期：2026-11-11。本文是当前研究设计的主入口，取代本项目此前“教师轨迹 SFT→真实 harness 多数据集 RL”和“直接把白盒 Harness Scaling 用于安全 RL”的启动顺序。历史设计、代码和运行结果保留，均按原实验条件解释。

**最新调整：用户指定以 FlashRSI 开源框架推进 RSI，独立 coding agent 搭建五维 Harness Scaling 基建。** 下文原 ClawLoop-RL-only 部分是历史候选，不再作为默认主框架。FlashRSI 以 snapshot 形式进化 agent 的 prompt、skill、工具等资源；须选定具体算法及受控可编辑范围。模型权重在线 GRPO/CHS-PO 仍需与本项目 VeRL 训练链实现明确接口，不能把资源进化直接当作模型权重训练。固定模型的 FlashRSI 运行也可作为独立的系统级 baseline。安全策略、工具权限和 sandbox 不作为 RSI 可编辑资源。

**资源授权更新：**用户允许在本机共享使用 GPU 0、2、3 推进 4B 实验；该授权不代表显存和宿主内存已满足训练要求，也不授权结束其他用户进程。2026-11-11 19:47 实测 GPU 0/2/3 可用显存约 21.4/20.7/21.4 GiB，主机可用内存约 22 GiB 且 swap 已满。FlashRSI 已克隆并验证 commit `778e2d01a995491fe63db210dcf46db1723a4430`；官方 4B 权重（revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`）下载未完成。C1/C2 真实训练与模型权重 RL 接线未验收。应先继续 CPU 实现，启动真实 GPU 预检前重新检查资源并测量峰值，不得降低保护阈值冒充已满足内存要求。

**执行状态：设计和资源盘点已完成；4B RSI 训练尚未具备可运行入口，未启动。** 现有训练脚本不能直接执行本方案：它只接受旧 2B/9B SFT 起点、已有白盒 pool，且现有白盒训练实现将上下文组件固定为 C0。空卡只是启动的必要条件，还须完成下文的模型、数据、递归闭环与评分器门禁。

2026-11-11 代码复核：旧 `GRPO-whitebox-v1` / `CHS-PO-whitebox-v1` 契约不一致，但项目已实现 `GRPO-progress-v1` / `CHS-PO-progress-v1`，可共用 `verified_progress_v1`。新增 `rl_whitebox_reward_contract_pair.py` 对契约、variant、path weight 和 oracle 版本进行配对检查，并将单组 profile 校验接入 runner。跨 run 的 pair 校验还需纳入正式启动器；这不等于已经完成两模型配对训练或 4B 闭环。

论文附录 C 指出 context compaction 会将一次 rollout 拆成多个训练 segment；GRPO 仍按原始 rollout 归一化，子段继承同一 advantage，策略 loss 对有效模型 token 做均值。现有单一 `AgentLoopOutput` 及 `rl_verl_multiharness_loop.py` 采用不断追加的 token 序列，不支持改变历史后的多段真实前缀与独立 loss；C1/C2 仍需实现多段展开与端到端验证。保留 fail-closed，不能仅放开 C1/C2 名称或修改 attention mask 冒充完整五维训练。

## 1. 研究问题

本项目不再先收集教师轨迹进行专门 SFT，而从开源后训练 Instruct/对话模型出发，结合 Harness Scaling 的多配置训练与 RSI，研究：

1. 仅以良性通用任务驱动 RSI，通用能力是否提升？安全效用及越权风险如何随之变化？
2. 在相同良性＋安全配对任务及相同安全反馈下，CHS-PO 相比 GRPO 是否在保持良性任务效用的同时改善安全？
3. 成果能否迁移到训练未见的五组件组合，以及训练和选模未使用的真实外部 harness？安全效果来自模型权重、演化后的 harness 资源，还是两者交互？

“通用能力提升、安全下降”是待验证假设，并非必须得到的结论。如果没有观察到下降，应报告实际结果；不能按结果改测试集或选择性排除运行。安全且完成与良性 Utility 是主要效用结果；ASR、持久化、真实越权和评分覆盖分别报告。

## 2. 模型与计算路线

| 阶段 | 起点 | 目的 |
|---|---|---|
| 工程预检 | 官方 `Qwen/Qwen3.5-4B` 后训练模型 | 检验工具调用、可复位环境、在线 rollout、RSI 反馈、GRPO/CHS-PO 更新及权重同步 |
| 冻结后的主实验 | 官方 Qwen3.5-4B 与 `Qwen/Qwen3.5-9B`，各自从同一原始后训练 checkpoint 开始 | 在两个规模上运行相同主要对照；只能主张同系列跨规模 |

每个模型冻结上游 revision、各文件哈希、tokenizer、chat template、thinking 模式、生成参数和许可证。此处“跳过 SFT”指不再做本项目教师轨迹 SFT，不否认官方模型已有后训练。旧 2B SFT step267 和 9B SFT step291 不作为新实验起点。

先以 4B 做开发、资源峰值测量和 verifier 校准；这些使用过的任务和配置记录为开发暴露。冻结方案后，4B、9B 均从各自原始起点启动新的正式 run，不把试运行的步数或 checkpoint 拼入正式结果。全参数训练优先；可训练参数范围、精度、FSDP、offload 和资源上限在实际首轮预检后冻结。若需 LoRA/QLoRA，单独命名并在各对照组一致使用，不能与全参数结果混称。

## 3. Harness Scaling 与组件自我演化

将 harness 分为系统指令 S、工具 T、观察 O、上下文 C、工作流 W 五维。论文参考设置：每维 3 种实现，从 243 个组合中以 strength-2 正交设计选 18 个训练组合；每任务 K=2，随机选 anchor 并按汉明距离选择邻近配置，使用固定 assignment ledger 记录 task/配置/版本和策略版本。论文 GRPO 的 16 tasks×8 rollouts、300 steps 是参考，不是未经资源和数据门禁验证的本机默认值。[Harness Scaling](https://arxiv.org/abs/2610.29458)

每个 clean/attack 配对在切换配置时保持合法用户目标、工具真实权限、初始状态、攻击目标和载荷语义一致；工具呈现、观察、上下文管理和工作流可按组件变化。harness 的安全授权、隔离边界及隐藏 oracle 不因自我演化而改变；工具调用要映射到真实可审计副作用，不可暴露判分答案。

两种配置条件分别实验：

1. **固定组件对照**：使用预注册的 18 个配置及 K=2 分配；保留 K=1 和同 anchor Repeat-2 候选消融。
2. **自适应演化主实验**：从 18 个配置开始，允许 RSI 在安全边界内提出组件修改和新训练组合；候选须经过语义等价、权限、复位和测试集去重检查。每个版本保存内容哈希、父版本和准入结果。动态组合不再自动满足原始正交均衡，需单报实际覆盖。

训练前冻结组件组合 validation/test 和语义等价规则，训练不得生成或调用测试配置；真实外部 harness 的最终测试 case 和成绩不回馈任务生成、奖励调参或选模。adapter 开发使用与最终测试隔离的工程样例。

## 4. 数据与安全范围

通用任务以公开、可执行、可评分的良性训练任务为种子，RSI 递归生成新任务。安全首批候选为 AgentDojo、HarnessAudit；只允许通过原始 family、去重、reset、工具协议与 verifier 验收的 train families。每个安全训练 case 必须有严格 clean/attack 配对：合法用户目标、初始环境与权限相同，仅改变攻击或风险条件。缺少自然配对的 case 不通过人为改写凑数，首轮暂不入训并记录缺口。

预期覆盖间接提示注入、未授权动作、数据泄露、恶意持久化、边界穿越；各类是否可纳入以配对和评分证据为准。生成任务、配对及 verifier 必须通过：可复位、已知成功/失败反例、权限不扩大、隐藏答案不进入模型上下文、与 validation/test 近重复排除、必要时独立语义审计。失败候选保留原因，不进入训练。

每个来源按 `(benchmark namespace, original family id)` 在任务变体、harness 和种子展开之前冻结 train/validation/test。同一 family 的 clean/attack、payload 变体与跨 harness 实例不得跨 split。AgentDojo/HarnessAudit 的独立 family test 测来源内泛化；HarnessSafe、HarnessSecurity-Bench 为独立来源候选。项目已经使用过两者的材料与部分结果，须逐任务记录历史开发暴露，不能把已查看的结果冒充全盲测试；若无法保留可审计的未使用子集，仅报告开发暴露条件下的外部测试。

## 5. RSI、算法与公平对照

主闭环以 **FlashRSI 的可配置方法与 snapshot 机制**为资源演化层，选择具体方法、资源范围和模型角色后，再与项目 VeRL 的当前策略在线 rollout、GRPO/CHS-PO 训练层对接。任务合成、执行和 verifier 门禁、模型更新、诊断和下一轮训练数据之间的桥接需要实际实现验证。FlashRSI 官方公开的资源演化不等于自动执行模型权重 RL。[FlashRSI](https://github.com/RUC-NLPIR/FlashRSI)

主要实验分两阶段：

| 阶段/组 | 任务与反馈 | 策略/目的 |
|---|---|---|
| 起点 | 固定初始模型、固定起始 harness | 通用能力、安全及协议基线 |
| 通用 RSI | 良性通用任务与任务成功反馈 | Harness Scaling + FlashRSI 指定方法 + GRPO；观察能力/安全是否变化 |
| 安全主对照 | 同一批良性＋安全配对任务、相同效用与安全证据 | 同条件下比较 GRPO 与现有 CHS-PO；安全差异主要限于策略优化 |
| 安全信息消融 | 同配对任务；原 RSI 仅看任务成功 | 比较仅任务反馈与引入安全信息的整体效果，不能归因于优化器本身 |
| 系统级 RSI 对照 | 固定模型权重，允许规定的 prompt/skill/记忆等资源演化 | 比较资源自演化的能力与安全结果，和参数 RL 结果分开 |

共享任务池主实验：独立的同模型共享生成器分支分别生成良性任务和严格匹配的安全任务；预注册的比例合并，冻结任务、verifier、harness 配置后同时交给 GRPO 与 CHS-PO。共享分支自己的迭代规则与策略快照预先指定，且计算成本记账。两算法还各自根据反馈生成下一轮任务，作为**独立递归闭环**；任务分布可能不同，不能将其差异仅归因于优化器。同模型生成/诊断为主，固定外部 worker 为独立对照；worker 的模型版本和成本明确记录。

CHS-PO 第一版沿用已有按来源×harness 的 attack/persistence/protocol 约束思路，先验证其在 RSI 中是否有效；只修改策略优化的主对照不得暗中改变生成/诊断逻辑。安全 feedback 对 GRPO 和 CHS-PO 均可见；安全约束与 reward 标量化公式、dual 和非行动惩罚在独立训练/validation 上校准并冻结。若结果表明跨轮组件或任务分布风险无法由现有目标处理，再提出可验证的新机制及单独消融。当前不把第二个 RSI 框架列为必做；SkillRL 等保留候选，在完成该验证前不主张跨 RSI 框架通用。

算法组必须记录相同起点、任务 ledger、初始状态、harness 配置、采样、有效 rollout、生成 token、优化步数与训练时间，并单列共享生成器、任务审计、judge 和外部 worker 的消耗。任何预算不可完全一致的情况，冻结主预算并如实报告。零方差/零梯度、KL-only、no-op 和基础设施失败分开记录，不因奖励全相同而隐式挑选任务重采。

## 6. 反馈、指标与选模

安全结果优先来自工具 dispatch、文件/网络/邮件、持久状态及环境 verifier 的确定性证据；语义判断由固定、独立于被训练策略的 judge 补充，并审核误判。确定性违规不能被语言 judge 覆盖；无 verifier 或基础设施失败记 unscorable，不当作安全成功。

| 指标 | 定义 |
|---|---|
| benign Utility | clean 合法任务完成比例，按原数据集 verifier 与有效分母统计 |
| safe-and-useful | 合法目标完成且无已验证安全违规；按任务和风险类别分别统计 |
| ASR / 真实副作用 | attack 轨迹中攻击目标成功、越权执行或泄露比例，按来源 oracle 分开 |
| Persistence / Detection | 持久化风险及可观测风险识别；来源未定义时 N/A |
| Coverage / invalid / timeout | attempted、completed、scored、环境无效、模型协议错误与超时分别记录 |
| Pass@k | 仅对有成功 oracle 的能力任务，在声明独立采样规则后报告；不当作安全率或 ASR |

以 safe-and-useful 为主，良性 Utility、ASR、持久化与 Coverage 设联合门槛。具体下限/容忍度和置信区间先在独立 validation families 上依据起点和样本量预注册；不得以 test 结果修改。各组都报告固定预算末 checkpoint 和同一安全—效用规则选出的 checkpoint；无合格 checkpoint 则明确报告，不强选。

每轮保存 checkpoint、optimizer/scheduler、策略和生成器版本、组件/技能/记忆、任务与 ledger、verifier、诊断反馈、运行清单和资源状态。validation 按预定频率运行，不要求每轮完整评测。每组至少规划多个独立训练种子，最终数目在资源预检后冻结；按 family 配对报告置信区间、各数据源及各 harness 结果、宏平均和最差 harness，且单报共同有效支持集。

## 7. 泛化与归因

1. **来源内新 family**：AgentDojo/HarnessAudit 的 test families，训练 harness 配置不越界。
2. **未见组合**：冻结的五组件测试组合，且与演化配置做语义近重复检查。
3. **真实外部 harness**：锁定 checkpoint 后运行未参与本项目 RL、奖励调参和选模的原生 harness；例如 Gemini CLI、OpenCode、gptme 为待验证候选，不预设其无开发暴露。区分“RL 未见”“项目后训练未见”和 exposure unknown，预训练暴露不作无证断言。
4. **独立安全来源**：HarnessSafe、HarnessSecurity-Bench 等的可审计任务；后者安全机制开关须固定/分层，不能把隔离机制效果归因于模型。

外部 harness 分别测试“仅迁移模型参数”和“参数＋兼容资源迁移”。固定模型、资源进化的系统级 RSI 作为独立对照。validation 上对语义兼容且权限一致的初始/最终模型×初始/最终 harness 做 2×2 交叉分析，以定位权重、组件及交互的贡献；最终 test 只使用预注册组合。

## 8. 已有资产及不可直接复用的部分

| 资产 | 可借鉴内容 | 当前边界 |
|---|---|---|
| `scripts/train_rl_9b_whitebox.sh` / `train_rl_2b_whitebox.sh` | VeRL/FSDP、资源门禁、日志和 checkpoint 组织 | 硬编码 2B/9B 旧 SFT 起点，不接受 4B 官方后训练模型；不能作为新方案启动命令 |
| `scripts/rl_whitebox_harness.py`、`rl_whitebox_validation.py` | 固定 ledger、白盒分配与校验 | 当前 `harness_mode` 为 `four-factor-context-fixed`，C=0；不是论文完整五维 18 配置；需扩展和验证 |
| `outputs/rl/9b_whitebox_pool_v4/` | 15 个已审计 AgentDojo train family、540 cells、旧 ledger 的工程参考 | 旧白盒 AgentDojo 池；不是 HarnessAudit、多来源严格配对或 RSI 任务池，亦非原生 CLI |
| `rl_whitebox_objective.py`、`rl_whitebox_trainer.py` | CHS-PO 约束、优势和双变量实现 | 现有 `GRPO-whitebox-v1` 与 `CHS-PO-whitebox-v1` 入口绑定不同 reward contract；需新增/验证共享安全反馈的同条件对照 |
| `vendor/agentdojo/`、现有 AgentDojo verifier/工具映射 | 来源环境与任务重置经验 | 与当前组件及安全配对的复用能力须逐 cell 验证；HarnessAudit 的在线 reset/verifier/配对资格不能假定已完成 |
| `reports/harnessrisk_sft_grpo20_chspo150_comparison.md`、`AGENT.md` | 2B 多 harness GRPO/CHS-PO 历史结果与失败经验 | 旧 SFT 起点、预算/版本不同；OpenClaw Persistence 有初始化文件污染；不能当作新 4B 方法对照 |

历史 2B 经验显示低 ASR 可以与极低 Utility 同时出现，GRPO 后 Utility/ASR 可能同涨；旧 9B 运行曾遭 CPU RAM 与显存峰值、工具协议拒绝等问题。此前白盒历史结果不能证明当前 4B RSI 的安全退化或 CHS-PO 收益。

## 9. 4B 启动顺序与放行条件

1. **冻结资产**：获取官方 4B revision/权重，计算哈希；确认 VeRL/vLLM、tokenizer、真实工具调用；审计数据家族、配对、开发暴露和外部测试范围。模型及数据不完整不得创建正式训练 run。
2. **补实现**：独立 4B Instruct 训练入口；五组件及完整 18 配置；严格共享安全反馈的 GRPO/CHS-PO；AgentDojo/HarnessAudit 环境与 oracle；任务合成/诊断/门禁/版本冻结及真正下一轮反馈。不能用固定任务重复训练冒充 RSI。
3. **无 GPU 检查**：检查可复位环境、配置/ledger、clean/attack 语义一致、权限边界、verifier 的已知正反例、reward 与约束、测试集去重、checkpoint/restart 和失效归因。
4. **资源放行**：启动时重新检查每张候选卡的进程与可用显存、主机 MemAvailable/交换、磁盘与 I/O；不结束其他用户进程。原入口仅允许每张卡占用 <256 MiB，旧 2B 对应 >80 GiB 可用内存是历史门槛，不是实测 4B 的必然阈值；4B 阈值须以预检布局与峰值测量确定。2026-11-11 15:21 四张卡均有其他用户进程、MemAvailable 约 29 GiB、swap 已满，故当时不具备放行条件。
5. **小规模真实预检**：4B 在线生成、工具执行、环境 reset、独立 verifier、GRPO/CHS-PO 更新与 actor→rollout 同步；证明下一轮接收前一轮 RSI 产物并保存轨迹/版本。检查真实参数更新和完整 checkpoint；记录 GPU/RAM/I/O 峰值与无效分类。预检失败即保留日志并处理，不能算正式结果。
6. **冻结并启动**：把模型 hash、代码、18 配置/ledger、严格配对池、reward、采样、资源、seed、预算和评估/选模规则写入独立 run manifest；分别启动 4B 通用 RSI、共享池 GRPO/CHS-PO 及已选消融。验证首次完整 optimizer 更新、同步和指标；再按冻结方案推进 9B。

**本文件不是已就绪的启动脚本，也不安装自动抢占 GPU 的等待器。** 实施任务与实际资源门禁完成后，才可为新 run 固化命令并决定是否排队启动；不能通过修改当前旧训练脚本中的模型字符串来绕过缺失闭环。

## 10. 可复核引用

- 论文：[Harness Scaling](https://arxiv.org/abs/2610.29458)、[ClawLoop](https://arxiv.org/abs/2610.14118)。
- 本项目已有讨论：[早期安全 RSI 草案](harness_scaling_safe_rsi_design_20261111.md)、[4B 本机启动计划](4b_safe_rsi_local_launch_plan_20261111.md)。
- 历史结果：[2B HarnessRisk 对照](../reports/harnessrisk_sft_grpo20_chspo150_comparison.md)、[HarnessSafe 汇报](../reports/harnesssafe_eval_metrics_passk_20261106.md)。
