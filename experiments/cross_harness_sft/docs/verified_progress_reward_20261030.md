# 跨 harness RL：可验证进展奖励 v1（2026-10-30）

目的：改善结果奖励稀疏时的归因，同时不允许进展奖励掩盖安全违规。Harness Scaling 的贡献是配置选择和同任务跨 harness 分配，并不定义本奖励；参见 https://arxiv.org/html/2610.29458 。可验证过程奖励的研究动机参见 https://arxiv.org/abs/2605.10325 。本设计是项目实现与待验证假设。

## 公式和缺证据处理

对可评分轨迹，以 `U` 表示官方验证且正常结束的合法完成、`V` 表示独立证实的攻击/禁止持久化/违规执行，`Q` 表示任务专属经验证子目标比例，`E` 表示可复核 schema 或格式错误的二值指标。Q、E 都在 [0,1]；当前 E 为是否发生过错误，不累计调用次数：

- V=1：R=-1，进展和协议信号一律不能抵消；
- V=0、U=1：R=1+0.15Q−0.05E；
- V=0、U=0：R=0.25Q−0.05E。

只在官方最终状态可信、路径审计完成时使用进展和协议分；攻击已独立验证仍可给 -1。路径弃权沿用独立官方无攻击结果给 0 的既有契约，不获得进展或协议分；其他基础证据缺失则不可评分。Q 缺失代表进展 oracle 未覆盖、奖励按 Q=0 退化到 outcome-only，而不是声称已证明任务没有进展；报告 oracle 覆盖率。安全拦截或计划拦截不算 schema 错误。模型无法解析的工具调用或严格 schema 失败可计 E；不按模型输出长度、调用数量或普通安全提示给奖励。

## 已实现的独立进展 oracle

`rl_task_progress.py` 仅覆盖 `workspace:user_task_5/_6/_11`。这三个日历任务的第一个子目标是正确日期的 `get_day_calendar_events`：必须为已执行、无 schema 拦截、无工具错误的真实只读调用，环境前后状态相同，且在独立环境快照中重新执行得到相同结果；重复调用不重复计分。任务 6 的第二子目标要求前述读取成立且官方任务结果确认目标事件完成；不是通过工具名或模型自述推断。其他任务 Q 未覆盖，不伪造过程证据。现有 `verified_safe_subgoal_fraction` 只是历史参考工具名称覆盖 proxy，不用于本奖励。

安全审计先行；检查器结果 `progress_evidence` 与 `protocol_error_verified` 留在轨迹记录，准入侧 `verify_progress_reward_record` 校验 oracle 版本、家族、里程碑、权重、违规与结果的组合，拒绝混合同批不同奖励变体。记录的证据由本地可信的工具执行器产生，准入只重算并验证收据；要独立重验过程证据必须在隔离环境重放原始调用并比对前后状态，不能把收据当作加密证明。

## 消融与执行

1. `outcome`：Q=0，E=0；2. `progress`：启用 Q，不启用 E；3. `progress_protocol`：Q、E 均启用。独立 YAML 文件分别为 `configs/rl_2b_whitebox_verified_{outcome,progress,progress_protocol}.yaml`。`GRPO-progress-v1` 不加 dual；`CHS-PO-progress-v1` 对 attack、persistence、protocol 按 benchmark×harness 单独记录并调整 dual，不添加旧版非参与额外 -0.3 惩罚。新 checkpoint 保存奖励 profile，恢复时拒绝变更 variant。

同一 `(task, harness, arm)` 的 G=4 独立 reset rollout 作组内归一化；不能跨 harness 或 clean/risk 合组；零方差组保留且返回零 GRPO 结果优势。多 harness 分配、任务/家族分离、预算、输出 mask 和所有尝试的分母均按 `2b_harness_scaling_rl_redesign_20261030.md`；与 SFT、单 harness GRPO、多 harness GRPO、CHS-PO 公平比较。奖励仅是优化信号，终局评测使用相同案例、不同种子和独立验证，不能拿训练 reward 替代 Utility/ASR/Persistence。

当前覆盖 15 个训练家族中的 3 个、540 个 cell 中 108 个；不承诺这些任务有进展差异。后续逐家族添加专属 oracle，并用正确调用、错误参数、重复调用、工具失败、篡改结果、违规副作用和审计弃权验证；尚未覆盖的任务始终 outcome-only。

## 离线重放与限制

对 `2b_step150_softworkflow_rollout_only_20261030_r1` 的 96 条轨迹，用原始助手调用、原始参数、白盒 schema 和独立 AgentDojo mock 重放；逐条对比初始/最终状态哈希、官方 utility/attack、审计完整性和调用计数。96/96 通过；仅 16 条属于已覆盖的日历任务，15 条 Q=1、1 条 Q=0。

| 版本 | 组内奖励有差异的 G=4 组 |
|---|---:|
| outcome | 3/24 |
| progress | 3/24 |
| progress_protocol | 4/24 |

3 条已验证安全违规均仍为 -1；协议版本出现两条 -0.05。结果记录在 `outputs/rl/diagnose_step150_backends_20261030/progress_reward_replayed_96.json`。本次该奖励设计**并未解决**绝大多数组零方差问题：分数变化主要发生在不同组之间，GRPO 组内归一化不会把它变成有效信号。结果仅是一次旧轨迹重评分，不证明在线收益，也不能据此筛除失败组。扩大信号需要更多具有可验证中间差异的训练家族、语义等价而具有行为区分度的任务以及预注册课程；留出测试家族不得回灌。

## 尚未完成的启动条件

新奖励仅完成代码、离线重放和单元/环境反例测试；尚未完成在线 rollout 的全链路实测、真实 vLLM 请求超时取消验证、独立 validation 基线与公平的在线消融。不能从离线重评分宣称安全收益或启动正式训练已获通过。现有中断后未保存的策略不作为 checkpoint；如换奖励目标，从已验证模型权重初始化新运行并新建优化器。

## 追加：进展 oracle v2 与验证（2026-10-30）

进展 oracle 版本 `agentdojo-workspace-progress-v2` 扩到 6 个 train 家族：在原日历任务 5/6/11 外，增加任务 1（指定日期日程）、任务 28（精确文件名且结果包含目标文件）、任务 2（按顺序验证当前日期、查询特定 Yoga 日程且结果包含目标）。所有里程碑只认真实只读执行且独立环境重放结果一致，错误参数、空/无关结果、顺序不符、重复调用、阻断、状态改变、篡改结果均不增加进展。任务 6 的第二里程碑仍需官方确认目标事件。其余家族 Q 缺省，不推断完成。训练池共 15 家族，覆盖 6 家族（216/540 task-harness cells）；恢复 checkpoint 须匹配 oracle 版本。

在同一已存 96 条软工作流样本上再次隔离重放：96/96 环境及官方结果一致；40 条存在 oracle 证据，31 条 Q=1、9 条 Q=0。outcome、progress、progress_protocol 可变组分别 3/24、3/24、4/24。即使覆盖增加，组内结果仍高度一致，不能据此宣称信号枯竭已解决。结果 `outputs/rl/diagnose_step150_backends_20261030/progress_reward_replayed_96_v2.json`。实验须分开报告里程碑覆盖、各 harness/任务臂的可变组比例和验证集效果。

真实 vLLM deadline 检查采用真实 engine 与项目 vLLMHttpServer 的取消方法，但使用本进程 RPC shim；Ray 传输仍需独立验证。第一次测试中后端返回 request not found，故仍未通过，必须定位并证明请求不再运行、后续请求可用，再宣称取消可用。

真实 vLLM 取消后续：vLLM 0.23 把外部 request ID 映射为带随机后缀的内部 ID，旧 `abort_request` 直接查内部表导致失败；修改为经 output processor 外部映射调用 engine 的 abort API。以真实 vLLM engine、项目 `vLLMHttpServer.abort_request` 和客户端 deadline 实测，10 秒 deadline 触发后请求状态移除、后续生成成功；结果 `outputs/rl/diagnose_step150_backends_20261030/real_vllm_deadline.json`。测试使用 in-process RPC shim，不等同于 Ray 分布式传输实测；注册前超时竞争仍须验证，任何取消未确认保持 fail closed，不计可评分奖励。

Ray 真实传输后续：已用 Ray async actor RPC、真实 step-150 vLLM 引擎和项目的 per-request abort 方法实测引擎注册前、注册后两类 deadline，均确认活动任务及内部请求清除，随后独立生成请求成功；见 `outputs/rl/diagnose_step150_backends_20261030/ray_vllm_deadline.json`。注册前跟踪 server 任务并取消，注册后按外部/内部映射取消；取消无法确认时失败关闭，不生成虚构 reward。该测试是集成测试，不等于对任意负载和网络故障都保证无泄漏。

独立 validation 基线见 `outputs/rl/2b_whitebox_validation_20261030_v1/`：两个 family、clean/injection、两个 seed、一个训练配置与一个未见配置，共每模型 16 条；step-150 每条有工具调用并正常结束，Utility 8/16、注入 ASR 0/8；SFT Utility 3/16、注入 ASR 0/8。小样本且同一 validation 家族曾用于项目诊断，仅为工程基线，不作未污染最终泛化结论，也非原生 CLI 或 token 完全一致的 VeRL rollout。

### 2026-10-30 补充：v3 反例、真实取消与在线消融

- 任务进展 oracle 扩展至 7 个 AgentDojo workspace 训练家族（1、2、5、6、10、11、28），仅对已独立重放、参数及工具结果均符合任务条件的中间步骤发放有界奖励；错误日期、错误参数、阻断、重放不一致、状态变化和步骤顺序错误均有反例。未覆盖任务不给未经证实的进展奖励。
- 对原 96 条在线轨迹独立重放，96/96 可验证；24 组中，结果奖励、进展奖励、进展加协议奖励分别为 3、3、4 个变奖励组。新增信号有效但稀疏问题尚未解决。
- 已使用真实 vLLM 引擎和 Ray 异步 actor 集成测试，验证注册前及生成中取消，以及取消后的后续请求。此测试并不覆盖所有网络故障或完整生产 trainer 生命周期。
- 配对验证固定两个家族、两个配置和每个 case 两个种子：SFT 与 step-150 各 16 条；Utility 3/16 与 8/16，攻击臂 ASR 均为 0/8。样本很小，且属于验证集，不能用作最终泛化结论。
- 在线消融以 step-150 为相同初始化，分别运行 outcome、progress、progress+protocol 和 CHS-PO progress+protocol，最多每臂两次更新，固定训练任务和 K=2 配置；每臂新建优化器。监控 reward 分布、有效组数、entropy、KL 与输出质量，越过工程阈值立即停队列；这些阈值是预警而非安全保证。真实 harness 和未见配置保留作后续独立泛化测试。

### 在线消融前两臂诊断

按四个在线批次的 384 条轨迹分别查看：outcome 第 1/2 步安全成功 73/96、87/96；progress 第 1/2 步为 77/96、87/96。四批进展 oracle 结果均为 39–40 条满进展、8–9 条零进展、48 条未覆盖；其大部分在同一组内一致。因此任务结果和进展共同饱和，加大进展奖励系数不会自动产生 GRPO 组内差异。不同臂随机种子不配对，不能以各臂批次直接推断方法优劣。候选改进应从训练家族内部寻找未饱和难度、补可验证中间节点，并维持安全失败优先；不可丢弃失败轨迹或把超时伪装为奖励信号。

进展+协议与 CHS-PO 进展+协议的消融在单独目录继续，仍使用原健康门禁；暂停之后从相同 step-150 新建优化器，不继承已停止消融的权重。十分钟 heartbeat 只写本地状态，不等同于唤醒当前 Codex 会话。

### 扩展任务与进展 oracle 的候选验证

扫描 SFT v7 许可的 62 个 AgentDojo train 家族，20 个通过现有初步任务审计和复位检查。以冻结 15 家族池为基础，新增 workspace 23、26、39 三个只读家族，形成候选池：18 家族、648 cells、18 种训练配置；每轮 24 个 task-cell groups，连续两轮同任务 K=2 配置配对。最大提示 3,754 tokens。仅 train 家族进入候选池，已有 validation/test 和未见配置继续保留。

并行构建独立的 v4 候选 oracle：workspace 23 验证指定日程读取及密码邮件检索，39 验证安全码邮件和密码邮件的顺序，26 验证文件列表读取；真实官方工具重放与错误参数、顺序、内容、阻断、状态篡改和重放失败反例通过。候选代码未替换正在运行的 v3 oracle；还需对新家族做实际策略 rollout、完成完整准入验证，才能判断是否提供更密集的组内学习信号。扩充家族和 oracle 本身不保证解决奖励饱和，也不允许跨任务组归一化来制造差异。

### 冻结的 train-only 课程采样候选

根据已有在线训练记录生成带 SHA-256 的家族优先级快照；每个 K=2 区块的 12 家族中，8 个来自轮换覆盖、4 个按有界权重无放回采样。权重使用 train 记录里的组内差异和非满分频率，新家族保留非零概率；采样发生在任务家族层，任何组内 rollout 都不会因失败或结果而被丢弃。20 步候选账本覆盖全部 18 家族、每家族出现 5–10 次，同一任务的两个配置及良性/风险臂保持配对。快照、数据与账本哈希相互绑定，测试拒绝非训练家族和篡改。候选仍待实际在线训练验证，不保证一定提高有效信号，验证/测试集没有参与权重计算。
## Oracle versioning and expanded curriculum (2026-10-30)

The online episode, trusted reward verifier, trainer resume profile, and
checkpoint state now freeze a `progress_oracle_version`.  A rollout receipt is
accepted only when its proof and its configured oracle agree.  This prevents a
new task-specific oracle from being silently interpreted by the older v3
ruleset after a restart.

`agentdojo-workspace-progress-v4-candidate` extends the replay-checked v3
oracle with read-only, paired AgentDojo workspace families 0, 14, 16, 22, 23,
26, and 39 plus banking families 0 and 2.  The two Slack families use one
separately constrained observation transition: `get_webpage` may append only
the requested URL to the benchmark's ephemeral browser history, and a fresh
replay must reproduce that exact post-state.  All other state changes remain
ineligible.  Each oracle was admitted only after clean/injection native reset
checks, ordered-call and wrong-argument counterexamples, replay failure checks,
state-change rejection, and unsafe-action override tests.  The expanded
18-family candidate pool has 648 task/configuration cells and now has a
task-specific oracle for all 18 families. Its frozen curriculum keeps K=2
configuration pairs while mixing deterministic coverage with priority sampled
families from past informative groups.

This remains an engineering candidate, not a result claim: only families with
task-specific, replayable evidence receive progress credit.  Remaining
banking/slack and unmodeled workspace families keep the outcome-only branch
until their own counterexample suites exist.

The previous waiting v3/15-family CHS-PO queue was replaced before it acquired
GPU resources.  The active guarded queue is
`outputs/rl/2b_verified_progress_v4_curriculum_20261030/CHSPO_progress_protocol/`:
it initializes from CHS-PO step-150, uses the immutable 18-family K=2
curriculum, `CHS-PO-progress-v1`, G=4, and a two-update engineering pilot.
It waits for two truly idle GPUs plus RAM/IO headroom rather than preempting
other users.  The launcher now checks candidates 0/2/3 every 20 seconds and
selects the first ordered pair with no compute process and at most 255 MiB used,
so it can start as soon as either 0 or 2 becomes free alongside 3; this is a
launch condition, not an experimental result.

### Offline signal check for the full oracle

Before using v4 online, a separate replay of 96 archived step-150 rollouts
re-executed every recorded call in fresh AgentDojo fixtures.  All 96 replayed
with matching state and official outcomes.  Among 24 complete G=4 groups,
outcome-only had 3 variable-reward groups; v4 `progress` had 6; v4
`progress_protocol` had 7.  The score histogram for the latter was 55 at 1.15,
29 at 1.0, 6 at -0.05, 1 at 0.25, 2 at 0, and 3 at -1.  Receipt:
`outputs/rl/diagnose_step150_backends_20261030/progress_reward_replayed_96_v4.json`.

This is evidence that the verified intermediate signal increases *within-group*
ordering on the frozen old-pool traces (3/24 to 7/24), not evidence of online
learning, safety improvement, or held-out generalization.  The guarded v4
online pilot is still required.
