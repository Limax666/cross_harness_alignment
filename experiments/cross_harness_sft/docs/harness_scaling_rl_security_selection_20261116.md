> 历史v4选型：用户后续要求万级规模与50/50，当前主方案见[数据v5](harness_scaling_rl_training_pool_v5_20261116.md)。SkillLife小池不再承担主训练规模，下方原决定和证据保留。

# Harness Scaling RL 安全任务池重新选型与实施决定

日期：2026-11-16。用户授权独立检索、subagent投票和最终拍板。本决定覆盖v3固定HarnessSafe/HarnessAudit的首轮来源要求；原代码、调研和检查证据保留，不混入新配方。代码接入继续推进，不需再等一次方案批准。

## 1. 最终决定

**首轮安全训练只选SkillLifeBench的autonomous lifecycle轨道；通用训练保留ClawGym官方RL精选2,000任务。采样为75%通用＋25%安全，安全内部clean/challenged各半。**HarnessSafe、HarnessAudit、NVIDIA IPI、AgentDyn等不进入首轮默认训练；不能为了凑来源数引入没有可靠任务效用的奖励。

这是一轮小规模机制验证，不是宣称找到了大规模完整安全训练集。真正独立安全机制只有12个模板，60个实例是12×5，clean/challenged、K=2、G=8均不增加独立模板数。

## 2. 独立投票及主负责人裁决

| 独立审阅 | 首选 | 次选 | 比例建议 |
|---|---|---|---|
| [A](research/security_data_vote_a_20261116.md) | SkillLifeBench lifecycle | HarnessSecurity-Benchmark模型可控子集 | 通用/安全80/20 |
| [B](research/security_data_vote_b_20261116.md) | SkillLifeBench lifecycle | NVIDIA RL-IPI，补真实Utility后 | 通用/安全70/30 |
| [C](research/security_data_vote_c_20261116.md) | SkillLifeBench lifecycle | AgentDyn，经oracle审查后 | 首轮一个安全源，沿用原通用/安全比例 |

三个独立投票的首选一致。负责人另外核查了官方manifest、backend、评分和隔离执行源码，决定采用SkillLifeBench；**比例不按投票取平均**，仍保留已批准的75/25和16组配额，避免同时更换数据源、比例及优化目标。0/25/50%安全对照不变；10/100%仍留后续扫描。投票只帮助选择，不是可执行/效果验收证据。

## 3. 为什么它更适合首轮

- 模型能检查、选择和重载技能，区分selected版本与实际loaded绑定；环境变更不是绕过模型自动触发的恶意插件。每动作调用实际隔离代码，状态快照回写供后续动作使用。[官方backend](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/backend.py)
- 合法任务成功和攻击目标达成分别判定，joint success要求前者真、后者假；攻击按全部执行prefix累计，后续修复不能抹去已发生的违规。无需把“读过某工具”或“CSS较高”伪装成合法任务完成。[官方测量定义](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/docs/evaluation.md)
- 公开代码将任务backend与模型会话分离，可保留原状态、权限及评分器，替换模型采样/消息呈现为本项目S/T/O/C/W执行器；不要求在Codex/OpenCode等产品内部训练。[runner](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/runner.py)
- 数据提供明确template与parent归属，可以按机制切分。选的是60条lifecycle自主任务，不是388源场景中的固定调用实验，也不选其89条AgentDojo-derived runtime轨道。[官方数据说明](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/docs/dataset.md)

原生隔离依赖user/network/PID namespace、只用于任务的文件系统及seccomp；本机仍需真实执行证据，不能关闭隔离来换吞吐。源码已固定到`57bc115a56baa483d4f68824d5a9c78f00a265dd`并稀疏拉取任务/代码，未运行其qualify/replay/run。[隔离实现](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/runtime/isolated_executor.py)

## 4. 其他候选为何不作为首轮默认

| 候选 | 实际发现 | 本轮决定 |
|---|---|---|
| HarnessSecurity-Benchmark | 有独立效用/安全oracle；九机制的公开manifest合计23题，但部分考查产品自动审批/信任行为 | 后续补模型可控子集，不能直接把产品机制移除后仍称同一任务 |
| SafeClawArena | 风险面接近HarnessRisk；原全量混有自动插件执行，主汇总并未将独立safe_results用作合法效用 | 保留系统迁移评测/后续精选，不将406全量直接入GRPO |
| NVIDIA Nemotron-RL IPI | 真实session状态、明确训练许可；但当前utility只检查required工具名称出现，不能证明合法写动作/终态成功 | 后续补完成oracle再扩量，首轮权重0 |
| AgentDyn | 新增动态任务有真实状态；独立审查发现至少一处攻击描述与security判据不一致，且主要是IPI | 审核/修复oracle后再用，非首轮默认 |
| ASB / InjecAgent | 本次核查的实现分别有expected-achievement字符串模拟、预填tool-response续写等限制 | 不当作已验证的stateful在线任务源 |
| AgentHarm | 官方README请求仅评测使用，不用其例子训练 | 留作评测，不入训练 |
| HarnessSafe / HarnessAudit | 可用作风险机制参考；当前接入要补独立Utility、native证据映射或适配小子集 | 保存已有工作，首轮权重0，不再作为必须打通的前置条件 |

证据与具体源码链接在三个独立note。关键原始资料：[NVIDIA IPI verifier](https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/indirect_prompt_injection/app.py)、[HSB](https://github.com/TsingPig/HarnessSecurity-Benchmark)、[SafeClawArena](https://github.com/sunblaze-ucb/SafeClawArena)、[AgentDyn](https://github.com/SaFo-Lab/AgentDyn)、[AgentHarm使用说明](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/agentharm/README.md)。这些实现局限不表示基准没有研究价值，而是不能直接当作本项目的可靠联合奖励。

## 5. 已落盘的真实模板划分与采样

实际读取官方60任务manifest后，在B/D/E三类各4个模板内，用`SHA256('20261116|template_id')`排序：前2模板train、第3 dev、第4 test。每模板5实例、两条件全部同split，任何重复rollout/新实体都不跨split。划分不依据模型成绩。

| 安全分区 | 模板数 | 实例数 | clean/challenged条件数 |
|---|---:|---:|---:|
| train | 6 | 30 | 60 |
| dev | 3 | 15 | 30 |
| test | 3 | 15 | 30 |

逐task/template/parent的固定名单与源manifest hash见[实际划分清单](research/skilllife_lifecycle_split_20261116.json)。这是一份元数据划分，不是本机运行验收或实验成绩。公开语料和预训练暴露未知；开发/实现时不把保留分区的隐藏oracle送入训练。

通用数据仍取ClawGym-Task官方RL精选2,000任务，模板簇内部90/10 train/dev，保留其200题ClawGym-Bench作终点。用户认可ClawGym通用方向，本实现采用其训练任务版本；不因Bench的HF split叫train就将评测题移入训练。

| 采样配方 | ClawGym通用 | SkillLife clean | SkillLife challenged |
|---|---:|---:|---:|
| R1/R2/R3主配方 | 75% | 12.5% | 12.5% |
| 每完整16组更新，各组8条 | 12组／96条 | 2组／16条 | 2组／16条 |
| 2,560次尝试，无额外失败 | 1,920 | 320 | 320 |
| R4，安全0% | 100% | 0% | 0% |
| R5，安全50% | 50% | 25% | 25% |

每条rollout从相同任务初态独立reset；episode内保留原业务状态与执行prefix。K=2在两个相继训练step用当前策略重采样，**不是沿用前一步被模型修改后的环境**，也不是把clean/challenged当作两配置。仍使用18个五维配置、配置内G=8、token级真实log-prob及相同预算。

## 6. 实施迁移

v3的ClawGym下载、元数据处理和通用任务adapter继续复用；旧HarnessSafe/HarnessAudit模块与检查证据保存，默认registry关闭其训练路由。新建SkillLife backend包裹官方执行/评分器，不运行官方产品harness作为rollout宿主，也不依赖已归档的模型轨迹。新pool/manifest/run必须用v4身份，禁止混用旧三源比例。

R1 reward维持`u−c_source−c_unauthorized_persistence−0.1*c_protocol`，其中u取官方task_success，c_source取官方累计goal_success；同一攻击事件的持久化不重复罚。先发生攻击后修复仍有c_source=1；缺快照/执行错误为invalid。官方原预算与本项目12回合/16K等先导限制的差异单列，不能将适配结果冒充论文原始运行。

C2仅在独立上下文目录写`.hsrl/context_summary.txt`，各C配置共享一个仅能读该文件的工具，不授予额外任意shell/文件访问，不把摘要文件混入原业务评分快照。源任务fixture、条件标签、参考动作、oracle始终在host侧。首次真实R1执行才能证明本机reset、源oracle、五维变化、参数更新与策略同步；选型投票和静态检查不替代这些证据。

## 7. HF通用源更正与实际绑定

用户确认使用 `RUC-AIBOX/ClawGym-Task`。原始JSONL已按固定HF revision bd9043cff88b913922a165d9c022a4796aee3f3a下载并校验，实际15,402条。首轮复用官方精选2,000的selection，全部prompt在HF一一对应；8,647既有资产字节相同，25个Git漏收输入（23个日志及2个任务文件）在独立目录从HF补齐。旧vendor及证据不改，新pool需绑定HF字节与来源后重建，不能把旧缺文件表现算模型失败。详见[HF绑定证据](research/clawgym_hf_source_binding_20261116.json)。ClawGym-Bench200保留测试，不入训；完整HF待后续能力准入，不宣称全量可执行。
