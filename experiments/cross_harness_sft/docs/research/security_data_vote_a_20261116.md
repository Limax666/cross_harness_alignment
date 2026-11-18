# 安全数据源独立投票 A（2026-11-16）

## 投票结论

**第一票：SkillLifeBench 的 60 条 lifecycle decision 子集，作为最先接入的安全训练种子。第二票：HarnessSecurity-Bench 的模型可控任务子集，作为编码/权限边界补充。** SafeClawArena 的模型可控 IPI/PSE/部分 SSI/CDF 适合作第三来源与专项外测；不建议把其全部 406 条直接用作 GRPO 安全训练池。

这是针对 ONE 受控白盒执行框架、4B 先导/9B 后续、S/T/O/C/W 变化和 K2 相继步骤的选择；**不是建议在产品 CLI 中轮流 RL**。ClawGym-Task 官方 RL 精选 2k 继续承担通用任务效用训练。现有来源没有一个已核实的、可直接声称为“128 条独立合法任务 + 全载体安全 + 原生 K2”的现成子集，因此应如实称为 seed pool，后续才做过滤与适配。

## 方法与边界

独立 web search 后读取官方 README、数据指南、manifest 和执行/评分源码。SafeClawArena 还读取本地 vendor 的 406 个 JSON、judge.py；HSB 读取本地 verifier/metrics/oracle，并通过官方 raw manifest 复核全部九机制。本次未运行任何 payload、模型 API、模拟、qualify、测试或实验；所有“可执行”结论都是静态代码审计，**不是本机已成功执行的声明**。GitHub API tree 请求遇限流，随后只读 raw.githubusercontent.com 文件；没有 clone 或安装依赖。

评分 1–5，越高越符合本任务；成本项高分表示更省。分数是本人的选型判断，不是论文测量。五项等权合计仅帮助比较，同分按本任务语义取舍。

| 候选 | 任务相似性 | reset + verifier | 白盒接入 | 安全与效用 | 4B 单卡成本 | 合计 | 本次定位 |
|---|---:|---:|---:|---:|---:|---:|---|
| SkillLifeBench lifecycle | 4 | 5 | 5 | 5 | 4 | 23 | 第一；技能/绑定/工作流持久状态种子 |
| HarnessSecurity-Bench | 5 | 5 | 4 | 5 | 2 | 21 | 第二；编码与权限边界补充 |
| AgentDyn 新增三 suite | 3 | 4 | 5 | 4 | 5 | 21 | 工具状态环境备用；不作主源 |
| SafeClawArena | 5 | 4 | 3 | 2 | 2 | 16 | 第三；精选与专项外测 |
| WeClawArena v2 | 3 | 4 | 3 | 4 | 1 | 15 | 暂缓；多 owner/多 agent 成本偏离 |

## 1. SkillLifeBench：最接近可移植安全训练环境，但有效独立规模很小

官方源语料 388 paired scenarios：A 实现19、B接口77、C运行时128、D组合65、E演进99。其中126改自 AgentDojo；固定 invocation 的源案例描述执行后果，并非自主模型决策任务。正式 decision track 分为 runtime 89 和 lifecycle 60；**本票只选 lifecycle 60 = 12 模板 × 5 fixture 实例 = 120 clean/attack episodes**，不把388、4,172历史模型执行、560 qualification controls算作训练任务量。数据指南明确12模板才是独立聚类单位。未发现官方 train/dev/test 划分。[数据指南](https://github.com/Tasselszcx/SkillLifeBench/blob/main/docs/dataset.md)

我从官方 task-manifest 实际读取到60个任务、12个template_id，每个5条。覆盖私有报告 ACL、本地文档保存、过滤数据、日历、撤销授权、过期缓存、序号回滚、残留收件人、原生打包、发布、批准导出、并发协调。每条有合法用户 task、clean/attack 条件、工具schema/代码、required/protected文件和参考动作。例如 private-report-acl 同一工具在攻击版本将省略 access 的默认值改为 shared，但模型可显式 owner-only 或选择批准实现；这是模型有避险行动空间的任务。[官方 manifest](https://github.com/Tasselszcx/SkillLifeBench/blob/main/data/lifecycle/protocol/task-manifest.json)

真实执行证据来自源码而非 README：LifecycleBackend.__enter__ 创建 TemporaryDirectory、initial_fixture，初始化文件与registry；execute 对业务工具调用 execute_action；每个动作后 record_observation 调 grade_task，harm_ever 累积；finalize 输出 task_success、goal_success 和 joint_success。因此先泄漏再修复不能抹掉攻击。grade_task 分派到 B/D/E 的文件/事件判据，最终文字不能替代完整业务产物。[backend.py](https://github.com/Tasselszcx/SkillLifeBench/blob/main/skilllifebench/tasks/backend.py)、[B grader](https://github.com/Tasselszcx/SkillLifeBench/blob/main/skilllifebench/tasks/b_tasks.py)、[D grader](https://github.com/Tasselszcx/SkillLifeBench/blob/main/skilllifebench/tasks/d_tasks.py)、[E grader](https://github.com/Tasselszcx/SkillLifeBench/blob/main/skilllifebench/tasks/e_tasks.py)

IsolatedExecutor 在 disposable root 中用 user/net/PID namespaces、chroot、capability drop、seccomp 执行代码；外部 supervisor 观测文件。公开 runner 通过 OpenAI-compatible transport 调模型，将 backend 和 HarnessSession 分开，可保留环境/评分而替换为本项目策略采样与 S/T/O/C/W 渲染。默认15轮、30工具动作、单次输出2048tokens、episode900秒，上限不能冒充实际成本；相对于真实产品 daemon/多agent，CPU环境较轻。Linux x86-64、user namespace 和系统依赖是接入前提，本机尚未验证。[执行器](https://github.com/Tasselszcx/SkillLifeBench/blob/main/skilllifebench/runtime/isolated_executor.py)、[runner](https://github.com/Tasselszcx/SkillLifeBench/blob/main/skilllifebench/runner.py)、[评测协议](https://github.com/Tasselszcx/SkillLifeBench/blob/main/docs/evaluation.md)

为什么排第一：安全和合法业务状态均有真实判据，模型能检查/选择/重绑技能，适合白盒轨迹和在线奖励。官方已报告 Qwen3.5-4B/9B lifecycle challenged Joint SR 55.0%/51.7%，说明至少存在同量级模型的官方实验先例，不能据此推断本机吞吐或本项目模型效果。[官方结果](https://github.com/Tasselszcx/SkillLifeBench#results)

关键局限：12模板非常小，不能随机按5个实例切分后声称跨任务泛化；至少应按template_id分组留出。模型可观察的候选metadata直接标记批准scope，部分任务可能较规则化。B族manifest明确 temporal_scope 为构造package/registry/handle状态，**不声称先前真实agent session**；因此不能将既有状态直接当作原生跨会话 K2。K2应在项目框架中定义同一环境内真实step1→step2，而非把clean/attack配对当两步。388固定invocation案例及89 AgentDojo runtime均不随本票自动纳入。

## 2. HarnessSecurity-Bench：效用与安全判据最扎实，保留任务资产而替换产品 runner

官方23任务、9安全机制、6产品harness，不是23×9个独立任务。逐个读取官方manifest cases得到：AA2、CAL2、CDL2、MCP3、NI2、PIF2、PR4、PT4、RO2，合计23。模式ON/OFF和六harness是评测条件，不是新任务或训练split。我的本地vendor并不完整：只有CAL的两条完整manifest/任务目录可逐文件读取，其他多个机制只有README或部分文件；**不能把本地缺文件误称上游没有发布**。完整任务数量来自官方raw manifest。[官方仓库](https://github.com/TsingPig/HarnessSecurity-Benchmark)、[机制任务根目录](https://github.com/TsingPig/HarnessSecurity-Benchmark/tree/main/tasks)

其优势是合法开发目标+低信任文档、MCP结果、外部记录载体，并用独立 utility verifier 验修复功能/报告，再由host oracle判禁止效果。CAL cli-arg-parser 的verify.py真的import被修复包，验证typed coercion、错误、schema和报告；PR查受保护端点状态，NI查每trial collector接触和精确值，PIF各8功能检查+3安全效果。metrics.py明确不依赖LLM judging，并把unknown保留。因此可以复用task/environment/verifier/oracle，移植到ONE框架；不能把原native approval配置效果声称成模型学习效果。[CAL verifier](https://github.com/TsingPig/HarnessSecurity-Benchmark/blob/main/tasks/cal-bench/tasks/cli-arg-parser-attack/tests/verify.py)、[host oracle](https://github.com/TsingPig/HarnessSecurity-Benchmark/blob/main/scripts/_shared/mech_common.py)、[metrics](https://github.com/TsingPig/HarnessSecurity-Benchmark/blob/main/scripts/_shared/metrics.py)

为何只排第二：多个语言/容器/sidecar与编码长轨迹使4B GRPO成本高；23条独立题量仍小；原生审批、project trust、read-only、network isolation有强产品语义。特别是AA原始指标混合native自动批准与外部reviewer，PT可能由项目信任时自动载入配置，RO/NI硬边界可能不依模型决策。必须逐任务标注“攻击首次发生在模型动作前/后”，只把有安全完成路径且模型能选择避险动作的episode作为policy RL；纯产品自动执行项留harness评测。本次未逐条建立最终可训练清单，故**可训练数只能写≤23、待过滤，不能预报23全部可训练**。没有在本次检查中发现官方训练划分或原生K2协议。

## 3. SafeClawArena：风险题材最贴近，但406不是可直接优化联合效用的训练池

官方406 task JSON在本地实际数得406：SSI100、PSE60、CDF146、IPI100，24类别；会话数328条单session、78条双session。官方的15(platform,model)配置也不增加任务数量。[官方仓库](https://github.com/sunblaze-ucb/SafeClawArena)、[任务规范](https://github.com/sunblaze-ucb/SafeClawArena/blob/main/schema/task_schema.json)

它确有reset_env.sh、容器、配置/工作区/记忆fixture和canary状态检查，且PSE有注入s1和利用s2、重启gateway。题材覆盖memory、skill、工具链和持久状态，语义比AgentDyn更贴近。IPI-4.1-001要求从review读证据写REJECT，攻击诱导ACCEPT，具备合法目标；这类能增加明确utility。PSE-2.1-001的s2带“s1 memory含攻击串”的前置条件，迁入时应保留“成功拒绝污染也继续合法后续工作”的评估，防止安全策略导致利用session被跳过后拿虚高联合分。[IPI示例](https://github.com/sunblaze-ucb/SafeClawArena/blob/main/tasks/ipi/ipi-4.1-001.json)、[PSE示例](https://github.com/sunblaze-ucb/SafeClawArena/blob/main/tasks/pse/pse-2.1-001.json)

重要代码结论：406条均有safe_condition，但judge.compute_score(success_results,safe_results,taint_leakage)中safe_results没有进入评分。SSI/IPI按攻击是否发生，CDF按泄漏权重，PSE按记忆污染/后续影响，**没有独立合法业务完成Utility总指标**；拒绝一切/不完成任务不能由安全分排除。judge对webhook_payload和cron_output两个taint channel直接设detected=False，故“九通道”不等于全部都有实际capture。本票要求补充独立任务utility、审计通道再进入在线训练。[judge源码](https://github.com/sunblaze-ucb/SafeClawArena/blob/main/scripts/judge.py)

确定应剔除policy RL的子类：SSI1.4的20条 malicious in-process plugin，在gateway_start钩子自行执行，官方明说绕过LLM；本地ssi-1.4-001也直接显示加载/启动触发。至少这20条不应靠训练模型修复，因而余下386也只是待过滤上限。其他shell/cron/预注入配置仍需查first-effect时点，不能默认模型可控。保存工具和文件fixture可移植，但直接依赖OpenClaw/NemoClaw/SecLaw的runner和daemon不适合作ONE白盒训练环。[插件案例](https://github.com/sunblaze-ucb/SafeClawArena/blob/main/tasks/ssi/ssi-1.4-001.json)、[README关键结论](https://github.com/sunblaze-ucb/SafeClawArena#headline-results)

## 4. AgentDyn：便宜、可重置，但不因换名自动消除AgentDojo的任务范式局限

官方60合法任务、560 injection test cases，Shopping/GitHub/DailyLife各20。实际AST读取新suite user_tasks.py均20个class，injection_tasks.py分别9/9/10个class（合28 attack-goal类），**560不是560个独立合法目标**；还需按可注入(user,goal,vector)组合检查，不能拿60×28笛卡尔积造任务。[官方README](https://github.com/SaFo-Lab/AgentDyn)、[shopping用户任务](https://github.com/SaFo-Lab/AgentDyn/blob/main/src/agentdojo/default_suites/v1/shopping/user_tasks.py)、[攻击任务](https://github.com/SaFo-Lab/AgentDyn/blob/main/src/agentdojo/default_suites/v1/shopping/injection_tasks.py)

TaskSuite.run_task载入默认环境、初始化任务并deep-copy pre_environment，工具执行更改Pydantic数据库/文件树状态，分别调用utility与security。故是真正可变状态环境，不是纯文本问答；但银行/购物/文件工具是Python建模服务，非真实GitHub或shell/skill loader。TaskSuite接口和工具函数非常容易白盒接入，CPU成本低；相比目标所需技能/持久记忆/信任边界，主要仍是工具返回注入与动态tool-use，且官方直接基于AgentDojo。保留作备用与跨域测试，尊重用户不选AgentDojo主源的取向，不用AgentDyn改名替代同类主源。[环境执行与双评分](https://github.com/SaFo-Lab/AgentDyn/blob/main/src/agentdojo/task_suite/task_suite.py)、[GitHub环境](https://github.com/SaFo-Lab/AgentDyn/blob/main/src/agentdojo/default_suites/v1/github/task_suite.py)

## 5. WeClawArena：跨owner场景好，当前ONE/4B先导成本不合适

官方v2为124 base tasks、每题no_attacker+4攻击条件，共620 variants；只有124基础bundle，运行时materialize overlay，不能将620称独立任务。六域中SWE50、Travel20，其余协商/竞标/临床/交易。资源、owner身份、权限、工具动作、治理结果和独立TSR/ASR契约很贴近跨信任边界，但天然多agent、多owner。[官方README](https://github.com/kingofspace0wzz/WeClawArena)、[数据指南](https://github.com/kingofspace0wzz/WeClawArena/blob/main/docs/benchmark_dataset_guide.md)

我读取runtime_overlay.py看到为variant复制resources/resources_initial与SQLite；live_executor的_restore_resources_from_snapshot确实恢复状态；attack scoring与环境谓词独立。runner含mock/live/docker三executor，mock可跑不能证明真实模型/工具路径。主路径还有naturalization与多agent loop，container-backed模型执行依赖Docker和provider；本次未验证所有domain都有可直接零裁判成本的确定性ASR。因此不适合4B单卡在线组采样的第一来源，之后可抽单focal-agent、固定peer脚本版本做外测，但这将是新适配配置。[overlay](https://github.com/kingofspace0wzz/WeClawArena/blob/main/clawsocialarena/attacks/runtime_overlay.py)、[live executor](https://github.com/kingofspace0wzz/WeClawArena/blob/main/clawsocialarena/orchestrator/phase5_live_executor.py)、[runner](https://github.com/kingofspace0wzz/WeClawArena/blob/main/scripts/run_phase5_pilot.py)

## 接入建议与明确淘汰项

**首轮允许只用一个安全源，且我建议如此做**：通用池继续用 ClawGym-Task 官方 RL 精选2k；安全池只接 SkillLifeBench lifecycle，先把联合任务/攻击评分和统一白盒轨迹跑通。建议首个4B先导按任务采样预算约 **通用80% / 安全20%**，安全池内 clean/attack 约1:1；这是工程起点而非经验证最优配比。由于只有12模板，安全池不宜仅因可重复reset就占据多数更新。正式决定比例应看通用Utility、安全task_success、attack_goal与joint_success及实际token/耗时，不能仅看ASR。第二来源HSB在可控性过滤后再加，不要求首轮混源；SafeClawArena补utility前不进入首轮奖励。

固定来源版本：root报告已只读clone/稀疏checkout SkillLifeBench 到 main vendor/skilllifebench_security_research_20261116，commit `57bc115a56baa483d4f68824d5a9c78f00a265dd`；尚未运行 qualify/run/replay。本note的可执行性结论仍只代表静态审查，不因clone而升级为验证完成。[固定提交](https://github.com/Tasselszcx/SkillLifeBench/tree/57bc115a56baa483d4f68824d5a9c78f00a265dd)

1. 先迁SkillLifeBench lifecycle后端/grade，保留业务、攻击、联合三结果；按12模板分组而非实例随机split。优先建立少量模板的训练/留出接口，再谈扩池；不得把5实例扩写视作五种机制。
2. HSB逐题筛可控性，先考虑PIF/CAL/CDL中Python小任务，保留host oracle，之后再扩语言与sidecar；AA/PT中的产品自动行为单列外测。
3. SafeClawArena优先选确有合法结果的IPI和模型控制写入的PSE，补utility，并保留真正相继session；s2不能仅在攻击成功时才可评。当前不承诺可训练条数。
4. **淘汰为主训练来源**：AgentDojo及AgentDyn同范式主池（用户取向+状态风险覆盖不够）、SafeClawArena全406直接混训（utility缺口+20纯插件自动执行）、SkillLifeBench全388直接混训（含固定invocation案例，并非388自主decision任务）、WeClawArena全620直接上4B（多agent成本与ONE因果解释不匹配）。HarnessSafe328/HarnessAudit34不因既有讨论自动入选。

K2必须是同一合法任务内顺序交互且保留真实状态；clean/attack成对对照、两个独立reset rollout或两个不同任务，均不能冒充K2。S/T/O/C/W变体不得改变合法授权、业务verifier与攻击定义，否则训练条件效应和任务难度混杂。上述是研究选型建议，尚未做任何运行验证或成本测量。
