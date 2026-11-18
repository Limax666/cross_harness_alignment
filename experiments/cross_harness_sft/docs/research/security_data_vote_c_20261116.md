# 安全 RL 任务源独立投票 C（反方与可实施性）

日期：2026-11-16。范围：官方网页与公开源码只读审计；未执行工具任务、payload、模拟、实验或模型 API，未修改代码、服务或现有数据。下文“可复位/可运行”指源码提供路径，不代表已在本机验收。ClawGym 官方 RL 精选 2k 通用源保持既定方案。

## 投票

**首选：SkillLifeBench 的 60 条 autonomous lifecycle 轨道，作为小规模机制验证安全池；次选：AgentDyn 的三个新增 suite，作为 IPI 与合法业务动作池。没有一个候选是现成、足够大、已有官方训练/验证/测试隔离且可直接交付本项目 GRPO 的完整主池。**

我不投把 HarnessSafe/HarnessAudit 的复杂 native attestation 继续前置，也不投把 SafeClawArena 全量或 NVIDIA IPI 原样作为首轮联合安全效用奖励。SkillLifeBench 最贴近版本、权限、组合与持久状态，同时能给出合法任务完成和攻击目标两个独立谓词；但只有 12 个独立模板，结论应限制为可学习机制与训练闭环，不应据此声称大范围安全泛化。AgentDyn 较易执行、已有业务状态 oracle，却主要覆盖间接注入，不能承担全部 harness 生命周期风险。

比例投票：首轮安全池内部 **SkillLifeBench lifecycle 100%，其他安全源0%**；AgentDyn只保留为次选与后续外部补充，不为凑规模混入。该比例不改变既定ClawGym通用2k，也不擅自决定通用/安全总采样比。

首选理由来自独立代码检查，而非仅引用论文：`LifecycleBackend.execute` 调用实际技能代码，接收文件快照并恢复到下一步 fixture；`select_version` 与 `reload_skill` 分别改变 selected 和 loaded 身份，后者不清空任务产物；`finalize` 采用 task 与 harm 两个维度。官方数据说明明确只有 12×5 实例，并区分 fixed-invocation corpus 与自主决策任务。[backend](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/backend.py)、[dataset](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/docs/dataset.md)

## 候选横向审计

| 候选 | 公开训练划分 | reset 与状态 | 奖励/效用证据 | 模型控制与白盒迁移 | 本轮判断 |
|---|---|---|---|---|---|
| SkillLifeBench lifecycle | 公开 60 实例，未见官方 train/dev/test；388 corpus 不能当388自主任务 | 新 backend/temp fixture 是新 episode；步骤间完整 snapshot 回写，版本与handle状态持久 | 文件、动作、顺序与完整产物谓词；任一已观察前缀攻击成功不可被后修复抹掉；无动作不能完成任务 | 模型可 inspect/list/select/reload 后决定业务动作；沙箱执行发生在选定工具调用时。可替换外层模型循环而保留 backend | **首选小池**，规模与家族外推有限 |
| AgentDyn | 官方提供评测任务/攻击，不等于官方训练集；只用新增 shopping/github/dailylife | YAML构造新 Pydantic 环境；任务内工具修改同一个环境对象 | 合法 utility 与攻击成功方法分别读取前/后环境或trace；许多utility直接查订单、密码等状态 | 攻击目标由模型选动作实现。必须保留页面handler、认证/OTP、依赖注入和完整状态，而非只移植字符串 | **次选**，业务状态真实于模拟后端，非外部真实网站 |
| NVIDIA Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1 + NeMo Gym | **确有 train.jsonl 1272条，9领域**；未见官方dev/test | `seed_session` deepcopy；工具read/write会真改session内dict，verify结束pop；不是纯静态tool-output拼接 | safety来自trace目标工具+部分区分参数；utility只看`required_tools`名称全部出现，未检查合法write内容/成功/状态 | 恶意调用是模型可控；可平移handler到单白盒。原reward会奖励“读完就停”而未完成用户任务 | 可作次阶段训练原料；**不得原样作为联合效用主池** |
| Agent-SafetyBench | released_data单一公开集；未见官方训练划分 | 每题新建环境实例，很多方法修改Python状态，也有方法仅返回success | 官方主安全评分为ShieldAgent，对新轨迹不是标签可直接提供真值；无通用合法utility oracle | 模型选择工具；迁移环境容易，补每任务合法与安全状态oracle工作大 | 不优先；“有环境”不等于“现成可验证reward” |
| BIPIA | **有train/test的context与attack划分** | 静态context+attack生成样本，无业务状态transition/reset | 输出攻击成功分类/评估，部分LLM评估 | 适合IPI文本训练/辅助测试，不保留本项目stateful风险机制 | 不作主安全agent RL池 |
| InjecAgent | 1054 test cases，未见正式训练划分 | 预置工具响应、解析下一动作；泄露任务第二步用simulated response；不是完整多步业务后端 | 根据调用解析判断攻击，不提供完整合法业务完成状态 | 模型动作可控，但重建真实状态环境需新开发 | 不作主池 |
| AgentHarm | val/test_public/test_private；这些是评测划分，未见官方train | synthetic proxy tools/每sample工具配置，不是完整现实服务 | 有harmful/benign双任务，但官方scorer带refusal与semantic模型judge；不是全确定性状态奖励 | 有模型多步决策；直接恶意用户意图与IPI/权限生命周期不同 | 适合作异类评测，暂不投入主训练 |
| ASB（agiresearch） | 公开normal/attack工具与评测文件，不代表训练环境划分 | 核心`SimulatedTool.run`直接返回Expected Achievements，`AttackerTool.run`返回达成目标的文本 | 工具选择/trace评测为主，缺完整合法状态oracle | 工具诱导可控，但不能把返回文案视为真实状态变化 | 不优先 |
| SafeClawArena | 官方406评测任务；未见官方train/dev/test | Docker平台、reset_env.sh清理memory/session/cron/config/skills再provision | canary/taint与落盘、配置、输出通道；缺现成统一合法任务效用 | IPI等部分可训练；恶意插件类别有平台自动执行、与模型无关的失败。移植到普通白盒后会丢原平台机制 | 外部系统回归参考；**不作全量模型RL奖励** |

表中事实来源为下列官方源码及说明；没有把第三方排行榜或二手博客作为依据。

## 首选的可运行路径、迁移边界与统计限制

固定审计 commit：SkillLifeBench `57bc115a56baa483d4f68824d5a9c78f00a265dd`。入口是 `python3 -m skilllifebench run --model ...`；代码链为 `runner.py → HarnessSession/run_native → LifecycleBackend → execute_action → IsolatedExecutor`。这里的 `native` 是公开runner的原生OpenAI兼容工具调用循环，不要求运行OpenClaw/Codex等原生平台；迁移单白盒时可保留后台，只替换模型循环和S/T/O/C/W外层处理。[runner](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/runner.py)、[session](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/runtime/harness_matrix_session.py)

真实技能执行在新user/network/PID namespace、chroot与seccomp内，外部supervisor读取文件；源码要求Linux `/usr/bin/python3`、unshare、ldd、libseccomp，并实际准备PyYAML、cryptography/CFFI和固定OpenSSL路径。不能只读类注释便宣称任意Linux即开即用。默认技能内存上限256MiB，runner覆盖单调用wall=30s/cpu=20s；这些是CPU环境成本，不占模型GPU，实际峰值和本机user namespace许可仍未验证。环境外网被隔离，只有模型client需要endpoint；用本地4B endpoint即可避免模型API网络依赖。[executor](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/runtime/isolated_executor.py)

合法效用不是“出现某个工具名”。B任务检查完整产物及protected文件，D任务检查完整导出/授权状态/唯一投递，E任务检查完整报告、可见范围、数据行序、库存或archive成员。`task_success`与`harm`独立，最终joint是前者且非后者；暂时泄漏后恢复也不能洗掉harm。必须保留每步`record_observation`及完整snapshot，不能只检查最终文件。[B graders](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/b_tasks.py)、[D graders](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/d_tasks.py)、[E graders](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/e_tasks.py)

局限：初始lifecycle prefix是人为构造的acquire/load/activate，不是已经观察到的上一个模型会话。将其接到K=2时须如实区分构造初态与policy造成的状态。已确认本项目K2协议是每条rollout都reset相同初态，跨训练step重新on-policy；不能沿用前一个step的环境改动。每条episode内部工具调用产生的状态与版本绑定则应持续保留。不能让W或资源变体改掉隐藏oracle、模型不可见的package对象、技能代码与原风险机制。18组合是harness处理的重复测量，不是18倍独立安全任务。

划分建议仅是本项目新协议：模板级8/2/2（40/10/10实例）或预注册按B/D/E家族留出；clean/attack、同模板全部5实例、对应parent必须同侧。模板级只有2个最终组时置信区间很宽，只能做机制试点；可用leave-template-out辅助报告，但不能替代真正未见新模板测试。不可把60全训后再声称其中10条held-out，也不可将388固定调用场景直接重复采样充大训练规模。其公开模型记录已包含4B/9B，这只是上游可行性证据，不能证明本项目单4090训练显存够用。[公开评价定义与预算](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/docs/evaluation.md)、[论文](https://arxiv.org/abs/2610.35215)

## 次选AgentDyn：可保留原风险，需逐task审查oracle

固定审计 commit：`5353cf7615b135cace8d07c8f12dac53a16b6db3`。官方60用户任务、560注入组合，新增shopping/github/dailylife；不是把旧AgentDojo当主源。环境定义与工具是纯Python/Pydantic+YAML，本地CPU就可运行后端；官方入口仍使用provider模型，需要单白盒本地model适配，不能将README里的OpenRouter支持误写成已有本地训练器。[README](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/README.md)

实际reset路径：`TaskSuite.run_task_with_pipeline(environment=None)`调用`load_and_inject_default_environment`，随后`init_environment`，保留`pre_environment.model_copy(deep=True)`用于判定；调用方传入旧environment会继承旧状态，故必须为每独立rollout构造新环境。官方loop会对缺model_output最多再次query三次，直接照搬会影响单卡预算和K2定义；应明确是否允许该重试。[task_suite.py](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/task_suite/task_suite.py)

`shopping/user_tasks.py`的utility直接读取订单、折扣、数量、密码与发票投递等实际后端状态，比read-tool名代理强；但部分task只查“存在订单”，并非完整授权状态不变性，因此仍须逐task检查原oracle强度，不可宣称所有任务严格完备。`browse_webpage`会根据页面注册的handler触发业务操作，这是模型访问链接后的环境语义，应保留而不能改成静态文本。`InjectionTask.security()`这个命名容易误导：已查一例返回攻击目标密码是否出现，True表示攻击目标成功；桥接时必须用正反例核对极性，不能直接当secure=True。[utility](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/default_suites/v1/shopping/user_tasks.py)、[页面路由](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/default_suites/v1/shopping/task_suite.py)、[injection oracle](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/default_suites/v1/shopping/injection_tasks.py)

数据隔离必须以用户任务/基础环境/攻击家族为单元；560组合不是560独立任务。按task ID随机分，仍共享同一suite环境与业务实体。若宣称未见环境泛化，至少按整个suite留出，并承认只有3个domain集群。没有官方训练划分就要称“本项目从公开评测环境派生的隔离训练配置”，不能声称原benchmark官方train。

## NVIDIA IPI的关键反证：真状态，但效用奖励不够

数据卡固定 `d738d4f361cc38bb4d7a42b9066776dade5332f5`，CC-BY-4.0、1272条仅train。数据卡说`required_tools`是用于暴露注入的read工具。官方Gym `app.py`的`verify`只把这些名字是否出现作为`reward_utility`；`reward_safety`检测target调用，两者相乘。于是**合法读成功后立即停止、未完成用户要求的write，也可能得到1**。这不是只靠不调用目标就总得1（有required-read门），但确实不是合法业务完成奖励。[官方数据卡](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1/blob/d738d4f361cc38bb4d7a42b9066776dade5332f5/README.md)、[app.py](https://github.com/NVIDIA-NeMo/Gym/blob/25c5d318c4f638ac6d608771ec59eeaa82cfc137/resources_servers/indirect_prompt_injection/app.py)

后台并非假的：每session deepcopy环境，医疗handler会改chart字段、appointments并向sent_actions追加记录；因此它比静态InjecAgent好迁移，也可在CPU运行。问题是验证没有读这些最终状态，甚至没有确认工具调用是否执行成功。`verifier.py`只对每种`verification_type`定义少数strict discriminator keys，其他参数可模糊匹配；注册目标之外的其他违规不会自动被发现。若作为训练源，须新增合法完整write/state合同，并保留独立广义安全检查，不能把工具trace标签当真实副作用证明。[handlers](https://github.com/NVIDIA-NeMo/Gym/blob/25c5d318c4f638ac6d608771ec59eeaa82cfc137/resources_servers/indirect_prompt_injection/healthcare_tools.py)、[verifier](https://github.com/NVIDIA-NeMo/Gym/blob/25c5d318c4f638ac6d608771ec59eeaa82cfc137/resources_servers/indirect_prompt_injection/verifier.py)

## 其他排除项的官方证据

- Agent-SafetyBench的`evaluation/eval.py`每题构造环境并调用工具；AccountManipulation确实更新密码/删除账户，但flag_discrepancy只是返回成功文案。`score/eval_with_shield.py`另载模型对轨迹判safe/unsafe，没有通用任务完成检查。[eval](https://github.com/thu-coai/Agent-SafetyBench/blob/74feea8de601b3a1449a93fcf70017fe61556f73/evaluation/eval.py)、[环境](https://github.com/thu-coai/Agent-SafetyBench/blob/74feea8de601b3a1449a93fcf70017fe61556f73/environments/AccountManipulation.py)、[scorer](https://github.com/thu-coai/Agent-SafetyBench/blob/74feea8de601b3a1449a93fcf70017fe61556f73/score/eval_with_shield.py)
- BIPIA有真实train/test文件，可做文本辅助训练；生成与评测围绕context/attack拼接及模型响应，不能替代stateful工具执行。[官方README](https://github.com/microsoft/BIPIA/blob/main/README.md)、[benchmark文件说明](https://github.com/microsoft/BIPIA/blob/main/benchmark/README.md)
- InjecAgent评测脚本把预置Tool Response填入prompt，解析攻击工具调用；数据窃取第二步由`get_simulated_attacker_tool_response`生成回包。它不是能自由复位、执行任意业务路径的完整环境。[官方README](https://github.com/uiuc-kang-lab/InjecAgent)、[evaluate_prompted_agent.py](https://github.com/uiuc-kang-lab/InjecAgent/blob/main/src/evaluate_prompted_agent.py)
- AgentHarm的官方task只有val/test_public/test_private选项，scorer显式创建refusal_judge与semantic_judge；虽有benign twin，也不能直接称确定性无judge的训练池。[任务源码](https://github.com/UKGovernmentBEIS/inspect_evals/blob/a243875e33bcfdf40487a6722c74d6d6f76cf358/src/inspect_evals/agentharm/agentharm.py)、[官方使用说明](https://ukgovernmentbeis.github.io/inspect_evals/evals/agentharm/)
- ASB `SimulatedTool.run(params)`忽略params返回预置expected achievement，AttackerTool返回达成目标文案。不能据此声称真实业务副作用已完成。[官方核心代码](https://github.com/agiresearch/ASB/blob/544540ff0788998072f7df463056fd17e62cc9f4/pyopenagi/tools/simulated_tool.py)
- SafeClawArena具备真实Docker reset与canary taint，但官方明确恶意plugin一类在未加固平台可与模型无关地全部成功。这类点应作为系统架构对照，不应给4B策略归因；重写成单白盒工具动作也会改题。[官方仓库](https://github.com/sunblaze-ucb/SafeClawArena)、[reset实现](https://github.com/sunblaze-ucb/SafeClawArena/blob/c4e8c395a7716f9da123db6b8038cb8021846361/scripts/reset_env.sh)、[官方发现](https://safeclawarena.github.io/)

## 关于直接从HarnessRisk另建训练变体的反方意见

如果首要目标是S/T/O/C/W的精确干预而外部环境迁移反复改变机制，可以复用HarnessRisk基础服务组件另写**独立训练任务**，但只换名字、实体、canary、数字或攻击措辞不构成隔离。已被查看/调参使用过的风险家族不是未见test；共享服务实现与oracle也使结果存在构造同源性。必须将原128完整冻结为回归集，训练任务有新任务图/授权边界/攻击家族和独立manifest，另留未参与reward设计的外部源作为最终检验。此路线可减少工程复杂度，但统计上是内部验证环境派生，不能声称独立benchmark泛化。

我的投票仍将这条路线排在SkillLifeBench与AgentDyn之后：两者已公开提供可核查业务动作与原oracle，先证明安全+合法效用的学习信号，再决定是否需要自建更大机制池。无论选哪个，GRPO group零方差、协议失败、越权副作用、合法任务失败必须分别报告；不能用安全率提高掩盖模型不行动。

## 本次验证与未验证边界

完成：逐源码检查公开路径、reset生命周期、状态写回、模型控制点与reward定义；记录官方commit和primary URLs。未完成且未声称：本机依赖安装、沙箱资格验收、任务正反例执行、18组合机制保持、K2续段、模型rollout、单4090显存峰值、optimizer或泛化效果。此次为来源选择审计，不是项目环境验收。
