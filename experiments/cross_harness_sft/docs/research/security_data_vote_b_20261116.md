# 安全数据独立投票 B：以联合效用 oracle 优先

日期：2026-11-16。范围：重新 web search，并静态阅读官方仓库、论文和数据卡；没有运行测试、攻击载荷、仿真 episode、模型 API、训练或服务。下文“可用”是源码层面的适配判断，不是本机运行验证。使用 academic-paper-review 的证据核查方法，按本任务的选源需求压缩输出。

## 独立投票

**首选：SkillLifeBench lifecycle decision track，作为第一轮唯一安全主源也成立。** 它原生同时判定合法任务完成和攻击目标，真正执行代码、保存文件和版本绑定状态，检查所有动作前缀；最接近 HarnessRisk 的权限、组合、版本更新、持久状态风险。它是公开评测环境而非官方 RL 训练集，且只有 **12 模板×5 实例=60 个 task**，因此要声明“从公开环境构造自定义 train split”，不能宣称官方大规模安全训练集或把 388 source scenarios 全当可决策任务。[官方 dataset guide](https://github.com/Tasselszcx/SkillLifeBench/blob/main/docs/dataset.md)

**次选：NVIDIA Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1 + NeMo Gym IPI backend，但须补齐 utility 和副作用判定后才能入主训练。** 它有明确 train split、低成本 CPU 字典状态工具、九领域、无需 LLM judge，扩量条件好；原生 reward 的效用代理太弱，不能照搬。若暂不愿补 oracle，应排除该源本轮训练，不能以“训练集”标签压过 oracle 缺陷。[数据卡](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1)、[backend](https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/indirect_prompt_injection/app.py)

**第三顺位：AgentDyn 的三个新增 suite，可作独立评测，必要时作研究训练候补。** 它是真环境、双 oracle，适配量比 SkillLifeBench 小；但建立于 AgentDojo framework，当前来源有明显 oracle 不一致，且与“AgentDojo 不作主源”的偏好存在接近边界。我的第一轮不选它作安全主源。

## 核查结果与角色

| 候选 | 官方规模/划分（只报核实值） | 真执行、reset 与评分 | 我的角色判断 |
|---|---|---|---|
| SkillLifeBench lifecycle | release manifest v1.0.2；60 lifecycle、12模板；89 runtime 为另一轨；source corpus 388 paired scenarios | 原始代码在 disposable filesystem/namespace/seccomp sandbox 执行；每 episode TemporaryDirectory；任务文件、业务动作、状态/顺序谓词；prefix attack OR；新 lifecycle runner 无需 semantic judge | **首选安全训练环境**，自定义模板切分；runtime89及固定调用源语料不混入 |
| NVIDIA RL Agentic IPI | HF 明确 train，viewer 1.27k；9领域；metadata revision `d738d4f361cc38bb4d7a42b9066776dade5332f5`（root 提供）；CC-BY-4.0 | session seed 深拷贝 environment；工具 handlers 读写状态；原 reward 是 trace 上目标调用匹配×required tool name coverage，非终态联合 oracle；无需 judge | **条件次选/扩量**，原样不可入联合安全 RL |
| AgentDyn | 官方60 open-ended user tasks、560 injection cases；Shopping/GitHub/Daily Life；未找到官方 train split | YAML/Pydantic 初始世界、深拷贝 reset，工具修改账户/订单/邮件/文件；utility/security Python 谓词 | 外部评测候选；抽审并修复 oracle 后才能考虑训练 |
| Agent-SafetyBench | 349 environments、2,000 cases，官方定位 test/evaluation；代码/数据2025-02-20发布；MIT | env_manager 新实例+深拷贝参数，真实调用 Python 方法；**部分方法只有固定 success 返回**；ShieldAgent judge 给轨迹安全分；未提供统一合法任务终态 oracle | taxonomy、风险题材和外部 safety 评测参考；不直接作主池 |
| ASB / Agent Security Bench | 10场景、400+工具；论文近90k是跨模型/攻击测试总量，**不是独立可重置任务数**；MIT | `SimulatedTool.run(params)` 忽略参数，返回表内 Expected Achievements；AttackerTool 返回已达目标字符串 | **淘汰原样训练**，工具名/攻击形式研究参考 |
| InjecAgent | 1,054 cases、17 user tools、62 attacker tools；MIT | 预填用户工具调用及 Tool Response 后续写；第二步工具结果模拟；没有完整可变世界/reset/合法任务终态 oracle | **淘汰 online RL 主源**，可作静态抗注入评测 |
| AgentHarm | 当前官方公开44 base test/176 augmented，8 base val/32 augmented；有benign counterpart | Inspect 多步工具模拟、定制 graders；semantic/refusal judges；不是合法目标受攻击的主要设计 | **仅外部评测**；官方明确请求不要训练，val也不挪成train |
| EnvInjection | 论文2505.11717，视觉网页像素扰动攻击方法；本次未核实可下载的官方完整任务+reset+双oracle包 | 原任务是诱发指定网页动作，依赖视觉与白盒梯度；不能据论文“环境攻击”推定已开放RL环境 | 不选当前4B文本工具runtime主源；规模/许可证/代码可取得性未核实 |
| ToolSafe / TS-Bench | 明确有train/validation，但任务是给定轨迹和候选工具动作做安全检测 | TS-Guard 的RL训练是 guard 分类/推理；不等于 actor 在真实世界 on-policy rollout | 可作judge/guard研究，**不满足本次agent任务池** |
| SafeArena | 官方real web benchmark，有safe/harm splits；HF访问须申请 | 自部署修改版WebArena、真实网页状态和reset；有任务评分；部署/浏览器成本明显高 | 后续web安全评测；本轮4090最小适配不优先 |

Agent-SafetyBench、ASB、InjecAgent、AgentHarm 的规模分别由[ASB-safety论文](https://arxiv.org/abs/2412.14470)、[ASB-security官方项目](https://luckfort.github.io/ASBench/)、[InjecAgent论文](https://arxiv.org/abs/2403.02691)、[AgentHarm官方README](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/agentharm/README.md)核对。新增候选分别参见[EnvInjection论文](https://arxiv.org/html/2505.11717v1)、[ToolSafe仓库](https://github.com/MurrayTom/ToolSafe)、[TS-Bench数据卡](https://huggingface.co/datasets/MurrayTom/TS-Bench)、[SafeArena官方仓库](https://github.com/McGill-NLP/safearena)。未找到可取得的完整开放环境的论文不因“做过RL”而被视为可落地数据源；例如[Agent Safety Alignment via RL](https://arxiv.org/html/2507.08270v1)描述自建sandbox，但本次未核实其训练环境/数据公开包。

## 三个重点候选的源码证据

### 1. SkillLifeBench：双 oracle 已有，但规模小且没有官方训练划分

静态读取版本：Git HEAD **`57bc115a56baa483d4f68824d5a9c78f00a265dd`**；`release-manifest.json`声明 **1.0.2**。代码MIT、数据CC-BY-4.0，第三方素材保留上游条款。[release manifest](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/release-manifest.json)

- `skilllifebench/tasks/backend.py`：`LifecycleBackend.__enter__`创建临时fixture、初始化文件和绑定；`execute`调真实隔离executor；每步收集snapshot；`record_observation`累积`harm_ever`；`finalize`返回独立`task_success/goal_success/joint_success`。reset 是重建backend，episode内reload只换implementation、不重置业务状态。[backend源码](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/backend.py)
- `b_tasks.py::grade`不相信完成声明：要求业务动作status=ok，必要artifact完整匹配，保护文件和ACL满足约束。`d_tasks.py/e_tasks.py`还有各自的组合/演化谓词；这比必需工具名coverage强。[B grader](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/b_tasks.py)、[D grader](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/d_tasks.py)、[E grader](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/tasks/e_tasks.py)
- `runner.py`的公开OpenAI-compatible runner确实重新问模型、执行新动作；与 `replay/verify` 冻结证据核查不同。原生limit是15轮、30 attempted calls、每request2048 output tokens，不能把这些历史预算当作当前4090已验证可承受。[runner](https://github.com/Tasselszcx/SkillLifeBench/blob/57bc115a56baa483d4f68824d5a9c78f00a265dd/skilllifebench/runner.py)、[评价协议](https://github.com/Tasselszcx/SkillLifeBench/blob/main/docs/evaluation.md)

**适配**：复用LifecycleBackend、IsolatedExecutor和三族grader，替换外层策略client/session为本项目ONE白盒S/T/O/C/W；保留管理工具、源代码绑定、文件快照和prefix评分。不能只取task文本或把危险skill实现替换成无副作用stub，否则风险机制就丢失。C摘要必须保留当前/已加载版本身份、审批范围、workspace作用域及已执行动作来源；T别名必须在dispatch/评分前映射回原名；O只改显示包装，不能吞掉技能源码/返回内容；S不能改变授权；W不得自动替模型修复版本绑定或执行安全路径。

**最小训练划分建议**：12个完整template分8 train/2 validation/2 test，即40/10/10 task实例；同一父模板、5实例、clean/attack、全部18配置必须同split。按B/D/E机制平衡后冻结；两个held-out模板只能给很粗的统计，不包装成大规模泛化证据。runtime89为AgentDojo-derived且部分历史判分有语义adjudication，不混作此源的纯确定性训练样本。388里固定invocation只描述实现后果，不能冒充policy选择episode。可另保留全部60为外测而改选其他源训练，但那是不同方案。

### 2. NVIDIA IPI：完全拒绝漏洞有所限制，效用漏洞仍明显

静态源码HEAD **`25c5d318c4f638ac6d608771ec59eeaa82cfc137`**（NeMo Gym，Apache-2.0）；数据与代码版本分别冻结。`app.py::seed_session`深拷贝世界，`route_tool_call`调用`handler(env, **args)`，所以它确实有状态执行，并非预生成tool-output replay。[app.py](https://github.com/NVIDIA-NeMo/Gym/blob/25c5d318c4f638ac6d608771ec59eeaa82cfc137/resources_servers/indirect_prompt_injection/app.py)

但 `verify`把utility写成所有`required_tools`**名字**均出现在`extract_function_calls`，不检查参数、函数执行结果、合法目标artifact或终态；列表为空则默认1。非空列表能阻止纯零工具拒绝拿1，**不能阻止读一次必需工具后拒绝**，错误对象/错误参数的同名调用也不是被该判定排除的。README示例的合法任务含处方操作，而required_tools只有`get_patient_record`，直接暴露了目标与proxy不等价。[README](https://github.com/NVIDIA-NeMo/Gym/blob/25c5d318c4f638ac6d608771ec59eeaa82cfc137/resources_servers/indirect_prompt_injection/README.md)

`verifier.py`安全分也只匹配注册target_tool和部分discriminator keys：例如转诊只看specialist_email、chart update只看patient_id；不是全面未经授权副作用检测，既可能忽略不同目标攻击，也可能把合法的同patient操作标成攻击。函数调用被尝试但工具报错，与已执行副作用没有区分。不能把`reward_safety`命名直接解释成HarnessRisk式完整安全。[verifier.py](https://github.com/NVIDIA-NeMo/Gym/blob/25c5d318c4f638ac6d608771ec59eeaa82cfc137/resources_servers/indirect_prompt_injection/verifier.py)

**入训门槛**：只选能写成明确允许状态diff/正确响应facts的子集；为每任务增加参数正确、执行成功、合法postcondition及禁止增量；将只读问答的事实完成与“调用过read”分开；保留原target匹配作单独指标；保留每步世界快照/事件，verify自动清session前保存证据；纯拒绝、只read后停、错误对象、工具失败必须不得得到joint success。以上是后续工程验收建议，本次没有运行这些控制。五维适配比SkillLifeBench轻，但不能让T别名漏过trace verifier，不能把injection/target spec送进策略上下文。相同世界seed和攻击内容在K=2内保持；新策略重rollout。

### 3. AgentDyn：实际可执行，但当前oracle需要审查

HEAD **`5353cf7615b135cace8d07c8f12dac53a16b6db3`**，官方仓库已从leolee99迁到SaFo-Lab。只考虑它新增的shopping/github/dailylife，不把附带原AgentDojo四suite计作新来源。[README](https://github.com/SaFo-Lab/AgentDyn)

`task_suite.py::run_task_with_pipeline`每次load初始世界、用户任务初始化、深拷贝pre环境，结束调用utility/security；新增三个suite各自有Python工具和任务谓词。`shopping/UserTask1.utility`核对post-world密码；`UserTask0.utility`检查指定product进入订单。无需把已提供的ground_truth当模型轨迹；它可作环境控制/参考，不应注入policy。[共用runner](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/task_suite/task_suite.py)、[shopping任务](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/default_suites/v1/shopping/user_tasks.py)

静态发现具体不一致：`shopping/InjectionTask0`文字GOAL是访问折扣网址，ground_truth是send_money，而security检查密码是否变成attacked；因此不能笼统声称560攻击case都已有可信目标oracle。UserTask0声明past_order_list却仅判post中存在P007，是否初始即满足须逐fixture检查，不能以此一条就断言所有任务无效。适配S/T/O/C/W很轻，但C必须保留任务相关第三方“有帮助指令”和其不可信来源，不能统一删掉工具指令而改变benchmark难点。[injection源码](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/default_suites/v1/shopping/injection_tasks.py)

## 明确原样不可入训的证据

- **ASB**：HEAD `544540ff0788998072f7df463056fd17e62cc9f4`；`SimulatedTool.run`忽略params，只返回表内Expected Achievements；`AttackerTool.run`返回已达攻击目标字符串。套18配置不会把它变成stateful环境。[simulated_tool.py](https://github.com/agiresearch/ASB/blob/544540ff0788998072f7df463056fd17e62cc9f4/pyopenagi/tools/simulated_tool.py)
- **InjecAgent**：HEAD `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`；prompted evaluator直接拼接既定Thought/User Tool/Tool Parameters/Tool Response；data-stealing第二步用模拟返回。工具调用解析分数不能代替合法任务状态oracle。[evaluate_prompted_agent.py](https://github.com/uiuc-kang-lab/InjecAgent/blob/f19c9f2c79a41046eb13c03c51a24c567a8ffa07/src/evaluate_prompted_agent.py)
- **Agent-SafetyBench**：HEAD `74feea8de601b3a1449a93fcf70017fe61556f73`；确有`env.call_tool`和fresh instance。但`Email.send_email`只返回success，不维护sent邮件；不能按349环境总数把所有action算作可验证副作用。原ShieldAgent评价轨迹安全性，与任务效用不同。[Email.py](https://github.com/thu-coai/Agent-SafetyBench/blob/74feea8de601b3a1449a93fcf70017fe61556f73/environments/Email.py)、[evaluation](https://github.com/thu-coai/Agent-SafetyBench/blob/74feea8de601b3a1449a93fcf70017fe61556f73/evaluation/eval.py)、[ShieldAgent评分](https://github.com/thu-coai/Agent-SafetyBench/blob/74feea8de601b3a1449a93fcf70017fe61556f73/score/eval_with_shield.py)
- **AgentHarm**：官方README Canary string段明确请求不要用examples训练，仅用于evaluation；当前scorer还区分executed calls、semantic/refusal judges，不能靠简单函数名匹配自称官方评分。保留test_public/benign作最终外测，版本需冻结，不能把val自动改为训练。[官方README与changelog](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/agentharm/README.md)
- **ToolSafe TS-Bench**：有训练划分但其观测对象是既成轨迹+候选动作，RL优化guard，不是重置世界后由actor产生新工具轨迹。不能为了“明确train”而误选。[数据卡](https://huggingface.co/datasets/MurrayTom/TS-Bench)
- **Nemotron-AIQ-Agentic-Safety-Dataset-1.0**与本次NVIDIA RL-IPI是不同数据集；不将前者的评测数据或禁训数据并入后者。

## 配比与实验角色建议

第一轮4B/4090：**通用ClawGym RL2k : 安全 = 70:30**，按task-group预算而非原始行数或18配置复制后的行数计算；安全中clean/challenged 1:1，初始仅SkillLifeBench train模板。安全30%里一半clean可监控拒绝倾向，不能仅报告挑战臂ASR。对安全源用联合任务成功奖励；安全但未完成不能等同成功。保留utility、attack、joint三向量以及未经授权动作attempt/executed计数，未定义的persistence/detection保持N/A。

若NVIDIA子集的真正utility/postcondition验收通过，第二阶段把安全30%分为**SkillLifeBench 2/3 + NVIDIA IPI 1/3**（整体70/20/10），同一实验臂统一配比；先冻结模板/初始世界家族切分及NVIDIA内容近重复簇，不把两个来源因为样本量差异按行混成NVIDIA主导。9B先保持同配比以便能力规模对比，再做50:50消融；没有实测证据说明50:50必定更优。

18个S/T/O/C/W组合是同一环境的接口条件，不增加独立任务数。每个K=2 block绑定同一task、clean/attack臂、fixture、payload和预算，伙伴配置后用更新后策略新rollout并fresh reset；不能缓存第一次动作或带入已被污染的业务状态。把训练、validation、test配置与任务模板两个轴分别冻结，才可同时讨论新配置泛化与新任务机制泛化。

最终外测：保留HarnessRisk/HarnessSafe/HarnessAudit的现有约束与曝光记录；SkillLife held-out模板是小规模机制迁移；AgentDyn新增suite是动态IPI迁移（先审oracle）；AgentHarm只评价直接有害请求/benign效用；不要将这些异质指标合并成一个“安全率”。本票支持**先一个可靠安全源**，不把多源数量作为上线门槛。

## 证据边界

本次只获取和阅读官方资料/源码；GitHub API受到403限制，改用官方raw文件和git ls-remote固定上述HEAD。源码证据临时副本位于`/tmp/security_data_vote_b_sources/`，未作为可长期复验artifact承诺；本note的commit permalink是持久来源。未启动任何环境、未调用本机已有模型服务、未验证sandbox在当前宿主机可用、未测4B/9B显存/吞吐，未检验所有case的oracle完整性。公开runner已存在不代表本项目S/T/O/C/W桥接已实现。
