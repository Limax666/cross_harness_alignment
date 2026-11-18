# 万级 Harness Scaling RL 安全环境独立审阅

日期：2026-11-16。用户要求约13.5K级总池、通用与安全各半。本审阅只读检索官方仓库、数据卡和代码，并在内存解析公开 JSON/Parquet；没有运行任务代码、攻击载荷、模型、judge、模拟实验或GPU操作。未下载75GB任务档案，以下不是本机运行准入结果。

## 裁决票

**修订票：第一候选为AgentDoG 1.5独立的lightweight RL runtime任务包；第二候选OpenART公开任务包；ToolHazard显式alignment train与SkillLifeBench作机制补充。**这次修订依据负责人与另一审阅新发现并实际盘点的独立RL包，不是把AgentDoG 1.0的轨迹分类数据改名为在线环境。我额外核读了该runtime HF数据卡，确认它与1.0监督分类集是不同资产。

目标可设约7K ClawGym＋7K安全**任务条件记录**，但现阶段不可宣称7K安全环境已完成runtime准入；同family的clean/attack/query-target是不同条件，不是新的独立基础合法任务。相较OpenART约75GB且外部服务reset需自建，lightweight runtime只需加载已合成Python状态机，更适合单4B首轮；仍须补全predicate/exception/隐私与真实执行审查。

## 0. 新增AgentDoG 1.5 runtime独立包：与分类训练集严格分开

原始源：[quantumfr/agentic-lightweight-envs-runtime-20260528](https://huggingface.co/datasets/quantumfr/agentic-lightweight-envs-runtime-20260528/tree/main)。[官方AgentDoG仓库的Agentic Safety Training说明](https://github.com/AI45Lab/AgentDoG#agentic-safety-training)、[AgentDoG 1.5论文](https://arxiv.org/abs/2605.29801)分别将这一应用定位于可扩展安全SFT/RL。HF作者quantumfr是其论文作者账号之一；资产license标Apache-2.0，具体代码/数据版本仍按下载manifest冻结。

本审阅独立读到HF README：catalog15774＝clean3552＋attack9994＋query_target2228，semantic-local groups9715；提供runtime_catalog、task_drafts、semantic_local_pairing、type-normalization summary和SHA256SUMS。它是预构建runtime任务/代码包，使用时不必重新跑LLM合成流水线。README明确reward来自末态/domain-outcome判据，不取工具自报success。`--backend placeholder`是其服务器的backend名字，不能仅据这个名字认定它是静态模拟回答，也不能反过来认为所有工具均有现实软件保真；是否真的执行Python状态转移应按代码/轨迹证明。

**以下详细数字来自负责人及另一位独立审阅的实际catalog盘点，我未重算整份158MB压缩catalog：**对local pairing里的重复clean去重后13212个任务条件，2931个base family；按family80/10/10后train2344个clean＋6748个attack＋1524个query_target＝10616唯一条件。依据这份受控训练分区，拟选5000 attacked＋1000 clean＋1000 pure malicious与ClawGym7000组成14000记录，规模有原始资产依据；仍需排除失效判据、不可执行依赖与数据泄漏，并报告最终family/机制覆盖。

对接ONE白盒执行器应复用reset、Python工具实现、init_state和domain-outcome checker，而非复用原产品agent或既有轨迹。T名称/参数反向映射后调用原source action；O/C变化不改变实际业务状态与oracle，C2摘要文件不得混进业务评分状态。这个方向比共享外部软件服务更易实现独立reset与多样工具schema。

**一个必须单列的奖励问题：**query_target为恶意用户目标时，不可把“恶意目标完成”作为正向合法u，再与相同违规c相减导致执行/拒绝均为0。该stratum需明确u的合法目标/N/A及拒绝/违规成本合同，并保留原source指标；若尚未冻结这部分，应先用同训练分区的合法目标＋attack条件补足，而非修改标签假称安全。Guard/judge用于训练反馈和域状态违规oracle仍是两回事，分类模型打分不是最终状态事实。

本审阅认可该runtime包优先于OpenART作为单卡主候选；不把本段源码/数据卡核读或他人静态盘点称为本机执行、真实RL更新或完整oracle验收。

## 1. OpenART：规模与任务真实性最匹配，不能原样调用其产品CLI充当白盒RL

官方：[仓库](https://github.com/AI45Lab/OpenART)、[HF任务包](https://huggingface.co/datasets/dongdongunique/openart-planner-tasks)、[HF工具库](https://huggingface.co/datasets/dongdongunique/openart-tools)。本次观察代码HEAD `addc45570f08cc047079f61e7e89f3e233cd17a9`；HF revision `b4089b22b629cd2c62078dfffee8700bca3fb52b`。任务发布档案含workspace、scenario_model、tool_use_graph、safety_contract、私有evaluator及其测试，不是纯攻击字符串或既有模型轨迹。

### 1.1 实际读取三个metadata文件后的计数

| HF config | 实际记录 | Planner profile | 档案体量（数据卡） |
|---|---:|---|---:|
| mid | 1,046 | stress | 69MB |
| mid-v5 | 3,553 | moderate-risk-diverse-30-45 | 70.2GB |
| large | 5,551 | high-tool-calls-60-100 | 4.97GB |
| 总计 | **10,150** | 三个生成批组 | 约75GB |

这是本次真实解析的记录数，与数据卡相符；不是10150个已成功执行的环境，也不是8种attack条件乘积。三个物理split都叫train，但项目定位是benchmark，没有官方train/dev/test语义划分，研究入训须自行先分组切分。未发现“禁止训练”条款；同样不能仅凭HF split名称宣称作者指定这些是训练题。[官方数据卡](https://huggingface.co/datasets/dongdongunique/openart-planner-tasks/blob/b4089b22b629cd2c62078dfffee8700bca3fb52b/README.md)

### 1.2 独立性审查

- 全部10150个task_id唯一、benign_goal精确文本唯一；`(source_run_id, source_task_name)`也10150唯一。因此并非明显的同task-ID跨config重复，但精确文本唯一不证明语义独立。
- source_task_name只有6553个不同值，名称如`opencode-planner-run-001`会在不同生成run复用，不能直接拿它当跨来源唯一parent。
- mid-v5的3553条中，`original_scenario`仅3518个不同值，35组同scenario各出现两次。至少这些必须共同split，不应称为新增独立机制。
- large的5551条`task_content_sha256`全部为空，且没有original_scenario字段；不能把空hash相等当重复，也不能拿这个字段声称已做artifact去重。需从bundle的scenario_model、graph、workspace inventory派生hash与家族关系。
- 现有metadata没有统一可信的parent/family ID。训练split建议按可用original_scenario、规范化目标、risk-contract结构、tool-graph与workspace引用图联合聚类；完整保留task ID、source run、source_task_name。原始domain字符串共有457种（含空值与粒度差异），不能把README概述50个domain当当前文件恰50个分类。

### 1.3 工具确会执行，但有本地函数、真实服务和guide-only混合

不是用LM编造每条工具响应：官方managed store包含有实际source_files的CLI工具，例如[document_extract_pairs_csv.py](https://github.com/AI45Lab/OpenART/blob/addc45570f08cc047079f61e7e89f3e233cd17a9/openart-tools/document.extract_pairs_csv/scripts/document_extract_pairs_csv.py)真实读PDF/写CSV；[dtap_request.py](https://github.com/AI45Lab/OpenART/blob/addc45570f08cc047079f61e7e89f3e233cd17a9/openart-tools/dtap.atlassian/scripts/dtap_request.py)执行HTTP读取/写入。也有仅文档的guide-only工具，不能把63697个materialized tools全部当可执行函数。实际能力由每题tool_use_graph选出的实现决定。[managed-tool规则](https://github.com/AI45Lab/OpenART/blob/addc45570f08cc047079f61e7e89f3e233cd17a9/OpenART/docs/07_capabilities_tools_mcp.md)

**关键门槛：上游明确不负责外部service的start、seed、reset或snapshot。**它只从host环境传声明的端点/凭据。本项目不能直接复用共享GitLab/邮箱等服务并声称rollout已独立reset；必须建per-rollout命名空间、初态恢复、状态快照和只通向本地任务服务的网络。原workspace copy/diff/snapshot不等于外部服务reset。必须将ONE白盒执行器接在任务/工具层，替换产品CLI宿主，而非把其OpenCode runner作为训练harness。[原文服务边界](https://github.com/AI45Lab/OpenART/blob/addc45570f08cc047079f61e7e89f3e233cd17a9/OpenART/docs/03_task_and_config_model.md#service-backed-managed-tools)

### 1.4 安全与Utility必须拆开映射

[框架evaluator](https://github.com/AI45Lab/OpenART/blob/addc45570f08cc047079f61e7e89f3e233cd17a9/OpenART/framework/components/evaluators.py)会识别`unsafe_leak_detected`与`safe_completed`；其归一化后的score=1表示攻击成功。**不能把通用score直接当正向RL奖励。**[release-sync示例evaluator](https://github.com/AI45Lab/OpenART/blob/addc45570f08cc047079f61e7e89f3e233cd17a9/OpenART/examples/tasks/high-complexity-release-sync/utils/evaluator.py)原始score又可能在safe_completed或unsafe_leak时都取1，而且一旦泄漏会提前把safe_completed置False。联合reward应独立保留u与违规c，不能沿该提前返回将“任务完成但泄漏”误当成单纯任务失败。

可复用安全contract，独立验收合法产物/服务终态checker；若必须改造u，要明确这是本项目新增完成判据，不冒充原生独立Utility。保留原安全结果及全部prefix，不能靠擦除泄漏产物后评分。官方Strict ASR要求deterministic与LLM judge双确认；训练若只用确定性cost，需独立命名，最终评测不能宣称已完成官方strict judge。检查函数/私有contract不送target，只有声明的公共task与fixture进入观察。

### 1.5 许可与预算

代码AGPL-3.0；HF数据卡为`license: other`，明确任务/工具artifact可能继承各上游许可，没有另授统一dataset license。需逐artifact保存来源/许可，不把整库称作MIT/CC-BY训练集。

原任务长程，metadata的mid样例约26–42工具节点、large可达100。既有12个模型回合/8192生成token先导协议不能未经分析套全库，否则大批截断会变成伪能力差；应给长程segment、C1/C2及统一跨算法预算。批准调整前不能把超预算题偷偷换为短题后仍叫完整10K源。

## 2. ToolHazard：官方已做GRPO，适配最直接，实际独立训练量远小于概述

官方仓库：[MurrayTom/ToolHazard](https://github.com/MurrayTom/ToolHazard)，本次固定`544b73b12a25431cb0be3eb43df41b4aacce5335`，MIT。公开train_set与test_set分开；这里不是Agent Security Bench/ASB的expected-result字符串工具。

### 2.1 真实JSON盘点

| 文件 | 记录数 | 去掉`_IPI_策略后缀`的基础task数 | 实际env数 |
|---|---:|---:|---:|
| train_set/rl_tasks_IPI_all_new.json | 329 | 64 | 17 |
| train_set/sft_tasks_IPI_all_new.json | 711 | 138 | 43 |
| train_set/sft_tasks_with_check_functions.json | 497 | 494个不同task_id | 100 |
| test_set/merged_6_categories.json | 502 | 87 | 28 |
| 顶层envscaler_rl_scenario_metadata.json | 2,550 | 2,550 | 51 |

前两项合计1040个adversarial records，但只有202个base task ID；六种attack策略不是六个独立合法任务。40_rl_train_env与100_sft_train_env是环境库存40与100，不是任务数量。2550项没有attack字段，不能直接称2550安全训练题。**实际env-set交叉核对发现：这2550题覆盖env_141_rl…env_191_rl共51环境，与显式train40及SFT100均0交集，却覆盖test_set全部28环境。不能因文件名带rl就入训，否则污染官方环境留出。**329条显式RL攻击记录的17环境全部包含于train40，且与SFT100、test28均0交集。原目录包括SFT轨迹，但这里的可迁移对象是init_config、真实环境代码、task与检查函数，不是把轨迹拿来离线冒充在线RL。

### 2.2 源码证明与需修复的默认行为

[base_env.py](https://github.com/MurrayTom/ToolHazard/blob/544b73b12a25431cb0be3eb43df41b4aacce5335/rl/roll/pipeline/agentic/env/envscaler_env/base_env.py)有fresh reset、deepcopy初态、真正getattr调用Python工具、每步state snapshot/diff、末态check_func计算；[env_util.py](https://github.com/MurrayTom/ToolHazard/blob/544b73b12a25431cb0be3eb43df41b4aacce5335/rl/roll/pipeline/agentic/env/envscaler_env/utils/env_util.py)从环境源码构建类，`get_state_info`读取真实对象状态，不是LLM模拟工具返回。

官方RL入口只读取checklist_with_func做任务Utility，没有把injected_checklist_with_func接成安全cost。需要复用公开注入目标检查函数另算c，保留合法/攻击结果；不能直接宣称上游GRPO原配置就是本项目安全对齐目标。代码执行异常默认reward=0，checker执行失败None也可能通过分母间接计0，这些在本项目须变invalid并保留原因。动态exec源码/判据应在隔离进程中执行，不照搬host exec。它不需要LLM judge才能计算状态判据，但生成新环境/新攻击的模型费用另记。

## 3. 其余候选与否决证据

| 候选 | 原始规模/用途 | 本轮判断 |
|---|---|---|
| Agent-SafetyBench | 实际released_data 2000条；1265 unfulfillable、735 fulfillable；环境库存349 | 可作辅助风险指令，不能直接当2000个具备独立正常效用的安全训练任务。Bank.py很多transfer/send_money仅return success、无对应状态变更；官方以ShieldAgent模型评安全，无统一独立Utility checker。 |
| ToolEmu | 144例/36 toolkit | 官方明确用强LM模拟工具执行并用LM评安全/helpfulness，不符合当前要求的真实状态checker主池；规模也不足。 |
| SafeAgentBench | 750 embodied规划任务，AI2-THOR环境 | 真环境但物理/视觉状态与本项目文件、工具、skill执行风险不同；不是首轮4B单卡文本harness任务扩量的主源。 |
| AI45Research/AgentDoG1.0-Training-Data | 此特定发布物是轨迹安全分类/诊断SFT（两个4K配置） | 不能直接计作在线环境；此判断不适用于上方独立的AgentDoG1.5 Agentic RL runtime包。 |
| AgentRisk SemEval 2027 | 568train＋143dev轨迹与拟执行动作 | 官方任务是风险分类/干预预测，没有公开在线环境，不能当711个真实tool执行任务。 |
| MT-AgentRisk | 365基础harmful任务的single/multi条件；需请求访问 | 真实工具方向接近，但量不大、主要恶意用户请求且需独立合法Utility；不当7K主池，保留外测。 |

原始资料：[Agent-SafetyBench源码](https://github.com/thu-coai/Agent-SafetyBench/blob/74feea8de601b3a1449a93fcf70017fe61556f73/environments/Bank.py)、[数据](https://github.com/thu-coai/Agent-SafetyBench/blob/74feea8de601b3a1449a93fcf70017fe61556f73/data/released_data.json)、[官方评分说明](https://github.com/thu-coai/Agent-SafetyBench)、[ToolEmu](https://github.com/ryoungj/ToolEmu)、[SafeAgentBench](https://github.com/shengyin1224/SafeAgentBench)、[AgentDoG Training Data](https://huggingface.co/datasets/AI45Research/AgentDoG1.0-Training-Data)、[AgentRisk](https://huggingface.co/datasets/AgentRisk/agentrisk-data-2027)、[MT-AgentRisk](https://huggingface.co/datasets/CHATS-Lab/MT-AgentRisk)。

## 4. 给负责人的落地建议

1. 优先核验AgentDoG1.5独立runtime包的完整oracle、reset、权限与family划分；OpenART作为第二安全大池候选，完整盘点10150 bundles，但先按scenario/graph/原始parent分组并封存dev/test，再审查依赖/许可/工具权限/私有oracle。不先保证最终train一定7K。
2. 复用ClawGym完整HF源而非把官方精选2K当永久上限，目标约7K真正通过准入的通用训练实例。两源可执行联合任务才按50/50配额入训；候选文件条数不替代准入数。
3. ToolHazard alignment source作第二个安全机制来源，按base task/env家族切分，1040攻击记录与202基础任务都报告；允许迁移SFT来源任务的环境/判据，但不使用既有轨迹替代在线采样。SkillLife只作小池机制锚点，不靠复制30题扩成数千“新任务”。
4. 安全桶定义为“有合法目标并带可判安全约束的任务”，不是把所有harmful用户请求或无attacker的正常任务都叫ASR任务。边界风险、注入攻击与clean条件分别报告。
5. 14K池如果每task-condition都G8×K2完整一遍，需要224000 rollouts。旧2560尝试预算只够160个task-condition分配位置；可以保留为短预算开发，但不能宣称训练覆盖整个14K池。

所有判断是源码/数据结构审查；未证明实际reset/sidecar隔离、源utility可靠性、4B可完成率或在线参数更新。服务生命周期、独立u、许可和长程协议均是具体接入工作，不能用放松安全门禁或把失败记0来跳过。
