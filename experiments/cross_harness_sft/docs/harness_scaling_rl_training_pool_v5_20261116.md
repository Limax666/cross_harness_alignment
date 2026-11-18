> **2026-11-18执行覆盖：** 当前14K READY/50:50与真实v9参数RL已运行；旧GPU、小池及未启动描述是历史。请先读[当前执行规格](harness_scaling_safe_rsi_current_20261118.md)与[跨服务器交接](new_server_handoff_20261118.md)，不要照旧命令直接启动。

# Harness Scaling 安全 RL：万级任务池与 50/50 主配方 v5

日期：2026-11-16。用户明确否决仅30个安全训练实例的小池，要求约13.5K量级、安全与通用各半。本决定覆盖v4数据源、75/25主配方及“2K精选作为通用上限”；S/T/O/C/W统一白盒、18配置、相继step的K2、配置内G8、参数RL/RSI资源分离和双卡职责继续保留。当前是实际来源与源码审计，不是已完成14K准入或RL训练。

## 1. 负责人最终选择

**主建设目标：14,000条训练task-condition记录＝7,000 ClawGym通用任务＋7,000 AgentDoG 1.5独立RL runtime安全条件。** 先用现成有执行代码/状态/oracle的训练环境解决量级，SkillLife/近邻小集退为机制与迁移诊断；不靠复制小集、18配置、K2或G8凑数据条数。

通用来源是[RUC-AIBOX/ClawGym-Task](https://huggingface.co/datasets/RUC-AIBOX/ClawGym-Task)的完整固定HF文件，实际15,402条，不是永远只用官方2,000精选。安全来源是[AgentDoG Agentic Safety Training](https://github.com/AI45Lab/AgentDoG/tree/main/Agentic%20Safety%20Training)所指向的[Lightweight Agentic RL Runtime Environments](https://huggingface.co/datasets/quantumfr/agentic-lightweight-envs-runtime-20260528/tree/main)，不是AgentDoG guard轨迹分类/SFT数据。

| 主训练桶 | 建设配额 | 含义 |
|---|---:|---|
| ClawGym一般能力 | 7,000 | 独立合法请求、原始输入与代码checker；不含Bench200 |
| AgentDoG attacked-benign | 5,500 | 合法用户目标受到环境侧攻击；保留任务效用和风险调用判据 |
| AgentDoG clean controls | 1,000 | 从上述已选攻击任务的基础家族中选匹配clean，每clean只收一次 |
| AgentDoG query_target | 500 | 恶意用户请求；正确拒绝/安全处理，不奖励完成恶意目标 |
| **总计** | **14,000** | 通用7,000、安全7,000；安全内部三类分别报告 |

这些是待完整准入后冻结的目标配额。安全7K不是7K独立原目标，多个攻击条件可属于同一个base family；清洁配对也不增加独立目标。必须同时公布unique condition、unique base task/family、风险模板、工具环境与实际rollout数。单纯变更实体名、payload外观或harness配置不能变成新的独立机制。

## 2. 已核实的数量与划分

### 2.1 ClawGym完整源

HF revision `bd9043cff88b913922a165d9c022a4796aee3f3a`，文件405,688,752字节，SHA256 `0455f13c4f7a4839d39c3c1da9704c4959046ed529040d44f46f8f9bd07f7774`。本地15,402条JSON全部可解析；“约13.5K”是README概述，盘点使用实际文件实数。原精选2,000均能一一映射，8,647既有资产同字节，补齐25个Git漏收输入：23日志、1任务校验脚本、1任务fixture配置。fixture配置只作任务文件，绝不导入主机环境。

当前仅精选2K的HF adapter做过静态准入：1,403通过，其中train1,263/dev140；其余597是排除或待修复，不代表整个15,402只有1,403可用。尤其397个writable-input/gold契约pending是适配工作，不是原任务天然无效，不能永久丢弃来缩池。扩到全源须重新materialize、依赖/联网能力审计、原始checker与任务授权写范围审查和近重复划分。不能将联网要求改成离线编造回答，或因缺依赖默默给模型0分。

完整源先按源模板、规范化目标、输入/评分代码相似性及来源关系建簇，再按固定seed划分研究者train/dev；Bench200仅评测。目标7K从train合格集选，不能从dev/test回填或把同目标的安全派生放另一split。候选与准入不足配额时保留缺口并修复，不采样复制达标。

### 2.2 AgentDoG RL实际盘点

HF revision `7c263216d05afcd2a31185f6d102b35504418b3c`，数据卡Apache-2.0。固定gzip字节按发布SHA256SUMS核验；源码/数据审查详情及artifact路径见[独立大源审阅](research/safety_scale_rl_sources_20261116.md)。数据license不能自动外推为整个GitHub代码仓库的许可证。

| 范围 | clean | attack | query_target | 合计 |
|---|---:|---:|---:|---:|
| 全catalog | 3,552 | 9,994 | 2,228 | 15,774 |
| 去重复clean引用后的配对索引覆盖 | 2,931 | 8,398 | 1,883 | 13,212 |
| family隔离train候选 | 2,344 | 6,748 | 1,524 | **10,616** |
| dev候选 | 293 | 870 | 192 | 1,355 |
| test候选 | 294 | 780 | 167 | 1,241 |

配对索引有9,715行，但clean反复引用、单clean最多185次；这些行不能直接当独立任务。共有2,931个base_family_key。按`SHA256('20261116|base_family_key')`固定排序后80/10/10整family切分；0个task ID跨family冲突。没进入该配对索引的其余catalog记录单列，不盲目追加。这里test是研究者保留分区，不把HF物理train字段当作者官方未见测试。

源工具通过真正Python实现读写episode state，初态deepcopy。全catalog静态读取的效用规则包括state_subtree_only_matching_records_changed(11,115)、state_subtree_new_any_match(3,527)、state_subtree_any_match(1,115)、tool_result_nonempty(17)。没有发现always/constant-true顶层规则；主准入应使用可验证状态规则，17条仅tool_result_nonempty需单独审查。工具源码出现71,447次、21,866种source hash；AST可解析、source非空、dispatcher存在不等于全部语义正确，真实执行和反例/终态一致性还要验收。

攻击risk规则是history_call_sequence_contains，query_target是history_call_matches，clean是never。它们检查调用名/参数/历史，**不是已经证明真实泄漏或破坏的副作用oracle**；clean的never是攻击目标不适用，不是“所有行为安全”的证明。默认集合式链匹配不应叫严格有序序列，只有order_sensitive才有顺序约束。历史累积使本release已有调用匹配不会因事后修复消失；不能把每步重算变量误诊成已有擦除伤害bug。

7K安全候选已经实际选出：[候选ID索引](research/agentdog_rl_safety7000_candidate_index_20261116.json)，规范化内容hash `69c289ffabf39c8ea0c88754743b15658f84dfbb0faf0090f81843e112dd75ea`。5,500 attack覆盖1,298 family，1,000 clean来自其中1,000 family，500 query覆盖500 family；合计7,000唯一ID、1,420去重semantic family，每family1–10条件。Root已独立重算hash、80/10/10分区和配额，见[复核证据](research/agentdog_rl_safety7000_root_verification_20261116.json)。匹配是同源语义family关系，不保证initial fixture字节完全一致，不能直接宣称严格反事实配对。

完整runtime catalog及配对索引已下载到`data/rl_sources/agentdog_runtime_hf_7c26321/`，catalog158,049,160字节，SHA256 `d6d51b20a6a7d6597305af13e1a775233b94fd4c6da7a4ad79b3ad1632e7ee0c`；另有README、SHA256SUMS与summary，下载manifest保留。生成草案非运行必需，未下载。候选选择依据是源身份/类型/结构，未看模型成绩。必须在隔离、隐私、评分与依赖门禁通过后成为运行准入清单，不能只因ID存在就宣布可训练。

## 3. 新适配必须解决的具体问题

1. **只接task/backend，不接其固定harness或guard分类任务。** 任务公共指令/工具schema进入同一S/T/O/C/W执行器，T别名先转回canonical action再执行与评分，O只呈现原tool_result，C不得改变源oracle或权限。任务来源与harness配置分开抽样。
2. **外层隔离。** 官方AST受限的worker thread不是OS沙箱。工具实现与评分分别在本项目隔离进程/容器中；不透传主机env、密钥、源目录和Docker socket，禁止访问其他服务状态。每rollout新建初态，episode内部保持状态与history。
3. **防隐藏oracle泄漏。** server内部episode view可能带initial_state/state/success_rule/risk_success_rule/risk.goal/reward_info。target adapter严格白名单，只给公共user request、公开工具schema和应公开的工具结果；不能把整episode或fallback整响应发给模型。私有规则、目标标签和reward始终host-side，逐请求记录审计。
4. **保留分数语义。** source Utility、调用目标达成、judge safety和新增state-effect审计分别保存。未接实际泄漏/持久化oracle的项标N/A，不能用调用risk冒充HarnessRisk规则或judge ASR。unsafe attempts、拒绝访问和实际效果分开。
5. **judge与失败策略。** 初期复用用户授权的升算云凭据加载方式，配置中不保存值；沿用已有可用deployment的选择需另冻结训练judge模型/prompt/输出schema，不冒充原论文judge。失败、余额/解析/服务故障为invalid，不照抄上游静默改奖励公式；已成功精确输入hash不重复调用。本次研究没有调用judge API。
6. **恶意请求的Utility mask。** query_target中的恶意目标完成不是合法Utility。保留原源字段但训练合法U=N/A；安全目标使用经冻结的安全处理/违规反馈。GRPO-S可用`−c_source`，utility-only臂该类PG目标显式无效/零优势，不用恶意完成给正奖励；全部方法保持同数据成本并记录mask。benign/attacked-benign的合法U单独计算，不让“全部拒绝”获得joint-success。
7. **奖励新版本一致。** 新来源不能直接套仅SkillLife的reward。先固定canonical合法U、调用风险和judge安全映射及适用mask；GRPO/DAPO/Dr.GRPO/CHS-PO在同数据/证据/预算下比较。若采用`max(源调用违规,judge不安全)`等合并规则，须新版本冻结、保留各分量，不能直接称与上游官方recipe完全相同。独立效果oracle缺判不填0。

这些是接入工作，不是本次已实现/已验证的声明。不因要凑到7K放松门禁，也不退回旧小池而仍叫v5。

## 4. 其他开源来源的最终定位

| 来源 | 本次核实规模 | 定位与具体原因 |
|---|---:|---|
| [RUBAS](https://github.com/thu-coai/RUBAS) | 2,353正式训练记录；82环境、811工具 | 次级RL安全来源/生成器。2,000 harmful，235 sensitive，118 benign；不让纯拒绝题主导良性任务安全效用。stage1未含完整reward/trainer；根许可证与原工具来源需记录，不能自称已完整开箱接入。 |
| [OpenART](https://github.com/AI45Lab/OpenART) | 10,150任务包 | 第二阶段harness近邻大源。约75GB、部分长程/外部服务；官方不负责服务reset/snapshot，Utility与unsafe score需拆开，完整去重未做。不是本轮首选单卡轻环境。 |
| [ToolHazard](https://github.com/MurrayTom/ToolHazard) | RL329变体/64基础目标；SFT来源任务711变体/138基础目标 | 高质量stateful注入补充，显式train可接原状态/判据，不能拿SFT轨迹代替在线采样。默认2550良性文件覆盖官方test环境，禁止误入训；checker异常0分与未接安全cost需修。 |
| [AgenticQwen-Data](https://huggingface.co/datasets/alibaba-pai/AgenticQwen-Data) | HF实际37,401 train，非简介80K | 真正agentic RL数据、可扩量；官方mock_tool用LM模拟虚拟工具与用户并以rubric评分。适合作另一环境模型baseline，本轮优先可直接观测CPU状态的runtime，避免把模拟器变化混进harness效应。 |
| [NVIDIA RL-IPI](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1) | 1,272 | 明确训练数据、真实session状态；原Utility仅要求工具名出现，补真正完成checker后再用。 |
| SkillLifeBench、SkillSafetyBench、AuthBench等 | 60 lifecycle实例；155 cases/47 carriers；120权限题 | 保留为机制、近邻风险与迁移诊断候选；有真实双oracle的小源不能因此承担万级规模，不能称全部研究流程未见。 |

三份独立审阅并非同一结论的机械投票：[大规模RL源](research/safety_scale_rl_sources_20261116.md)、[可执行环境](research/safety_scale_env_sources_20261116.md)、[harness近邻](research/safety_scale_harness_sources_20261116.md)。近邻审阅更倾向“Claw合法目标＋开源风险素材”自建双oracle任务，因最贴近HarnessRisk；负责人选择先接AgentDoG RL现成大池满足规模与可执行性，近邻自建后续作为风险覆盖扩展。自建任务必须另记为新增构造，不能伪称原公开库已含数千独立目标。

## 5. 50/50采样、训练覆盖和成本

每个完整更新16个task-condition组×G8：8个通用组/64 rollout＋8个安全组/64 rollout。K2在相邻配置的相继更新重新reset、用当时策略采样；两次都计预算。安全5,500/1,000/500为11:2:1，按7个K-block（14个更新）兑现44/8/4组，避免每步小数配额造成丢类；来源、意图、条件、有效token和loss份额独立记账。50%记录/rollout不保证50%梯度，长轨迹token份额另报。

14,000条每条完整K2×G8一遍需要224,000训练episode尝试、875个K-block、1,750个128-rollout step（无失败时）。这远大于旧v4的2,560尝试/20step；旧预算最多160个task-condition位置，只能标短预算开发证据，不能叫万级池全量训练。数据规模、实际unique覆盖、epochs、更新数、失败重试和GPU小时必须同时报告。正式等预算比较暂沿用全局方案每臂38,400尝试上限，不因池扩大会自动增加API/GPU预算：至多300个128-rollout step、2,400个K2/G8 task-condition分配位置。完整一遍224K需另冻结资源预算并对齐所有比较臂。实际吞吐尚未测量，不能据元数据承诺工期；不因池文件大就宣称已覆盖，不在保留集上调训练时长。

GPU1继续FlashRSI固定权重基线；GPU0只在既有完整算法边界、等待器已登记后交接RL。新的数据研究没有停任何评测、切卡、热替换服务权重或恢复cron。4B/9B确认使用相同新数据manifest、组件规范、reward和预算口径；先有4B真实reset/评分/更新/策略同步证据，再扩大实验，不能用本次静态数据审计替代。

## 6. 当前交付与尚未完成

已完成：完整Claw原始文件下载/校验；新大安全源的版本、数量、family split与规则/工具代码静态审查；三份独立来源比较；v5主配方与源适配要求。AgentDoG完整runtime输入已落盘，7K候选ID索引已冻结并由Root复核，数量与家族见上文。

尚未完成：完整Claw7K运行准入、AgentDoG候选的隔离工具/隐私白名单/私有评分与judge集成、最终两源14K运行准入manifest，以及v5真实在线RL。当前不能说14K已收齐并全部可训；可以说已经找到了有实数基础、明确数据来源和具体接入路径的万级主方案。旧pool、失败和修复证据保留，不与新源混统计。
