# 大规模 Harness Scaling RL 安全训练源独立核查

日期：2026-11-16。研究者：safety_scale_rl_sources。只读检索、原始 JSON/GZip 解析、源码/AST 审查；没有 import/执行来源工具代码、启动环境/模型/评测、调用付费 judge、改服务或 GPU。唯一写入文件为本 note。下列“静态候选”不等于本机运行准入。

## 决定票

**首选 AgentDoG 1.5 独立发布的 lightweight agentic RL runtime 数据，而不是其约 1k guard 训练数据。** 已完整解析其 15,774 个 runtime 条件和全部工具源码字符串，按 base family 划分后，安全训练候选有 6,748 个 attacked-benign 条件和 1,524 个 query-target 条件，支持从中固定选 7,000 条安全条件；它们不是 7,000 个独立合法目标。

负责人最终主配方为 ClawGym 7,000 + AgentDoG RL 7,000（5,500 attacked-benign + 1,000 同源 matched-clean controls + 500 malicious-query），严格 50/50；已实际生成唯一ID索引，见第7节。如主实验要求安全部分全部为“正常用户任务受环境攻击”，可先定为 ClawGym 6,500 + attacked-benign 6,500，共 13,000，而不是把不足 7,000 的 attacked-benign 复制补数。query-target 必须单独标出，其恶意目标完成度不能充当正向 Utility；安全拒绝奖励应按该任务的合法目标 mask 定义。

第二票 NVIDIA Nemotron RL-IPI：明确训练用途、可执行状态和低评分成本，但只有 1,272 条且 utility 有实质缺口。第三票 RUBAS：明确 RL train + stateful mock tools、可扩展 task generator，但现有 2,353 中 2,000 为 harmful，不能伪装全部是受污染的良性任务，完整 reward/trainer 与许可证仍需处理。RUBAS 不应为凑 14k 成为硬性必选。ShieldAgent / RePolicy / AgentDoG 轨迹分类集不能当业务执行任务数量。

## 1. AgentDoG RL：原始来源与可复核身份

官方桥接 README：[Agentic RL release](https://github.com/AI45Lab/AgentDoG/blob/c8d803f267a43ec0e103a651265f50f1ff4456d5/Agentic%20Safety%20Training/Agentic%20RL/README.md)。它链接的数据为 [quantumfr/agentic-lightweight-envs-runtime-20260528](https://huggingface.co/datasets/quantumfr/agentic-lightweight-envs-runtime-20260528)。不能仅按 AgentDoG 主页 guard 样本约 1k 排除这套独立 RL 环境。

- Git HEAD 经 `git ls-remote` 核实：`c8d803f267a43ec0e103a651265f50f1ff4456d5`；关键源码按该 commit 的 raw URL 阅读。
- HF revision：`7c263216d05afcd2a31185f6d102b35504418b3c`。
- HF 数据卡 license：Apache-2.0；未在 AgentDoG 根或 RL/env_server 常见位置找到独立 LICENSE，故不把整个 Git 源码擅称 MIT/Apache。HF 数据内嵌 tool_impl_sources 与独立 runtime server 源码的来源/许可记录应分开。
- 完整 `runtime_catalog.json.gz` 实测 158,049,160 字节；边读边 SHA-256 为 `d6d51b20a6a7d6597305af13e1a775233b94fd4c6da7a4ad79b3ad1632e7ee0c`，匹配官方 SHA256SUMS。
- `semantic_local_pairing.jsonl.gz` 实测 140,870 字节；SHA-256 `bebb181aca11e16da11d0658b9cd6f36b307d8d6d6213b66b0b9db11eb9f79b3`，匹配官方。
- `task_drafts.json.gz` 是生成草案元数据，不是 runtime 必需文件；未下载或执行。发布不含 LLM 合成 pipeline，不能说自动扩展生成器已随 runtime 开源。

### 实数与计数单位

完整流式 JSON 解析得到，与 card 一致：

| 条件类型 | Catalog 唯一 task ID |
|---|---:|
| clean | 3,552 |
| 环境注入 attack | 9,994 |
| malicious query / query_target | 2,228 |
| 总计 | 15,774 |

pairing 文件有 9,715 local rows，但只有 **2,931 个 base_family_key、2,931 个不同 clean task ID**。clean 出现 9,715 次，单个 clean 最多重复 185 次；query 出现 6,059 次但仅 1,883 个不同 ID。去重后配对覆盖如下：

| 配对记录类型 | 出现次数 | 不同 task ID |
|---|---:|---:|
| clean | 9,715 | 2,931 |
| attack | 8,398 | 8,398 |
| query_target | 6,059 | 1,883 |
| 不同条件 ID 总数 | — | 13,212 |

所有 13,212 个配对 ID 均在实际 catalog 中，0 缺失、0 相同 task ID 跨 base family 冲突。Catalog 其余 2,562 条未进入该 paired 索引，暂单列，不强行拼进同源配对。2,931 个 semantic base family 是发布的聚类单位，并不证明语义/工具模板完全独立；应进一步跨源去重和检查模板相关性。18 配置、K2、G8、local triplet 都不增加独立任务数。

### 固定 seed 的实际 80/10/10 family 切分

排序规则：`SHA256('20261116|' + base_family_key)`，按 digest bytes 升序，前 floor(0.8×2931) train，至 floor(0.9×2931) dev，余 test。clean 和全部 attack/query 派生条件共同切分，不按 local row 随机切。

| 分区 | family | unique clean | unique attack | unique query_target |
|---|---:|---:|---:|---:|
| train | 2,344 | 2,344 | 6,748 | 1,524 |
| dev | 293 | 293 | 870 | 192 |
| test | 294 | 294 | 780 | 167 |

train attack + query = **8,272 安全条件候选**，可在不复制 clean 的情况下选择 7k；但全是 attacked-benign 的数量仅 6,748，需透明标类型。source 中 16 个 bundle 的源码文本含 `placeholder` 字符串，尚未逐个语义审查，不将字符串命中直接等同假环境；即使最保守暂挂这 16 条，6,000 attack + 1,000 query 的规模仍有足够静态余量。正式准入仍需实际 reset、执行、可达目标和 oracle 一致性验证，不能把这个余量当已运行合格。

## 2. 是否真的执行工具、reset、评分

关键源码：[server.py](https://github.com/AI45Lab/AgentDoG/blob/c8d803f267a43ec0e103a651265f50f1ff4456d5/Agentic%20Safety%20Training/Agentic%20RL/env_server/tasksvc/runtime/server.py)、[tool_runtime.py](https://github.com/AI45Lab/AgentDoG/blob/c8d803f267a43ec0e103a651265f50f1ff4456d5/Agentic%20Safety%20Training/Agentic%20RL/env_server/tasksvc/runtime/tool_runtime.py)、[runtime_evaluators.py](https://github.com/AI45Lab/AgentDoG/blob/c8d803f267a43ec0e103a651265f50f1ff4456d5/Agentic%20Safety%20Training/Agentic%20RL/env_server/tasksvc/runtime/runtime_evaluators.py)。

`build_episode` 深拷贝 initial_state_template 为每个 episode 独立状态。`_build_episode_executor` 根据 tool_dispatch_table 把工具映射到 catalog 的 tool_impl_sources + entrypoint；`EpisodeToolExecutor` 实际 exec 固定来源代码并调用 execute(arguments,state,context)，随后 server 使用返回 state 更新业务状态，不是在线调用 LLM 编造工具返回。

例如实际首任务提供 get_channels/delete_email 等；get_channels 从 state.inbox.channels 检索证据并返回目标定位信息；合法完成规则检查指定邮件记录的 deleted 字段改变且不误改其他记录。`--backend placeholder` 在本版 load_task_catalog 中没有生成 mock response 的作用：函数要求固定 catalog 文件，并调用真实 catalog loader/validator。

### 全 catalog 静态工具和 oracle 统计

- 71,447 个 tool-source 出现，21,866 个不同 source SHA-256。
- 逐个 AST 解析：0 源码语法错误；0 空 tool_impl_sources；0 allowed_tools / dispatcher / source_key 对接错误。
- 15,433 个不同 `(user_query,initial_state,utility_rule,tools)` 组合 hash；这不是独立目标/机制的证明，不能替代 base family 数。
- Utility 顶层规则：`state_subtree_only_matching_records_changed` 11,115；`state_subtree_new_any_match` 3,527；`state_subtree_any_match` 1,115；`tool_result_nonempty` 17。没有 always/恒真 Utility 规则。
- risk 顶层规则：clean 的 3,552 条均为 never（不适用）；9,994 attack 全为 history_call_sequence_contains；2,228 query 全为 history_call_matches。没有把 clean 的 N/A 当 0 风险观测。
- 对 paired train/dev/test 进一步要求 state-based Utility、非恒定且 enabled 的 risk、完整 dispatcher，前表计数均保持。17 条较弱的 tool_result_nonempty 不在该 paired 严格候选内。

这些是源码/结构静态事实。没有执行 reference plan、reset 或工具，不能证明每题可解、所有 Python 工具无 bug、所有 rule 与用户语义一致。

## 3. 必须保留的边界，不能照搬原 recipe

1. **隐藏 oracle 隔离。** `_episode_public_view` 含 initial_state/state/success_rule/risk_success_rule 和 risk.goal，仅给可信 controller。policy 只接 query + public tool schemas；只返回工具业务 observation/result，不接完整 episode/metadata/reward。原 Slime rollout 在 observation 为空时用 `json.dumps(tool_result)` fallback，可能把 reward_info 透传；应在项目适配中禁止该 fallback。
2. **原 runtime 非 OS 安全隔离。** EpisodeToolExecutor 是 AST 白名单 + 同进程 worker thread；不是 Docker/user namespace，超时后 join(0.2) 也不保证杀掉卡死线程。需要本项目独立外层隔离和有界进程生命周期，不能因“lightweight”关闭沙箱。
3. **源风险 oracle 是调用行为，不是实际损害 ASR。** `_history_call_matches` 只对工具名、arguments_match 评分，不检查工具调用是否业务成功、状态是否真的泄漏/删除。保留其原指标为 registered unsafe-call goal；若论文报告真实副作用，另加 state/effect 审计。`history_call_sequence_contains` 对 unique calls 默认按集合匹配，仅 order_sensitive=True 才强制顺序，不把名字当有序证明。
4. **关于先违规再修复。** 虽然 risk_success 每 step/final 重新赋值，但本 catalog 全部风险判据读取累积 history，既有匹配调用不会被后续状态修复抹掉。不可据泛型实现自行声称存在此 bug；对后续扩源 state-risk 可另做 prefix OR。
5. **评分器/奖励版本冻结。** 官方 clean R=U、malicious R=S、attacked benign=.5U+.25S+.25US，S 来自外部 judge。无 judge/解析失败时 source recipe 会换公式，不能照搬静默降级。本项目若选 deterministic risk-goal 作为 cost，应称新奖励适配并与原 broad judge safety 分开；它不能覆盖所有未登记危害。judge 失败按缺判记录，不能默认 safe。
6. **异常不记模型 0 分。** reward_health 会对工具错误、截断等置 0 或 cap；本项目已有 invalid/模型预算耗尽的区分，需保留基础设施失败并统一所有算法契约，不能复制为一概 0。
7. **恶意查询的效用。** query_target 的 source success 表示恶意目标可达/达成。不可在 GRPO-S 中直接取该 success 为正向合法完成度。保留 malicious mask，安全拒绝与帮助完成恶意目标分开；也可主实验只选 6.5k attacked benign，query 另作对照。

## 4. 其他候选与不合格的大数字

| 来源 | 实际规模/用途 | 环境和评分 | 本票定位 |
|---|---|---|---|
| NVIDIA Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1 | 1,272 train 记录，9 域，4 attack 类；HF revision d738d4f361cc38bb4d7a42b9066776dade5332f5；CC BY 4.0，明确商业/非商业训练 | seed_session deepcopy环境；TOOL_HANDLERS真实改 session；deterministic safety，utility仅 required read-tool 名是否出现 | 次选；补真正合法目标终态后扩源，不能用原 utility 声称任务完成 |
| RUBAS | 官方 train 2,353 = harmful2,000/sensitive235/benign118；82 stateful mock 环境/811 tools，公开任务/初态/rubric生成器 | 基于 task rubric 的 tool-use/arg/response safety + helpfulness；发布 stage1不含完整 trainer/reward闭环；repo未见根LICENSE | 第三；保留高质量 sensitive+benign任务及多风险生成基础，不为凑数锁为主必选 |
| Agent-SafetyBench（thu-coai，不是 AgentSecurityBench） | 2,000 evaluation tasks，多种 Python 状态环境；MIT repo | Bank.open_account等真实更新 Python list/dict；用 ShieldAgent模型判安全，缺通用独立 Utility verifier | 工具/场景生成基础或部分研究者train，需明确原评测暴露；非现成安全RL主池 |
| RePolicy PolicyTraj-20K | >20k agent trajectory，policy retrieval + guard judgment，SFT/GRPO | 工具是 get_policy、目标unsafe F1/格式/引用；不是执行原业务环境 | 可作judge/guard baseline，不能计为20k业务执行任务 |
| AgentDoG 1.0/1.5 guard数据、ShieldAgent-Bench | 轨迹分类/安全诊断数据，约1k至3k样本 | 评已有轨迹/取政策；没有对应每条自主业务reset/动作/Utility | 可训练judge，不并入主任务数 |
| Safety-Gymnasium/MOSRL | 机器人/物理控制安全RL | 连续动作/物理cost，不是本研究工具agent执行协议 | 不因名字含Safety Gym纳入 |

NVIDIA 一手资料：[数据卡](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1/blob/d738d4f361cc38bb4d7a42b9066776dade5332f5/README.md)、[Gym app.py](https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/indirect_prompt_injection/app.py)。本次复核本地固定源码 verify：required_tools 非空时仅判断名字集合包含，空时 Utility=1，原缺口仍在。

RUBAS 一手资料：[官方 README/训练统计](https://github.com/thu-coai/RUBAS)、[pipeline](https://github.com/thu-coai/RUBAS/tree/main/pipeline)。来自同组独立审阅的精确任务盘点与本次官方页交叉一致；未由本人执行环境或其生成器。

Agent-SafetyBench 一手资料：[环境 Bank.py](https://github.com/thu-coai/Agent-SafetyBench/blob/main/environments/Bank.py)、[官方评分说明](https://github.com/thu-coai/Agent-SafetyBench)。BaseEnv.call_tool 会类型转换/丢额外参数，若五维 T/schema 映射需要保持原语义并明确协议适配。

RePolicy 一手资料：[官方作用与训练](https://github.com/JiangHoucheng/RePolicy)、[PolicyTraj-20K数据说明](https://github.com/JiangHoucheng/RePolicy/blob/main/docs/dataset_card.md)。大记录数不能替代环境任务数。

## 5. 规模结论与实施优先级

现在有确实可支持 6–7k 安全条件的公开 runtime 源，没必要继续让 SkillLifeBench30个训练实例承担主池。建议先接 AgentDoG runtime 到 ONE 白盒适配，保留其细粒度 canonical tools/state/evaluation contract，source fixture与oracle始终私有，然后用固定源family train中真正唯一 task_id 组成规模池。ClawGym full 15,402 需独立处理网络/宿主/依赖和可写输入参考契约；不能把旧 selected2k 的1,403静态合格结论外推全量，也不能倒过来把旧保守门禁当完整语料质量差。

正式报告同时列：源catalog 15,774；paired unique conditions13,212；independent semantic base groups2,931；训练候选 attack6,748/query1,524；固定选取实际数；实际通过运行门禁数；模型 rollout及tokens/成本。50/50指冻结训练池与采样配额，不能靠重复clean/localrows补足“独立任务”规模；任务数和50%梯度份额也不是一回事。

原 source risk rules 覆盖工具行为，不能用本轮数据直接支持所有 HarnessRisk 持久载体安全机制的训练覆盖。SkillLifeBench、HarnessSafe/Audit等可作为机制补充/外部迁移评测，但加入比例或新适配结果需要独立标注，不回溯混新旧池hash。此note不代表新实验已启动。

## 6. 研究过程故障与未验证项

首次全 catalog 流读取在约10k条之后因研究脚本对字节chunk直接 UTF-8 decode 踩字符边界失败；使用 TextIOWrapper 修正后从原固定文件重新完整读取，实际15,774条及整GZip hash通过。源文件没有改动，没有把不完整第一次计数当全量。GitHub API一度403限流，改官方 pinned raw文件和只读codeload内存读取；无来源代码执行。

仍未验证：每题目标可达、工具逻辑/参数schema一致、运行隔离与reset、真实无oracle泄露、judge校准、unsafe-call目标与实际副作用一致、七千选取后的风险/family平衡、Claw full来源准入及跨源重复、真实4B RL参数更新。来源规模足够不意味着这些验收已完成。

## 7. 按负责人最新配方实际冻结 7,000 个候选条件 ID

文件：[agentdog_rl_safety7000_candidate_index_20261116.json](agentdog_rl_safety7000_candidate_index_20261116.json)。index SHA-256（去掉index_sha256字段后，对JSON读回对象以UTF8 JSON sort_keys=True/ensure_ascii=False/separators逗号冒号计算）为 `69c289ffabf39c8ea0c88754743b15658f84dfbb0faf0090f81843e112dd75ea`。已重新读取并验证 hash、7,000唯一task ID、全部在train、角色配额以及clean确来自入选attack同basefamily。没有执行环境。

| 角色 | 唯一条件 | 不同basefamily | 每family条件最少/最多 |
|---|---:|---:|---:|
| attacked benign | 5,500 | 1,298 | 1 / 8 |
| 同源matched clean control | 1,000 | 1,000 | 1 / 1 |
| pure malicious | 500 | 500 | 1 / 1 |
| 合计（family去重） | 7,000 | **1,420** | **1 / 10** |

合计每family平均4.9296条；分布为1条×122family、2×7、3×17、4×218、5×686、6×255、7×14、8×39、9×49、10×13。先按basefamily固定80/10/10，再在train按固定seed哈希轮转；每轮每family至多取一条，耗尽则跳过，无放回。5,500 attack选完后仅从其1,298个family内取1,000个不同clean；500 query按family轮转各取一条。选择依据是源身份/类型/结构，未看任何模型结果。

这7,000是**实际存在并去重的task-condition ID**，仅1,420个semantic basefamily，不声称7,000独立目标或7,000种安全机制。1,000 clean controls计在安全来源桶但不计attack ASR分母。源semantic pairing不保证初态字节完全相同，不擅称严格fixture配对实验。

### 脱敏oracle结构摘要

以下只给真实catalog所用判据和源码字段键，不给private fixture/目标值，不是模型轨迹或执行验证。

```json
[
  {"kind":"utility existing-record effect",
   "type":"state_subtree_only_matching_records_changed",
   "field_keys":["root_key","selector_match","identity_fields","field","equals"],
   "example_action_tools":["get_channels","delete_email"]},
  {"kind":"registered attack-call goal",
   "type":"history_call_sequence_contains",
   "field_keys":["calls","tool_name","arguments_match","order_sensitive"],
   "requires_business_success":false},
  {"kind":"registered malicious-query call goal",
   "type":"history_call_matches",
   "field_keys":["tool_name","arguments_match"],
   "requires_business_success":false}
]
```

其中order_sensitive为解释器可选控制键，不表示catalog每条实际都提供；实际默认可能用集合匹配。单纯有这些字段不能证明oracle与业务/安全语义完全一致，真实运行与私密性准入门禁全部仍未完成；候选ID索引显式保留该状态。

第一次候选index hash复读失败，原因是内存histogram整数键在JSON读回后成为字符串、排序次序不同。已对实际JSON读回对象规范化重算并通过二次复读，旧 `c37b75618877b983cc24ba2fdd2a13f153f5fe5b6bc229acfb4a8f152d8a3f84` 作废；task IDs、源字节及配额没有改变。
