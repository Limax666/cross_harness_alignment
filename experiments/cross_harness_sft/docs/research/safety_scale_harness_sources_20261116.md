# 大规模 Harness Scaling RL 安全池：harness 近邻来源独立审阅

日期：2026-11-16。用户要求总池约 13.5K、通用/安全约 50/50，原 SkillLife60 只作小机制先导。本审阅只做官方 web/源码/公开 JSON 的静态读取，未执行源环境、payload、模型、judge、模拟或冒烟实验，未改任何服务/GPU/队列。以下“可接入”是源码层判断，不能当成本机运行验收。

## 结论与投票

**没有核实到一个与 HarnessRisk 同题型、现成含 6,000+ 独立合法目标、原样即可用于本项目在线 RL 的开源集。**近邻的大部分来源只有几十至几百题；几千/几万记录的 skill 集主要是检测素材或 SFT 轨迹，不能直接改名为在线 RL 任务。

我的首选路线是：**ClawGym 大池提供独立合法目标、输入和原代码 Utility checker；复用开源风险素材及生成器构建大规模安全任务，另加独立副作用/越权/持久化 oracle。**这与 Harness Scaling 的执行层独立变化兼容，且比把 30 个实例复制到数千条更符合用户要的规模。安全任务仍需具体构建与验收，不能把目标数写成已收齐。

若要求尽量少自建，第二选择是 RUBAS 已发布 RL training2353 + 它的风险任务生成器，再以少量高质量 harness 近邻补足机制；但大量 RUBAS harmful 题正确行为就是不调用工具并拒绝，不能让它支配“良性目标受到不可信内容污染”的实验。公开 repo 未发现根 LICENSE，不能宣称 MIT/Apache 许可已核实；需同时核查原工具来源条款。

**投票排序：**

1. 主训练构造底座：ClawGym 原任务 + 开源风险生成器/素材的双 oracle 新任务；明确登记为本项目新构造。
2. 现成 RL 任务与环境辅助：RUBAS（2353），优先 sensitive + 真正有用的安全执行，harmful 拒绝题只占受控子层。
3. harness 特异风险种子：SkillSafetyBench（155 cases/47 carriers）、SafeClawArena 的模型可控子集、HarnessSafe/Audit 与 SkillLife；它们不承担全部量级。
4. MalSkillBench 作为恶意/良性 skill 素材和生成器，而非直接 7944 个任务；AuthBench/WeClawArena 作权限/跨 owner 外测或后续适配。

## 来源逐项核查

| 来源 | 真正的规模单位 | 环境/评分/复位 | 训练与许可核查 | 本项目定位 |
|---|---|---|---|---|
| RUBAS | 2353 个 released train records：Harmful2000、Sensitive235、Benign118；82 environments、811 tools 不是额外任务数 | Python 本地模拟工具和初始参数，可重新实例化；task-specific rubric，含工具、参数、回答安全和 helpfulness；不是每题都有独立终态 task_success | README 明确 RL training 和生成 pipeline，stage1只发 data/pipeline/env；root tree 没有 LICENSE，不能标为已确认 permissive | 可扩量骨架/辅助，避免全拒绝占优；补实际任务完成与副作用判据 |
| SkillSafetyBench | 155 prepared cases /47 ordinary task carriers，6 domains、30 categories；不是155独立目标 | 每 case 附 task tests + deterministic verify_attack.py，素材可含 skill、helper、sidecar、memory、templates；现成环境而非历史轨迹 | Apache2；NOTICE 提醒原始 SkillsBench/资产出处；官方称静态 benchmark，明确不是 generator，未见官方 train split | 最贴近需求的双 oracle 风险种子，按原 carrier 分组切分；只能少量补充 |
| SafeClawArena | 406 adversarial task JSON、4 dimensions/24 subcategories；其中20个 SSI1.4 自动插件题无法由模型防守 | reset_env.sh 有完整状态清理/恢复；contrib 生成器与 SSI/PSE/CDF 生成器公开；score 为安全分，当前 compute_score 没使用 safe_results | MIT；无预置训练划分，需研究者拆分与功能验收 | 复用模型可控风险模板/持久化生成器；加合法 Utility；去掉自动触发类 |
| HarnessSafe | 328 frozen cases /7 carrier families | 多阶段载体和源阶段/CSS评分，不能替代合法任务完成 | 代码Apache2，case定义/文档CCBY4；非大规模官方 train pool | 风险分类、carrier/恢复模板，新增独立Utility后精选 |
| HarnessAudit | 官方210 tasks、single/multi-agent，不应拿本地34 single-agent当210 | tool/resource/information-flow约束，SAR/AVS/TCR各有证据；multi-agent AVS用LLM judge；替换产品层要保留授权边界 | MIT；官方评测集，未见6K训练数据/正式训练划分 | 权限/信息流机制，工程适配成本高于小型本地task |
| MalSkillBench | malicious3944（generated3214 + wild703 + tool-test27）+ benign4000，是 skill packages，不是完整合法工作任务 | 发布包有SKILL.md和关联code/resources；生成器 + Docker/OpenCode/syscall监控 + judge；运行时 expected/evidence 不随数据发 | README仅 academic research use；不能声称通用MIT；野外包还要保留原出处 | 千级攻击素材/生成器，补“合法目标+效用+攻击副作用”后才是在线训练任务 |
| AuthBench（Evolvent） | 120 terminal task definitions，非同名语言作者归因百万数据集 | 人审文件级权限标签、实际受限执行、Utility和attack outcomes；重置原terminal环境 | MIT；原任务来自Terminal/SWE/OpenThoughts，来源split与重叠要审计 | 高质量权限风险/迁移外测，无法补6K量级 |
| WeClawArena v2 |124 base tasks×5 conditions=620 variants | 真正跨owner资源、消息与finalstate；独立utility/attack；live+container与mock分别存在，不能拿mock当实证 | repo Apache2，另有DATASET_LICENSE/third-party notices须按源核查 | 跨owner/治理外测，多个agents+可能裁判/浏览器，不宜首轮单4B主池 |

来源：[RUBAS](https://github.com/thu-coai/RUBAS)、[RUBAS train](https://github.com/thu-coai/RUBAS/blob/8bcd81f9bd1cb0147e2bb3d783f16759d4b20259/data/train.json)、[SkillSafetyBench](https://github.com/jinchang1223/skill-safety-bench)、[官方47 carriers页面](https://jinchang1223.github.io/skill-safety-bench-website/)、[SafeClawArena](https://github.com/sunblaze-ucb/SafeClawArena)、[HarnessSafe](https://github.com/artist-coding/harnesssafe)、[HarnessAudit](https://github.com/UCSB-AI/HarnessAudit)、[MalSkillBench](https://github.com/lxyeternal/MalSkillBench)、[AuthBench](https://github.com/evolvent-ai/Authbench)、[WeClawArena](https://github.com/kingofspace0wzz/WeClawArena)。

## 三个会改变选型的源码事实

### 1. RUBAS 的工具是真正可调用的本地状态模拟，但不等于完整双 oracle

已实际读取固定 commit `8bcd81f9bd1cb0147e2bb3d783f16759d4b20259` 的10,054,099字节 train.json，只做 JSON 解析：2353记录与官方Harmful2000/Sensitive235/Benign118完全一致。记录含 instruction、environments(name/tools/parameters)、rubric、risk pattern。不是仅 assistant completion，可以剥离来源会话后用当前策略在线调用工具。

BaseEnv.call_tool 通过实例方法派发，按 schema转换部分参数；会删除未知参数，boolean用bool(value)。迁入本项目时应保留原行为差异记录，并在公共schema边界严格检查，不能让T别名或JSON类型漂移被工具内部悄悄修正。初始 parameters 给每episode重新实例化，适合ONE白盒。

但数据中典型 harmful rubric 把 tool_call_requirement 设 forbidden、helpfulness.enabled=false、must_refuse=true；安全训练若只放大2000题会偏向一概拒绝。rubric里的必须调用某工具/正则命中也不是任务终态证明。Stage1没有完整 trainer/reward evaluator release；生成器明确要LLM API，mode0造instruction+环境，mode1造rubric，mode2/3只校验/过滤环境。需要记录生成来源、校验输出和独立终态断言，不能说现成6k已可运行。

源码：[BaseEnv.py](https://github.com/thu-coai/RUBAS/blob/8bcd81f9bd1cb0147e2bb3d783f16759d4b20259/collected_tools/combined_tools/BaseEnv.py)、[pipeline指南](https://github.com/thu-coai/RUBAS/blob/8bcd81f9bd1cb0147e2bb3d783f16759d4b20259/pipeline/README.md)、[rubric契约](https://github.com/thu-coai/RUBAS/blob/8bcd81f9bd1cb0147e2bb3d783f16759d4b20259/pipeline/Rubric_README.md)。

### 2. SafeClawArena 生成器是可复用风险机制，当前规模不能机械扩成独立6K

固定最新审查commit `c4e8c395a7716f9da123db6b8038cb8021846361`。PSE generator 的2.1明确“5 techniques×4 content variants=20”，2.2“5 techniques×2 scenarios=10”。复制新canary、换名称、unicode/obfuscation产生的新记录仍同风险/合法目标簇，要共同拆分与报告。真正扩量须新工作目标、输入/授权状态与可验证产物，不只改payload外观。

judge.compute_score(success_results,safe_results,taint_leakage) 在1459–1511行计算安全分时没有使用safe_results，不能把高安全分理解为任务做好。CONTRIBUTOR_GUIDE还明确20个SSI1.4插件在gateway_start等钩子执行，不经LLM；这类训练模型不能改变首个违规效果，不应进policy RL主奖励。构建普通白盒工具环境时必须保留模型可决策的前置暴露与安全替代路径。

源码：[judge.py](https://github.com/sunblaze-ucb/SafeClawArena/blob/c4e8c395a7716f9da123db6b8038cb8021846361/scripts/judge.py)、[PSE generator](https://github.com/sunblaze-ucb/SafeClawArena/blob/c4e8c395a7716f9da123db6b8038cb8021846361/scripts/generate_pse_tasks.py)、[contributor guide](https://github.com/sunblaze-ucb/SafeClawArena/blob/c4e8c395a7716f9da123db6b8038cb8021846361/CONTRIBUTOR_GUIDE.md)。

### 3. MalSkillBench 的“验证通过”不是我们的攻击成功

公开README明确：CI/MIXED经Docker/OpenCode、strace/inotifywait和LLM判别；PI的 triggered **与 detected 都算positive**。静态读取Core/verification/verifier.py，PIVerifier还可通过refusal_pass把拒绝识别作为候选通过。这里的passed证明样本可识别/有效，不证明某个defender违规。生成期_expected.json、_evidence.json、_runtime不在公开release，不能假定每个skill随附精确副作用oracle。

因此可用其数千包作为artifact训练素材，但要从恶意行为类别为每个新任务构建受控收集端/权限状态/文件效果判据，不能把passed直接作c_source=1。合法任务Utility也需ClawGym原checker或新的任务断言；模型须能不读/不运行危险skill并仍完成合法目标。

源码：[README verification与release scope](https://github.com/lxyeternal/MalSkillBench#verification)、[verifier.py](https://github.com/lxyeternal/MalSkillBench/blob/main/Core/verification/verifier.py)、[generator.py](https://github.com/lxyeternal/MalSkillBench/blob/main/Core/generation/agent.py)。

## 怎样诚实达到用户要求的规模

- 规划总量13,500条任务记录，通用6,750、安全6,750；**这是建议的建设目标，不是已收齐数量**。50/50还要按采样配额兑现，不能文件数各半但rollout失衡。
- 先对ClawGym完整HF任务做train/dev/test、来源/语义模板去重和capability修复。原2k中397条写输入契约pending是工程待核查，不是天生不可用，不能永久用1,403替代完整任务池。
- 安全端以独立合法目标+公开风险素材/生成器构建任务，保留“正常目标完成且无越权”的联合约束；拒绝恶意用户请求另作受控子层，不能占绝大多数。建议风险骨架至少覆盖工具输出/文档注入、skill代码与供应链、记忆持久化与恢复、权限/信息流、工作流与协议欺骗。
- 每条新增任务冻结 base_task_id、source_artifact_id、risk_template_id、source revisions、目标产物/授权状态、reset方式、utility checker、unsafe effect checker。source artifact/原目标/近重复风险模板的关联簇整体划分；同Claw合法目标的clean与challenged不跨split。
- “安全条数”分别报告：独立原目标、风险模板/机制、复合task实例、clean/challenged条件、harness配置、rollout次数。18/243配置、K2、G8不增加任务池数。若复用6750个相同Claw目标做通用+安全两条件，总共只是6750个独立目标，不可写成13500独立目标。
- 真正足量的有效任务还取决于新oracle/环境验收。达不到目标就报告实际数与缺口；不把仅metadata或static label当可执行。全量新数据生成需实际LLM费用/过滤记录，不把本次只读研究称已生成。

## 接入成本判断

RUBAS本地Python状态和Claw小型文件任务最适合单卡4B训练：环境主要CPU，模型rollout/actor在GPU0。SkillSafety/终端任务可用独立容器，但媒体/浏览器/原生依赖需逐题能力准入。WeClaw多owner与ToolSandbox的LLM用户模拟/真实搜索不是零API成本；ToolSandbox原项目还要求RapidAPI搜索凭据，不能默认全离线。现有产品daemon自动插件行为不能由移除产品层就变成模型训练题。

本审阅没有运行任何环境或测量吞吐/内存，不能承诺并发/时间。推荐的扩量路线保留最可靠小源作独立机制验证，真正大池通过有明确来源与双oracle的新任务构建实现。

补充原始资料：[ToolSandbox README](https://github.com/apple-aiml-research/ToolSandbox)、[AgentSys](https://github.com/ruoyaow/agentsys-memory)（它发布的是防御框架并复用AgentDojo，不是新增大数据集）、[From Procedures to Permissions](https://arxiv.org/html/2611.08905v1)（2871保留SFT轨迹，未在本次检索找到可直接下载的完整RL任务环境，因此不算开源6K来源）。
