> 历史v3研究记录：首轮来源已由[11/16独立重新选型v4](harness_scaling_rl_security_selection_20261116.md)覆盖。下方HarnessSafe/Audit准入要求不再是当前首轮前置，原研究证据保留。

# Harness Scaling RL 训练数据调研与首轮配方

日期：2026-11-16。结论用于[4B单卡设计v3](4b_harness_scaling_rl_single_gpu_design_20261116.md)及后续4B/9B同协议对照。本文核查公开论文、官方数据卡、代码和本地文件；没有运行训练或实验评测。

## 1. 首轮只选三个明确来源

**安全源：HarnessSafe＋HarnessAudit single-agent；通用源：ClawGym-Task官方RL精选2,000任务。主配比为25%安全、75%通用，安全内部两来源各半，即全局75% / 12.5% / 12.5%。**这是本项目首轮设计值，不是论文给出的最佳比例，也不是实验结论。

| 数据源 | 具体版本和子集 | 已核实规模 | 首轮用途 |
|---|---|---:|---|
| HarnessSafe | 本地`vendor/harnesssafe/runs/active/**/case_meta.json`，commit `d6aaae97f36f226c5ea715fb76380cdd9d7bf185` | 328个不同case ID，7个canonical suite | 按family拆分后的train部分进入安全池，dev/test分离 |
| HarnessAudit single-agent | 项目根`HarnessAudit/single_agent/tasks/**/*.yaml`及配套工具/bank，commit `6317162590aeeb1c8dde32b880ac199933343e4a` | 34个任务文件，7领域 | 按family拆分后的train部分进入安全池；不是全量multi-agent版本 |
| ClawGym-Task RL selected | `ClawGym/ClawGym-Agents` commit `0e8526d8f5e0999896c74cee4d6b998fb8a0429e`，`RL/data/clawgym_train/` | 官方文档列2,000训练任务；已直接核查其中一个任务与reward入口，未下载/验收全部2,000 | 在官方训练来源内部再留开发集，剩余进入通用池 |

本地源的文件数、领域分布、实际内容聚合hash及远端元数据在[只读盘点记录](research/harness_scaling_rl_dataset_inventory_20261116.json)。328/34/2,000是**来源规模**，不等于完成适配、排重、划分后的可训练数量；也不是2,362个独立family。安全源原为评测基准，本项目建立研究者训练划分，不声称官方提供了相同train/test划分。

## 2. 为什么确定用ClawGym-Task作为通用源

Harness Scaling在训练章节明确使用ClawGym-Task的workspace任务。本轮从该任务体系出发，可减少“通用任务与研究场景无关”的偏差；单卡先用上游精选版本，后续才扩大到全量，不能称精确复现论文的完整训练规模。[Harness Scaling §4.1](https://arxiv.org/html/2610.29458v1#S4.SS1)

官方RL文档提供2,000任务的训练目录和200任务的评测目录；精选任务关注工具操作覆盖并排除web_search/web_fetch相关任务，适合当前离线、可复位的workspace执行器。任务包含输入文件和代码评分入口，不需要用安全clean副本凑满通用配额。[官方数据说明](https://github.com/ClawGym/ClawGym-Agents/blob/0e8526d8f5e0999896c74cee4d6b998fb8a0429e/RL/data/DATASET_README.md)

上游RL实现通过OpenClaw黑盒执行。本项目**仅复用任务、输入和checker，不复用其产品harness训练环**；任务交给自己的S/T/O/C/W白盒执行器，评分读取相同工作空间产物。上游文档确认其RL训练和评测以代码verifier计分；本轮同样将代码reward与后续可选judge分开。[官方RL说明](https://github.com/ClawGym/ClawGym-Agents/blob/0e8526d8f5e0999896c74cee4d6b998fb8a0429e/RL/README.md)

本次直接核查的示例`task_0000`要求读取两份CSV、按条件连接与排序，并输出提醒表和JSON汇总，属于可观察的多步文件/数据工作流；这只证明一个源任务的结构可读取，不证明全池真实执行已通过。[固定版本任务示例](https://github.com/ClawGym/ClawGym-Agents/blob/0e8526d8f5e0999896c74cee4d6b998fb8a0429e/RL/data/clawgym_train/task_0000/data_entry.json)

## 3. 安全源的具体范围与局限

### HarnessSafe

本地按canonical suite盘点：memory_runtime 72、skill_runtime 84、tool_mcp_poisoning 70、memory_to_skill 36、subagent_poisoning 30、compaction_resume_poisoning 30、shared_artifact_supply_chain 6，合计328。多阶段和控制条件归属于原case，不能展开成互相独立的训练/测试任务。元数据中84份缺少显式status字段，不据此直接判成inactive，也不据此保证合格。

保留N0–N5b/CSS等源风险指标；安全训练要另有合法目标完成checker，不能将风险阶段分当作Utility。源码与case定义采用分路径许可，复用时保留对应归属和来源。[官方说明](https://github.com/artist-coding/harnesssafe)、[许可映射](https://github.com/artist-coding/harnesssafe/blob/main/LICENSES.md)

### HarnessAudit

首轮精确使用本地34个single-agent控制任务：finance 6、healthcare 6、office 5、ecommerce 5、daily_life 4、social_interaction 4、legal_compliance 4。保留角色授权、资源参数范围、信息流规则、领域bank和完成度checker；SAR tool/resource/flow、AVS、TCR分别存储，不将它们更名成HarnessRisk的ASR。[官方指标及single-agent说明](https://github.com/UCSB-AI/HarnessAudit)

进一步核查发现，官方README指向两个HF地址。本次匿名请求`eric-ai-lab/HarnessAudit`返回401；可公开读取的`LCZZZZ/HarnessAudit`固定revision为`753706812572bb161ddb0f9e5799c6d7606c9264`，API文件清单确认**210个tasks YAML、8个tools YAML**。它们是完整任务发布与目录，不能把218个YAML全算任务，更不能把210个基础任务冒充本地34个single-agent控制。[公开数据卡](https://huggingface.co/datasets/LCZZZZ/HarnessAudit)

完整多agent、扰动与多模态任务留待后续新池版本，先验收角色隔离和轨迹契约；本轮不无记录扁平化成单agent任务，不因此改变已经明确的两个安全来源权重。

## 4. 主配比及消融配方

`Dρ=(1−ρ)D_ClawGym + (ρ/2)D_HarnessSafe + (ρ/2)D_HarnessAudit-SA`。

| 实验角色 | 安全ρ | 通用ClawGym占全部数据 | HarnessSafe占全部数据 | HarnessAudit-SA占全部数据 |
|---|---:|---:|---:|---:|
| 通用-only对照 | 0% | 100% | 0% | 0% |
| 后续低安全比例 | 10% | 90% | 5% | 5% |
| **首轮主配方** | **25%** | **75%** | **12.5%** | **12.5%** |
| 首轮高安全对照 | 50% | 50% | 25% | 25% |
| 后续仅安全端点 | 100% | 0% | 50% | 50% |

比例是配置展开前task-condition抽样配额；G=8、K=2固定且无额外失败时，对应相同比例的episode尝试。完整16组更新在25%下为12 ClawGym组、2 HarnessSafe组、2 HarnessAudit组；即96/16/16条rollout。10%通过多个K=2块兑现，不按每步整数四舍五入改成别的配方。

首轮每臂2,560次尝试，无额外失败时主配方为ClawGym 1,920、HarnessSafe 320、HarnessAudit 320；这是执行预算，不是独立题数。对应最多120/20/20个K=2任务分配位置，HarnessSafe其中若全为可比配对才有10 clean＋10 attack位置。失败、过滤及补采成本仍全部记录；不把有效样本比例冒充名义配方。

安全源按来源等权可以避免34任务的HarnessAudit在自然拼接后几乎失去权重，但会提高其重复暴露，需要记录每case/family重复次数与留出性能。安全clean只在配对stratum内平衡，不挤占ClawGym的独立通用配额。后续调整安全内部q时另立消融，不和ρ同时无记录变化。

## 5. 明确留作评测或不进入首轮的数据

| 数据 | 首轮角色 | 理由与约束 |
|---|---|---|
| HarnessRisk 128 | 已暴露全量回归，训练权重0 | 对齐现有FlashRSI初始/RSI快照；不用于RL选模，不能叫未见家族测试 |
| ClawGym-Bench 200 | 封存通用终点评测，训练权重0 | 官方RL的`clawgym_eval`即此基准；HF虽标`train`，语义仍是评测集，不能据split名称纳入训练 |
| HarnessSafe/HarnessAudit研究者dev/test | 同源开发/保留泛化 | case的所有条件/阶段/实体变体一起按family隔离，不重标为独立来源迁移 |
| SafeClawArena | 独立安全系统迁移的候选评测，训练权重0 | 系统级隔离/插件等场景需原生平台机制；不能无损改写成普通白盒任务后直接用来训练 |
| τ²-bench、BFCL、SkillsBench | 首轮训练权重0，后续独立能力扩展 | 控制首轮接入规模；各自正式测试继续封存，不把评测题收进通用池 |
| Agent-World | 首轮训练权重0 | 有可执行环境，但还需另冻结任务/反馈契约；其SFT轨迹数量不是现成在线训练任务数，首轮不改做轨迹SFT |
| AgentDojo | 首轮训练权重0 | 按用户研究方向退出主任务池，不作为另两源未适配时的默认替代 |

ClawGym评测与训练的对应以[上游RL数据说明](https://github.com/ClawGym/ClawGym-Agents/blob/0e8526d8f5e0999896c74cee4d6b998fb8a0429e/RL/README.md)为准；HF的200行及split显示见[ClawGym-Bench数据卡](https://huggingface.co/datasets/RUC-AIBOX/ClawGym-Bench)。SafeClawArena的原生系统场景见[官方仓库](https://github.com/sunblaze-ucb/SafeClawArena)；其他候选边界依据[Agent-World](https://github.com/RUC-NLPIR/Agent-World)、[τ-bench](https://github.com/sierra-research/tau2-bench)、[SkillsBench](https://github.com/benchflow-ai/skillsbench)的官方资产说明。

## 6. 转为可训练任务前必须解决的具体事项

1. **划分：**安全两源按跨源family/近重复簇作70/15/15的研究者划分；ClawGym只在官方2,000训练任务内按模板簇作90/10 train/dev，不从200评测题取开发样本。seed固定20261116，先处理历史暴露再分配；实际簇规模可能不等于整数目标，最终清单逐ID报告，不伪造1800/200或230/49/49的精确合格数。
2. **family标识：**优先原始attack-family/origin-template；精选目录缺失模板字段时，补取原始任务元数据匹配或审计模板簇，不能用每条case ID代替family以获得虚假的无泄漏划分。
3. **grader隔离：**上游ClawGym样例将`reward/check.py`放在workspace的reward目录。本项目改为独立、target不可读的评分进程和只读产物快照；保留原checker逻辑及hash，路径适配另记录。目标看不到参考答案、隐藏判据或评分源码，也不能修改checker；checker出错为invalid，不沿用脚本解析失败时返回0的默认。
4. **共享底座：**三源全部通过同一个S/T/O/C/W执行器，工具映射回canonical后再由源checker评分；不启动上游OpenClaw黑盒训练器。配置不按来源绑定。
5. **放行：**三来源各自有合格task、reset/隐私/评分与完整五维真实更新证据，才能运行该首轮配方。某源不足不自动把其权重分给另一源，也不改为“通用任务待定”；先修复或记录新的有名配方版本。

公开资源选定、元数据可读与采样比例明确，不等于完整任务包已下载、许可及适配已验收或训练已运行。ClawGym根LICENSE和HF数据卡没有在本次核查中取得明确数据授权文本；冻结时需保存实际来源的授权/归属依据，不能替上游宣称未核实的许可。当前工作产物是有明确来源、版本和数值的实验配方及只读盘点记录。
