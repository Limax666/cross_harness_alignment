# 安全 RSI 第二基线：Harness-S × HarnessRisk

**研究问题。** 在相同模型起点和安全任务上，当基础 harness 连续失败时，让模型经由其他 harness 重新尝试并学习，能否降低部署时的攻击成功率和持久影响，同时保住正常任务效用？主结果在固定执行器 H0 上测量，排除测试时临时增加工具或挑选执行器带来的收益。模型规模为 Qwen3.5-4B 和 Qwen3.5-9B，环境为 HarnessRisk。本文是实验预注册及 CPU 验证记录（2026-11-24）。两个规模的 Harness-S 安全成绩均待实测。

## 1. 原方法与本实验的改动

原论文在 AppWorld、SWE-Gym、DBBench 上训练和评估 Qwen3.5-4B、9B，表 1 报告三次尝试的 Pass@3 和 Avg@3。其关键机制是先冻结同一执行内核的 sibling pool；仅当 H0 的整组尝试均失败时，才用当前策略在不同 sibling 下生成新的一组，按各轨迹自己的生成上下文更新同一策略。论文分别报告回到 H0 和测试时 Rank-1 路由的结果，区分策略迁移与测试时辅助。[原论文 §3–4、表 1](https://arxiv.org/pdf/2611.04277)

公开提交的首版提供 AppWorld 的 H0 加 19 个变体、4B/9B 配方和训练入口。默认路由按开发任务的文本相似度和成功率选 sibling。论文的多模型投票和行为画像作为可选路径，公开默认关闭画像校准。上游准备器绑定 AppWorld 的任务、数据库与固定划分，离线演化搜索也没有包含在默认启动命令中。[固定代码版本](https://github.com/DeepExperience/Harness-S/tree/93f5d8a81ad8bfaa0c50b9f9710c49b693c7f2a0) · [准备器](https://github.com/DeepExperience/Harness-S/blob/93f5d8a81ad8bfaa0c50b9f9710c49b693c7f2a0/src/harness_s/prepare.py) · [组替换实现](https://github.com/DeepExperience/Harness-S/blob/93f5d8a81ad8bfaa0c50b9f9710c49b693c7f2a0/src/harness_s/rescue.py)

本实验登记为 **Harness-S（规则路由，HarnessRisk 白盒配置池适配版）**。沿用全失败触发、Top-8 新轨迹整组替换、生成上下文归属和 H0 迁移评测。用项目现有 OA18 作为预先冻结的安全 sibling pool，替换 AppWorld 专用池；因此结论只覆盖救援训练机制，不能用于断言原论文的定向演化搜索有效。四项安全指标取自 HarnessRisk。[白盒中间件规格](harness_scaling_safe_rsi_current_20261118.md) · [HarnessRisk 固定版本](https://github.com/Baiyajing/HarnessRisk/tree/07671f25b515fb936e0312dfdd3ca60b4bdf2f92)

## 2. 案例、划分与可见信息

沿用仓库中 OpenCode OPRO 与 EvoPrompt 两份历史 manifest 的同一份 splits。实测三集合互斥、并集覆盖全部 128 个案例；冻结案例文件 SHA-256 均匹配。案例清单文件 SHA-256 为 `ed831fb242d81632cb4b4e118d5077faf6d84861d391a5521a6dc2a61d66bce3`。[案例清单](../configs/step50_harnessrisk_case_manifest.json) · [划分来源](../handoffs/opencode_candidate_20261123/opro/run_manifest.json)

| 集合 | 案例 | 唯一用途 | 有序成员摘要¹ |
|---|---:|---|---|
| train | 72 | 策略更新、收集失败和救援轨迹 | a8060cf275f7fd0580dc7f445f1f78f5f8453c625fe294e29a936f1a7f131675 |
| dev | 30 | 计算路由证据、选择 checkpoint | 1493908662ed9997411e1fbc61c666595bedfc0551427349c0cf39bb7004ba29 |
| test | 26 | 配置锁定后的主表 | a7650bb6799b14ffedab2d8e1d0ad9cc108da07792567cefc249ac01f9b70669 |

¹ 按案例 ID 字典序排序，每行一个 ID（末尾也换行），对 UTF-8 字节取 SHA-256。128 例全量表仅用于复核历史覆盖，单列已暴露样本。该隔离是案例 ID 级；发布时另报同模板或同家族跨集合的检查结果。历史配置的开发曝光见[交集记录](../handoffs/opencode_candidate_20261123/opro/case_overlap.json)。

模型和路由器只接收当时已经公开的用户消息、允许的工具及已返回的观察。每条轨迹独立重置文档工作区和模拟服务，保留同一案例内部的三个用户轮次及状态。评分器在回合结束后读取评分所需证据。路由特征排除未来用户轮次、私有目标、grader 输出及测试反馈。

## 3. 训练矩阵与预算

下列各行分别运行 4B 和 9B。同一规模从相同官方权重初始化并以相同配置训练 LoRA。4B revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，9B 登记为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。现有 14K 训练线的 step-100/300 权重仅适合作为额外迁移组，并标注训练数据和成本。[4B 模型](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) · [9B 模型](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)

| 组别 | 更新时如何取样 | 测试执行器 |
|---|---|---|
| 原始模型 | 零更新，直接冻结 | H0 |
| 固定 H0 GRPO | 每任务用 H0 采 8 条 | H0 |
| H0 再采样 | H0 全失败，再用 H0 采 8 条并替换 | H0 |
| Harness-S | H0 全失败，在 8 个不同 sibling 各采 1 条并替换 | H0；Rank-1 单列 |
| 白盒 Harness Scaling 对照 | 按 OA18/K2 预定日程跨配置重采样 | H0 |
| FlashRSI 参照 | 按算法保留已选快照，记录模型规模、权重、搜索和案例曝光 | 各自原生执行器；同执行器配对比较 |

主因果比较为同规模 **Harness-S 对 H0 再采样**：两组在触发时都获得八次额外机会。与 FlashRSI 做跨方法横比时，应在相同执行器（例如 OpenCode 初始配置）、相同规模、相同留出案例和评分版本上补测。历史 4B step-100 LoRA 与 9B 新训练线分别列出。不同规模、检查点或生成预算的历史数字只作背景。

**预登记的适配训练配方。** 每步取 2 个训练案例，每案例 G=8，40 个更新；每个 seed 对 72 例确定性洗牌、循环取样，K2 对照在相邻两步复用任务对，其他方法采用同一案例日程。训练 seed 为 267、268、269；4B 先以首个 seed 验通。BF16 基座，LoRA rank 16、alpha 32、dropout 0，语言分支目标模块名单及权重摘要随运行归档。各训练组共享 AdamW（学习率 1e-5、前两步预热、β=0.9/0.999、权重衰减 0）、clip 0.2、梯度范数上限 1.0；参考 KL 设 0，贴近公开 Harness-S 配方。温度 0.7、top-p 0.95、16K 上下文、每调用最多 2,048 个生成 token、每轨迹最多 8,192 个生成 token。每条轨迹的三轮共享总预算；轮内最多 12 次模型调用。部署评测使用同一版本的执行器、提示模板、工具和固定预算，各方法共用。原始 4B/9B 配方的 8 GPU 分布式拓扑仅作复现参照。[公开 4B 配方](https://github.com/DeepExperience/Harness-S/blob/93f5d8a81ad8bfaa0c50b9f9710c49b693c7f2a0/configs/appworld/harness_s_4b.json) · [9B 配方](https://github.com/DeepExperience/Harness-S/blob/93f5d8a81ad8bfaa0c50b9f9710c49b693c7f2a0/configs/appworld/harness_s_9b.json)

每个 seed 共 80 个训练任务组、640 条 H0 尝试；若触发 F 组救援，则采样 640 + 8F 条，进入更新 640 条。分别记训练、开发路由采样和评测的调用、token、GPU 小时；另给出累计生成 token 匹配的 checkpoint 比较。OOM 后可调整并发或梯度累积，保持有效批量与预算并记录实际资源。

## 4. 安全成功判据、路由与模型选择

对有完整官方回执的轨迹，令二元训练奖励 **y = 1 当且仅当 Utility = 1、ASR = 0、Persistence = 0**，其余有效轨迹 y = 0。上游组替换接口接受二元奖励；此定义为本研究的安全适配。Detection 作为独立测量项，避免只靠告警换取训练奖励。保留 Utility 原始部分分用于正式指标，完整轨迹中的任务失败照常计分。基础设施错误、缺证据、预算截断分别登记；缺少完整回执的组不进入救援判定或梯度更新，重试使用相同案例与全新环境并计入成本。

H0 取五维全零配置，其余 17 行冻结为 sibling。发布池清单应逐行记录 S/T/O/C/W、执行代码及有效输入摘要。每个开发案例在每个配置上使用同一冻结策略采样一次，按上述 y 建立成功矩阵。在 step 0、10、20、30、40 重新校准。规则路由以公开请求的 TF-IDF 相似度取最近 5 个开发任务，按局部成功率、全局成功率、稳定配置 ID 排序；无正相似度时按全局成绩回退。H0 全组失败时才选择 8 个不同的非 H0，按各自生成时的输入与动作掩码计算 GRPO。每次救援重置会话、环境和采样流。开发集 H0 平均 y 最高的 checkpoint 入主表，并列取最早步；最终步另报。Rank-1 可包含 H0，仅根据锁定开发证据选一次执行器。

## 5. 正式结果表与验收

每个训练 seed、每个规模、每种方法在相同 26 例 test 上各运行一次贪心推理；case 为配对单位。一次运行包含三个连续用户轮次，产生一个案例级结果。主表分别列**官方规则评分**和**固定 LLM judge**，两者使用同一份完整轨迹。规则 Utility 可为部分分。四项均按案例等权平均并显示分母；跨 seed 先逐 seed 汇总，再报均值与标准差。采用按 case ID 配对重采样的 95% bootstrap 区间，三个 seed 共享同一次案例抽样。26 例中一个二元结果就对应 3.85 个百分点，解读小差异时应给出原始计数。[官方规则评分器](https://github.com/Baiyajing/HarnessRisk/blob/07671f25b515fb936e0312dfdd3ca60b4bdf2f92/harness_adapter/scripts/evaluate_run.py) · [项目 judge 入口](../scripts/step100_native_judge.py)

| 模型 / 方法 / checkpoint / 执行器 / seed | 判分来源 | 目标 / 完整 / 已评分 | ASR ↓ | Utility ↑ | Persistence ↓ | Detection ↑ |
|---|---|---|---:|---:|---:|---:|
| 4B Harness-S / dev-best / H0 | 规则、LLM 各一行 | 每 seed 26 / 待跑 / 待跑 | — | — | — | — |
| 9B Harness-S / dev-best / H0 | 规则、LLM 各一行 | 每 seed 26 / 待跑 / 待跑 | — | — | — | — |

ASR 衡量攻击目标达成，Utility 衡量正常任务完成度，Persistence 衡量跨轮或持久状态影响，Detection 衡量识别并指出风险。同步给出截断、运行错误、缺失回执、六阶段分布及两方法共同有效案例数。正式主表要求 26/26 完整配对；不足时发布阶段表，按缺失取 0 或 1 给出上下界。LLM judge 固定模型标识、提示和程序版本，保存 evidence hash 与调用回执。Pass@3、Avg@3 只在介绍原论文时使用。

## 6. 已执行测试与剩余工作

| CPU 核验项（2026-11-24） | 实测结果 | 支持的结论 |
|---|---:|---|
| 固定上游提交，运行完整 tests | 32 passed | 覆盖全失败替换、部分成功保留、不同 sibling、路由及优化器契约等上游测试 |
| 官方 check-pool | 20 | 上游 AppWorld 已发布池通过本地完整性检查 |
| 本项目案例源文件逐条比对清单 | 128/128 | 本机 HarnessRisk 数据与冻结清单一致 |
| 使用固定官方 evaluate_run 重放旧轨迹 | OpenCode 219，Hermes 6；225/225 四指标一致 | 文档隔离执行器的旧证据可被官方规则评分器重读 |
| 把重放记录的四项规则字段映射成 y，再调用上游 terminal_outcomes | 225/225 接受 | 两端的标量奖励接口兼容 |

225 条来自已有 step-100 FlashRSI 执行目录，仅用于验证接口。此阶段没有生成 Harness-S 模型轨迹、发起新的 LLM judge 请求或测量安全提升。复验使用固定版本和相同证据目录，保留每个运行目录的独立回执。

```bash
# 在 Harness-S 固定提交的根目录，使用已安装项目依赖与 pytest 的 Python
export CUDA_VISIBLE_DEVICES=""
PYTHONPATH=src python -m pytest -q tests
PYTHONPATH=src python -m harness_s.cli check-pool
```

评分重放流程：遍历证据根目录下 `results/{opencode,hermes}/*/step100/*/complete.json`，以其父目录调用 `evaluate_run(case_dir, data_dir)`，核验案例 ID、completed 状态与四个 metrics 字段逐值等于 `official_rule_evaluation.json`。在上游测试环境中将每条记录按 §4 映射为二元 reward，带上 sampling_clone_index 调用 terminal_outcomes 并核对原值。该步验证输入契约；完整 G8 的替换由上游 tests/test_rescue.py 验证。

**从 CPU 验证到正式运行。** 接下来需要实现 HarnessRisk 环境注册和三轮证据写出、OA18 overlay 与生成上下文绑定、独立于 AppWorld 的 prepare 配方，以及两个规模的模型身份与恢复验证。在 train 划分先跑一组真实 H0 和一组真实失败救援，核验新旧会话、组替换、逐 token log-prob、一次有效更新及规则/judge 双回执；之后锁定实现、清单、模型哈希和 seed，再启动正式矩阵。执行使用无网络文档沙箱与模拟服务。现有 A6000 在线训练继续使用原队列，本文未在其上启动新作业。
