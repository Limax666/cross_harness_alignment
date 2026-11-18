# Codex 接手：Harness Scaling 安全 RSI（2026-11-18）

这是当前入口，优先于归档的旧 GPU 分工、旧小池方案和“继续小时巡检”等历史指令。

先读：

1. [当前实验规格](experiments/cross_harness_sft/docs/harness_scaling_safe_rsi_current_20261118.md)：三阶段、14K/50:50、五维 OA18/K2、奖励、已验收结果和限制。
2. [新服务器接手步骤](experiments/cross_harness_sft/docs/new_server_handoff_20261118.md)：依赖、数据/模型/检查点迁移、身份门禁与目标 GPU 适配。
3. [发布时点观察](experiments/cross_harness_sft/reports/hsrl_source_handoff_observation_20261118.json)：源服务器 live 状态，不是可直接恢复的停机备份。

## 当前事实与第一步

源服务器仍在运行 4B R1 GRPO-S；本次 Git 同步不停止它，不在新机重复启动同一个 R1。2026-11-18 12:34 的观察计数为 updates7/effective7、下一步16条joint；最近完整独立小时验收为 update6。这两个口径不可混淆，实时状态只从源服务器实际 journal/进程/current_run 获取。

用户已于11:39要求删除小时巡检loop。cron已删除、monitor/disabled存在；禁止在任何服务器自动恢复 cron、Codex唤醒、旧FlashRSI维修队列。取消巡检不等于暂停或完成训练。

接到在新服务器继续训练的指令后，Codex自主完成环境准备、正式数据消费者和源隔离验收、GPU/路径适配，再在安全交接边界正式续训。不要只交付计划或以mock/模拟/冒烟替代真实交互与更新。源机仍活跃时先准备资产，不双开同一实验；迁移要求完整checkpoint、累计预算和无未知inflight。

## 代码与不可默改的契约

- 当前正式代码入口：`experiments/cross_harness_sft/scripts/train_hsrl_v9.py`。v3/v5/v7/v8和旧SkillLife小池保留历史，不是可互换的启动器。
- 官方 `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，独立文件authority在 `configs/hsrl_official4b_851bf6e.json`。不要换成旧SFT或merged checkpoint。
- READY14K＝7K ClawGym-Task＋7K AgentDoG（5500 attack/1000 clean/500 query_target）；manifest SHA `220521662b8c40127929dbe07ad74a631d2d564a700954d1c1382e0615373c99`。完整发布资产不在Git；不能仅下载JSONL冒充READY。
- 16组×G8＝128真实rollouts/更新；每完整步8通用/8安全；五维S/T/O/C/W各三实现、OA18 strength-2、K2同任务不同配置各用当前策略重采样。不是轮流调用产品CLI训练，也不能把C固定0。
- 当前LoRA r16/alpha32/dropout0；BF16基座冻结，496个FP32可训张量。GRPO-S按源适用掩码使用效用减注册风险减0.1协议代价；纯恶意utility不适用与infra缺评分须严格区分。不能用旧GRPO/CHS-PO不同奖励契约拼接结果。
- seed267；16K context/2048 call/8192 episode/12 rounds/30 tool steps；8环境、模型batch≤8、padded budget32768；累计最多38400尝试/300更新，失败/隔离不清零。源Bug修复有120次隔离开销，299只是逐更新上界，完整K2终点/对照等预算要单独审计，不得偷偷加预算。
- 原 source namespace 缺标准异常类导致v8失败；训练必须用 `hsrl_source_repair_v1.adapter.Dataset`。源修复只增加ValueError/TypeError/Exception，原 grader/reward/只读/网络/工具AST门禁保持。旧112 valid未更新＋8 invalid保留隔离，不拼入新128。

## 新服务器不要误用的本机绑定

v9当前只接受物理GPU1，且绑定原绝对路径、lease、CUDA内核和模型authority。单卡GPU0新机需要新版本化入口/独立cohort，不能把0伪装1、删掉guard、全局替换manifest哈希或编辑冻结源码来假装exact resume。按交接文档复用原采样/优化/持久化主体，独立记录真实GPU UUID/UID/socket/内核与路径，核对adapter/AdamW/scheduler/RNG及已耗预算。跨硬件不声称bitwise重放。

原参考机器上active源码worktree是 `/data/home/liumingxiao/hsrl_v3_worktree_20261116`，运行数据在主项目 `experiments/cross_harness_sft/outputs/rl/`；这些是源机地址，不是让新机照抄的本地PID/目录。source pin、环境快照和第三方补丁均已收入configs/patches。Git不包含venv、权重、私有数据/secret/grader输出、完整run或CUDA二进制。

## 结果验收与下一阶段

进程存在/attempted不是完成。每个案例查source valid/done、官方checker、privacy/reset/cleanup、逐token action/old/reference logprob；模型任务失败可真实负奖励，infra失败不可计零。checkpoint必须COMPLETE且有resume.pt才可恢复，ARCHIVED只有adapter不能exact resume。保持initial/最近两份/显式anchor，保留所有轨迹和adapter，不为清盘删复现锚点。

update6独立验收：128/128、496张量变化、loss -0.027500533、preclip19.841835、实际clip约1；完整K2配对/复位通过。原历史梯度尖峰2913.166根因未证明，不因后续有限就称修复。旧较宽QA缺最终回执未验收，14定向JUnit与真实API/源组/optimizer路径证据保留。训练源注册风险目标不是独立ASR，当前没有证明安全/通用能力提升。

先完成当前4B GRPO-S和匹配单配置/效用对照，再推进DAPO/Dr.GRPO/CHS-PO与9B。FlashRSI × HarnessRisk初始/六算法基线属于独立系统评测线：同协议全量身份/规则/judge分别验收，OpenCode SkillOpt/Evo-Harness N/A，选回初始如实记H_r=H_0。不要把尚未完成矩阵写成全量成功，不恢复已停队列抢训练卡。

源码修改沿用 [项目验证skill](.agents/skills/verify-hsrl-dataset/SKILL.md)。不改活代码，不绕过隔离/隐私/资源门禁，不泄露密钥，不删难例，不混协议，不触碰其他用户GPU。新增/修复行为要验证真实受影响CLI/API与实际副作用；测试是经验验证，不是形式正确性证明。

历史交接、旧评测线与旧GPU状态在 [归档AGENT](experiments/cross_harness_sft/docs/archive/AGENT_before_server_handoff_20261118.md)，仅供追溯，不覆盖本页。
