# FlashRSI 与安全 RL 接入状态

用户指定使用上游 FlashRSI 推进 RSI，五维 Harness Scaling 由独立 coding agent 并行构建。上游仓库为 `https://github.com/RUC-NLPIR/FlashRSI`，本次开始获取时远端 HEAD 为 `778e2d01a995491fe63db210dcf46db1723a4430`；获取完成后还须用本地 `git rev-parse HEAD` 核对，未完成下载不能视为已安装。

## 定位与实验边界

FlashRSI 原生对象是可版本化的 agent Snapshot、可编辑资源和 RSI 搜索方法；本项目 VeRL 的 GRPO/CHS-PO 负责模型参数更新。FlashRSI 自身的 harness 资源演化不等价于 VeRL 在线权重训练，也不自动提供 CHS-PO。

首个候选基线为 FlashRSI 的 AHE（迭代式 harness 修订），并保留其自身原始配置与 snapshot/evolution 记录；最终选择以前置可运行性审计和冻结配置为准。训练模型、worker、judge、配置、方法版本及成本必须分开记录。Qwen3.5-4B 官方后训练权重为目标起点，未获取/核验前不得启动模型任务。

FlashRSI 文档将 Codex 描述为基于固定 Rust 版本的 Python 语义移植，OpenCode 描述为纯 Python Snapshot runtime；这些不能冒充原生 Codex/OpenCode CLI。原生外部 harness 作为独立泛化评估，需实际执行与协议验收。

## 接线与准入顺序

1. 核对官方代码来源与本地 commit，保留原始 checkout。检查本机 Python/依赖，不覆盖已有 VeRL 环境；官方 `pyproject.toml` 要求 Python >=3.11。
2. 运行不调用模型的 `--check-only` 或 `--prepare-only`，只证明配置/依赖级准备；FlashRSI 自身明确不因此验证远端 API 或真实模型。
3. 以工程用任务和独立沙箱建立 FlashRSI Snapshot → 审核后五维 harness 配置的适配。固定受信系统安全规则、工具权限、模型端点、文件系统和网络隔离；RSI 候选资源必须验证路径、哈希、语义及任务留出，失败拒收。
4. 对 AgentDojo / HarnessAudit 单独验收真实 reset、同目标 clean/attack 配对、状态/副作用 oracle 与安全反馈；没有受验资格的训练 cell 不接入。禁止训练/test 家族和测试 harness 泄漏给 worker。
5. 接通已冻结任务及版本的在线 VeRL rollout 与参数更新，区分模型权重更新与 FlashRSI 资源编辑；严格匹配 GRPO 与 CHS-PO 的 reward profile，记录生成器/worker/rollout/评测成本及每轮 snapshot。
6. 4B 小规模真实预检确认任务与资源迭代确实反馈到下一轮，并完成 optimizer 更新、actor→rollout 同步、checkpoint 重启、分段 log-prob 和 oracle 验收。直到五维 C1/C2 真正接入 VeRL 动态分段和全局 token 损失计算，持续保留旧训练路径的 fail-closed。
7. 冻结公平预算与验证门槛后，才启动正式对照。记录安全且完成、benign Utility、ASR、真实越权/持久化与有效覆盖；独立外部 harness 的结论不能从 FlashRSI 语义移植推断。

## 当前状态

- 独立 Codex 任务 `01a1c99c-48ff-7f42-a000-c16142ac2308` 负责五维配置/分段基建；其 CPU 原型并不代表 C1/C2 已接入真实训练。
- FlashRSI 已克隆至 `vendor/FlashRSI`，本地 HEAD 为 `778e2d01a995491fe63db210dcf46db1723a4430`，工作树干净。用既有 Python 3.11 环境对上游本地 SearchQA quickstart 执行 `--check-only`：SkillOpt 与 AHE 配置均报告 train=6、dev=3、test=3；两者都使用不可连接的本地占位端点，命令明确没有模型调用、没有执行任务，也没有验证真实 API、权限隔离或训练。AHE 示例允许 worker 编辑 target 声明的资源，正式安全实验必须先实现资源角色与权限的受限映射，不可直接运行默认演化。
- FlashRSI 与本项目训练链、AgentDojo/HarnessAudit 及 4B 模型仍需实际接入和验证；没有启动或声称已完成训练。4B 官方权重下载仍在进行。
