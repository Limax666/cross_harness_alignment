# FlashRSI 4B 安全基线：范围与运行清单

目标是上游 7 个 Target Harness × 项目列出的可执行安全 benchmark 的完整任务条件。FlashRSI commit `778e2d01a995491fe63db210dcf46db1723a4430` 的 31 个内置 benchmark 没有专门的安全类 benchmark，以下安全源均需独立接入。所用模型固定为 `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；模型训练和资源演化均未启动。

## 2026-11-12：切换至 FlashRSI 算法 B1

用户要求暂停正在执行的固定 Snapshot B0，并转向真实 FlashRSI 算法实验。B0 的 AgentDojo MiniAgent、OpenCode、OpenClaw、Pi 以及 AgentHarm MiniAgent 五个采集入口已停止；输出目录保留，停止记录见 `outputs/b0_pause_record_20261112.json`。Qwen 4B vLLM 服务保留，`/v1/models` 已确认服务可访问。没有删除已有 B0 样本。

B1 采用 FlashRSI 官方 **GEPA**，固定 Qwen 4B 同时担任 Target 与 Worker，每个合格的 harness × benchmark 独立运行；仅编辑 Snapshot 的 `prompt` 组件，并锁定安全策略段、工具权限、环境和 verifier。GEPA 要求 prompt 组件和 instruction file；上游七个 Target 都具备该接口。每 cell 采用一个 epoch、reflection minibatch size 3，度量调用上限为 `2 × dev_count + 2 × min(3, train_count)`，最终 test 只在 Snapshot 冻结后执行。

当前 B1 有 49 个目标格子（7 harness × 7 个已列出的安全数据源），**正式验收通过为 0，B1 算法调用尚未启动**。AgentDojo 现有校准脚本将样本标为候选校准、非正式准入；其他格子仍缺执行器、复位证明、独立评分控制或 harness 接入。不能绕过验收门槛。逐格状态见 `outputs/b1_gepa_security_20261112/matrix_ledger.jsonl`，资源与阻塞见同目录 `preflight.json`。

## Harness 进度

| FlashRSI Target | AgentDojo v1.2.2，1,046 条 | 说明 |
|---|---|---|
| MiniAgent | B0 已暂停；B1 待正式验收 | AgentDojo 原生工具；受限 snapshot |
| OpenCode | B0 已暂停；B1 待正式验收 | FlashRSI 内置 Python 语义适配器，非原生 CLI；工具注册表接入 AgentDojo |
| OpenClaw | B0 已暂停；B1 待正式验收 | FlashRSI 内置 Python 语义适配器，非原生 CLI；工具注册表接入 AgentDojo |
| Hermes | B0 当前未运行；B1 待正式验收 | FlashRSI 内置 Python 适配器，非原生 CLI；AgentDojo 安全路径待验收 |
| Pi | B0 已暂停；B1 待正式验收 | 已有 Python 适配器和流式工具桥；完整安全资格待验收 |
| Codex | 未接入 | 上游为 Python 语义适配器；Responses 协议和工具分发需接入 |
| Claude Code | 未接入 | 上游为闭源 SDK 集成；需可隔离的工具/MCP 路径及模型协议适配 |

各队列的冻结清单和逐任务记录分别位于 `outputs/b0_full_agentdojo_20261112_v2/`、`outputs/b0_full_agentdojo_opencode_20261112_v2/`、`outputs/b0_full_agentdojo_openclaw_20261112/`、`outputs/b0_full_agentdojo_hermes_20261112/`。每条轨迹都记录 snapshot、工具认证、官方评分及路径审计状态；无路径 oracle 的结果不可视作综合安全通过。测试 family 只用于预先冻结的基线对照，不向后续训练、演化或选模提供结果。

## 安全 benchmark 数据源

| 来源 | 本机数据及任务规模 | 状态 |
|---|---|---|
| AgentDojo | 官方 v1.2.2，97 条干净、949 条攻击条件；4 个环境 | B0 已暂停；已有适配器的 family、路径审计和评分仍需逐格正式验收 |
| AgentHarm | Inspect Evals 0.28.0 固定数据 revision `e23b3fe60a0da9037314b88e5ee3a0c054970dad`，val 64、test_public 352 | B0 MiniAgent 已暂停；独立语义裁判和其他 Harness 尚未接入，未评分不能当安全 |
| HarnessAudit | 仓库 `fba81545003bc32743f6e75d6bb4a0ad7e1d0d3c`，210 个多代理任务、34 个单代理任务 | 已取得源码与任务；原生执行器、状态环境、裁判及 FlashRSI 适配器尚未接入 |
| HarnessRisk | 公开数据 revision `54589cd6dca0da004b603ce15cdb4fb5358ff52e`，128 个 case | parquet 中 `full_case_json` 已按 ID 重建并写入哈希清单；环境、评测和 FlashRSI 适配器尚未接入 |
| ActBench | 交接包包含固定源码和 300 个任务目录 | 数据条件、mock 服务及 FlashRSI 适配器尚待接入和计数验收 |
| SafeClawArena | 交接包含 406 个任务文件 | 隔离执行、工具环境、评分器与 FlashRSI 适配器尚未接入 |
| WildClawBench | 仓库 `0c6551f798f8028bdfdd065f10d8d6b048a7fd06` | 已取得源码；安全子集与版本计数、隔离执行和评分器尚待验收 |
| HarnessSafe / HarnessSecurity-Bench | 历史文档提及 | 可执行官方来源与版本尚待核实；不得虚构任务或结果 |

本地隔离采用网络命名空间、Landlock 和 seccomp 的实验实现，尚未经过独立安全审计；不能当成正式安全隔离的证明。环境不提供可用 Docker，不能直接将需要容器的真实代码执行任务放到宿主机运行。出现 API 协议错误、模型超时、不可执行任务、评分器缺失或路径审计弃权时应单独统计，均不记为安全通过。
