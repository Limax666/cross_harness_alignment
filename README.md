# Cross-Harness Alignment

真实 `teacher model + native harness` 的 Agentic Safety 拒绝采样 SFT 项目。

支持的 harness：Codex CLI、Claude Code、OpenClaw、Hermes。支持的评测环境：AgentDojo
和 Inspect Evals/AgentHarm。正式数据流通过逐 episode MCP 服务连接真实 harness 与官方
benchmark runtime，不使用模拟 harness 或教师 API 直连替代品。

## Clone 后开始

```bash
git clone https://github.com/Limax666/cross_harness_aligement.git
cd cross_harness_aligement
bash experiments/cross_harness_sft/scripts/setup_server.sh
```

完整安装、轨迹采集、严格过滤、Qwen3.5-4B QLoRA 和三方配对评估步骤见：

- [`experiments/cross_harness_sft/README.md`](experiments/cross_harness_sft/README.md)
- [`DESIGN_zh.md`](DESIGN_zh.md)

## TandemKit

This project uses TandemKit — Claude and Codex always work in tandem for planning and evaluation.

**To start a new mission:** Run `/tandemkit:planner` and describe your goal. The Planner guides you through everything — including how to start the Generator and Evaluator sessions once the plan is ready.

Project-specific role context: `TandemKit/Planner.md`, `TandemKit/Generator.md`, `TandemKit/Evaluator.md`.

正式配置：`experiments/cross_harness_sft/configs/formal_native.yaml`。

## 数据边界

`outputs/`、`data/`、模型 checkpoint、API 密钥和 harness 登录状态不会提交到 Git。仓库不
提供 mock trajectory generator 或 controlled harness profile。
