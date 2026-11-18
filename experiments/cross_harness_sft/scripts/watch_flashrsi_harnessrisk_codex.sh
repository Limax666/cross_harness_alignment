#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/home/liumingxiao/cross_harness_alignment
OUT="$ROOT/experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_v1_20261113/monitor"
THREAD=01a1aa6a-0266-7d60-9abf-997271f68415
mkdir -p "$OUT"
exec 9>"$OUT/wake.lock"
if ! flock -n 9; then printf '[%s] wake already running; skipped\n' "$(date -Is)"; exit 0; fi
if [[ -f "$OUT/disabled" ]]; then printf '[%s] monitor disabled by marker\n' "$(date -Is)"; exit 0; fi
PROMPT='这是用户授权的每小时双卡闭环巡检。读取 AGENT.md 和 experiments/cross_harness_sft/docs/flashrsi_matrix_repair_loop_20261114.md，优先检查 GPU0 上 Hermes GEPA v11 与 GPU1 上 OpenCode gpu1 v1 初始/OPRO/GEPA/EvoPrompt/ACE 队列及其后续真实重跑；OpenCode SkillOpt/Evo-Harness 为 N/A。检查真实进程、端口、GPU、attempts、逐案例隐私审计与官方规则、result、selected Snapshot、judge manifest 和队列日志；有进程或 attempted 不等于有效完成。扫描 matrix.json、repair_tickets.json、worker.json 和 repair_journal.md；检查 worker.lock 和活进程，若已有维修器工作，不并行修改相同代码或启动重复任务。失败、卡死、接线和 judge 问题应保存证据、定位根因、自主修复并在独立目录安全重跑完整真实评测；禁止盲目增加 token、放松沙箱/隐私、泄露密钥、错误计零、删难例、混协议或触碰其他人的 GPU。已完成 judge 不重复调用。检查 OpenCode 初始结束后四种算法是否实际衔接，失败项不得遗忘。每次将时间、覆盖、失败、实际修复与恢复结果追加 hourly_report.md。两条当前队列及其适用算法均满足同一运行的128案例身份、隐私、规则和judge验收才结束双卡巡检；其他未完成矩阵项如未纳入本次双卡范围不得冒充完成。'
printf '[%s] queueing Codex thread %s\n' "$(date -Is)" "$THREAD"
timeout 60 /data/home/liumingxiao/.local/bin/codex queue --thread "$THREAD" --message "$PROMPT"
printf '[%s] queued successfully\n' "$(date -Is)"
