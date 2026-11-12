# FlashRSI GEPA B1: AgentDojo × MiniAgent

完整运行归档 `gepa_miniagent_agentdojo_complete.tar.gz` 保留每条 case 证据、每次有效 Snapshot、Worker 轨迹、算法决定、配置、结果和用量汇总。SHA256SUMS 从仓库根目录校验。`source_run_manifest.json` 是运行采用的 B0 冻结 case 清单；`qualification.json` 是两个 calibration family 的验收结果，历史暴露未知；`matrix_at_launch.jsonl` 为启动时快照。

GEPA 使用固定 Qwen3.5-4B 和 FlashRSI pinned revision，dev 94 条，train 反思批量 3，max metric calls 194，实际 100（test 96 次另计）。Worker 提议因改动安全策略被拒，保留初始 Snapshot。实现仍对原 Snapshot 重跑了 candidate minibatch；此举不是对新候选的评估。196 次 Target 尝试 144 次获得官方评分，52 次出错。test verifier coverage 64/96，path audit coverage 0；无法将含 fail-closed 填充值的平均风险当作完整覆盖下的 ASR。完整运行只覆盖一个 harness × benchmark 单元。
