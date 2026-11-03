# SafeClawArena 留出家族：SFT 与 CHS-PO step-150（最终重评）

2026-10-16 23:07 CST 完成。使用 v7 未入 SFT 的 PSE:2.4（10 题）与 SSI:1.2（20 题），NemoClaw、SecLaw 各题 3 次独立复位；两模型按相同 case/rep/seed 由 SafeClawArena 官方 `judge.py` 评分。四组各 90/90 条均有有效 verdict，GPU 已释放。最终证据在 `outputs/safeclawarena_chspo150_eval_schemafix_v3_20261016/` 的 `provenance.json`、`matrix.log` 和逐 case JSON；旧版和前两次修复试跑均不计入本表。

| Harness | 模型 | 官方安全分均值↑ | 攻击成功率（逐条 ASR）↓ | ASR_any@3（30 题）↓ | Detection↑ | 可见协议拒绝* | 回合耗尽回复* |
|---|---|---:|---:|---:|---:|---:|---:|
| NemoClaw | SFT | 0.922 | 7/90（7.8%） | 6/30（20.0%） | 0/90 | 32/90 | 56/90 |
| NemoClaw | CHS-PO 150 | 0.767 | 21/90（23.3%） | 10/30（33.3%） | 2/90 | 44/90 | 16/90 |
| SecLaw | SFT | 1.000 | 0/90（0.0%） | 0/30（0.0%） | 0/90 | 0/90 | 88/90 |
| SecLaw | CHS-PO 150 | 0.911 | 8/90（8.9%） | 5/30（16.7%） | 5/90 | 6/90 | 35/90 |

相对 SFT，CHS-PO 在 **NemoClaw 的逐条 ASR 增加 15.6 个百分点**，在 **SecLaw 增加 8.9 个百分点**；两种未见 harness 均未显示所需的攻击安全泛化。Detection 略有增加，但不足以抵消 ASR 回退。

\*“可见协议拒绝”仅统计官方结果 `sessions[].agent_response` 中出现 `Tool call rejected:` 的案例，不等于所有模型原始生成中的协议错误。服务器日志仍能看到缺必填参数、未知工具名；官方结果的 `sessions[].tool_calls` 字段不能单独用于判断真实工具执行覆盖率。“回合耗尽回复”统计回复包含 `allowed turns` 的案例。两项都是诊断指标，不参与官方 ASR 计算。SafeClawArena 是攻击任务集，没有配对良性 Utility；尤其 SecLaw/SFT 的 88/90 回合耗尽意味着 0 ASR **不能**视为成功完成合法任务或稳健防御。

这是“未见运行 harness、但 SafeClawArena 来源曾在 SFT 的 OpenClaw 轨迹中出现”的迁移诊断，不是完全未见 benchmark；它也不是与 GRPO 等预算严格匹配的 CHS-PO 消融。NemoClaw/SecLaw 本轮结果已被用于诊断，后续调参不能再将相同 case 当作盲测最终集。
