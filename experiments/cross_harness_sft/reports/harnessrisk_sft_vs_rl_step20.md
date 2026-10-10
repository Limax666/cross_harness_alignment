# HarnessRisk：全参数 SFT 与在线 RL step-20 对照

SFT 为 Qwen3.5-2B v7 step 267；SFT+RL 是从该权重继续在线 GRPO 20 步的 R11 合并模型。
每个模型×Harness 均使用同一 128 case、3 次独立 rollout（384 条），单 case 超时 300 秒；四项指标均由 HarnessRisk LLM judge `openai/gpt-5.4-nano` 评分。失败和超时轨迹保留在分母。

| Harness | 指标 | SFT | SFT+RL step 20 | 变化（百分点） |
|---|---|---:|---:|---:|
| hermes | Utility ↑ | 4/384 (1.04%) | 20/384 (5.21%) | +4.17 |
| hermes | ASR ↓ | 6/384 (1.56%) | 13/384 (3.39%) | +1.82 |
| hermes | Persistence ↓ | 3/384 (0.78%) | 5/384 (1.30%) | +0.52 |
| hermes | Detection ↑ | 16/384 (4.17%) | 14/384 (3.65%) | -0.52 |
| nanobot | Utility ↑ | 30/384 (7.81%) | 79/384 (20.57%) | +12.76 |
| nanobot | ASR ↓ | 12/384 (3.12%) | 35/384 (9.11%) | +5.99 |
| nanobot | Persistence ↓ | 50/384 (13.02%) | 112/384 (29.17%) | +16.15 |
| nanobot | Detection ↑ | 4/384 (1.04%) | 6/384 (1.56%) | +0.52 |
| openclaw | Utility ↑ | 0/384 (0.00%) | 44/384 (11.46%) | +11.46 |
| openclaw | ASR ↓ | 0/384 (0.00%) | 16/384 (4.17%) | +4.17 |
| openclaw | Persistence ↓ | 88/384 (22.92%) | 118/384 (30.73%) | +7.81 |
| openclaw | Detection ↑ | 2/384 (0.52%) | 4/384 (1.04%) | +0.52 |

四项指标方向分别为 Utility↑、ASR↓、Persistence↓、Detection↑。下表的 `any@3` 指同一 case 的三轮中至少一次为 1。

| Harness | Utility any@3：SFT → RL | ASR any@3：SFT → RL | 超时：SFT → RL |
|---|---:|---:|---:|
| hermes | 4/128 (3.12%) → 18/128 (14.06%) | 6/128 (4.69%) → 12/128 (9.38%) | 90 → 18 |
| nanobot | 29/128 (22.66%) → 59/128 (46.09%) | 10/128 (7.81%) → 29/128 (22.66%) | 7 → 0 |
| openclaw | 0/128 (0.00%) → 37/128 (28.91%) | 0/128 (0.00%) → 14/128 (10.94%) | 0 → 0 |

## 判断

- **hermes**：改善 Utility ↑；退化 ASR ↓, Persistence ↓, Detection ↑。
- **nanobot**：改善 Utility ↑, Detection ↑；退化 ASR ↓, Persistence ↓。
- **openclaw**：改善 Utility ↑, Detection ↑；退化 ASR ↓, Persistence ↓。

OpenClaw 的初始化文件此前被 judge 计作 Persistence 证据；该列须经初始化文件排除后再判断真实持久化变化。低 ASR 若伴随低 Utility，不能单独解释为安全能力提升。论文使用 GPT-5.4，当前 judge 是 GPT-5.4-nano，因此本表仅用于同一 judge 下的 SFT→RL 配对对照。

原始记录：`outputs/harnessrisk_qwen35_2b_full_v7/runs/` 与 `outputs/harnessrisk_qwen35_2b_rl_step20/runs/`。
