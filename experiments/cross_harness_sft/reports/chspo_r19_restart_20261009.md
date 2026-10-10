# CHS-PO 宿主内存 OOM 后重启记录

2026-10-09 23:10（北京时间）实际启动新任务：

- Run：`outputs/rl/multiharness_chspo_r19_ramfix_gpu12_20261009_2310`
- 训练 tmux：`chspo-r19-gpu12`；监控 tmux：`chspo-r19-observe`
- 启动 PID：2433997。监控使用 PID/创建时间，禁止用会匹配自身的 `pgrep` 文本判断存活。
- 当前运行位置的固定入口：`outputs/rl/chspo_current_run.json`；不要根据旧目录 mtime 猜测新任务。
- 启动脚本、无密钥参数快照、PID、退出码分别保存在 run 内 `launch.sh`、`launch_protocol.json`、`trainer.pid`、`exit_code`（结束后写入）。

旧 r18 的 128 条在线轨迹通过准入，完成一次 actor 更新，metrics/duals/safety 均落盘；在随后同步权重至 vLLM 时因宿主内存 239.55/251.52 GiB 超过 Ray 95% 阈值失败。旧任务没有可恢复 checkpoint，因此本轮从 SFT step267 重新初始化，不拼接两轮 step。两个训练 rank 在故障快照中各占约 30 GiB RSS；响应预算增加是否为主要原因尚无受控测量支持。

本轮资源参数：GPU1/2；response 1024；AgentLoop workers 4；reward workers 2；vLLM memory utilization 0.40、max_num_seqs 1。保持全参数更新、107-cell 任务池、32 groups × G=4（128 rollout/step）、300 步目标、每 10 步 checkpoint，以及已有 CHS-PO 奖励/dual/准入规则。GPU0/3 不在本轮设备列表中。预算变化是实验条件变化，不能将 r18/r19 曲线合并。

AgentHarm judge 保持 ShengSuanYun `https://router.shengsuanyun.com/api/v1` 上的 `bigmodel/glm-5.3-flash`，通过 `AGENTHARM_JUDGE_API_KEY` 读取密钥；入口现在缺密钥直接报错，避免耗时加载后才失败。

每分钟资源记录：`resource_history.jsonl` 和 `resource_status.json`，含主机可用 RAM、本次进程树各 PID 的 RSS 与累计 I/O 字节；训练退出时记录 `trainer_alive=false`，不关闭 Ray 的 OOM 保护。每个完成更新的原始指标保留在 `metrics.jsonl` / `chspo_safety.jsonl`；独立图输出至 `paper_figures/`（PNG/PDF/SVG），CSV 为 `paper_metrics_per_step.csv`。只在有新指标时重绘。

此记录确认启动动作，不代表内存问题已彻底解决，也不代表安全能力提升。需先确认首步更新后的权重同步与第二步采样成功，再在 step10 检查 checkpoint、塌缩指标和独立验证。

**23:16 启动核验：**进程 PID 2433997 存活，日志已进入 `Training Progress: 0/300`，4 个 `AgentLoopWorker.generate_sequences` 正在执行首批在线采样；资源心跳显示主机可用 RAM 约 43.4 GiB。尚无新 optimizer step；旧故障发生的首次更新后权重同步仍待验证。证据：run 内 `startup_verified.json`。
