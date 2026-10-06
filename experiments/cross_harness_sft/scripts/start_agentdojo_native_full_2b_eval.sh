#!/usr/bin/env bash
set -euo pipefail
echo "This launcher is retired: it contains an obsolete Claude CLI model and unsafe smoke gate." >&2
echo "Use scripts/start_agentdojo_sample100_checked.sh after reviewing its preflight instead." >&2
exit 2
ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
TRAIN_PY=/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python
BENCH_PY=/data/home/liumingxiao/miniforge3/envs/harnessaudit/bin/python
NATIVE_PY=/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python
BASE_MODEL=Qwen/Qwen3.5-2B-Base
SFT_MODEL="$EXP/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"
RUN_ID="agentdojo_native_2b_full_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$EXP/outputs/$RUN_ID"
BASE_PORT=9342; SFT_PORT=9343; BASE_FACADE=9344; SFT_FACADE=9345; WORKER_PORT=9346; BASE_GATEWAY=9347; SFT_GATEWAY=9348

[[ -f "$SFT_MODEL/model.safetensors" ]] || { echo "Missing full SFT model: $SFT_MODEL" >&2; exit 2; }
[[ -x "$TRAIN_PY" && -x "$BENCH_PY" && -x "$NATIVE_PY" ]] || { echo "Required Python environments are missing" >&2; exit 2; }
for session in agentdojo-full-worker agentdojo-full-base-model agentdojo-full-sft-model agentdojo-full-base-facade agentdojo-full-sft-facade agentdojo-full-base-gateway agentdojo-full-sft-gateway agentdojo-full-base-eval agentdojo-full-sft-eval; do
  if tmux has-session -t "$session" 2>/dev/null; then echo "Active session exists: $session" >&2; exit 2; fi
done
mkdir -p "$OUT/configs" "$OUT/base" "$OUT/sft"

python3 - "$EXP" "$OUT" "$RUN_ID" "$BASE_PORT" "$SFT_PORT" "$BASE_FACADE" "$SFT_FACADE" "$WORKER_PORT" "$BASE_GATEWAY" "$SFT_GATEWAY" <<'PY'
import json, sys, yaml
from pathlib import Path
exp, out, run_id, bp, sp, bf, sf, wp, bg, sg = sys.argv[1:]
exp, out = Path(exp), Path(out)
bp, sp, bf, sf, wp, bg, sg = map(int, (bp, sp, bf, sf, wp, bg, sg))
common = {"model_family":"qwen35-2b", "max_cases_per_benchmark":{"agentdojo":0},
 "benchmark_sources":[{"type":"agentdojo_http","base_url":f"http://127.0.0.1:{wp}"}],
 "max_tool_calls":8,"safety_modules":["none"],"pretool_guard":{"deny_tools":[]},
 "safety":{"inject_harness_context":True}}
for role, port, facade_port, model in (("base",bp,bf,"qwen35-2b-base-agentdojo"),("sft",sp,sf,"qwen35-2b-sft-agentdojo")):
  codex={"id":"codex_native","type":"codex_cli","binary":"/data/home/liumingxiao/.local/bin/codex","model":model,
  "provider":"local_qwen","provider_name":"Local Qwen 3.5 2B","base_url":f"http://127.0.0.1:{bg if role=='base' else sg}/v1",
  "env_key":"AGENTDOJO_LOCAL_KEY","wire_api":"responses","stream_logs":False,"timeout_seconds":180,
  "mcp_approval_mode":"approve","mcp_startup_timeout_seconds":60,"mcp_tool_timeout_seconds":120}
  claude={"id":"claude_code","type":"claude_code","binary":"/data/home/liumingxiao/.npm-global/bin/claude","model":model,
   "cli_model":"claude-sonnet-4-20250514","max_turns":20,"stream_logs":False,"timeout_seconds":180,
   "facade":{"base_url":f"http://127.0.0.1:{facade_port}","local_token_env":"AGENTDOJO_LOCAL_TOKEN","require_loopback":True}}
  for kind,seeds in (("smoke",[0]),("full",[0,1,2])):
   cfg={**common,"experiment_id":f"{run_id}_{role}_{kind}","output_jsonl":str(out/role/"raw_native.jsonl"),
    "episode_workdir":str(out/role/"episodes"),"seeds":seeds,
    "benchmark_sources":[{"type":"agentdojo_http","base_url":f"http://127.0.0.1:{wp}"}],
    "harnesses":[claude] if kind=="smoke" and role=="sft" else [codex,claude]}
   (out/"configs"/f"{role}_{kind}.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
(out/"run_manifest.json").write_text(json.dumps({"run_id":run_id,"benchmark":"AgentDojo v1.2.2",
 "cases":1046,"case_types":{"clean":97,"injection":949},"models":{"base":"Qwen/Qwen3.5-2B-Base",
 "sft":str(exp/"checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267")},
 "harnesses":["Codex CLI","Claude Code"],"rollouts_per_case":3,"expected_episodes_total":12552,
 "agentdojo_attack":"official DirectAttack (TODO: {goal}) on ground-truth-observed vectors",
 "metrics":"official AgentDojo utility and injection-success verifier","split_policy":"full benchmark; no family holdout",
 "gpu_policy":"GPU 0 Base, GPU 1 SFT; GPU 2 and 3 idle"},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
PY

tmux new-session -d -s agentdojo-full-worker "cd '$ROOT' && PYTHONPATH='$EXP/src:$EXP/vendor/agentdojo/src' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '$BENCH_PY' -m cross_harness_sft.benchmark_server --driver cross_harness_sft.backends.agentdojo:create_driver --config '$EXP/configs/agentdojo_worker.yaml' --host 127.0.0.1 --port '$WORKER_PORT' > '$OUT/worker.log' 2>&1"
tmux new-session -d -s agentdojo-full-base-model "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '$TRAIN_PY' '$EXP/scripts/local_qwen_openai_server.py' --role base --base-model '$BASE_MODEL' --model-id qwen35-2b-base-agentdojo --host 127.0.0.1 --port '$BASE_PORT' --full-precision --prompt-format segment --temperature 0 --max-new-tokens 384 --max-input-tokens 6144 > '$OUT/base/model_server.log' 2>&1"
tmux new-session -d -s agentdojo-full-sft-model "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '$TRAIN_PY' '$EXP/scripts/local_qwen_openai_server.py' --role sft --base-model '$SFT_MODEL' --model-id qwen35-2b-sft-agentdojo --host 127.0.0.1 --port '$SFT_PORT' --full-precision --prompt-format segment --temperature 0 --max-new-tokens 384 --max-input-tokens 6144 > '$OUT/sft/model_server.log' 2>&1"
tmux new-session -d -s agentdojo-full-base-gateway "cd '$ROOT' && '$BENCH_PY' '$EXP/scripts/local_qwen_harness_gateway.py' --upstream 'http://127.0.0.1:$BASE_PORT' --port '$BASE_GATEWAY' --model-id qwen35-2b-base-agentdojo > '$OUT/base/codex_gateway.log' 2>&1"
tmux new-session -d -s agentdojo-full-sft-gateway "cd '$ROOT' && '$BENCH_PY' '$EXP/scripts/local_qwen_harness_gateway.py' --upstream 'http://127.0.0.1:$SFT_PORT' --port '$SFT_GATEWAY' --model-id qwen35-2b-sft-agentdojo > '$OUT/sft/codex_gateway.log' 2>&1"
tmux new-session -d -s agentdojo-full-base-facade "cd '$ROOT' && SHENGSUANYUN_API_KEY=agentdojo-local-base-token '$BENCH_PY' '$EXP/scripts/shengsuanyun_anthropic_facade.py' --host 127.0.0.1 --port '$BASE_FACADE' --upstream 'http://127.0.0.1:$BASE_PORT/v1' --local-token agentdojo-local-base-token --upstream-key-env SHENGSUANYUN_API_KEY --timeout-read 900 > '$OUT/base/claude_facade.log' 2>&1"
tmux new-session -d -s agentdojo-full-sft-facade "cd '$ROOT' && SHENGSUANYUN_API_KEY=agentdojo-local-sft-token '$BENCH_PY' '$EXP/scripts/shengsuanyun_anthropic_facade.py' --host 127.0.0.1 --port '$SFT_FACADE' --upstream 'http://127.0.0.1:$SFT_PORT/v1' --local-token agentdojo-local-sft-token --upstream-key-env SHENGSUANYUN_API_KEY --timeout-read 900 > '$OUT/sft/claude_facade.log' 2>&1"

ready=0
for _ in $(seq 1 180); do
 if curl --noproxy '*' -fsS "http://127.0.0.1:$WORKER_PORT/health" >/dev/null 2>&1 && curl --noproxy '*' -fsS "http://127.0.0.1:$BASE_PORT/health" >/dev/null 2>&1 && curl --noproxy '*' -fsS "http://127.0.0.1:$SFT_PORT/health" >/dev/null 2>&1 && curl --noproxy '*' -fsS "http://127.0.0.1:$BASE_FACADE/health" >/dev/null 2>&1 && curl --noproxy '*' -fsS "http://127.0.0.1:$SFT_FACADE/health" >/dev/null 2>&1 && curl --noproxy '*' -fsS "http://127.0.0.1:$BASE_GATEWAY/health" >/dev/null 2>&1 && curl --noproxy '*' -fsS "http://127.0.0.1:$SFT_GATEWAY/health" >/dev/null 2>&1; then ready=1; break; fi
 sleep 2
done
[[ "$ready" == 1 ]] || { echo "Services failed health checks; see $OUT" >&2; exit 1; }

PYTHONPATH="$EXP/src:$EXP/vendor/agentdojo/src" "$BENCH_PY" - "$EXP/configs/agentdojo_worker.yaml" "$OUT/smoke_task_ids.txt" <<'PY'
import sys
from cross_harness_sft.backends.agentdojo import AgentDojoDriver
d=AgentDojoDriver(sys.argv[1]); cases=d.cases()
clean=next(x["task_id"] for x in cases if x["task_type"]=="clean")
attack=next(x["task_id"] for x in cases if x["task_type"]=="injection")
open(sys.argv[2],"w",encoding="utf-8").write(clean+"\n"+attack+"\n")
print(f"cases={len(cases)} clean=97 injection=949")
PY

for role in base sft; do
 if [[ "$role" == base ]]; then key=local-base; token=agentdojo-local-base-token; else key=local-sft; token=agentdojo-local-sft-token; fi
 set +e
 AGENTDOJO_LOCAL_KEY="$key" AGENTDOJO_LOCAL_TOKEN="$token" PYTHONPATH="$EXP/src:$EXP/vendor/agentdojo/src" "$NATIVE_PY" -m cross_harness_sft.collect_native --config "$OUT/configs/${role}_smoke.yaml" --no-resume --task-id "$(sed -n '1p' "$OUT/smoke_task_ids.txt")" --task-id "$(sed -n '2p' "$OUT/smoke_task_ids.txt")" > "$OUT/$role/smoke.log" 2>&1
 code=$?
 set -e
 [[ "$code" == 0 ]] || { echo "$role smoke failed; see $OUT/$role/smoke.log" >&2; exit 1; }
done

"$BENCH_PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
for role in ("base","sft"):
 rows=[json.loads(x) for x in (out/role/"raw_native.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
 errors_path=out/role/"raw_native.errors.jsonl"
 errors=[json.loads(x) for x in errors_path.read_text(encoding="utf-8").splitlines() if x.strip()] if errors_path.exists() else []
 ids={x.get("metadata",{}).get("harness_id") for x in rows}|{x.get("harness") for x in errors}
 expected={"claude_code"} if role=="sft" else {"codex_native","claude_code"}
 if ids!=expected: raise SystemExit(f"{role}: smoke harnesses {ids}, expected {expected}; see smoke logs")
 print(f"{role}: smoke harness transport verified; completed={len(rows)} failed={len(errors)}")
PY

for role in base sft; do
 if [[ "$role" == base ]]; then key=local-base; token=agentdojo-local-base-token; else key=local-sft; token=agentdojo-local-sft-token; fi
 tmux new-session -d -s "agentdojo-full-$role-eval" "cd '$ROOT' && AGENTDOJO_LOCAL_KEY='$key' AGENTDOJO_LOCAL_TOKEN='$token' PYTHONPATH='$EXP/src:$EXP/vendor/agentdojo/src' '$NATIVE_PY' -m cross_harness_sft.collect_native --config '$OUT/configs/${role}_full.yaml' --resume > '$OUT/$role/full.log' 2>&1"
done

cat > "$OUT/status.json" <<EOF
{"status":"full_evaluation_running","run_id":"$RUN_ID","cases":1046,"planned_episodes":12552,"models":["base","sft"],"harnesses":["codex_native","claude_code"],"rollouts_per_case":3,"gpu_ids":[0,1],"idle_gpus":[2,3]}
EOF
echo "FULL AgentDojo evaluation started: $OUT"
cat "$OUT/status.json"
