#!/usr/bin/env bash
# Base/SFT x Codex/Claude Code: fail-closed smoke before the fixed 100-task run.
set -euo pipefail
ROOT=/data/home/liumingxiao/cross_harness_alignment
EXP="$ROOT/experiments/cross_harness_sft"
TRAIN_PY=/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python
BENCH_PY=/data/home/liumingxiao/miniforge3/envs/harnessaudit/bin/python
NATIVE_PY="$TRAIN_PY"
BASE_MODEL=Qwen/Qwen3.5-2B-Base
SFT_MODEL="$EXP/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"
SAMPLE="$EXP/outputs/agentdojo_native_2b_sample100_20260927T150131Z/sample_manifest.json"
RUN_ID="agentdojo_sample100_checked_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$EXP/outputs/$RUN_ID"
BASE_PORT=9442; SFT_PORT=9443; BASE_FACADE=9444; SFT_FACADE=9445
WORKER_PORT=9446; BASE_GATEWAY=9447; SFT_GATEWAY=9448
SESSIONS=()
FULL_STARTED=0

fail() { echo "ERROR: $*" >&2; exit 1; }
cleanup() {
  if [[ "$FULL_STARTED" == 0 ]]; then
    for name in "${SESSIONS[@]}"; do tmux kill-session -t "$name" 2>/dev/null || true; done
    if [[ -d "$OUT" ]]; then printf '{"status":"smoke_failed_or_interrupted"}\n' > "$OUT/status.json"; fi
  fi
}
trap cleanup EXIT
start() {
  local name="${RUN_ID}_$1" command="$2"
  tmux new-session -d -s "$name" "$command"
  SESSIONS+=("$name")
}
[[ -f "$SFT_MODEL/model.safetensors" ]] || fail "missing SFT checkpoint"
[[ -f "$SAMPLE" ]] || fail "missing fixed 100-task manifest"
[[ -x "$TRAIN_PY" && -x "$BENCH_PY" ]] || fail "missing Python environments"
command -v tmux >/dev/null || fail "tmux unavailable"
command -v nvidia-smi >/dev/null || fail "nvidia-smi unavailable; cannot verify GPU occupancy"
"$BENCH_PY" -c 'import yaml, httpx' || fail "benchmark environment missing dependencies"
for port in "$BASE_PORT" "$SFT_PORT" "$BASE_FACADE" "$SFT_FACADE" "$WORKER_PORT" "$BASE_GATEWAY" "$SFT_GATEWAY"; do
  if ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then fail "port $port is occupied"; fi
done
mkdir -p "$OUT/configs"
"$BENCH_PY" - "$SAMPLE" "$OUT" "$RUN_ID" <<'PY'
import json, sys, yaml
from pathlib import Path
source, out, run_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sample = json.loads(source.read_text(encoding="utf-8"))
ids = sample["task_ids"]
assert len(ids) == len(set(ids)) == 100
assert sum(x.endswith(":clean") for x in ids) == 9
assert sum(not x.endswith(":clean") for x in ids) == 91
smoke = ["workspace:user_task_10:clean", "slack:user_task_13:injection_task_3"]
assert all(x in ids for x in smoke)
manifest = {**sample, "run_id":run_id, "smoke_task_ids":smoke,
            "status":"smoke_pending", "cli_transport_model":"sonnet",
            "upstream_models":{"base":"qwen35-2b-base-agentdojo","sft":"qwen35-2b-sft-agentdojo"}}
(out/"run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
for role in ("base", "sft"):
    model = manifest["upstream_models"][role]
    for harness in ("codex_native", "claude_code"):
        if harness == "codex_native":
            backend = {"id":harness,"type":"codex_cli","binary":"/data/home/liumingxiao/.local/bin/codex",
                "model":model,"provider":"local_qwen","provider_name":"Local Qwen 3.5 2B",
                "base_url":f"http://127.0.0.1:{9447 if role=='base' else 9448}/v1",
                "env_key":"AGENTDOJO_LOCAL_KEY","wire_api":"responses","stream_logs":False,
                "timeout_seconds":300,"mcp_approval_mode":"approve",
                "mcp_startup_timeout_seconds":60,"mcp_tool_timeout_seconds":120}
        else:
            backend = {"id":harness,"type":"claude_code","binary":"/data/home/liumingxiao/.npm-global/bin/claude",
                "model":model,"cli_model":"sonnet","max_turns":20,"stream_logs":False,
                "timeout_seconds":300,"facade":{"base_url":f"http://127.0.0.1:{9444 if role=='base' else 9445}",
                "local_token_env":"AGENTDOJO_LOCAL_TOKEN","require_loopback":True}}
        for kind, selected, seeds in (("smoke",smoke,[0]),("full",ids,[0,1,2])):
            dest = out/kind/role/harness
            dest.mkdir(parents=True, exist_ok=True)
            config = {"experiment_id":f"{run_id}_{kind}_{role}_{harness}",
                "model_family":"qwen35-2b","output_jsonl":str(dest/"raw_native.jsonl"),
                "episode_workdir":str(dest/"episodes"),"task_ids":selected,"seeds":seeds,
                "max_cases_per_benchmark":{"agentdojo":0},
                "benchmark_sources":[{"type":"agentdojo_http","base_url":"http://127.0.0.1:9446"}],
                "max_tool_calls":8,"safety_modules":["none"],"pretool_guard":{"deny_tools":[]},
                "safety":{"inject_harness_context":True},"harnesses":[backend]}
            (out/"configs"/f"{kind}_{role}_{harness}.yaml").write_text(
                yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
PY

start worker "cd '$ROOT' && PYTHONPATH='$EXP/src:$EXP/vendor/agentdojo/src' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '$BENCH_PY' -m cross_harness_sft.benchmark_server --driver cross_harness_sft.backends.agentdojo:create_driver --config '$EXP/configs/agentdojo_worker.yaml' --host 127.0.0.1 --port '$WORKER_PORT' > '$OUT/worker.log' 2>&1"
for role in base sft; do
  if [[ "$role" == base ]]; then gpu=0; model="$BASE_MODEL"; port="$BASE_PORT"; facade="$BASE_FACADE"; gateway="$BASE_GATEWAY"; slug=qwen35-2b-base-agentdojo; token=agentdojo-local-base-token
  else gpu=1; model="$SFT_MODEL"; port="$SFT_PORT"; facade="$SFT_FACADE"; gateway="$SFT_GATEWAY"; slug=qwen35-2b-sft-agentdojo; token=agentdojo-local-sft-token; fi
  start "$role-model" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '$TRAIN_PY' '$EXP/scripts/local_qwen_openai_server.py' --role '$role' --base-model '$model' --model-id '$slug' --host 127.0.0.1 --port '$port' --full-precision --prompt-format segment --temperature 0 --max-new-tokens 384 --max-input-tokens 16384 > '$OUT/$role-model.log' 2>&1"
  start "$role-gateway" "cd '$ROOT' && '$BENCH_PY' '$EXP/scripts/local_qwen_harness_gateway.py' --upstream 'http://127.0.0.1:$port' --port '$gateway' --model-id '$slug' > '$OUT/$role-gateway.log' 2>&1"
  start "$role-facade" "cd '$ROOT' && SHENGSUANYUN_API_KEY='$token' '$BENCH_PY' '$EXP/scripts/shengsuanyun_anthropic_facade.py' --host 127.0.0.1 --port '$facade' --upstream 'http://127.0.0.1:$port/v1' --upstream-model '$slug' --local-token '$token' --upstream-key-env SHENGSUANYUN_API_KEY --timeout-read 900 > '$OUT/$role-facade.log' 2>&1"
done
ready=0
for _ in $(seq 1 180); do
  ready=1
  for port in "$BASE_PORT" "$SFT_PORT" "$BASE_FACADE" "$SFT_FACADE" "$WORKER_PORT" "$BASE_GATEWAY" "$SFT_GATEWAY"; do
    curl --noproxy '*' -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 || ready=0
  done
  [[ "$ready" == 1 ]] && break
  sleep 2
done
[[ "$ready" == 1 ]] || fail "service health check failed; see $OUT/*.log"

for role in base sft; do
  if [[ "$role" == base ]]; then key=local-base; token=agentdojo-local-base-token
  else key=local-sft; token=agentdojo-local-sft-token; fi
  for harness in codex_native claude_code; do
    echo "Smoke: $role / $harness"
    AGENTDOJO_LOCAL_KEY="$key" AGENTDOJO_LOCAL_TOKEN="$token" PYTHONPATH="$EXP/src:$EXP/vendor/agentdojo/src" \
      "$NATIVE_PY" -m cross_harness_sft.collect_native --config "$OUT/configs/smoke_${role}_${harness}.yaml" --no-resume \
      > "$OUT/smoke/$role/$harness/collector.log" 2>&1 || fail "smoke collector failed: $role/$harness"
  done
done
"$BENCH_PY" "$EXP/scripts/check_agentdojo_eval.py" --root "$OUT/smoke" --manifest "$OUT/run_manifest.json" --smoke \
  | tee "$OUT/smoke_gate.log" || fail "smoke gate failed; full evaluation was NOT started"
if rg -q "\[context-truncated\]" "$OUT/base-model.log" "$OUT/sft-model.log"; then
  fail "model input was truncated during smoke; full evaluation was NOT started"
fi

printf '{"status":"full_running","planned_trajectories":1200,"gpu_ids":[0,1]}\n' > "$OUT/status.json"
for role in base sft; do
  if [[ "$role" == base ]]; then key=local-base; token=agentdojo-local-base-token
  else key=local-sft; token=agentdojo-local-sft-token; fi
  start "$role-eval" "cd '$ROOT' && AGENTDOJO_LOCAL_KEY='$key' AGENTDOJO_LOCAL_TOKEN='$token' PYTHONPATH='$EXP/src:$EXP/vendor/agentdojo/src' '$NATIVE_PY' -m cross_harness_sft.collect_native --config '$OUT/configs/full_${role}_codex_native.yaml' --no-resume > '$OUT/full/$role/codex_native/collector.log' 2>&1 && AGENTDOJO_LOCAL_KEY='$key' AGENTDOJO_LOCAL_TOKEN='$token' PYTHONPATH='$EXP/src:$EXP/vendor/agentdojo/src' '$NATIVE_PY' -m cross_harness_sft.collect_native --config '$OUT/configs/full_${role}_claude_code.yaml' --no-resume > '$OUT/full/$role/claude_code/collector.log' 2>&1"
done
FULL_STARTED=1
echo "Started after 4/4 smoke groups passed: $OUT"
