"""Paired A6000 HarnessRisk evaluation with the FlashRSI OpenCode and Hermes kernels."""
import argparse
import asyncio
import base64
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from flashrsi_harnessrisk_privacy import assert_no_oracle
from harnessrisk_inert_workspace import InertWorkspace
from step50_pair_evidence import MAX_CALLS, MAX_NEW, PROTOCOL, OwnerTurn, digest, read, score_case, write
from step50_native_case_tools import isolate_native_tools, resources, run_owner

SCRIPTS = Path(__file__).resolve().parent
PINS = SCRIPTS.parent / "configs/step50_hermes_opencode.json"
SKILL = "skills/autonomous-ai-agents/hermes-agent/SKILL.md"
SIDES = ("base", "step50")


def check_files(root, hashes):
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f"Identity mismatch: {name}")


def verified_runtime(args, pins):
    for root, revision in [(args.flashrsi, pins["flashrsi_commit"]), (args.bench, pins["harnessrisk_commit"])]:
        if subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip() != revision:
            raise ValueError(f"Checkout revision differs: {root}")
    check_files(args.bench, {"harness_adapter/scripts/evaluate_run.py": pins["scorer_sha256"]})
    sys.path.insert(0, str(args.bench / "harness_adapter/scripts"))
    native = args.flashrsi / "flashrsi/agents" / args.harness
    if subprocess.check_output(["git", "-C", args.flashrsi, "status", "--porcelain",
                                "--untracked-files=all", "--", str(native.relative_to(args.flashrsi))]).strip():
        raise ValueError("Modified FlashRSI runtime")
    sys.path.insert(0, str(native / "runtime"))
    isolate_native_tools(args.harness)
    return native


def rebuild_snapshot(harness, group, native, temp):
    recipe = read(PINS.with_name("step50_native_snapshot_recipe.json"))[harness]
    selected = recipe["groups"][group]
    overrides = selected["text_overrides"]
    if set(overrides) - {"AGENTS.md", "SOUL.md", SKILL}:
        raise ValueError("Only prompt and skill text can be supplied by the recipe")
    files = {}
    for key, metadata in recipe["layout"].items():
        path = Path(key)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid native resource path")
        raw = overrides[key].encode() if key in overrides else (native / key).read_bytes()
        files[key] = {**metadata, "content": base64.b64encode(raw).decode("ascii")}
    snapshot = {**selected["header"], "files": files}
    path = temp / (group + ".json")
    path.write_bytes((json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return path


def project(snapshot, spec, native, root, harness):
    check_files(snapshot.parent, {snapshot.name: spec["file_sha256"]})
    data = read(snapshot)
    if data["id"] != spec["id"]:
        raise ValueError("Unexpected Snapshot ID")
    files = {}
    for key, node in data["files"].items():
        name = Path(key)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unexpected Snapshot path")
        value = base64.b64decode(node["content"], validate=True)
        if key.startswith("runtime/"):
            if (native / key).read_bytes() != value:
                raise ValueError(f"Snapshot and installed kernel differ: {key}")
        elif key in {"SOUL.md", "config.yaml", "AGENTS.md", SKILL} and not node["executable"]:
            assert_no_oracle(value)
            (root / key).parent.mkdir(parents=True, exist_ok=True)
            (root / key).write_bytes(value)
            files[key] = hashlib.sha256(value).hexdigest()
        else:
            raise ValueError(f"Unsupported Snapshot component: {key}")
    expected = {"SOUL.md", "config.yaml", SKILL if harness == "hermes" else "AGENTS.md"}
    if set(files) != expected:
        raise ValueError("Snapshot resource layout changed")
    return files


def prepare(args, temp):
    pins = read(PINS)
    native = verified_runtime(args, pins)
    case_manifest = (args.artifacts / "case_manifest.json" if args.artifacts else
                     PINS.with_name("step50_harnessrisk_case_manifest.json"))
    check_files(case_manifest.parent, {case_manifest.name: pins["case_manifest_sha256"]})
    rows = read(case_manifest)["cases"]
    if len(rows) != 128 or len({r["case_id"] for r in rows}) != 128:
        raise ValueError("Expected 128 unique cases")
    data = args.bench / "data/HarnessRisk"
    for row in rows:
        cid = row["case_id"]
        if not re.fullmatch(r"(setup|skill|daily|memory|action|recovery)_\d{3}", cid):
            raise ValueError("Unexpected case ID")
        check_files(data, {cid + ".json": row["sha256"]})
        case = read(data / (cid + ".json"))
        if case["case_id"] != cid or len(case["task"]["user_messages"]) != 3:
            raise ValueError("Invalid owner turns")
        InertWorkspace(case["world"])
        assert_no_oracle(case["task"]["user_messages"])
    check_files(args.adapter, pins["model"]["file_sha256"])
    check_files(args.base, pins["base_files"])
    from safetensors import safe_open
    with safe_open(str(args.adapter / "adapter_model.safetensors"), framework="pt", device="cpu") as weights:
        if len(weights.keys()) != pins["model"]["lora_tensors"]:
            raise ValueError("Unexpected adapter tensor count")
    empty = temp / "case"
    empty.mkdir()
    groups = {}
    snapshots = {}
    unique = {}
    for group, spec in pins["snapshots"][args.harness].items():
        root = temp / group
        snapshot = (args.artifacts / spec["path"] if args.artifacts else
                    rebuild_snapshot(args.harness, group, native, temp))
        files = project(snapshot, spec, native, root, args.harness)
        bundle = resources(args.harness, root, empty)
        skill = (root / SKILL).read_text() if args.harness == "hermes" else None
        # Verify the frozen kernel and effective inputs before coalescing a reselected Snapshot.
        key = (spec["id"], tuple(sorted(files.items())), bundle.system_prompt)
        canonical = unique.setdefault(key, group)
        snapshots[group] = {**spec, "canonical_group": canonical, "resource_sha256": files,
                            "prompt_sha256": hashlib.sha256(bundle.system_prompt.encode()).hexdigest()}
        if group == canonical:
            groups[group] = (root, bundle, skill)
    code = {p.name: digest(p) for p in [Path(__file__), SCRIPTS / "step50_pair_evidence.py",
            SCRIPTS / "step50_native_case_tools.py", SCRIPTS / "harnessrisk_inert_workspace.py",
            SCRIPTS / "local_qwen_openai_server.py", SCRIPTS / "flashrsi_harnessrisk_privacy.py",
            SCRIPTS / "flashrsi_harnessrisk_canonical_judge_v2.py", SCRIPTS / "step50_native_judge.py"]}
    manifest = {"protocol": PROTOCOL, "harness": args.harness, "cases": 128, "case_manifest_sha256": pins["case_manifest_sha256"],
                "snapshots": snapshots, "model": pins["model"], "base_files": pins["base_files"], "code": code,
                "flashrsi_commit": pins["flashrsi_commit"], "scorer_sha256": pins["scorer_sha256"],
                "owners": 3, "max_calls_per_owner": MAX_CALLS, "max_new_tokens": MAX_NEW, "precision": "bfloat16",
                "thinking": False, "sampling": "greedy", "pairs_per_group": 128,
                "arm_trajectories": 256 * len(groups)}
    return data, rows, groups, manifest


async def audit_first_requests(args, data, rows, groups, empty):
    """Materialize each real first request at the native provider boundary without generating a reply."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)

    class Captured(BaseException):
        pass

    result = {}
    for group, (root, bundle, skill) in groups.items():
        hashed = hashlib.sha256()
        for row in rows:
            case = read(data / (row["case_id"] + ".json"))
            memory = InertWorkspace(case["world"])
            observed = []
            async def capture(messages, *, model, tools, tool_choice):
                assert_no_oracle([messages, tools])
                names = {tool["function"]["name"] for tool in tools}
                allowed = memory.allowed | ({"skill_view"} if skill is not None else set())
                if names != allowed or messages[0]["role"] != "system" or case["task"]["user_messages"][0] not in str(messages):
                    raise ValueError("First request lost case tools or the owner")
                tokens = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                    enable_thinking=False, add_generation_prompt=True, return_dict=True)["input_ids"]
                import struct
                observed.append(hashlib.sha256(struct.pack("<" + "I" * len(tokens), *tokens)).digest())
                raise Captured()
            turn = SimpleNamespace(memory=memory, chat=capture)
            try:
                await run_owner(args.harness, bundle, root, empty, case["task"]["user_messages"][0], turn, skill)
            except Captured:
                pass
            if len(observed) != 1:
                raise RuntimeError(f"Native first request missing for {group}/{row['case_id']}")
            hashed.update(observed[0])
        result[group] = {"cases": len(rows), "token_ids_set_sha256": hashed.hexdigest()}
    return result


async def case_run(tokenizer, model, case, group, path, harness, data):
    memory = InertWorkspace(case["world"])
    root, bundle, skill = group
    history, transcript, events, outcomes = [], [], [], []
    with TemporaryDirectory(dir="/tmp", prefix="flashrsi-owner-") as work:
        for index, user in enumerate(case["task"]["user_messages"], 1):
            turn = OwnerTurn(tokenizer, model, memory, path, index, history, transcript, events)
            transcript.append({"role": "user", "content": user})
            status, answer, error = await run_owner(harness, bundle, root, Path(work), user, turn, skill)
            if turn.failure is not None:
                raise turn.failure
            if status != "finished" or not turn.calls or not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(f"Owner {index} status={status} detail={error or 'no completed answer'}")
            history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
            outcomes.append({"owner": index, "physical_calls": turn.calls, "native_stop": status,
                             "clipped": turn.clipped, "call_limit": turn.cap, "answer": answer})
    return score_case(case, path, memory, transcript, events, outcomes, harness, data)


@contextmanager
def gpu_lane():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one allocated physical GPU UUID")
    lock_dir = Path("/tmp") / ("step50-gpu-" + str(os.getuid()))
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    if lock_dir.is_symlink() or lock_dir.stat().st_uid != os.getuid():
        raise RuntimeError("Invalid GPU lock directory")
    with (lock_dir / uuid).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pids = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid",
                                         "--format=csv,noheader,nounits"], text=True).strip()
        if pids:
            raise RuntimeError("The selected GPU has a compute process; its owner must release it first")
        yield


def launch(args, data, rows, groups, manifest):
    from local_qwen_openai_server import build_server
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    import torch
    if {name for _, name in socket.if_nameindex()} != {"lo"} or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("Inference requires an offline private network namespace")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select exactly one BF16 GPU")
    gpu = torch.cuda.get_device_properties(0)
    manifest.update({"gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": gpu.name, "torch": str(torch.__version__)})
    run_file = args.output / "run_manifest.json"
    if run_file.exists() and read(run_file) != manifest:
        raise RuntimeError("Output belongs to a different run")
    tokenizer, model = build_server(SimpleNamespace(base_model=str(args.base), adapter=str(args.adapter), full_precision=True))
    if model.dtype != torch.bfloat16:
        raise RuntimeError("Loaded model did not retain BF16 precision")
    expected = load_file(str(args.adapter / "adapter_model.safetensors"))
    live = get_peft_model_state_dict(model)
    if set(expected) != set(live) or not all(torch.equal(live[k].cpu(), v.to(live[k].dtype)) for k, v in expected.items()):
        raise RuntimeError("Live LoRA state differs from the pinned archive")
    del expected, live
    write(run_file, manifest)
    signature = digest(run_file)
    failures, valid = [], 0
    for group, config in groups.items():
        for index, row in enumerate(rows):
            cid = row["case_id"]
            for arm in (SIDES if index % 2 else reversed(SIDES)):
                path = args.output / group / arm / cid
                identity = {"arm": arm, "case_id": cid, "case_sha256": row["sha256"], "group": group,
                            "harness": args.harness, "snapshot_id": manifest["snapshots"][group]["id"],
                            "adapter_enabled": arm == "step50", "run_manifest_sha256": signature}
                if path.exists():
                    if not (path / "complete.json").is_file():
                        failures.append(str(path.relative_to(args.output)))
                        continue
                    receipt = read(path / "complete.json")
                    if any(receipt.get(k) != v for k, v in identity.items()) or not read(path / "privacy_audit.json")["passed"] or read(path / "official_rule_evaluation.json")["case_id"] != cid:
                        raise RuntimeError("Existing completion lacks matching evidence")
                else:
                    path.mkdir(parents=True)
                    write(path / "identity.json", identity)
                    try:
                        with model.disable_adapter() if arm == "base" else nullcontext():
                            if model.get_model_status().enabled is not (arm == "step50"):
                                raise RuntimeError("Active adapter state mismatch")
                            outcome = asyncio.run(case_run(tokenizer, model, read(data / (cid + ".json")),
                                                           config, path, args.harness, data))
                        write(path / "complete.json", {**identity, **outcome})
                    except Exception as exc:
                        write(path / "error.json", {"type": type(exc).__name__, "detail": str(exc)[:1600]})
                        failures.append(str(path.relative_to(args.output)))
                        if len(failures) >= 3 and valid == 0:
                            raise RuntimeError("Three failures before any valid arm. Inspect preserved evidence.") from exc
                        continue
                valid += 1
    write(args.output / "rollouts_finished.json", {"valid_arms": valid, "expected_arms": manifest["arm_trajectories"],
                                                  "unresolved": failures})
    if failures:
        raise SystemExit("Partial coverage; see rollouts_finished.json")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("prepare", "run"))
    cli.add_argument("--harness", choices=("opencode", "hermes"), required=True)
    cli.add_argument("--artifacts", type=lambda s: Path(s).resolve(),
                     help="Optional unpacked historical harnessrisk directory for direct Snapshot verification")
    for name in ("flashrsi", "bench", "base", "adapter", "output"):
        cli.add_argument("--" + name, type=lambda s: Path(s).resolve(), required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "worker.lock").open("a+") as lock, TemporaryDirectory(dir="/tmp", prefix="step50-native-") as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, rows, groups, manifest = prepare(args, Path(tmp))
        if args.action == "prepare":
            requests = asyncio.run(audit_first_requests(args, data, rows, groups, Path(tmp) / "case"))
            write(args.output / "input_audit.json", {**manifest, "native_first_requests": requests})
            print(args.harness, "cases", len(rows), "groups", list(groups), "planned arms", manifest["arm_trajectories"])
        else:
            with gpu_lane():
                launch(args, data, rows, groups, manifest)


if __name__ == "__main__":
    main()
