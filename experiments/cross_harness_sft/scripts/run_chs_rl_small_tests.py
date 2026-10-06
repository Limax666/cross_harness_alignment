#!/usr/bin/env python3
"""Run the bounded, GPU-free gates before attempting native RL rollouts."""

import argparse
import runpy
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-verl", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    count = 0
    for name in ("test_rl_verifier.py", "test_rl_native_evidence.py", "test_rl_tool_attempt_audit.py",
                 "test_rl_agentdojo_path_audit.py", "test_rl_agentdojo_injection_fixture.py",
                 "test_rl_agentdojo_start_api.py", "test_rl_group_sampler.py", "test_local_qwen_tool_calls.py"):
        namespace = runpy.run_path(str(project / "tests" / name))
        for key, function in namespace.items():
            if key.startswith("test_") and callable(function):
                function()
                count += 1
    print(f"contract and parser assertions passed: {count}", flush=True)
    subprocess.run([sys.executable, str(project / "scripts" / "smoke_chs_rl_cpu.py")], check=True)
    if not args.without_verl:
        subprocess.run([sys.executable, str(project / "scripts" / "smoke_verl_advantages_cpu.py")], check=True)
    print("all CPU RL gates passed; native harness/GPU update remains a separate gate", flush=True)


if __name__ == "__main__":
    main()
