#!/usr/bin/env bash
# Fetch the exact benchmark and VeRL sources used by the online RL pilot.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_REF=02a318c303a7b60871cb63e4ce2f779b522874d9
AGENTDOJO_REF=089ed468cf3ed0322acc66b0211f26d9d90dbf60
PATCH="$ROOT/patches/verl_project_local_02a318c3.patch"
CUDA_PATCH="$ROOT/patches/verl_nonzero_cuda_visible_devices.patch"
mkdir -p "$ROOT/vendor"

if [[ ! -d "$ROOT/vendor/verl/.git" ]]; then
  git clone https://github.com/verl-project/verl.git "$ROOT/vendor/verl"
fi
if [[ ! -d "$ROOT/vendor/agentdojo/.git" ]]; then
  git clone https://github.com/ethz-spylab/agentdojo.git "$ROOT/vendor/agentdojo"
fi

git -C "$ROOT/vendor/verl" fetch origin "$VERL_REF"
git -C "$ROOT/vendor/verl" checkout --detach "$VERL_REF"
if git -C "$ROOT/vendor/verl" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "VeRL project patch already applied"
else
  git -C "$ROOT/vendor/verl" apply --check "$PATCH"
  git -C "$ROOT/vendor/verl" apply "$PATCH"
fi
if git -C "$ROOT/vendor/verl" apply --reverse --check "$CUDA_PATCH" 2>/dev/null; then
  echo "VeRL nonzero visible-GPU patch already applied"
else
  git -C "$ROOT/vendor/verl" apply --check "$CUDA_PATCH"
  git -C "$ROOT/vendor/verl" apply "$CUDA_PATCH"
fi

git -C "$ROOT/vendor/agentdojo" fetch origin "$AGENTDOJO_REF"
git -C "$ROOT/vendor/agentdojo" checkout --detach "$AGENTDOJO_REF"
printf 'VeRL %s with project patch; AgentDojo %s\n' "$VERL_REF" "$AGENTDOJO_REF"
