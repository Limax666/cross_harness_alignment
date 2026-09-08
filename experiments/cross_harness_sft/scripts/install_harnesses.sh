#!/usr/bin/env bash
set -euo pipefail
: "${CODEX_VERSION:?Set exact CODEX_VERSION}"
: "${CLAUDE_CODE_VERSION:?Set exact CLAUDE_CODE_VERSION}"
: "${OPENCLAW_VERSION:?Set exact OPENCLAW_VERSION}"
: "${HERMES_REF:?Set exact HERMES_REF}"
npm install -g "@openai/codex@$CODEX_VERSION" "@anthropic-ai/claude-code@$CLAUDE_CODE_VERSION" "openclaw@$OPENCLAW_VERSION"
HERMES_DIR="${HERMES_DIR:-/opt/hermes-agent}"
[[ -d "$HERMES_DIR/.git" ]] || git clone https://github.com/NousResearch/hermes-agent.git "$HERMES_DIR"
git -C "$HERMES_DIR" fetch --tags origin
git -C "$HERMES_DIR" checkout --detach "$HERMES_REF"
python3 -m pip install -e "$HERMES_DIR"
codex --version
claude --version
openclaw --version
hermes --version
