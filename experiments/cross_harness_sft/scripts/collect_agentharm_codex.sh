#!/usr/bin/env bash
# Compatibility entry point. The old direct `codex exec` collector did not
# mount AgentHarm tools or run its graders, so all collection now goes through
# the official worker + isolated MCP path.
set -euo pipefail
exec "$(dirname "$0")/run_agentharm_codex_native.sh" "$@"
