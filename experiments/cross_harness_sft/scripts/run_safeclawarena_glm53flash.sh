#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$EXPERIMENT/.agentharm_token_plan.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing local provider env file: $ENV_FILE" >&2
  exit 2
fi
set -a
source "$ENV_FILE"
set +a
if [[ -z "${SHENGSUANYUN_API_KEY:-}" ]]; then
  echo "SHENGSUANYUN_API_KEY is unset" >&2
  exit 2
fi
umask 077
exec python3 "$SCRIPT_DIR/collect_safeclawarena_glm53flash.py" "$@"
