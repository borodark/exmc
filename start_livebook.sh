#!/usr/bin/env bash
# Start Livebook locally with exmc notebooks.
#
# Usage:
#   ./start_livebook.sh
#   ./start_livebook.sh --port 8888
#
# Opens http://localhost:8080 (or custom port) with notebooks/ available.
# Requires: mix (Elixir), livebook escript installed via:
#   mix escript.install hex livebook

set -euo pipefail

PORT="${1:-8080}"
[[ "${1:-}" == "--port" ]] && PORT="${2:-8080}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Install livebook escript if missing
if ! command -v livebook &>/dev/null && ! [ -f "$HOME/.mix/escripts/livebook" ]; then
  echo "Installing Livebook escript..."
  mix escript.install hex livebook --force
fi

LIVEBOOK_BIN="livebook"
if ! command -v livebook &>/dev/null; then
  LIVEBOOK_BIN="$HOME/.mix/escripts/livebook"
fi

export LIVEBOOK_TOKEN_ENABLED=false

exec "$LIVEBOOK_BIN" server \
  --port "$PORT" \
  --home "$SCRIPT_DIR" \
  "$SCRIPT_DIR/notebooks"
