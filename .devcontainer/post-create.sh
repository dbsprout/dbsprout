#!/bin/bash
set -euo pipefail

CLAUDE_DIR="/home/node/.claude"
HOST_DIR="/tmp/.claude-host"

# Seed Claude state volume from host (only on first create / empty volume)
if [ ! -f "${CLAUDE_DIR}/settings.json" ]; then
    cp "${HOST_DIR}/settings.json" "${CLAUDE_DIR}/settings.json"
    cp "${HOST_DIR}/credentials.json" "${CLAUDE_DIR}/.credentials.json"

    # Strip hooks that reference host-only scripts.
    # Plugin hooks (Letta/claude-subconscious) load from their own hooks.json.
    python3 -c "
import json, pathlib
p = pathlib.Path('${CLAUDE_DIR}/settings.json')
d = json.loads(p.read_text())
d.pop('hooks', None)
p.write_text(json.dumps(d, indent=2))
"
fi

# Sync plugins from host (also runs on every start via postStartCommand)
bash /workspace/.devcontainer/sync-plugins.sh

# Export GH_TOKEN from mounted host token file (idempotent)
grep -q '/.gh-token' "${HOME}/.zshrc" 2>/dev/null || \
    echo '[ -f /tmp/.gh-token ] && export GH_TOKEN=$(cat /tmp/.gh-token)' >> "${HOME}/.zshrc"

# Source workspace env vars into shell profile (idempotent)
grep -q '/workspace/.envrc' "${HOME}/.zshrc" 2>/dev/null || \
    echo '[ -f /workspace/.envrc ] && source /workspace/.envrc' >> "${HOME}/.zshrc"

# Install project dependencies and git hooks
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

# Start Claude connection watchdog as a background loop (no cron needed)
if [ -x /usr/local/bin/claude-watchdog.sh ]; then
    # Kill any previous watchdog loop
    pkill -f "claude-watchdog-loop" 2>/dev/null || true
    # Run every 5 minutes in the background
    (
        exec -a claude-watchdog-loop bash -c '
            while true; do
                /usr/local/bin/claude-watchdog.sh >> /tmp/claude-watchdog.log 2>&1
                sleep 300
            done
        '
    ) &
    disown
fi
