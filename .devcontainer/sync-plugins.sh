#!/bin/bash
set -euo pipefail

CLAUDE_DIR="/home/node/.claude"
HOST_DIR="/tmp/.claude-host"

# Copy plugins from host into writable volume (refreshes on every start)
if [ -d "${HOST_DIR}/plugins" ]; then
    rsync -a --delete "${HOST_DIR}/plugins/" "${CLAUDE_DIR}/plugins/"
else
    echo "No host plugins found at ${HOST_DIR}/plugins — skipping sync"
    exit 0
fi

# Rewrite host home paths to container paths in plugin registry
if [ -f "${CLAUDE_DIR}/plugins/installed_plugins.json" ]; then
    sed -i "s|/home/fflores/|/home/node/|g" "${CLAUDE_DIR}/plugins/installed_plugins.json"

    # Fix project paths: /home/node/PycharmProjects/dbsprout → /workspace
    sed -i "s|/home/node/PycharmProjects/dbsprout|/workspace|g" "${CLAUDE_DIR}/plugins/installed_plugins.json"
fi

# Remove orphan markers left over from previous path mismatches
find "${CLAUDE_DIR}/plugins" -name ".orphaned_at" -delete 2>/dev/null || true

# Patch: /dev/tty crash in claude-subconscious (ENXIO in containers)
find "${CLAUDE_DIR}/plugins" -name "session_start.ts" -path "*/claude-subconscious/*" \
    -exec sed -i "s|tty = fs.createWriteStream('/dev/tty');|tty = fs.createWriteStream('/dev/tty'); tty.on('error', () => { tty = null; });|g" {} + 2>/dev/null || true

# Install plugin dependencies (tsx for claude-subconscious)
for pkg in "${CLAUDE_DIR}"/plugins/cache/*/*/*/package.json; do
    dir=$(dirname "$pkg")
    if [ -f "$pkg" ] && grep -q '"tsx"' "$pkg"; then
        npm install --prefix "$dir" --omit=dev 2>/dev/null || true
    fi
done

echo "Plugin sync complete"
