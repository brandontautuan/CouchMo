#!/usr/bin/env bash
# run.sh — one-command launcher for the CouchMo sim (macOS / Linux).
# The sim UI is served via noVNC at http://localhost:6081/vnc.html,
# so no X11 forwarding / XQuartz is needed.

set -euo pipefail

MODE="${1:-sim}"   # sim | preview | shell | headless

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running preflight check..."
python3 scripts/bringup_check.py

NOVNC_URL="http://localhost:6081/vnc.html"

case "$MODE" in
  sim)
    echo "Starting Gazebo sim. Open: $NOVNC_URL"
    docker compose up sim
    ;;
  preview)
    echo "Starting URDF preview. Open: $NOVNC_URL"
    docker compose up preview
    ;;
  headless)
    echo "Starting headless sim (no noVNC, no GUI)."
    docker compose up sim_headless
    ;;
  shell)
    echo "Opening interactive shell inside the sim container..."
    docker compose run --rm sim bash
    ;;
  *)
    echo "Usage: $0 [sim|preview|shell|headless]"
    echo "  sim      — full Gazebo sim, view in browser at $NOVNC_URL"
    echo "  preview  — URDF/RViz preview, view in browser at $NOVNC_URL"
    echo "  headless — sim without noVNC (batch dataset generation)"
    echo "  shell    — interactive bash inside the sim container"
    exit 2
    ;;
esac
