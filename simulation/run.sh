#!/usr/bin/env bash
# run.sh — one-command launcher for Couchmo simulation on macOS
# Handles XQuartz setup, X11 auth, and Docker display forwarding

set -e

MODE="${1:-sim}"   # sim | preview | shell

# ── 1. XQuartz check ─────────────────────────────────────────
if ! ls /Applications/Utilities/XQuartz.app &>/dev/null; then
  echo ""
  echo "  XQuartz is not installed. Installing via Homebrew..."
  brew install --cask xquartz
  echo ""
  echo "  XQuartz installed. You MUST LOG OUT and back in once"
  echo "  before the display server works. Do that now, then re-run."
  exit 0
fi

# ── 2. Start XQuartz if not running ──────────────────────────
if ! pgrep -x Xquartz &>/dev/null; then
  echo "Starting XQuartz..."
  open -a XQuartz
  sleep 3
fi

# ── 3. Allow Docker to connect to X11 display ────────────────
# XQuartz must have "Allow connections from network clients" enabled.
# This sets it via defaults and restarts if needed.
XQUARTZ_PREF=$(defaults read org.xquartz.X11 nolisten_tcp 2>/dev/null || echo "1")
if [ "$XQUARTZ_PREF" = "1" ]; then
  echo "Enabling XQuartz network connections (requires XQuartz restart)..."
  defaults write org.xquartz.X11 nolisten_tcp 0
  pkill Xquartz 2>/dev/null || true
  sleep 2
  open -a XQuartz
  sleep 3
fi

# ── 4. Set DISPLAY and allow localhost ───────────────────────
export DISPLAY=host.docker.internal:0
xhost +localhost 2>/dev/null || true
xhost + 2>/dev/null || true   # fallback: allow all (local dev only)

echo ""
echo "  DISPLAY=$DISPLAY"
echo "  XQuartz ready."
echo ""

# ── 5. Launch the requested service ──────────────────────────
case "$MODE" in
  sim)
    echo "Launching: Gazebo + ROS 2 simulation..."
    echo "(First run downloads ~2.5 GB image + packages. Grab a coffee.)"
    echo ""
    DISPLAY=$DISPLAY docker compose up sim
    ;;
  preview)
    echo "Launching: URDF preview in RViz (no Gazebo)..."
    DISPLAY=$DISPLAY docker compose up preview
    ;;
  shell)
    echo "Opening interactive ROS 2 shell..."
    docker run --rm -it \
      -e DISPLAY=$DISPLAY \
      -e ROS_DOMAIN_ID=42 \
      -v "$(pwd)/src":/ros2_ws/src \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      --network host \
      osrf/ros:humble-desktop-full \
      bash -c "source /opt/ros/humble/setup.bash && bash"
    ;;
  *)
    echo "Usage: $0 [sim|preview|shell]"
    echo "  sim     — full Gazebo simulation (default)"
    echo "  preview — RViz URDF viewer only (fast)"
    echo "  shell   — interactive ROS 2 shell"
    ;;
esac
