#!/bin/bash
set -e

# Start virtual framebuffer (1600x900 display, no physical screen needed)
Xvfb :99 -screen 0 1600x900x24 &
export DISPLAY=:99
sleep 1

# Start VNC server on the virtual display (no password for local dev)
x11vnc -display :99 -nopw -forever -shared -bg -quiet

# Start noVNC web interface (browser connects to localhost:6080)
websockify --web /usr/share/novnc 6080 localhost:5900 &
sleep 1

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Couchmo RViz Preview                            ║"
echo "║  Open in your browser:                           ║"
echo "║    http://localhost:6080/vnc.html                ║"
echo "║  Click 'Connect' — no password needed            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Launch ROS nodes
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

ros2 launch couchmo_description display.launch.py
