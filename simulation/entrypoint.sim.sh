#!/bin/bash
set -e

# Virtual display
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
sleep 1

# VNC + noVNC
x11vnc -display :99 -nopw -forever -shared -bg -quiet
websockify --web /usr/share/novnc 6080 localhost:5900 &
sleep 1

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Couchmo — Phase 3 Sim + Nav2 + SLAM            ║"
echo "║  Open: http://localhost:6081/vnc.html            ║"
echo "║  Teleop (new terminal):                          ║"
echo "║    docker exec -it couchmo_sim bash              ║"
echo "║    source /opt/ros/humble/setup.bash             ║"
echo "║    ros2 run teleop_twist_keyboard teleop_twist_keyboard ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Launch Gazebo + robot + RViz
ros2 launch couchmo_description gazebo.launch.py &
GAZEBO_PID=$!

# Wait for Gazebo to be up before starting Nav2 + SLAM
sleep 8

# SLAM Toolbox — builds the map from /scan in real time
ros2 launch slam_toolbox online_async_launch.py \
    use_sim_time:=true \
    slam_params_file:=/ros2_ws/install/couchmo_nav/share/couchmo_nav/config/nav2_params.yaml &

sleep 3

# Nav2 — obstacle avoidance + path planning
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=true \
    params_file:=/ros2_ws/install/couchmo_nav/share/couchmo_nav/config/nav2_params.yaml &

wait $GAZEBO_PID
