#!/usr/bin/env bash
# Copy the onboard detector's code to the Jetson (~/vision_onboard). The engine is built THERE, once:
#   python3 build_engine.py models/rfdetr560_op16.onnx models/rfdetr560_mixed.engine
# Nothing is installed: system Python 3.8 + JetPack's TensorRT 8.5 + ROS 2 Foxy, all already on the robot.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
sshpass -p "${ROBOT_PASS:-123}" ssh $O unitree@192.168.123.164 'mkdir -p ~/vision_onboard/models'
sshpass -p "${ROBOT_PASS:-123}" scp $O "$HERE"/{vision_onboard.py,trt_rfdetr.py,apple_depth.py,build_engine.py} \
    unitree@192.168.123.164:vision_onboard/
echo "deployed to the Jetson: ~/vision_onboard"
