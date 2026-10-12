#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
SIM2REAL TOY TEST 2 -- reach a REAL apple the camera sees, with the waist reach
policy (AppleReachSpindle-v3: right arm + waist yaw), through g1_fruits_picker's
perception and robot interface.

This is toy_reach.py (same robot path, safety watch, E-STOP, home pose, logs) with
the camera as the target source:
  1. arm HOME, waist to 0 (the arm out of the camera's view);
  2. the picker's perception -- live_locator (RF-DETR on the head camera) ->
     fruits_to_torso (camera -> torso extrinsics) -> its DetectionManager (tracks
     kept across waist turns) -- is sampled for --detect_s while the arm holds;
  3. every fruit is listed: position, radius, how steadily it was seen, and whether
     its reach target is inside the policy's training workspace (waist envelope);
     the nearest reachable, steady one is chosen (or --pick N);
  4. ENTER -> the policy reaches it, the torso turning as trained (--waist follow,
     the waist streamed with the arm), until the palm is at the touch distance of the
     detected apple surface -> hold -> back home.
The apple's position is fixed when it is chosen (pelvis frame): the hand hides it
from the camera on the way in. The palm's contact force fed to the policy is
computed from that detected position (--touch virtual), not measured.

Terminals (each: distrobox enter humble-box; conda activate g1_dex3_sim2real):
  1  cd ~/g1_fruits_picker/apple_picker && ./ssh_robot_launch.sh          (robot: cameras + daemon)
  2  cd ~/g1_fruits_picker/apple_picker && source robot_ros_env.sh && python3 live_locator.py --cameras unitree
  3  cd ~/g1_fruits_picker/apple_picker && source robot_ros_env.sh && python3 fruits_to_torso.py
  4  cd ~/g1_fruits_picker/apple_picker && python3 dashboard/server.py --net-if enp62s0   (STOP / reset)
  5  cd ~/dex3_rl_manipulation && python3 sim2real/toy_apple.py --speed 0.3
     python3 sim2real/toy_apple.py --speed 0.3 --shadow     # list the fruit + the plan, nothing moves
     python3 sim2real/toy_apple.py --speed 0.3 --pick 1     # that fruit of the list

Offline: python3 sim2real/toy_apple.py --mock --speed 0.5 --mock_apples "0.33,-0.21,0.28;0.30,0.05,0.25"

Every toy_reach.py option applies (python3 sim2real/toy_reach.py -h).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy_reach  # noqa: E402

DEFAULTS = {"--target": ["detection"], "--waist": ["follow"]}

if __name__ == "__main__":
    if any(x in ("-h", "--help") for x in sys.argv[1:]):
        print(__doc__)
    argv = sys.argv[1:]
    for flag, val in DEFAULTS.items():
        if flag not in argv:
            argv = argv + [flag] + val
    sys.argv = [sys.argv[0]] + argv
    toy_reach.main()
