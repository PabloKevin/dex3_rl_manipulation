#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
TEST ONLY: fake_detections.py for a TURNING waist -- apples fixed in the world (given in the torso frame at
waist 0), published on /fruits_detected_torso in the CURRENT torso frame (the waist yaw from rt/lowstate,
motor 12) and only while inside the head camera's horizontal field of view (--hfov_deg) and depth range.
For toy_harvest.py's waist scan over loopback. 3 mm noise, the apple's radius 4 cm.
  python3 sim2real/fake_detections_world.py --net-if lo --apples "0.38,-0.12,0.05;0.40,-0.45,0.02"
"""
import argparse
import json
import math
import os
import threading
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--apples", required=True, help="'x,y,z;x,y,z' torso frame at waist 0 [m]")
ap.add_argument("--net-if", default="lo")
ap.add_argument("--hfov_deg", type=float, default=70.0, help="the camera's horizontal view (the D435i at 424 px: 70)")
ap.add_argument("--image_w", type=float, default=424.0, help="picture width [px] of the boxes (fx from --hfov_deg)")
ap.add_argument("--hz", type=float, default=10.0)
ap.add_argument("--duration", type=float, default=900.0)
a = ap.parse_args()

import subprocess  # noqa: E402
import sys  # noqa: E402

import rclpy  # noqa: E402
from std_msgs.msg import String  # noqa: E402

# the waist yaw from rt/lowstate in a CHILD process (the SDK's DDS and ROS clash in one process here)
# on 'lo' (no multicast) every DDS participant takes an index, by default at most 9: with the fake motors, the
# daemon, its publisher, the runner, the toy and the dashboard there are not enough -- this one may go higher
YAW_READER = ("import sys,time\nimport unitree_sdk2py.core.channel as ch\n"
              "ch.ChannelConfigHasInterface = ch.ChannelConfigHasInterface.replace('</General>', "
              "'</General><Discovery><MaxAutoParticipantIndex>60</MaxAutoParticipantIndex></Discovery>', 1)\n"
              "from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber\n"
              "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_\n"
              "ChannelFactoryInitialize(0, sys.argv[1])\nW = [0.0]\n"
              "ChannelSubscriber('rt/lowstate', LowState_).Init(lambda m: W.__setitem__(0, float(m.motor_state[12].q)), 1)\n"
              "while True:\n    print(W[0], flush=True)\n    time.sleep(0.05)\n")
W = {"yaw": 0.0}
# without ROS's libraries on the path: its libddsc is not the one the SDK's cyclonedds was built against
env = dict(os.environ, LD_LIBRARY_PATH=":".join(x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":")
                                                if x and "/opt/ros" not in x))
child = subprocess.Popen([sys.executable, "-c", YAW_READER, a.net_if], stdout=subprocess.PIPE, text=True, env=env)


def _read_yaw():
    for line in child.stdout:
        try:
            W["yaw"] = float(line)
        except ValueError:
            pass


threading.Thread(target=_read_yaw, daemon=True).start()
rclpy.init()
node = rclpy.create_node("fake_detections_world")
pub = node.create_publisher(String, "/fruits_detected_torso", 10)
apples = [np.array([float(v) for v in t.split(",")]) for t in a.apples.split(";")]
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
print(f"[fake_det_world] {len(apples)} apples fixed in the world, view +-{a.hfov_deg / 2:.0f} deg", flush=True)
t_end = time.time() + a.duration
while time.time() < t_end:
    psi = W["yaw"]
    c, s = math.cos(-psi), math.sin(-psi)
    objs = []
    for p0 in apples:
        p = np.array([c * p0[0] - s * p0[1], s * p0[0] + c * p0[1], p0[2]])     # the torso turned by psi
        bearing = math.degrees(math.atan2(p[1], p[0]))
        ang_r = math.degrees(math.asin(min(1.0, 0.04 / max(np.linalg.norm(p[:2]), 0.05))))
        if abs(bearing) > a.hfov_deg / 2 + ang_r or p[0] < 0.15:
            continue
        # the picture box: the sphere projected (bearing only), cut at the picture's edges as the detector's
        fx = (a.image_w / 2) / math.tan(math.radians(a.hfov_deg / 2))
        uc = a.image_w / 2 - fx * math.tan(math.radians(bearing))
        rp = fx * 0.04 / max(p[0], 0.05)
        x1, x2 = max(0.0, uc - rp), min(a.image_w - 1, uc + rp)
        if x2 - x1 < 4:
            continue
        q = p + np.random.normal(0.0, 0.003, 3)
        objs.append({"class_label": "red_apple", "confidence": 0.9, "camera": "unitree", "radius_m": 0.04,
                     "stamp_s": time.time(), "cartesian": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
                     "bbox": [int(x1), 100, int(x2), 140]})
    m = String()
    m.data = json.dumps({"objects": objs})
    pub.publish(m)
    time.sleep(1.0 / a.hz)
