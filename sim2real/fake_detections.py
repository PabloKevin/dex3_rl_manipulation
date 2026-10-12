#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
TEST ONLY: publish /fruits_detected_torso as g1_fruits_picker's fruits_to_torso does
(JSON objects with class_label, cartesian x/y/z in the torso frame, radius_m), for
toy_apple.py over loopback without the camera. Positions are fixed in the torso frame
(true while the waist stays at 0, as it does while the toy samples), 3 mm noise.

  python3 sim2real/fake_detections.py --apples "0.33,-0.21,0.28;0.30,0.05,0.25"
"""
import argparse
import json
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--apples", required=True, help="'x,y,z;x,y,z' torso frame [m]")
ap.add_argument("--label", default="red_apple")
ap.add_argument("--hz", type=float, default=5.0)
ap.add_argument("--duration", type=float, default=600.0)
a = ap.parse_args()

import rclpy  # noqa: E402
from std_msgs.msg import String  # noqa: E402

rclpy.init()
node = rclpy.create_node("fake_detections")
pub = node.create_publisher(String, "/fruits_detected_torso", 10)
apples = [np.array([float(v) for v in t.split(",")]) for t in a.apples.split(";")]
t_end = time.time() + a.duration
print(f"[fake_det] {len(apples)} '{a.label}' on /fruits_detected_torso at {a.hz} Hz", flush=True)
while time.time() < t_end:
    objs = []
    for p in apples:
        q = p + np.random.normal(0.0, 0.003, 3)
        objs.append({"class_label": a.label, "confidence": 0.9, "camera": "unitree", "radius_m": 0.04,
                     "cartesian": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2])}})
    m = String()
    m.data = json.dumps({"objects": objs})
    pub.publish(m)
    time.sleep(1.0 / a.hz)
