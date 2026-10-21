#!/usr/bin/env python3
"""vision_onboard.py -- the apple detector ON THE JETSON (its GPU, TensorRT), for the RL runs only.

What the laptop's live_locator does for the head camera, done next to the camera:
  /unitree/color/image_raw + /unitree/aligned_depth_to_color/image_raw + /unitree/color/camera_info (local, raw)
  -> RF-DETR (trt_rfdetr.Detector: the laptop's checkpoint as a TensorRT engine, same pre/post-processing)
  -> per-class thresholds and the apple's own depth (apple_depth: the laptop's code, verbatim)
  -> /fruits_detected_raw: the SAME JSON live_locator publishes (camera frame, the image's capture stamp)
  -> /apple_picker/annotated/compressed: the picture with the boxes, for the dashboard (8 Hz at most)
Everything downstream on the laptop (fruits_to_torso, the picker's tracker, the toy, the dashboard) is unchanged.

ONE detector at a time: the launcher starts this one INSTEAD of the laptop's live_locator (g1_launch --vision onboard);
the default (and the original test) keeps the laptop's. This process only reads the cameras -- it never starts,
stops or reconfigures them.

    source /opt/ros/foxy/setup.bash && source ~/cyclonedds_ws/install/setup.bash
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=~/ros2_cameras/cyclonedds_cameras.xml \
        python3 vision_onboard.py [--engine models/rfdetr560_mixed.engine]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

import apple_depth as AD
from trt_rfdetr import Detector

HERE = os.path.dirname(os.path.abspath(__file__))
#: live_locator's per-class thresholds (after the detector's own 0.3) and class filter
CLASS_THR = {"red_apple": 0.40, "green_apple": 0.40, "orange": 0.50}
CAMERA = "unitree"
NS = "/unitree"
#: a depth frame pairs with a colour frame when their stamps are this close (s)
PAIR_S = 0.05
COLORS = {"red_apple": (0, 0, 255), "green_apple": (0, 200, 0), "orange": (0, 140, 255)}


def _stamp(msg) -> float:
    return float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)


def _image_to_np(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 1)[:, :msg.width * 3]
        a = a.reshape(msg.height, msg.width, 3)
        return a if enc == "rgb8" else a[:, :, ::-1]          # -> RGB
    if enc in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
    raise ValueError(f"unsupported encoding {msg.encoding}")


class VisionOnboard(Node):
    def __init__(self, a):
        super().__init__("vision_onboard")
        self.a = a
        self.det = Detector(a.engine, threshold=0.3)
        self.lock = threading.Lock()
        self.color, self.depths, self.K = None, [], None
        self.new = threading.Event()
        ns = a.ns
        self.create_subscription(Image, f"{ns}/color/image_raw", self._on_color, qos_profile_sensor_data)
        self.create_subscription(Image, f"{ns}/aligned_depth_to_color/image_raw", self._on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, f"{ns}/color/camera_info", self._on_info, qos_profile_sensor_data)
        self.pub = self.create_publisher(String, a.topic, 10)
        self.pub_ann = self.create_publisher(CompressedImage, a.annotated_topic, qos_profile_sensor_data)
        self._ann_t = 0.0
        self.n, self.t_log, self.ms = 0, time.monotonic(), []
        self.get_logger().info(f"up: engine {os.path.basename(a.engine)} ({self.det.W}x{self.det.H}), "
                               f"publishing {a.topic}; waiting for {ns}/color/image_raw")
        threading.Thread(target=self._loop, daemon=True).start()

    def _on_color(self, msg):
        with self.lock:
            self.color = msg
        self.new.set()

    def _on_depth(self, msg):
        with self.lock:
            self.depths.append(msg)
            self.depths = self.depths[-6:]

    def _on_info(self, msg):
        if self.K is None:
            self.get_logger().info(f"camera_info: {msg.width}x{msg.height} fx {msg.k[0]:.1f} fy {msg.k[4]:.1f} "
                                   f"cx {msg.k[2]:.1f} cy {msg.k[5]:.1f}")
        self.K = (float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))

    def _loop(self):
        while rclpy.ok():
            if not self.new.wait(0.5):
                continue
            self.new.clear()
            with self.lock:
                cm, deps, K = self.color, list(self.depths), self.K
            if cm is None or K is None:
                continue
            t0 = time.perf_counter()
            try:
                self._process(cm, deps, K)
            except Exception as e:
                self.get_logger().warn(f"frame failed: {type(e).__name__}: {e}", throttle_duration_sec=5.0)
            self.ms.append(1000 * (time.perf_counter() - t0))
            self.n += 1
            if time.monotonic() - self.t_log > 10.0:
                self.get_logger().info(f"{self.n / (time.monotonic() - self.t_log):.1f} frames/s, "
                                       f"{np.median(self.ms):.0f} ms per frame (median)")
                self.n, self.t_log, self.ms = 0, time.monotonic(), []

    def _process(self, cm, deps, K):
        rgb = _image_to_np(cm)
        ts = _stamp(cm)
        dm = min(deps, key=lambda d: abs(_stamp(d) - ts)) if deps else None
        depth = _image_to_np(dm) if dm is not None and abs(_stamp(dm) - ts) <= PAIR_S else None
        fx, fy, cx0, cy0 = K
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        objects, drawn = [], []
        for x1, y1, x2, y2, conf, cid, label in self.det.detect(np.ascontiguousarray(rgb)):
            if conf < CLASS_THR.get(label, 1.1):
                continue
            center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            depth_m, dq = None, None
            if depth is not None:
                depth_m, dq = AD._apple_depth(depth, (x1, y1, x2, y2), float(0.5 * (fx + fy)), rgb_bgr=bgr,
                                              red=("red" in label), depth_scale_for_int=0.001)
                if depth_m is None:
                    d_raw = AD._robust_depth_at(depth, int(center_x), int(center_y), window=5)
                    if d_raw is not None:
                        depth_m = float(d_raw) * 0.001
            drawn.append((x1, y1, x2, y2, label, conf))
            if depth_m is None:
                continue                                     # live_locator drops these too (no cartesian)
            th = math.atan2(center_x - cx0, fx)
            ph = math.atan2(cy0 - center_y, fy)
            o = {"class_label": label, "confidence": float(conf),
                 "cartesian": {"x": depth_m * math.tan(th), "y": depth_m * math.tan(ph), "z": float(depth_m)},
                 "bbox": [int(round(v)) for v in (x1, y1, x2, y2)]}
            if isinstance(dq, dict):
                o["depth_quality"] = {k: v for k, v in dq.items() if k != "clusters"}
            objects.append(o)
        if objects:
            m = String()
            m.data = json.dumps({"header": {"stamp": {"sec": int(cm.header.stamp.sec),
                                                      "nanosec": int(cm.header.stamp.nanosec)},
                                            "frame_id": cm.header.frame_id},
                                 "camera": CAMERA, "objects": objects, "source": "vision_onboard"})
            self.pub.publish(m)
        now = time.monotonic()
        if now - self._ann_t >= 1.0 / 8.0:
            self._ann_t = now
            vis = bgr.copy()
            for x1, y1, x2, y2, label, conf in drawn:
                c = COLORS.get(label, (255, 255, 255))
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                cv2.putText(vis, label, (int(x1), max(10, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                am = CompressedImage()
                am.header.stamp = cm.header.stamp
                am.format = "jpeg"
                am.data = buf.tobytes()
                self.pub_ann.publish(am)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=os.path.join(HERE, "models", "rfdetr560_mixed.engine"))
    ap.add_argument("--topic", default="/fruits_detected_raw")
    ap.add_argument("--ns", default=NS, help="the camera's topic namespace")
    ap.add_argument("--annotated_topic", default="/apple_picker/annotated/compressed")
    a = ap.parse_args()
    rclpy.init()
    node = VisionOnboard(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
