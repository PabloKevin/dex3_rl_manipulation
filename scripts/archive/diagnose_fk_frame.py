#!/usr/bin/env python3
# Copyright 2026 - G1 Dex3 Apple Grasp Project
"""
Direct FK-agreement check between cuRobo's kinematic model and Isaac Lab's
ground truth — no planning, no IK solve, just forward kinematics compared
at the robot's ACTUAL live joint angles.

Why this exists: reach_apple_ik.py has been converging tightly (~30 mm) to
the goal it's given, yet the arm visually ends up far from the real apple.
That means the SOLVE is fine — the goal handed to cuRobo (apple position
transformed into "torso frame") doesn't correspond to the same frame cuRobo
actually uses internally, or cuRobo's right_hand_palm_link FK doesn't match
the real robot for some other reason (wrong link, wrong joint order/sign,
stale collision-pruning of the kinematic chain, etc). Guessing at this via
calibration bias curve-fitting has already burned a lot of time; this script
answers it directly:

  1. Reads the robot's LIVE state from run_env.py (current arm joint angles,
     actual right_hand_palm_link pose in WORLD frame, torso pose in WORLD
     frame) — all already published on rt/arm_tune/state.
  2. Feeds those EXACT joint angles into cuRobo's CudaRobotModel.get_state(),
     using the identical base_link=torso_link override reach_apple_ik.py /
     reach_apple_curobo.py use, to get cuRobo's own FK for right_hand_palm_link
     relative to torso_link.
  3. Independently converts Isaac Lab's actual world-frame ee pose into the
     torso-local frame using the same world_to_torso() helper every other
     script here uses.
  4. Prints both and the discrepancy. If cuRobo's FK and Isaac Lab's actual
     torso-relative EE pose agree (small mm / few deg), the frame pipeline is
     correct and the bug is elsewhere (apple position, offset, ee_link choice).
     If they disagree by something roughly CONSTANT across several joint
     configs, that constant IS the correction to apply. If it varies with
     joint config, something deeper is wrong (wrong link/joint mapping) and
     bias-fitting can never fix it.

Usage:
    Terminal 1: ~/IsaacLab/isaaclab.sh -p scripts/run_env.py
    Terminal 2: python3 scripts/archive/diagnose_fk_frame.py
    (optionally move the arm around with reach_apple_ik.py in a third
    terminal between samples — this script re-samples every 3s so you can
    watch whether the discrepancy stays constant across different poses)
"""
import json
import math
import os
import time
import threading

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q
    return torch.stack([w, -x, -y, -z])


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[0]
    q_vec = q[1:]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + w * t + torch.cross(q_vec, t, dim=-1)


def world_to_torso(pos_world: torch.Tensor, torso_pos: torch.Tensor, torso_quat: torch.Tensor) -> torch.Tensor:
    delta = pos_world - torso_pos
    return quat_rotate(quat_conjugate(torso_quat), delta)


def quat_log_deg(q: torch.Tensor) -> float:
    """Angle (degrees) of the rotation represented by unit quaternion q (w,x,y,z)."""
    if q[0] < 0:
        q = -q
    w = torch.clamp(q[0], -1.0, 1.0)
    return math.degrees(2.0 * torch.acos(w).item())


# ── DDS: same pattern as every other script here ─────────────────────────────
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    ChannelFactoryInitialize(1)
    DDS_OK = True
except Exception as e:
    print(f"[DDS] OFFLINE: {e}")
    DDS_OK = False

_state_lock = threading.Lock()
_latest_state = None


def _state_callback(msg):
    global _latest_state
    try:
        with _state_lock:
            _latest_state = json.loads(msg.data)
    except Exception as e:
        print(f"[DDS] state parse error: {e}")


if DDS_OK:
    try:
        state_sub = ChannelSubscriber("rt/arm_tune/state", String_)
        state_sub.Init(_state_callback, 1)
        print("[DDS] Subscribed to rt/arm_tune/state")
    except Exception as e:
        print(f"[DDS] Could not subscribe: {e}")
        DDS_OK = False


def read_state():
    with _state_lock:
        return dict(_latest_state) if _latest_state is not None else None


def wait_for_state(timeout_s: float = 15.0) -> dict:
    print("Waiting for state from run_env.py...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = read_state()
        if s is not None:
            return s
        time.sleep(0.1)
    raise RuntimeError("No state received from run_env.py within timeout.")


# ── cuRobo ─────────────────────────────────────────────────────────────────
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
EE_LINK_NAME = "right_hand_palm_link"
BASE_LINK_NAME = "torso_link"
ROBOT_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "envs", "assets", "cuRobo", "g1_dex3_right_arm.yml",
)


def build_kinematic_model(tensor_args):
    robot_cfg_dict = load_yaml(ROBOT_CFG_PATH)
    kinematics = robot_cfg_dict["robot_cfg"]["kinematics"]
    kinematics["base_link"] = BASE_LINK_NAME
    kinematics["lock_joints"] = {}
    robot_cfg = RobotConfig.from_dict(robot_cfg_dict["robot_cfg"], tensor_args)
    return CudaRobotModel(robot_cfg.kinematics)


def main():
    if not DDS_OK:
        print("Cannot run: DDS unavailable.")
        return

    tensor_args = TensorDeviceType()
    print("Building cuRobo kinematic model (base_link=torso_link, same override "
          "reach_apple_ik.py / reach_apple_curobo.py use)...")
    kin_model = build_kinematic_model(tensor_args)
    print(f"✅ Ready. Comparing cuRobo FK({BASE_LINK_NAME} -> {EE_LINK_NAME}) against "
          f"Isaac Lab's actual reported ee_pos/ee_quat, at the SAME live joint angles.\n"
          f"Move the arm between samples (e.g. run reach_apple_ik.py in another terminal) "
          f"to see whether the discrepancy is constant or pose-dependent.\n")

    wait_for_state()

    last_q = None
    while True:
        state = read_state()
        q = state["arm_joint_pos"]
        if q != last_q:
            last_q = q
            q_t = torch.tensor([q], device=tensor_args.device, dtype=torch.float32)

            fk = kin_model.get_state(q_t)
            curobo_pos = fk.ee_position[0]
            curobo_quat = fk.ee_quaternion[0]

            torso_pos = torch.tensor(state["torso_pos"], device=tensor_args.device, dtype=torch.float32)
            torso_quat = torch.tensor(state["torso_quat"], device=tensor_args.device, dtype=torch.float32)
            ee_pos_world = torch.tensor(state["ee_pos"], device=tensor_args.device, dtype=torch.float32)
            ee_quat_world = torch.tensor(state["ee_quat"], device=tensor_args.device, dtype=torch.float32)

            isaac_pos_local = world_to_torso(ee_pos_world, torso_pos, torso_quat)
            isaac_quat_local = quat_mul(quat_conjugate(torso_quat), ee_quat_world)

            pos_diff = curobo_pos - isaac_pos_local
            pos_diff_mm = torch.norm(pos_diff).item() * 1000.0
            err_quat = quat_mul(quat_conjugate(isaac_quat_local), curobo_quat)
            orient_diff_deg = quat_log_deg(err_quat)

            print(f"q = {[round(v, 3) for v in q]}")
            print(f"  cuRobo FK   (torso-local): pos={[round(v, 4) for v in curobo_pos.tolist()]}  "
                  f"quat={[round(v, 4) for v in curobo_quat.tolist()]}")
            print(f"  Isaac Lab   (torso-local): pos={[round(v, 4) for v in isaac_pos_local.tolist()]}  "
                  f"quat={[round(v, 4) for v in isaac_quat_local.tolist()]}")
            print(f"  DISCREPANCY: pos={pos_diff_mm:.1f} mm  "
                  f"(vec={[round(v, 4) for v in pos_diff.tolist()]})  orient={orient_diff_deg:.2f} deg\n")

        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
