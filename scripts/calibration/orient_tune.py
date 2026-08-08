#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Fast orientation tuning loop. Solves IK for ONE candidate orientation
(no trajectory planning) and, if successful, publishes that static pose
directly to run_env.py so you see it in the viewer immediately.

Edit ROLL_DEG / PITCH_DEG / YAW_DEG below, rerun, look at the sim, repeat.
Much faster than reach_apple_curobo.py's full plan_single() for this purpose.

Usage (run_env.py must already be running):
    python3 scripts/calibration/orient_tune.py
"""
import json
import math
import os
import time
import threading
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.types import WorldConfig, Sphere, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import load_yaml
from curobo.util.logger import setup_curobo_logger

setup_curobo_logger("error")

# ── EDIT THESE, RERUN, LOOK AT THE SIM ────────────────────────────────────
ROLL_DEG, PITCH_DEG, YAW_DEG = 0.0, -75.0, 0.0
ROTATION_THRESHOLD = 0.08
# ────────────────────────────────────────────────────────────────────────

EE_LINK_NAME = "right_hand_palm_link"
BASE_LINK_NAME = "torso_link"
# Load the repo's OWN tracked copy directly — see reach_apple_curobo.py's
# ROBOT_CFG_PATH comment for why (get_robot_configs_path() resolves to
# cuRobo's package-installed copy, which drifts out of sync with this one).
ROBOT_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "envs", "assets", "cuRobo", "g1_dex3_right_arm.yml",
)
APPLE_RADIUS = 0.04
APPLE_COLLISION_RADIUS = APPLE_RADIUS + 0.02
APPROACH_Z_OFFSET = -APPLE_RADIUS * 3.0

ARM_DEFAULT_POSE = [0.3, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
ARM_SCALE = 0.5
HAND_OPEN_RAW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]


def quat_from_axis_angle(axis, angle_rad):
    axis = axis / torch.norm(axis)
    half = angle_rad / 2.0
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w.unsqueeze(0), xyz])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def euler_xyz_deg_to_quat(roll_deg, pitch_deg, yaw_deg, device):
    rx = torch.tensor(math.radians(roll_deg), device=device)
    ry = torch.tensor(math.radians(pitch_deg), device=device)
    rz = torch.tensor(math.radians(yaw_deg), device=device)
    qx = quat_from_axis_angle(torch.tensor([1.0, 0.0, 0.0], device=device), rx)
    qy = quat_from_axis_angle(torch.tensor([0.0, 1.0, 0.0], device=device), ry)
    qz = quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0], device=device), rz)
    return quat_mul(quat_mul(qx, qy), qz)


def quat_conjugate(q):
    w, x, y, z = q
    return torch.stack([w, -x, -y, -z])


def quat_rotate(q, v):
    w = q[0]
    q_vec = q[1:]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + w * t + torch.cross(q_vec, t, dim=-1)


def world_to_torso(pos_world, torso_pos, torso_quat):
    return quat_rotate(quat_conjugate(torso_quat), pos_world - torso_pos)


def absolute_to_raw_arm(absolute_joint_angles):
    return [(a - d) / ARM_SCALE for a, d in zip(absolute_joint_angles, ARM_DEFAULT_POSE)]


# ── DDS ──────────────────────────────────────────────────────────────────
ChannelFactoryInitialize(1)
publisher = ChannelPublisher("rt/arm_tune/targets", String_)
publisher.Init()

_state_lock = threading.Lock()
_latest_state = None


def _cb(msg):
    global _latest_state
    with _state_lock:
        _latest_state = json.loads(msg.data)


sub = ChannelSubscriber("rt/arm_tune/state", String_)
sub.Init(_cb, 1)

print("Waiting for state from run_env.py...")
t0 = time.time()
while time.time() - t0 < 15.0:
    with _state_lock:
        if _latest_state is not None:
            break
    time.sleep(0.1)
else:
    raise RuntimeError("No state received — is run_env.py running?")

state = dict(_latest_state)

tensor_args = TensorDeviceType()
torso_pos  = torch.tensor(state["torso_pos"], device=tensor_args.device, dtype=torch.float32)
torso_quat = torch.tensor(state["torso_quat"], device=tensor_args.device, dtype=torch.float32)
apple_pos_world = torch.tensor(state["apple_pos"], device=tensor_args.device, dtype=torch.float32)
current_arm_pos = state["arm_joint_pos"]

apple_local = world_to_torso(apple_pos_world, torso_pos, torso_quat)
goal_world  = apple_pos_world + torch.tensor([0.0, 0.0, APPROACH_Z_OFFSET], device=tensor_args.device)
goal_local  = world_to_torso(goal_world, torso_pos, torso_quat)
goal_quat   = euler_xyz_deg_to_quat(ROLL_DEG, PITCH_DEG, YAW_DEG, tensor_args.device)

ground_center_world = torch.tensor([0.0, 0.0, -0.05], device=tensor_args.device)
ground_pos_local  = world_to_torso(ground_center_world, torso_pos, torso_quat)
ground_quat_local = quat_conjugate(torso_quat)

robot_cfg_dict = load_yaml(ROBOT_CFG_PATH)
kin = robot_cfg_dict["robot_cfg"]["kinematics"]
kin["base_link"] = BASE_LINK_NAME
kin["lock_joints"] = {}
if "pelvis" in kin.get("collision_link_names", []):
    kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n != "pelvis"]

world_cfg = WorldConfig(
    cuboid=[Cuboid(name="ground", pose=ground_pos_local.tolist() + ground_quat_local.tolist(), dims=[5.0, 5.0, 0.1])],
    sphere=[Sphere(name="apple",
                    pose=[apple_local[0].item(), apple_local[1].item(), apple_local[2].item(), 1.0, 0.0, 0.0, 0.0],
                    radius=APPLE_COLLISION_RADIUS)],
)

ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg_dict["robot_cfg"],  # IKSolverConfig expects the UNWRAPPED dict (has "kinematics" at top level) —
                                  # unlike MotionGenConfig.load_from_robot_config, which auto-unwraps a
                                  # {"robot_cfg": {...}} dict internally. Passing the wrapped dict here
                                  # is what caused KeyError: 'kinematics'.
    world_model=world_cfg,
    tensor_args=tensor_args,
    ee_link_name=EE_LINK_NAME,
    rotation_threshold=ROTATION_THRESHOLD,
    position_threshold=0.01,
    num_seeds=30,
    use_cuda_graph=False,
)
ik_solver = IKSolver(ik_config)

print(f"Testing roll={ROLL_DEG}  pitch={PITCH_DEG}  yaw={YAW_DEG} deg ...")
result = ik_solver.solve_single(Pose(position=goal_local, quaternion=goal_quat))
success = result.success.item()
print(f"IK success: {success}")

if not success:
    print("❌ No solution at this orientation/tolerance. Try a different angle, or")
    print("   loosen ROTATION_THRESHOLD slightly (but see the caveat in reach_apple_curobo.py).")
else:
    joint_angles = result.solution.squeeze(0).cpu().tolist()
    print(f"Solved joint angles: {joint_angles}")
    raw_arm = absolute_to_raw_arm(joint_angles)
    print(f"Publishing static pose (raw action): {raw_arm}")
    print("Holding for 5 seconds — check the sim window now.")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        publisher.Write(String_(data=json.dumps({"arm": raw_arm, "hand": HAND_OPEN_RAW})))
        time.sleep(0.02)
    print("Returning to default pose.")
    for _ in range(20):
        publisher.Write(String_(data=json.dumps({"arm": [0.0]*7, "hand": HAND_OPEN_RAW})))
        time.sleep(0.02)