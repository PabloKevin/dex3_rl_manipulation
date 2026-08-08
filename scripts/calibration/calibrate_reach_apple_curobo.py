#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Calibration script for reach_apple_curobo.py.

Builds cuRobo and warms it up ONCE, then commands the right arm to N test
poses — position varied, orientation kept close to the "palm up, fingers
forward" convention already used in reach_apple_curobo.py (ROLL=90, PITCH=0,
YAW=0) — and for each one records:

  - the GOAL sent to cuRobo, in torso-local frame (the frame the planner
    actually solves in)
  - the ACTUAL pose of right_hand_palm_link once the robot settles, read
    back from run_env.py's DDS state broadcast ("ee_pos"/"ee_quat", world
    frame), then converted into the same torso-local frame

It reports mean/std/max position + orientation error, and prints a ready
-to-paste systematic bias you drop into reach_apple_curobo.py.

Usage:
    Terminal 1: ~/IsaacLab/isaaclab.sh -p scripts/run_env.py
    Terminal 2: python3 scripts/calibration/calibrate_reach_apple_curobo.py --n 10

Requires the version of run_env.py that publishes "ee_pos"/"ee_quat" on
rt/arm_tune/state (the one you're already using with reach_apple_curobo.py).
"""
import argparse
import json
import math
import os
import time
import threading

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────
#  Quaternion helpers — identical convention to reach_apple_curobo.py
# ─────────────────────────────────────────────────────────────────────────
def quat_from_axis_angle(axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.norm(axis)
    half = angle_rad / 2.0
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w.unsqueeze(0), xyz])


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q
    return torch.stack([w, -x, -y, -z])


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[0]
    qvec = q[1:]
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + w * t + torch.cross(qvec, t, dim=-1)


def euler_xyz_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float, device) -> torch.Tensor:
    rx = torch.tensor(math.radians(roll_deg), device=device)
    ry = torch.tensor(math.radians(pitch_deg), device=device)
    rz = torch.tensor(math.radians(yaw_deg), device=device)
    qx = quat_from_axis_angle(torch.tensor([1.0, 0.0, 0.0], device=device), rx)
    qy = quat_from_axis_angle(torch.tensor([0.0, 1.0, 0.0], device=device), ry)
    qz = quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0], device=device), rz)
    return quat_mul(quat_mul(qx, qy), qz)


def world_to_torso(pos_world: torch.Tensor, torso_pos: torch.Tensor, torso_quat: torch.Tensor) -> torch.Tensor:
    delta = pos_world - torso_pos
    return quat_rotate(quat_conjugate(torso_quat), delta)


def quat_log(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (w,x,y,z) -> rotation vector (axis * angle), radians."""
    if q[0] < 0:
        q = -q
    w = torch.clamp(q[0], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-12))
    if s < 1e-6:
        return torch.zeros(3, device=q.device)
    axis = q[1:] / s
    return axis * angle


def quat_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Inverse of quat_log: rotation vector (axis * angle) -> unit quaternion."""
    angle = torch.norm(rotvec)
    if angle < 1e-8:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=rotvec.device)
    axis = rotvec / angle
    return quat_from_axis_angle(axis, angle)


# ─────────────────────────────────────────────────────────────────────────
#  DDS
# ─────────────────────────────────────────────────────────────────────────
try:
    from unitree_sdk2py.core.channel import (
        ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    )
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    ChannelFactoryInitialize(1)
    publisher = ChannelPublisher("rt/arm_tune/targets", String_)
    publisher.Init()
    DDS_OK = True
    print("[DDS] Publisher ready on rt/arm_tune/targets")
except Exception as e:
    print(f"[DDS] OFFLINE: {e}")
    DDS_OK = False

_state_lock = threading.Lock()
_latest_state = None


def _state_callback(msg):
    global _latest_state
    try:
        data = json.loads(msg.data)
        with _state_lock:
            _latest_state = data
    except Exception as e:
        print(f"[DDS] state parse error: {e}")


if DDS_OK:
    try:
        state_sub = ChannelSubscriber("rt/arm_tune/state", String_)
        state_sub.Init(_state_callback, 1)
        print("[DDS] Subscribed to rt/arm_tune/state")
    except Exception as e:
        print(f"[DDS] Could not subscribe to state: {e}")
        DDS_OK = False


def read_state():
    with _state_lock:
        return dict(_latest_state) if _latest_state is not None else None


def wait_for_state(timeout_s: float = 15.0):
    print("Waiting for state from run_env.py (make sure it's running, past env.reset())...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = read_state()
        if s is not None:
            return s
        time.sleep(0.1)
    raise RuntimeError("No state received from run_env.py within timeout.")


def publish(arm_raw: list, hand_raw: list) -> None:
    if not DDS_OK:
        return
    try:
        publisher.Write(String_(data=json.dumps({"arm": arm_raw, "hand": hand_raw})))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
#  cuRobo
# ─────────────────────────────────────────────────────────────────────────
try:
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.geom.types import WorldConfig, Cuboid
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.util_file import load_yaml
    from curobo.util.logger import setup_curobo_logger

    setup_curobo_logger("error")
    CUROBO_OK = True
except Exception as e:
    print(f"[cuRobo] OFFLINE: {e}")
    CUROBO_OK = False


# ─────────────────────────────────────────────────────────────────────────
#  Constants — mirror reach_apple_curobo.py exactly
# ─────────────────────────────────────────────────────────────────────────
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
EE_LINK_NAME = "right_hand_palm_link"
BASE_LINK_NAME = "torso_link"
# Load the repo's OWN tracked copy directly — see reach_apple_curobo.py's
# ROBOT_CFG_PATH comment for why (get_robot_configs_path() resolves to
# cuRobo's package-installed copy, which drifts out of sync with this one —
# calibrating against the wrong file makes the resulting bias meaningless).
ROBOT_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "envs", "assets", "cuRobo", "g1_dex3_right_arm.yml",
)

ARM_DEFAULT_POSE = [0.3, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
ARM_SCALE = 0.5

# "Palm up, fingers forward" — same convention as reach_apple_curobo.py.
# Kept fixed on purpose; we only jitter yaw/pitch a little so we calibrate
# THIS pose family (the one actually used to grasp), not arbitrary wrist
# orientations the task never uses.
ROLL_DEG, PITCH_DEG, YAW_DEG = 90.0, 0.0, 0.0
ROTATION_THRESHOLD = 0.08

HAND_OPEN_RAW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]
CONTROL_HZ = 50
STEP_DT = 1.0 / CONTROL_HZ


def absolute_to_raw_arm(absolute_joint_angles) -> list:
    return [(a - d) / ARM_SCALE for a, d in zip(absolute_joint_angles, ARM_DEFAULT_POSE)]


def build_robot_cfg_dict():
    robot_cfg_dict = load_yaml(ROBOT_CFG_PATH)
    kinematics = robot_cfg_dict["robot_cfg"]["kinematics"]
    kinematics["base_link"] = BASE_LINK_NAME
    kinematics["lock_joints"] = {}
    if "pelvis" in kinematics.get("collision_link_names", []):
        kinematics["collision_link_names"] = [
            n for n in kinematics["collision_link_names"] if n != "pelvis"
        ]
    return robot_cfg_dict


def generate_calibration_points(n, center_world, device, pos_radius=0.12, z_radius=0.10,
                                 yaw_jitter_deg=30.0, pitch_jitter_deg=5.0, seed=42):
    """N test poses: position jittered around center_world (use the apple's
    position — it's guaranteed inside the reachable workspace), orientation
    kept close to the nominal palm-up pose."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        dx = float(rng.uniform(-pos_radius, pos_radius))
        dy = float(rng.uniform(-pos_radius, pos_radius))
        dz = float(rng.uniform(-z_radius, z_radius))
        yaw = YAW_DEG + float(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        pitch = PITCH_DEG + float(rng.uniform(-pitch_jitter_deg, pitch_jitter_deg))
        offset = torch.tensor([dx, dy, dz], device=device, dtype=torch.float32)
        points.append({"goal_world": center_world + offset, "roll": ROLL_DEG, "pitch": pitch, "yaw": yaw})
    return points


def main():
    parser = argparse.ArgumentParser(description="Calibrate reach_apple_curobo.py")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--pos_radius", type=float, default=0.05, help="XY jitter radius, metres")
    parser.add_argument("--z_radius", type=float, default=0.05, help="Z jitter radius, metres")
    parser.add_argument("--settle_time", type=float, default=1.5, help="Seconds to hold + settle before reading")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="calibration_results.json")
    args = parser.parse_args()

    if not DDS_OK or not CUROBO_OK:
        print("Cannot run: DDS or cuRobo unavailable.")
        return

    state = wait_for_state()
    print("✅ Got initial state from run_env.py.")
    tensor_args = TensorDeviceType()
    device = tensor_args.device

    torso_pos0 = torch.tensor(state["torso_pos"], device=device, dtype=torch.float32)
    torso_quat0 = torch.tensor(state["torso_quat"], device=device, dtype=torch.float32)
    apple_pos0 = torch.tensor(state["apple_pos"], device=device, dtype=torch.float32)

    # ── Build + warm up cuRobo EXACTLY ONCE ─────────────────────────────
    print("Initializing cuRobo (one-time build + warmup)...")
    robot_cfg_dict = build_robot_cfg_dict()
    ground_center_world = torch.tensor([0.0, 0.0, -0.05], device=device)
    ground_pos_local = world_to_torso(ground_center_world, torso_pos0, torso_quat0)
    ground_quat_local = quat_conjugate(torso_quat0)
    world_cfg = WorldConfig(cuboid=[Cuboid(
        name="ground",
        pose=ground_pos_local.tolist() + ground_quat_local.tolist(),
        dims=[5.0, 5.0, 0.1],
    )])
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_dict,
        world_model=world_cfg,
        tensor_args=tensor_args,
        ee_link_name=EE_LINK_NAME,
        use_cuda_graph=False,
        rotation_threshold=ROTATION_THRESHOLD,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    print("✅ cuRobo ready — warmed up once, reused for every calibration point below.\n")

    points = generate_calibration_points(
        args.n, apple_pos0, device,
        pos_radius=args.pos_radius, z_radius=args.z_radius, seed=args.seed,
    )

    records = []
    for i, pt in enumerate(points):
        print(f"── Point {i+1}/{len(points)} ──────────────────────────────")

        # Re-read live state so each plan starts from the ROBOT'S ACTUAL
        # current joint angles (not a stale/assumed value).
        state = read_state()
        torso_pos = torch.tensor(state["torso_pos"], device=device, dtype=torch.float32)
        torso_quat = torch.tensor(state["torso_quat"], device=device, dtype=torch.float32)
        current_arm_pos = state["arm_joint_pos"]

        goal_local = world_to_torso(pt["goal_world"], torso_pos, torso_quat)
        goal_quat_local = euler_xyz_deg_to_quat(pt["roll"], pt["pitch"], pt["yaw"], device)

        start_state = JointState.from_position(
            torch.tensor([current_arm_pos], device=device, dtype=torch.float32),
            joint_names=RIGHT_ARM_JOINT_NAMES,
        )
        goal_pose = Pose(position=goal_local, quaternion=goal_quat_local)

        result = motion_gen.plan_single(
            start_state=start_state,
            goal_pose=goal_pose,
            # See reach_apple_curobo.py's comment on this: enable_finetune_trajopt=False
            # is only guaranteed collision-free/within-limits, NOT guaranteed to actually
            # hit the goal pose — exactly the huge position/orientation error you'd be
            # calibrating (i.e. curve-fitting) against otherwise.
            plan_config=MotionGenPlanConfig(enable_finetune_trajopt=True, max_attempts=10, num_trajopt_seeds=12),
        )
        if not result.success.item():
            print(f"  ❌ planning failed ({result.status}) — skipping point")
            records.append({"index": i, "success": False, "status": str(result.status)})
            continue

        trajectory = result.interpolated_plan.position.squeeze(0).cpu().tolist()
        print(f"  ✅ planned, streaming {len(trajectory)} waypoints...")

        t0 = time.perf_counter()
        for k, target_arm_abs in enumerate(trajectory):
            publish(absolute_to_raw_arm(target_arm_abs), HAND_OPEN_RAW)
            elapsed = time.perf_counter() - t0
            sleep_t = (k + 1) * STEP_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # Hold + settle before trusting the state readback (PD controller /
        # physics need a moment to stop moving).
        final_arm_raw = absolute_to_raw_arm(trajectory[-1])
        for _ in range(int(args.settle_time * CONTROL_HZ)):
            publish(final_arm_raw, HAND_OPEN_RAW)
            time.sleep(STEP_DT)

        final_state = read_state()
        actual_world_pos = torch.tensor(final_state["ee_pos"], device=device, dtype=torch.float32)
        actual_world_quat = torch.tensor(final_state["ee_quat"], device=device, dtype=torch.float32)

        # Compare in TORSO-LOCAL frame — this is the frame cuRobo solves in,
        # and the frame the correction must be applied in.
        actual_local = world_to_torso(actual_world_pos, torso_pos, torso_quat)
        actual_quat_local = quat_mul(quat_conjugate(torso_quat), actual_world_quat)

        pos_err_vec = actual_local - goal_local
        pos_err_m = torch.norm(pos_err_vec).item()

        err_quat = quat_mul(quat_conjugate(goal_quat_local), actual_quat_local)
        err_rotvec = quat_log(err_quat)
        orient_err_deg = math.degrees(torch.norm(err_rotvec).item())

        print(f"  goal(local)   = {[round(v, 4) for v in goal_local.tolist()]}")
        print(f"  actual(local) = {[round(v, 4) for v in actual_local.tolist()]}")
        print(f"  pos_error = {pos_err_m*1000:.1f} mm   orient_error = {orient_err_deg:.2f} deg\n")

        records.append({
            "index": i, "success": True,
            "goal_pos_local": goal_local.tolist(),
            "goal_quat_local": goal_quat_local.tolist(),
            "actual_pos_local": actual_local.tolist(),
            "actual_quat_local": actual_quat_local.tolist(),
            "goal_pos_world": pt["goal_world"].tolist(),
            "actual_pos_world": actual_world_pos.tolist(),
            "actual_quat_world": actual_world_quat.tolist(),
            "pos_error_vec_local": pos_err_vec.tolist(),
            "pos_error_m": pos_err_m,
            "orient_error_rotvec_local": err_rotvec.tolist(),
            "orient_error_deg": orient_err_deg,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    ok = [r for r in records if r.get("success")]
    if not ok:
        print("No successful points — nothing to summarize.")
        return

    pos_errors = np.array([r["pos_error_m"] for r in ok])
    orient_errors = np.array([r["orient_error_deg"] for r in ok])
    pos_err_vecs = np.array([r["pos_error_vec_local"] for r in ok])
    rot_err_vecs = np.array([r["orient_error_rotvec_local"] for r in ok])

    mean_bias_pos_local = pos_err_vecs.mean(axis=0)  # systematic offset (torso frame)
    mean_bias_rot_local = rot_err_vecs.mean(axis=0)  # systematic small-rotation (torso frame)

    summary = {
        "n_points": len(records),
        "n_success": len(ok),
        "pos_error_mean_m": float(pos_errors.mean()),
        "pos_error_std_m": float(pos_errors.std()),
        "pos_error_max_m": float(pos_errors.max()),
        "orient_error_mean_deg": float(orient_errors.mean()),
        "orient_error_std_deg": float(orient_errors.std()),
        "orient_error_max_deg": float(orient_errors.max()),
        "mean_bias_pos_local_m": mean_bias_pos_local.tolist(),
        "mean_bias_rot_local_rotvec": mean_bias_rot_local.tolist(),
    }

    print("=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"  points: {len(ok)}/{len(records)} succeeded")
    print(f"  position error:    mean={summary['pos_error_mean_m']*1000:.1f} mm  "
          f"std={summary['pos_error_std_m']*1000:.1f} mm  max={summary['pos_error_max_m']*1000:.1f} mm")
    print(f"  orientation error: mean={summary['orient_error_mean_deg']:.2f} deg  "
          f"std={summary['orient_error_std_deg']:.2f} deg  max={summary['orient_error_max_deg']:.2f} deg")
    print(f"  systematic position bias (torso frame, m):      {mean_bias_pos_local}")
    print(f"  systematic rotation bias (torso frame, rad, axis*angle): {mean_bias_rot_local}")
    print("=" * 60)

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "points": records}, f, indent=2)
    print(f"\nSaved full results to {args.out}")

    print("\nPaste these two lines near the top of reach_apple_curobo.py:")
    print(f"CALIBRATION_POS_BIAS_LOCAL = {mean_bias_pos_local.tolist()}")
    print(f"CALIBRATION_ROT_BIAS_LOCAL = {mean_bias_rot_local.tolist()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        publish([0.0] * 7, [0.0] * 7)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        publish([0.0] * 7, [0.0] * 7)
