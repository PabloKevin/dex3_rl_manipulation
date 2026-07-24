#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
cuRobo IK-only reacher — fast, single-shot inverse kinematics (NOT full
trajectory optimization). Sends the solved joint target to the running sim
via DDS. Does NOT launch Isaac Sim.

Usage:
1. Terminal 1: ~/IsaacLab/isaaclab.sh -p scripts/run_env.py
2. Terminal 2: python3 scripts/reach_apple_ik_curobo.py

Why this instead of reach_apple_curobo.py (MotionGen) or reach_apple_ik.py
(hand-rolled Jacobian DLS):
  - reach_apple_curobo.py runs full collision-aware TRAJECTORY OPTIMIZATION
    (MotionGen). Correct and thorough, but motion_gen.warmup() alone takes
    several seconds, and every solve plans an entire multi-waypoint path
    with collision checking against the world.
  - reach_apple_ik.py is a hand-rolled real-time Jacobian (damped
    least-squares) closed-loop controller. Joint limits are only clamped
    AFTER the fact, not built into the solve, and for this arm's kinematics
    near this particular goal it turned out to be numerically unstable —
    it kept winding multiple joints to their hard limits simultaneously
    instead of converging (see the diagnostics from earlier runs).
  - cuRobo's IKSolver sits in between: a proper CONSTRAINED nonlinear solve
    (joint limits are baked into the optimization itself, not bolted on
    afterward) that solves ONE target configuration directly — no
    trajectory, no requirement for a populated collision world. Its "warmup"
    is just lazily capturing a CUDA graph on the first solve() call (a few
    hundred ms, once), not MotionGen's multi-second explicit warmup(). Both
    self-collision checking and world-obstacle checking are disabled below
    (IK_SELF_COLLISION / IK_WORLD_COLLISION) for maximum speed — flip them
    on if you want the solution to also avoid the apple/torso/tree.

Once solved, the target joint angles are streamed to run_env.py smoothly
(smoothstep interpolation from the current arm pose over MOVE_DURATION_S)
rather than snapped to instantly, so the PD controller isn't asked to track
a sudden large step.

IMPORTANT — branch selection: a 7-DOF arm reaching a 6-DOF (position +
orientation) target still has multiple valid, DISCONNECTED joint-space
solutions (elbow-up vs. elbow-back vs. arm-swung-behind, etc.). Without
guidance, cuRobo's multi-seed random search is free to return any of them —
which can look like the arm reaching somewhere completely unrelated to the
goal, even though the solve itself succeeded with near-zero error. This
script seeds AND regularizes the solve with the robot's actual current arm
pose (retract_config + seed_config, per cuRobo's own solve_single(goal,
retract_config, seed_config) usage) so it converges to the nearby, natural
branch instead of an arbitrary one.

Only the right arm (7 DOF) is driven. The hand is held open throughout.
"""
import json
import math
import time
import threading

import torch


def quat_from_axis_angle(axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (w, x, y, z) for a rotation of angle_rad about axis."""
    axis = axis / torch.norm(axis)
    half = angle_rad / 2.0
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w.unsqueeze(0), xyz])


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, both (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def euler_xyz_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float, device) -> torch.Tensor:
    """Degrees -> quaternion (w, x, y, z). Same convention as reach_apple_curobo.py:
    intrinsic rotations about X, then the NEW Y, then the NEWEST Z (q = qx * qy * qz).
    """
    rx = torch.tensor(math.radians(roll_deg), device=device)
    ry = torch.tensor(math.radians(pitch_deg), device=device)
    rz = torch.tensor(math.radians(yaw_deg), device=device)
    qx = quat_from_axis_angle(torch.tensor([1.0, 0.0, 0.0], device=device), rx)
    qy = quat_from_axis_angle(torch.tensor([0.0, 1.0, 0.0], device=device), ry)
    qz = quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0], device=device), rz)
    return quat_mul(quat_mul(qx, qy), qz)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q
    return torch.stack([w, -x, -y, -z])


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v (3,) by quaternion q (4,), wxyz convention."""
    w = q[0]
    q_vec = q[1:]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + w * t + torch.cross(q_vec, t, dim=-1)


def world_to_torso(pos_world: torch.Tensor, torso_pos: torch.Tensor, torso_quat: torch.Tensor) -> torch.Tensor:
    """Express a world-frame point in the torso_link local frame."""
    delta = pos_world - torso_pos
    return quat_rotate(quat_conjugate(torso_quat), delta)


# ── 1. DDS Setup ─────────────────────────────────────────────────────────────
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
_latest_state = None  # dict: torso_pos, torso_quat, arm_joint_pos, apple_pos, ...


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


def wait_for_state(timeout_s: float = 15.0) -> dict:
    """Block until run_env.py's first state message arrives."""
    print("Waiting for state from run_env.py (make sure it's running and past")
    print("env.reset(), and both scripts use ChannelFactoryInitialize(1) on")
    print("the same network interface)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        with _state_lock:
            if _latest_state is not None:
                return dict(_latest_state)
        time.sleep(0.1)
    raise RuntimeError(
        "No state received from run_env.py within timeout. Check: (1) run_env.py is "
        "running and past env.reset(), (2) it's publishing rt/arm_tune/state, "
        "(3) both scripts use ChannelFactoryInitialize(1) on the same network interface."
    )


# ── 2. cuRobo Setup ──────────────────────────────────────────────────────────
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    from curobo.util.logger import setup_curobo_logger

    setup_curobo_logger("error")
    CUROBO_OK = True
except Exception as e:
    print(f"[cuRobo] OFFLINE: {e}")
    CUROBO_OK = False


# ── 3. Constants ─────────────────────────────────────────────────────────────
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
EE_LINK_NAME = "right_hand_palm_link"
BASE_LINK_NAME = "torso_link"
ROBOT_CFG_FILENAME = "g1_dex3_right_arm.yml"

# Must match apple_grasp_env_cfg.py ActionsCfg + robot init_state right arm.
ARM_DEFAULT_POSE = [0.3, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
ARM_SCALE = 0.5

APPLE_RADIUS = 0.04
APPROACH_Z_OFFSET = -APPLE_RADIUS * 3.0  # goal sits below the apple centre, cupping from underneath

# Same palm orientation convention as reach_apple_curobo.py.
ROLL_DEG, PITCH_DEG, YAW_DEG = 90.0, 0.0, 0.0
ROTATION_THRESHOLD = 0.15   # rad — looser than MotionGen's: a single IK solve
                             # doesn't need to discriminate between whole arm
                             # branches the way a trajectory does, so this can
                             # relax a bit. Tighten if the palm angle matters.
POSITION_THRESHOLD = 0.005  # 5 mm

# Both OFF by default for max speed, per your "no collision aware" ask. Flip
# either on later if the solved pose needs to dodge the apple/torso/tree.
IK_SELF_COLLISION = False
IK_WORLD_COLLISION = False
NUM_SEEDS = 20  # parallel random restarts cuRobo tries internally per solve

HAND_OPEN_RAW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]

CONTROL_HZ = 50
STEP_DT = 1.0 / CONTROL_HZ
MOVE_DURATION_S = 2.5  # smoothstep from current pose to the IK solution


def publish(arm_raw, hand_raw) -> None:
    if not DDS_OK:
        return
    try:
        publisher.Write(String_(data=json.dumps({"arm": list(arm_raw), "hand": list(hand_raw)})))
    except Exception:
        pass


def absolute_to_raw_arm(absolute) -> list:
    """Convert absolute joint radians -> pre-scale raw action for the arm."""
    return [(a - d) / ARM_SCALE for a, d in zip(absolute, ARM_DEFAULT_POSE)]


def build_robot_cfg_dict():
    """Load the bundled yml and override kinematics for a torso-based, arm-only chain."""
    cfg_path = join_path(get_robot_configs_path(), ROBOT_CFG_FILENAME)
    robot_cfg_dict = load_yaml(cfg_path)
    kinematics = robot_cfg_dict["robot_cfg"]["kinematics"]

    kinematics["base_link"] = BASE_LINK_NAME
    kinematics["lock_joints"] = {}  # nothing left to lock once base_link = torso_link

    if "pelvis" in kinematics.get("collision_link_names", []):
        kinematics["collision_link_names"] = [
            n for n in kinematics["collision_link_names"] if n != "pelvis"
        ]

    return robot_cfg_dict


def main():
    if not DDS_OK or not CUROBO_OK:
        print("Cannot run: DDS or cuRobo unavailable.")
        return

    state = wait_for_state()
    print("✅ Got initial state from run_env.py.")

    tensor_args = TensorDeviceType()

    torso_pos = torch.tensor(state["torso_pos"], device=tensor_args.device, dtype=torch.float32)
    torso_quat = torch.tensor(state["torso_quat"], device=tensor_args.device, dtype=torch.float32)
    apple_pos_world = torch.tensor(state["apple_pos"], device=tensor_args.device, dtype=torch.float32)
    current_arm_pos = state["arm_joint_pos"]

    goal_world = apple_pos_world + torch.tensor([0.0, 0.0, APPROACH_Z_OFFSET], device=tensor_args.device)
    goal_local = world_to_torso(goal_world, torso_pos, torso_quat)

    print(f"  Torso pos (world):   {torso_pos.tolist()}")
    print(f"  Apple pos (world):   {apple_pos_world.tolist()}")
    print(f"  Goal pos  (torso):   {goal_local.tolist()}")
    print(f"  Current arm pos:     {current_arm_pos}")

    robot_cfg_dict = build_robot_cfg_dict()

    print("Building cuRobo IKSolver (lightweight — no MotionGen trajectory warmup, "
          "collision checking disabled)...")
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg_dict,
        None,  # world model — None disables world-obstacle collision checking
        rotation_threshold=ROTATION_THRESHOLD,
        position_threshold=POSITION_THRESHOLD,
        num_seeds=NUM_SEEDS,
        self_collision_check=IK_SELF_COLLISION,
        self_collision_opt=IK_SELF_COLLISION,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    print("✅ IKSolver ready (CUDA graph captures lazily on the first solve call "
          "below — no separate multi-second warmup() needed).")

    goal_quat = euler_xyz_deg_to_quat(ROLL_DEG, PITCH_DEG, YAW_DEG, tensor_args.device)
    goal_pose = Pose(position=goal_local, quaternion=goal_quat)

    # Seed AND regularize the solve with the robot's actual current arm pose.
    # Without this, cuRobo's multi-seed random search is free to converge to
    # any kinematic branch that satisfies the target pose — including ones
    # that swing the arm to a completely different (but technically valid)
    # configuration, which is what you were seeing. Shapes per cuRobo's own
    # example usage: retract_config (batch, dof), seed_config (batch, num_seeds, dof).
    current_js = torch.tensor([current_arm_pos], device=tensor_args.device, dtype=torch.float32)
    retract_config = current_js
    seed_config = current_js.unsqueeze(1)

    print("Solving IK (seeded from current arm pose so cuRobo prefers the nearby branch)...")
    t0 = time.perf_counter()
    try:
        result = ik_solver.solve_single(goal_pose, retract_config, seed_config)
    except Exception as e:
        print(f"  ⚠️  Seeded solve_single() call failed ({e}); falling back to the default")
        print("     unseeded multi-restart search — the solution may land on an unexpected branch.")
        result = ik_solver.solve_single(goal_pose)
    solve_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Solve time: {solve_ms:.1f} ms "
          "(first call includes one-time CUDA graph capture; later solves are much faster)")

    success = bool(result.success.reshape(-1)[0].item())
    if not success:
        print("❌ IK solve failed — no valid joint-limit-respecting configuration found")
        print("   near the current pose for this goal position/orientation. Try loosening")
        print("   ROTATION_THRESHOLD, or double-check the goal is actually within reach")
        print("   without switching arm branches.")
        return

    solution = result.solution.reshape(-1, len(RIGHT_ARM_JOINT_NAMES))
    target_abs = solution[0].cpu().tolist()
    pos_err = float(result.position_error.reshape(-1)[0].item())
    travel = math.sqrt(sum((a - b) ** 2 for a, b in zip(target_abs, current_arm_pos)))
    print(f"✅ IK solution found (position error: {pos_err * 1000:.2f} mm)")
    print(f"   Absolute joint angles (rad): {[round(v, 3) for v in target_abs]}")
    print(f"   Joint-space travel from current pose: {travel:.3f} rad (L2 norm)")
    if travel > 1.5:
        print("   ⚠️  That's a large joint-space jump for a seeded solve — double-check the")
        print("      goal/orientation before trusting this is the branch you expect.")

    print(f"\nStreaming to run_env.py over {MOVE_DURATION_S:.1f}s "
          "(smoothstep interpolation, hand held open)...")
    start_raw = absolute_to_raw_arm(current_arm_pos)
    end_raw = absolute_to_raw_arm(target_abs)
    n_steps = max(1, int(MOVE_DURATION_S * CONTROL_HZ))

    t0 = time.perf_counter()
    for i in range(n_steps):
        t = (i + 1) / n_steps
        s = 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3  # smoothstep
        raw = [a + (b - a) * s for a, b in zip(start_raw, end_raw)]
        publish(raw, HAND_OPEN_RAW)

        elapsed = time.perf_counter() - t0
        target_time = (i + 1) * STEP_DT
        sleep_t = target_time - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    print("Reach complete! Holding final pose for 2 seconds...")
    for _ in range(100):
        publish(end_raw, HAND_OPEN_RAW)
        time.sleep(STEP_DT)

    print("Returning to HOME pose (raw action zero == default pose)...")
    for _ in range(100):
        publish([0.0] * 7, HAND_OPEN_RAW)
        time.sleep(STEP_DT)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        publish([0.0] * 7, [0.0] * 7)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        publish([0.0] * 7, [0.0] * 7)