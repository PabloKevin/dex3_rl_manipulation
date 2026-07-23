#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Standalone cuRobo planner that sends joint targets to the running sim via DDS.
Does NOT launch Isaac Sim.

Usage:
1. Terminal 1: ~/IsaacLab/isaaclab.sh -p scripts/run_env.py
2. Terminal 2: python3 scripts/reach_apple_curobo.py

Key changes from the previous version:
  - base_link is overridden to "torso_link" instead of the yml's default "pelvis".
    The chain from torso_link -> right_hand_palm_link is EXACTLY the 7 cspace
    joints, so lock_joints can be empty — no more enumerating/locking every
    other joint in the body (that's what caused the earlier KeyError crashes).
  - The goal pose is computed live, in the torso frame, from state published
    by run_env.py — not a hardcoded pelvis-relative constant. This also means
    it stays correct if the waist or torso ever moves (e.g. once a locomotion
    controller is added).
  - Start state is the robot's ACTUAL current arm joint angles (from run_env.py's
    state broadcast), not a hardcoded retract_config guess.
  - cuRobo's planned joint angles are ABSOLUTE radians. run_env.py's action
    space uses use_default_offset=True with scale=0.5 (arm) / 0.3 (hand), so
    Isaac Lab applies: target = default_joint_pos + scale * raw_action.
    This script now converts absolute -> raw_action = (absolute - default) / scale
    before publishing. The previous version published absolute angles directly,
    which under-scaled every commanded delta by 2x (arm) — this was silently
    wrong even after the crash was fixed.
"""
import json
import time
import threading
import torch

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
_latest_state = None  # dict: torso_pos, torso_quat, arm_joint_pos, apple_pos


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


def wait_for_state(timeout_s: float = 15.0):
    """Block until run_env.py's first state message arrives."""
    print("Waiting for state from run_env.py (make sure it's running, and using the")
    print("updated version that publishes rt/arm_tune/state)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        with _state_lock:
            if _latest_state is not None:
                return dict(_latest_state)
        time.sleep(0.1)
    raise RuntimeError(
        "No state received from run_env.py within timeout. Check: (1) run_env.py is "
        "running and past env.reset(), (2) it's the updated version with _publish_state, "
        "(3) both scripts use ChannelFactoryInitialize(1) on the same network interface."
    )


# ── 2. cuRobo Setup ──────────────────────────────────────────────────────────
try:
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.geom.types import WorldConfig, Sphere, Cuboid
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.state import JointState
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
APPROACH_Z_OFFSET = -APPLE_RADIUS * 1.5  # target sits this far ABOVE the apple center (world +z)

# Hand stays open during the reach — raw action values, copied verbatim from
# harvest_apple.py's HOME/REACH stage (confirmed working over this same DDS link).
HAND_OPEN_RAW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]

CONTROL_HZ = 50
STEP_DT = 1.0 / CONTROL_HZ


def publish(arm_raw: list, hand_raw: list) -> None:
    if not DDS_OK:
        return
    try:
        publisher.Write(String_(data=json.dumps({"arm": arm_raw, "hand": hand_raw})))
    except Exception:
        pass


def absolute_to_raw_arm(absolute_joint_angles) -> list:
    """Convert absolute joint radians -> pre-scale raw action for the arm."""
    return [(a - d) / ARM_SCALE for a, d in zip(absolute_joint_angles, ARM_DEFAULT_POSE)]


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
    inv_quat = quat_conjugate(torso_quat)
    return quat_rotate(inv_quat, delta)


def build_robot_cfg_dict():
    """Load the bundled yml and override kinematics for a torso-based, arm-only chain."""
    cfg_path = join_path(get_robot_configs_path(), ROBOT_CFG_FILENAME)
    robot_cfg_dict = load_yaml(cfg_path)
    kinematics = robot_cfg_dict["robot_cfg"]["kinematics"]

    kinematics["base_link"] = BASE_LINK_NAME
    kinematics["lock_joints"] = {}  # nothing left to lock once base_link = torso_link

    # "pelvis" is upstream of torso_link now, so it's pruned from the chain too —
    # referencing it in collision_link_names would cause the same KeyError pattern.
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

    print("Initializing cuRobo...")
    tensor_args = TensorDeviceType()

    torso_pos  = torch.tensor(state["torso_pos"], device=tensor_args.device, dtype=torch.float32)
    torso_quat = torch.tensor(state["torso_quat"], device=tensor_args.device, dtype=torch.float32)
    apple_pos_world = torch.tensor(state["apple_pos"], device=tensor_args.device, dtype=torch.float32)
    current_arm_pos = state["arm_joint_pos"]

    apple_local  = world_to_torso(apple_pos_world, torso_pos, torso_quat)
    goal_world   = apple_pos_world + torch.tensor([0.0, 0.0, APPROACH_Z_OFFSET], device=tensor_args.device)
    goal_local   = world_to_torso(goal_world, torso_pos, torso_quat)

    # Ground plane: world z = -0.05 to 0.0 (0.1m thick, matches original dims).
    # cuRobo's primitive collision checker needs >=1 cuboid even when other
    # obstacles are spheres, so this also fixes "Primitive Collision has no
    # obstacles" — it's not just a nice-to-have safety margin.
    ground_center_world = torch.tensor([0.0, 0.0, -0.05], device=tensor_args.device)
    ground_pos_local  = world_to_torso(ground_center_world, torso_pos, torso_quat)
    ground_quat_local = quat_conjugate(torso_quat)  # keeps the box world-flat despite torso tilt

    print(f"  Torso pos (world):   {torso_pos.tolist()}")
    print(f"  Apple pos (world):   {apple_pos_world.tolist()}")
    print(f"  Apple pos (torso):   {apple_local.tolist()}")
    print(f"  Goal pos  (torso):   {goal_local.tolist()}")
    print(f"  Current arm pos:     {current_arm_pos}")

    robot_cfg_dict = build_robot_cfg_dict()

    world_cfg = WorldConfig(
        cuboid=[Cuboid(
            name="ground",
            pose=ground_pos_local.tolist() + ground_quat_local.tolist(),
            dims=[5.0, 5.0, 0.1],
        )],
        sphere=[Sphere(
            name="apple",
            pose=[apple_local[0].item(), apple_local[1].item(), apple_local[2].item(), 1.0, 0.0, 0.0, 0.0],
            radius=APPLE_RADIUS,
        )],
    )

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_dict,
        world_model=world_cfg,
        tensor_args=tensor_args,
        ee_link_name=EE_LINK_NAME,
        use_cuda_graph=False,
    )

    motion_gen = MotionGen(motion_gen_config)
    print("Warming up cuRobo (should take ~2-5 seconds)...")
    motion_gen.warmup()
    print("✅ cuRobo ready!")

    start_state = JointState.from_position(
        torch.tensor([current_arm_pos], device=tensor_args.device, dtype=torch.float32),
        joint_names=RIGHT_ARM_JOINT_NAMES,
    )

    # Orientation kept as identity in the torso frame for now — tune separately
    # once positional reach is confirmed to work.
    goal_pose = Pose(
        position=goal_local,
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=tensor_args.device),
    )

    print("Planning trajectory...")
    result = motion_gen.plan_single(
        start_state=start_state,
        goal_pose=goal_pose,
        plan_config=MotionGenPlanConfig(enable_finetune_trajopt=False, max_attempts=10, num_trajopt_seeds=12),
    )

    if not result.success.item():
        print(f"❌ cuRobo failed: {result.status}")
        trajectory = [current_arm_pos]
    else:
        print("✅ Plan successful!")
        trajectory = result.interpolated_plan.position.squeeze(0).cpu().tolist()

    print(f"Streaming {len(trajectory)} waypoints to run_env.py via DDS...")

    t0 = time.perf_counter()
    for i, target_arm_abs in enumerate(trajectory):
        publish(absolute_to_raw_arm(target_arm_abs), HAND_OPEN_RAW)

        elapsed = time.perf_counter() - t0
        target_time = (i + 1) * STEP_DT
        sleep_t = target_time - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    print("Reach complete! Holding final pose for 2 seconds...")
    final_arm_raw = absolute_to_raw_arm(trajectory[-1])
    for _ in range(100):  # 2 seconds at 50 Hz
        publish(final_arm_raw, HAND_OPEN_RAW)
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