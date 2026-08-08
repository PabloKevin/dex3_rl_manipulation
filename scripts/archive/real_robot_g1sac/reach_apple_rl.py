#!/usr/bin/env python3
# Copyright 2026 - G1 Dex3 Apple Grasp Project
"""reach_apple_rl.py
-----------------
Trajectory planning and execution using trained Stable-Baselines3 RL policies
for the Unitree G1 humanoid robot over Unitree DDS.

FIXES vs the previous version:
  1. FRAME: G1ArmEnv._fk() (and therefore everything the policy was trained
     on) returns poses in the PELVIS frame (g1_dex3_right_arm.yml's
     base_link), not the torso frame. The pelvis is rigidly fixed in world
     at (0, 0, ROBOT_ROOT_HEIGHT) with identity rotation (fix_root_link=True,
     no rotation in init_state) -- see apple_grasp_env_cfg.py. So converting
     world -> pelvis is just a constant subtraction, no rotation needed.
     (The cuRobo scripts use torso_link instead because THEY explicitly
     override base_link to torso_link in their own copy of the yml -- that
     override was never applied here.)
  2. ORIENTATION: run_env.py's DDS state never publishes an apple
     orientation (no apple_quat/apple_ori/target_quat key), so the old
     extract_target_rotation() silently fell back to np.eye(3) every call.
     That's an arbitrary, likely-unreachable orientation target. Replaced
     with a fixed, tunable approach orientation (same idea as
     scripts/calibration/orient_tune.py) via --roll-deg/--pitch-deg/--yaw-deg.
  3. SCALE: ARM_ACTION_SCALE/HAND_ACTION_SCALE now match
     apple_grasp_env_cfg.py's ActionsCfg (scale=1.0), not the old 0.5/0.3.

Sanity check before trusting a real apple reach: pick a random q_test,
compute pos_t, rot_t = env._fk(q_test), and command that as the target with
q_start = q_test's neighbour. If the arm reliably reaches a self-consistent
FK target, the control loop is correct and any remaining apple-specific
error is just approach-offset/orientation tuning, not a frame bug.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
from stable_baselines3 import SAC, TD3

# Import G1ArmEnv to guarantee 100% observation alignment with training
sys.path.append(os.path.abspath("/home/pablo_kevin/dex3_rl_manipulation/envs/g1_arm_policies"))
from g1_arm_env import G1ArmEnv


# ── 1. USER CONFIGURATION DEFAULTS ───────────────────────────────────────────
DEFAULT_CONFIG = "/home/pablo_kevin/dex3_rl_manipulation/envs/g1_arm_policies/config_right.yaml"
DEFAULT_CHECKPOINT = (
    "/home/pablo_kevin/dex3_rl_manipulation/envs/g1_arm_policies/g1_right_arm_sac.zip"
)
DEFAULT_MODE = "plan"  # Options: "plan" (step-by-step rollout) or "closed_loop" (real-time)
DEFAULT_STEP_DT = 0.02
DEFAULT_HOLD_TIME = 3.0

# ── 2. Action & Observation Constants ────────────────────────────────────────
# Must match apple_grasp_env_cfg.py's ActionsCfg exactly (both scale=1.0).
ARM_ACTION_SCALE = 1.0
HAND_ACTION_SCALE = 1.0
NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 7
MAX_JOINT_STEP = 0.05

# ── 3. Frame constants ────────────────────────────────────────────────────────
# Pelvis is fixed (fix_root_link=True) at this world pose with IDENTITY
# rotation -- see apple_grasp_env_cfg.py's G1_DEX3_CFG.init_state. If that
# ever changes (robot moved, rotated base, etc.) update this constant, or
# better: have run_env.py publish the pelvis pose directly over DDS.
ROBOT_ROOT_HEIGHT = 0.793
PELVIS_POS_WORLD = np.array([0.0, 0.0, ROBOT_ROOT_HEIGHT], dtype=np.float64)

# Apple grasp geometry: aim slightly below the apple's centre (cupping from
# underneath), same idea as APPROACH_Z_OFFSET in reach_apple_curobo.py.
APPLE_RADIUS = 0.04
APPLE_APPROACH_Z_OFFSET = + APPLE_RADIUS * 1.50

# Fixed target orientation (degrees), since no real apple orientation is
# ever published. This is a PELVIS-frame Euler triple (fixed base ==
# world-aligned), so it's not quite identical to the "torso frame" tuning
# in orient_tune.py -- torso is tilted from pelvis by waist_pitch (~0.1 rad,
# ~5.7 deg) in the default pose. Close enough to start; iterate with the
# CLI flags below if the palm approach still looks off.
DEFAULT_ROLL_DEG, DEFAULT_PITCH_DEG, DEFAULT_YAW_DEG = 90.0, 0.0, 0.0


#CALIBRATION RESULTS SUMMARY (pelvis-local frame)
CALIBRATION_VECTOR = [-0.0277, 0.007, -0.2137] # m [dx,dy,dz]


def world_to_pelvis(pos_world: np.ndarray) -> np.ndarray:
    """Constant-offset transform: pelvis is fixed in world with identity rotation."""
    return pos_world - PELVIS_POS_WORLD


class DDSRobotInterface:
    """Handles Unitree DDS communication with Isaac Lab / run_env.py."""

    def __init__(
        self,
        target_topic: str = "rt/arm_tune/targets",
        state_topic: str = "rt/arm_tune/state",
    ) -> None:
        self.target_topic = target_topic
        self.state_topic = state_topic
        self.state_lock = threading.Lock()
        self.latest_state: Optional[Dict[str, Any]] = None
        self.publisher = None
        self.subscriber = None
        self._init_dds()

    def _init_dds(self) -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

            ChannelFactoryInitialize(1)

            def _state_callback(msg: String_) -> None:
                try:
                    data = json.loads(msg.data)
                    with self.state_lock:
                        self.latest_state = data
                except Exception as e:
                    print(f"[DDS Interface] Error parsing incoming state JSON: {e}")

            self.subscriber = ChannelSubscriber(self.state_topic, String_)
            self.subscriber.Init(_state_callback, 1)

            self.publisher = ChannelPublisher(self.target_topic, String_)
            self.publisher.Init()
            print(
                f"[DDS Interface] Connected. Listening on '{self.state_topic}',"
                f" publishing to '{self.target_topic}'."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Unitree DDS channels: {e}") from e

    def get_state(self) -> Optional[Dict[str, Any]]:
        with self.state_lock:
            return self.latest_state.copy() if self.latest_state else None

    def wait_for_state(self, timeout: float = 10.0) -> Dict[str, Any]:
        print("[DDS Interface] Waiting for state broadcast...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self.get_state()
            if state is not None and "apple_pos" in state and "arm_joint_pos" in state:
                print("[DDS Interface] Valid state received!")
                return state
            time.sleep(0.05)
        raise TimeoutError("[DDS Interface] Timed out waiting for robot state.")

    def publish_joint_targets(
        self,
        arm_joint_angles: List[float],
        default_arm_pos: List[float],
        hand_joint_angles: Optional[List[float]] = None,
    ) -> None:
        if len(arm_joint_angles) != NUM_ARM_JOINTS or len(default_arm_pos) != NUM_ARM_JOINTS:
            raise ValueError(f"Arm joint angles and defaults must have length {NUM_ARM_JOINTS}")

        raw_arm = [
            (target - def_q) / ARM_ACTION_SCALE
            for target, def_q in zip(arm_joint_angles, default_arm_pos)
        ]
        raw_hand = (
            [0.0] * NUM_HAND_JOINTS
            if hand_joint_angles is None
            else [val / HAND_ACTION_SCALE for val in hand_joint_angles]
        )

        payload = {"arm": raw_arm, "hand": raw_hand}
        try:
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

            self.publisher.Write(String_(data=json.dumps(payload)))
        except Exception as e:
            print(f"[DDS Interface] Failed to publish action: {e}")


def compute_local_target(state: Dict[str, Any], target_rot_local: np.ndarray,
                          approach_z_offset: float) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame apple position -> pelvis-frame target the policy expects."""
    calibration_vector = np.array(CALIBRATION_VECTOR)
    apple_pos_world = np.array(state["apple_pos"], dtype=np.float64) - calibration_vector
    approach_point_world = apple_pos_world + np.array([0.0, 0.0, approach_z_offset])
    target_pos_local = world_to_pelvis(approach_point_world)
    return target_pos_local.astype(np.float32), target_rot_local


def build_observation(
    env: G1ArmEnv,
    q_current: np.ndarray,
    dq_current: np.ndarray,
    target_pos_local: np.ndarray,
    target_rot_local: np.ndarray,
    step_count: int = 0,
) -> np.ndarray:
    """Constructs the exact 21-dim observation vector expected by G1ArmEnv policies."""
    env.q = q_current.copy()
    env.dq = dq_current.copy()
    env._step_count = step_count
    env._target_pos = target_pos_local
    env._target_rot = target_rot_local

    pos_err, _, ori_err, _ = env._ee_pos_error()
    return env._obs(pos_err, ori_err)


def plan_trajectory_offline(
    model: Any,
    env: G1ArmEnv,
    q_start: np.ndarray,
    target_pos_local: np.ndarray,
    target_rot_local: np.ndarray,
    max_steps: int = 200,
) -> np.ndarray:
    """Unrolls the RL policy step-by-step using joint-space integration."""
    print(f"[Planner] Unrolling RL policy toward pelvis-frame target {np.round(target_pos_local, 3)}...")

    q_current = q_start.copy()
    dq_current = np.zeros(NUM_ARM_JOINTS, dtype=np.float64)
    q_trajectory = [q_current.copy()]

    for step in range(max_steps):
        obs = build_observation(
            env, q_current, dq_current, target_pos_local, target_rot_local, step_count=step
        )
        action, _ = model.predict(obs, deterministic=True)

        q_next = q_current + (action * MAX_JOINT_STEP)
        dq_current = (q_next - q_current) / env.dt

        q_trajectory.append(q_next.copy())
        q_current = q_next

    q_traj_arr = np.array(q_trajectory)
    print(f"[Planner] Generated {len(q_traj_arr)} trajectory waypoints.")
    return q_traj_arr


def execute_trajectory(
    dds: DDSRobotInterface,
    q_traj: np.ndarray,
    default_arm_pos: List[float],
    dt: float = 0.02,
    hold_time: float = 2.0,
) -> None:
    """Streams a planned joint trajectory over DDS."""
    print(f"[Executor] Streaming {len(q_traj)} trajectory points over DDS...")
    for q_cmd in q_traj:
        step_start = time.time()
        dds.publish_joint_targets(q_cmd.tolist(), default_arm_pos)
        time.sleep(max(0.0, dt - (time.time() - step_start)))

    print(f"[Executor] Holding final pose for {hold_time:.1f}s...")
    final_q = q_traj[-1].tolist()
    hold_steps = int(hold_time / dt)
    for _ in range(hold_steps):
        step_start = time.time()
        dds.publish_joint_targets(final_q, default_arm_pos)
        time.sleep(max(0.0, dt - (time.time() - step_start)))
    print("[Executor] Execution complete.")


def run_closed_loop_reactive(
    model: Any,
    env: G1ArmEnv,
    dds: DDSRobotInterface,
    target_rot_local: np.ndarray,
    approach_z_offset: float,
    control_dt: float = 0.02,
) -> None:
    """Runs real-time reactive control loop."""
    print("[Closed-Loop] Starting reactive control loop. Press Ctrl+C to stop...")
    step_count = 0
    prev_q = None

    try:
        while True:
            loop_start = time.time()
            state = dds.get_state()
            if state is None:
                time.sleep(0.01)
                continue

            current_q = np.array(state["arm_joint_pos"], dtype=np.float32)
            default_q = state.get("default_arm_pos", [0.0] * NUM_ARM_JOINTS)
            target_pos_local, _ = compute_local_target(state, target_rot_local, approach_z_offset)

            if prev_q is None:
                current_dq = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
            else:
                current_dq = (current_q - prev_q) / control_dt
            prev_q = current_q.copy()

            obs = build_observation(
                env,
                current_q,
                current_dq,
                target_pos_local,
                target_rot_local,
                step_count=step_count,
            )
            action, _ = model.predict(obs, deterministic=True)

            target_q = current_q + (action * MAX_JOINT_STEP)
            dds.publish_joint_targets(target_q.tolist(), default_q)

            step_count += 1
            time.sleep(max(0.0, control_dt - (time.time() - loop_start)))
    except KeyboardInterrupt:
        print("\n[Closed-Loop] Stopped by user.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reach Apple with Trained RL Policy over Unitree DDS"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to env config YAML")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to trained SB3 model checkpoint"
    )
    parser.add_argument(
        "--mode", choices=["plan", "closed_loop"], default=DEFAULT_MODE, help="Operation mode"
    )
    parser.add_argument("--step-dt", type=float, default=DEFAULT_STEP_DT, help="Time delay between DDS actions (s)")
    parser.add_argument("--hold-time", type=float, default=DEFAULT_HOLD_TIME, help="Seconds to hold pose at the end")
    parser.add_argument("--roll-deg", type=float, default=DEFAULT_ROLL_DEG, help="Target palm roll (pelvis frame, deg)")
    parser.add_argument("--pitch-deg", type=float, default=DEFAULT_PITCH_DEG, help="Target palm pitch (pelvis frame, deg)")
    parser.add_argument("--yaw-deg", type=float, default=DEFAULT_YAW_DEG, help="Target palm yaw (pelvis frame, deg)")
    parser.add_argument(
        "--approach-offset", type=float, default=APPLE_APPROACH_Z_OFFSET,
        help="Z offset (m) from apple centre to aim at, world frame (default: cup from below)"
    )
    args = parser.parse_args()


    target_rot_local = R.from_euler(
        "xyz", [args.roll_deg, args.pitch_deg, args.yaw_deg], degrees=True
    ).as_matrix()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = (
        os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config
    )
    checkpoint_path = (
        os.path.join(project_root, args.checkpoint)
        if not os.path.isabs(args.checkpoint)
        else args.checkpoint
    )

    print(f"[Setup] Initializing G1ArmEnv from '{config_path}'...")
    env = G1ArmEnv(config_path=config_path)

    algo_name = env.cfg.get("training", {}).get("algorithm", "SAC").upper()
    ModelCls = SAC if algo_name == "SAC" else TD3

    print(f"[Setup] Loading {algo_name} policy checkpoint from '{checkpoint_path}'...")
    model = ModelCls.load(checkpoint_path, env=env)

    dds = DDSRobotInterface()
    initial_state = dds.wait_for_state()
    default_arm_pos = initial_state.get("default_arm_pos", [0.0] * NUM_ARM_JOINTS)

    target_pos_local, _ = compute_local_target(initial_state, target_rot_local, args.approach_offset)
    print(f"[Setup] Apple (world): {initial_state['apple_pos']}")
    print(f"[Setup] Target (pelvis frame, after approach offset): {target_pos_local.tolist()}")

    if args.mode == "plan":
        current_q = np.array(initial_state["arm_joint_pos"], dtype=np.float32)

        q_traj = plan_trajectory_offline(
            model=model,
            env=env,
            q_start=current_q,
            target_pos_local=target_pos_local,
            target_rot_local=target_rot_local,
        )

        execute_trajectory(
            dds=dds,
            q_traj=q_traj,
            default_arm_pos=default_arm_pos,
            dt=args.step_dt,
            hold_time=args.hold_time,
        )
    elif args.mode == "closed_loop":
        run_closed_loop_reactive(
            model=model,
            env=env,
            dds=dds,
            target_rot_local=target_rot_local,
            approach_z_offset=APPLE_APPROACH_Z_OFFSET,
            control_dt=args.step_dt,
        )

    print("[Done] Script terminated cleanly.")


if __name__ == "__main__":
    main()