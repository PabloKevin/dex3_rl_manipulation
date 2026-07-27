#!/usr/bin/env python3
"""
calibrate_workspace.py
----------------------
Calibrates the RL-policy reach pipeline (reach_apple_rl.py) against the
running Isaac Sim instance, in the SAME pelvis-local frame the policy was
trained and is driven in (see reach_apple_rl.py's world_to_pelvis()).

Targets are sampled via env._fk() on random reachable joint configs -- same
technique validate_mujoco.py uses -- so they're guaranteed to already be in
the policy's native frame, and any error measured here is real robot/control
error, not test-data frame error.

For each target:
  1. command the policy to reach it via DDS,
  2. read back the real end-effector pose after it settles,
  3. convert that real pose into pelvis-local frame,
  4. report the resulting positional/angular error.

Use mean_pos_error[2] (Z) to sanity-check/tune reach_apple_rl.py's
--approach-offset: real_pos = target_pos + error, so if you want the palm to
land `d` metres higher, subtract `d` from mean_pos_error[2]'s sign when
adjusting the offset (or just nudge --approach-offset and re-run this script
until mean_pos_error trends toward zero).
"""

import os
import time
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC, TD3

from reach_apple_rl import (
    DDSRobotInterface,
    plan_trajectory_offline,
    execute_trajectory,
    world_to_pelvis,
    DEFAULT_CONFIG,
    DEFAULT_CHECKPOINT,
    NUM_ARM_JOINTS,
    DEFAULT_ROLL_DEG,
    DEFAULT_PITCH_DEG,
    DEFAULT_YAW_DEG,
    APPLE_APPROACH_Z_OFFSET,
)
from g1_arm_env import G1ArmEnv

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
NUM_TEST_POINTS = 10
HOLD_TIME = 2.0
OUTPUT_FILE = "workspace_calibration_data.json"


def generate_test_targets_around_apple(n, rng, approach_point_local, base_target_rot,
                                        pos_jitter=0.06):
    """Position-only jitter around the real approach point. Orientation is
    ALWAYS the fixed DEFAULT_ROLL/PITCH/YAW target — never perturbed — since
    that's exactly what reach_apple_rl.py commands in practice."""
    targets = []
    for _ in range(n):
        offset = rng.uniform(-pos_jitter, pos_jitter, size=3)
        target_pos = (approach_point_local + offset).astype(np.float32)
        targets.append((target_pos, base_target_rot))  # same rotation every time
    return targets


def quat_error_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic angular error (degrees) between two 3x3 rotation matrices."""
    diff = R.from_matrix(rot_a) * R.from_matrix(rot_b).inv()
    return float(np.abs(diff.magnitude()) * (180.0 / np.pi))


def main():
    print("[Calibration] Initializing DDS and loading RL policy...")
    dds = DDSRobotInterface()
    initial_state = dds.wait_for_state()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = (
        os.path.join(project_root, DEFAULT_CONFIG)
        if not os.path.isabs(DEFAULT_CONFIG) else DEFAULT_CONFIG
    )
    checkpoint_path = (
        os.path.join(project_root, DEFAULT_CHECKPOINT)
        if not os.path.isabs(DEFAULT_CHECKPOINT) else DEFAULT_CHECKPOINT
    )

    env = G1ArmEnv(config_path=config_path)
    print("[Check] training ee_frame :", env.cfg["urdf"]["ee_frame"])
    print("[Check] pin model frame id:", env.ee_frame_id, "->", env.model.frames[env.ee_frame_id].name)
    algo_name = env.cfg.get("training", {}).get("algorithm", "SAC").upper()
    ModelCls = SAC if algo_name == "SAC" else TD3
    model = ModelCls.load(checkpoint_path, env=env)

    default_arm_pos = initial_state.get("default_arm_pos", [0.0] * NUM_ARM_JOINTS)

    # Same approach point reach_apple_rl.py actually aims for -- apple centre
    # shifted by the tuned Z offset -- not the raw apple centre.
    apple_pos_world = np.array(initial_state["apple_pos"], dtype=np.float64)
    approach_point_local = world_to_pelvis(apple_pos_world) + np.array(
        [0.0, 0.0, APPLE_APPROACH_Z_OFFSET]
    )
    base_target_rot = R.from_euler(
        "xyz", [DEFAULT_ROLL_DEG, DEFAULT_PITCH_DEG, DEFAULT_YAW_DEG], degrees=True
    ).as_matrix()
    print(f"[Calibration] Approach point (pelvis-local): {np.round(approach_point_local, 3)}")

    rng = np.random.default_rng(42)
    targets = generate_test_targets_around_apple(
        NUM_TEST_POINTS, rng, approach_point_local, base_target_rot,
        pos_jitter=0.03,
    )
    results = []

    print(f"[Calibration] Starting execution across {NUM_TEST_POINTS} reachable targets...")
    for i, (target_pos, target_rot) in enumerate(targets, 1):
        print(f"\n--- Test Point {i}/{NUM_TEST_POINTS} ---")
        state = dds.get_state()
        current_q = np.array(state["arm_joint_pos"], dtype=np.float32)

        # 1. Plan trajectory to the (already pelvis-local) synthetic target
        q_traj = plan_trajectory_offline(
            model=model, env=env, q_start=current_q,
            target_pos_local=target_pos, target_rot_local=target_rot,
            max_steps=env.max_steps,   # 400, matches training — was silently capped at 200
        )

        # 2. Execute and hold
        execute_trajectory(dds=dds, q_traj=q_traj, default_arm_pos=default_arm_pos,
                            dt=0.02, hold_time=HOLD_TIME)

        # 3. Read real end-effector pose, convert to pelvis frame.
        #    Pelvis has identity rotation in world, so orientation needs no
        #    rotation at all -- only position needs the constant translation.
        settled_state = dds.get_state()
        real_pos_world = np.array(settled_state["ee_pos"], dtype=np.float32)
        real_quat_world = np.array(
            settled_state.get("ee_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32
        )  # w, x, y, z

        real_pos_local = world_to_pelvis(real_pos_world)
        real_rot_local = R.from_quat(
            [real_quat_world[1], real_quat_world[2], real_quat_world[3], real_quat_world[0]]
        ).as_matrix()

        # 4. Errors (Real - Target), both already pelvis-local
        pos_error = real_pos_local - target_pos
        euclidean_dist = float(np.linalg.norm(pos_error))
        ori_error_deg = quat_error_deg(target_rot, real_rot_local)

        print(f"Target Pos (pelvis) : {np.round(target_pos, 4)}")
        print(f"Real Pos   (pelvis) : {np.round(real_pos_local, 4)}")
        print(f"Error (X,Y,Z): {np.round(pos_error, 4)} m | "
              f"Total Distance: {euclidean_dist*100:.2f} cm | Ori Error: {ori_error_deg:.2f}°")

        results.append({
            "test_id": i,
            "target_pos": target_pos.tolist(),
            "target_rot": target_rot.tolist(),
            "real_pos": real_pos_local.tolist(),
            "real_rot": real_rot_local.tolist(),
            "error_pos_xyz": pos_error.tolist(),
            "error_dist_m": euclidean_dist,
            "error_ori_deg": ori_error_deg,
        })
        time.sleep(0.5)

    all_pos_errors = np.array([r["error_pos_xyz"] for r in results])
    mean_pos_error = np.mean(all_pos_errors, axis=0)
    std_pos_error = np.std(all_pos_errors, axis=0)
    mean_dist = np.mean([r["error_dist_m"] for r in results])
    mean_ori_deg = np.mean([r["error_ori_deg"] for r in results])

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"summary": {
            "mean_pos_error_xyz": mean_pos_error.tolist(),
            "std_pos_error_xyz": std_pos_error.tolist(),
            "mean_euclidean_error_m": mean_dist,
            "mean_ori_error_deg": mean_ori_deg,
        }, "raw_data": results}, f, indent=2)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS SUMMARY (pelvis-local frame)")
    print("=" * 60)
    print(f"Mean Directional Error (X, Y, Z) : {np.round(mean_pos_error, 4)} m")
    print(f"Error Standard Deviation         : {np.round(std_pos_error, 4)} m")
    print(f"Mean Euclidean Distance          : {mean_dist*100:.2f} cm")
    print(f"Mean Angular Error               : {mean_ori_deg:.2f}°")
    print(f"\nData saved to: {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 60)
    print("\nIf this mean error is small and roughly isotropic (all axes ~equal,")
    print("a few cm), that's just residual policy/PD tracking error -- expected,")
    print("don't over-correct for it. If Z is consistently large relative to X/Y,")
    print("that's your signal to keep nudging --approach-offset in reach_apple_rl.py.")


if __name__ == "__main__":
    main()