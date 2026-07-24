#!/usr/bin/env python3
"""
calibrate_workspace.py
----------------------
Commands the Unitree G1 right arm to N different workspace poses using reach_apple_rl.py,
records real vs. target end-effector states, calculates mean directional error, and generates
calibration offsets.
"""

import os
import time
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC, TD3

# Import core interfaces from your existing script
from reach_apple_rl import (
    DDSRobotInterface,
    plan_trajectory_offline,
    execute_trajectory,
    DEFAULT_CONFIG,
    DEFAULT_CHECKPOINT,
    NUM_ARM_JOINTS,
)
import yaml

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
NUM_TEST_POINTS = 10  # Number of random targets to test
HOLD_TIME = 2.0       # Seconds to settle before measuring real pose
OUTPUT_FILE = "workspace_calibration_data.json"

def generate_test_targets(n: int, base_pos: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generates N reachable targets perturbed around the default apple workspace."""
    targets = []
    np.random.seed(42)  # For reproducible calibration
    
    for _ in range(n):
        # Sample position within a +/- 15cm box around standard apple spawn
        pos_offset = np.random.uniform(-0.15, 0.15, size=3)
        target_pos = base_pos + pos_offset
        
        # Sample slight orientation tilts (up to 20 degrees roll/pitch/yaw)
        euler_tilt = np.random.uniform(-20, 20, size=3)
        rot = R.from_euler('xyz', euler_tilt, degrees=True)
        # Convert to [w, x, y, z] quaternion convention
        quat_xyzw = rot.as_quat()
        target_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        
        targets.append((target_pos.astype(np.float32), target_quat))
    return targets

def quat_error_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Calculates the geodesic angular error (in degrees) between two [w,x,y,z] quaternions."""
    # Convert [w, x, y, z] to scipy [x, y, z, w]
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    diff = r1 * r2.inv()
    return float(np.abs(diff.magnitude()) * (180.0 / np.pi))

def main():
    print("[Calibration] Initializing DDS and loading RL policy...")
    dds = DDSRobotInterface()
    initial_state = dds.wait_for_state()
    
    # Load Policy
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, DEFAULT_CONFIG) if not os.path.isabs(DEFAULT_CONFIG) else DEFAULT_CONFIG
    checkpoint_path = os.path.join(project_root, DEFAULT_CHECKPOINT) if not os.path.isabs(DEFAULT_CHECKPOINT) else DEFAULT_CHECKPOINT
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    algo_name = cfg.get("training", {}).get("algorithm", "SAC").upper()
    ModelCls = SAC if algo_name == "SAC" else TD3
    model = ModelCls.load(checkpoint_path)
    
    default_arm_pos = initial_state.get("default_arm_pos", [0.0] * NUM_ARM_JOINTS)
    base_apple_pos = np.array(initial_state["apple_pos"], dtype=np.float32)
    
    targets = generate_test_targets(NUM_TEST_POINTS, base_apple_pos)
    results = []
    
    print(f"[Calibration] Starting execution across {NUM_TEST_POINTS} workspace targets...")
    for i, (target_pos, target_quat) in enumerate(targets, 1):
        print(f"\n--- Test Point {i}/{NUM_TEST_POINTS} ---")
        state = dds.get_state()
        current_q = np.array(state["arm_joint_pos"], dtype=np.float32)
        default_q = np.array(default_arm_pos, dtype=np.float32)
        
        # 1. Plan trajectory to synthetic target
        q_traj = plan_trajectory_offline(model, current_q, default_q, target_pos, target_quat)
        
        # 2. Execute and hold
        execute_trajectory(dds, q_traj, default_arm_pos, dt=0.02, hold_time=HOLD_TIME)
        
        # 3. Read real end-effector position and orientation
        settled_state = dds.get_state()
        real_pos = np.array(settled_state["ee_pos"], dtype=np.float32)
        real_quat = np.array(settled_state.get("ee_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        # 4. Calculate directional errors (Real - Target)
        pos_error = real_pos - target_pos
        euclidean_dist = float(np.linalg.norm(pos_error))
        ori_error_deg = quat_error_angle(target_quat, real_quat)
        
        print(f"Target Pos : {np.round(target_pos, 4)}")
        print(f"Real Pos   : {np.round(real_pos, 4)}")
        print(f"Error (X,Y,Z): {np.round(pos_error, 4)} m | Total Distance: {euclidean_dist*100:.2f} cm | Ori Error: {ori_error_deg:.2f}°")
        
        results.append({
            "test_id": i,
            "target_pos": target_pos.tolist(),
            "target_quat": target_quat.tolist(),
            "real_pos": real_pos.tolist(),
            "real_quat": real_quat.tolist(),
            "error_pos_xyz": pos_error.tolist(),
            "error_dist_m": euclidean_dist,
            "error_ori_deg": ori_error_deg
        })
        time.sleep(0.5)

    # ── Calculate Mean Calibration Offsets ────────────────────────────────────
    all_pos_errors = np.array([r["error_pos_xyz"] for r in results])
    mean_pos_error = np.mean(all_pos_errors, axis=0)
    std_pos_error = np.std(all_pos_errors, axis=0)
    mean_dist = np.mean([r["error_dist_m"] for r in results])
    mean_ori_deg = np.mean([r["error_ori_deg"] for r in results])
    
    # Save dataset
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"summary": {
            "mean_pos_error_xyz": mean_pos_error.tolist(),
            "std_pos_error_xyz": std_pos_error.tolist(),
            "mean_euclidean_error_m": mean_dist,
            "mean_ori_error_deg": mean_ori_deg
        }, "raw_data": results}, f, indent=2)
        
    print("\n" + "="*60)
    print("CALIBRATION RESULTS SUMMARY")
    print("="*60)
    print(f"Mean Directional Error (X, Y, Z) : {np.round(mean_pos_error, 4)} m")
    print(f"Error Standard Deviation         : {np.round(std_pos_error, 4)} m")
    print(f"Mean Euclidean Distance          : {mean_dist*100:.2f} cm")
    print(f"Mean Angular Error               : {mean_ori_deg:.2f}°")
    print(f"\nData saved to: {os.path.abspath(OUTPUT_FILE)}")
    print("="*60)
    print("\n>>> NEXT STEP: Copy these constants into reach_apple_rl.py <<<")
    print(f"CALIBRATION_OFFSET_POS = np.array({np.round(mean_pos_error, 5).tolist()}, dtype=np.float32)")

if __name__ == "__main__":
    main()