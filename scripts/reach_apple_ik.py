#!/usr/bin/env python3
"""
Reach the apple using a custom Jacobian-based Inverse Kinematics (IK).
No cuRobo required. Pure PyTorch DLS IK.
Usage:
~/IsaacLab/isaaclab.sh -p scripts/reach_apple_ik.py
"""
import argparse
import sys
import os
import torch

# ── Isaac Lab app must be set up BEFORE any omni/isaacsim imports ────────────
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="IsaacLab IK reach-above-apple demo")
parser.add_argument(
    "--approach_height", type=float, default=0.12,
    help="meters above the apple's center to place the palm target",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab / torch ─────────────────────────────────────
import gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # registers gym tasks
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg_PLAY, ROBOT_ROOT_HEIGHT, APPLE_RADIUS
from envs.mdp.tree_utils import StemManager

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
EE_LINK_NAME = "right_hand_palm_link"

def main():
    # --- Build the env exactly like run_env.py does ---
    cfg = AppleGraspEnvCfg_PLAY()
    env = gym.make("AppleGrasp-Play-v0", cfg=cfg).unwrapped
    obs, _ = env.reset()
    
    # Attaches the apple to the branch
    stem = StemManager(env)
    robot = env.scene["robot"]
    apple = env.scene["apple"]
    device = env.device
    
    # Find joint and body indices
    arm_joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINT_NAMES, preserve_order=True)
    hand_joint_ids, _ = robot.find_joints(["right_hand.*"])
    default_arm_q = robot.data.default_joint_pos[0, arm_joint_ids].clone()
    ee_body_idx = robot.find_bodies(EE_LINK_NAME)[0][0]
    
    # Target position in world frame
    target_pos_world = apple.data.root_pos_w[0].clone()
    target_pos_world[2] += args_cli.approach_height
    
    print(f"Target world pos: {target_pos_world.cpu().numpy()}")
    
    current_q = robot.data.joint_pos[:, arm_joint_ids].clone()
    
    print("Starting IK reaching loop...")
    # DLS damping factor (higher = more stable but slower convergence)
    damping = 0.05 
    
    for step in range(300):
        # 1. Get current EE position in world frame
        current_ee_pos = robot.data.body_state_w[:, ee_body_idx, :3].clone()
        
        # 2. Get position Jacobian (3 x N) in world frame
        jacobian_pos = robot.data.jacobian_w[:, ee_body_idx, :3, arm_joint_ids].clone()
        
        # 3. Compute position error
        pos_error = target_pos_world - current_ee_pos[0]
        
        # 4. Damped Least Squares IK: dq = J^T (J J^T + lambda^2 I)^-1 * error
        J = jacobian_pos[0]  # Shape: [3, 7]
        I = torch.eye(3, device=device)
        dq = J.T @ torch.linalg.solve(J @ J.T + (damping**2) * I, pos_error.unsqueeze(-1)).squeeze(-1)
        
        # 5. Integrate to get next joint positions
        dt = env.physics_dt
        next_q = current_q + dq.unsqueeze(0) * dt
        
        # 6. Compute action (relative to default pose, as expected by ActionsCfg)
        arm_action = next_q - default_arm_q
        hand_action = torch.zeros(len(hand_joint_ids), device=device)
        actions = torch.cat([arm_action, hand_action]).unsqueeze(0)
        
        # 7. Step environment
        env.step(actions)
        
        # Update current_q for next iteration
        current_q = next_q
        
        pos_error_norm = torch.norm(pos_error)
        if step % 20 == 0:
            print(f"Step {step}, Pos Error: {pos_error_norm.item():.4f} m")
            
        # Stop if we are close enough
        if pos_error_norm.item() < 0.01:
            print("Reached target!")
            break
            
    print("Reach complete. Holding final pose -- Ctrl+C to exit.")
    final_actions = torch.cat([current_q[0] - default_arm_q, hand_action]).unsqueeze(0)
    
    while simulation_app.is_running():
        env.step(final_actions)
        
    env.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()