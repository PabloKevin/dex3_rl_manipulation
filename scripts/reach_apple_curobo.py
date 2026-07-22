#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Plan a right-arm reach ABOVE the apple using cuRobo, matching the proven 
configuration from the physical-superintelligence-lab/SIMPLE repository.
Usage:
~/IsaacLab/isaaclab.sh -p scripts/reach_apple_curobo.py
"""
import argparse
import sys
import os
import torch

# ── Isaac Lab app must be set up BEFORE any omni/isaacsim imports ────────────
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="cuRobo reach-above-apple demo")
parser.add_argument(
    "--approach_height", type=float, default=0.12,
    help="meters above the apple's center to place the palm target",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab / torch / cuRobo ────────────────────────────
import gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # registers gym tasks
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg_PLAY, ROBOT_ROOT_HEIGHT, APPLE_RADIUS
from envs.mdp.tree_utils import StemManager

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Cuboid, Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, join_path
from curobo.util.logger import setup_curobo_logger

setup_curobo_logger("error")  # Keep it clean, only show real errors

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
ROBOT_CFG_PATH = join_path(get_robot_configs_path(), "g1_dex3_right_arm.yml")

def main():
    cfg = AppleGraspEnvCfg_PLAY()
    env = gym.make("AppleGrasp-Play-v0", cfg=cfg).unwrapped
    obs, _ = env.reset()
    
    stem = StemManager(env)
    robot = env.scene["robot"]
    apple = env.scene["apple"]
    device = env.device
    
    arm_joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINT_NAMES, preserve_order=True)
    hand_joint_ids, _ = robot.find_joints(["right_hand.*"])
    default_arm_q = robot.data.default_joint_pos[0, arm_joint_ids].clone()
    
    # --- Apple position -> robot base_link (pelvis) frame ---
    env_origin = env.scene.env_origins[0]
    apple_pos_world = apple.data.root_pos_w[0]
    apple_pos_env = apple_pos_world - env_origin
    robot_base_pos = torch.tensor([0.0, 0.0, ROBOT_ROOT_HEIGHT], device=device)
    apple_pos_local = apple_pos_env - robot_base_pos
    
    target_pos_local = apple_pos_local.clone()
    target_pos_local[2] += args_cli.approach_height
    
    print(f"Apple world pos:            {apple_pos_world.cpu().numpy()}")
    print(f"Apple base-local pos:       {apple_pos_local.cpu().numpy()}")
    print(f"Target (above apple, base-local): {target_pos_local.cpu().numpy()}")
    
    # --- cuRobo setup (Matching SIMPLE repo proven config) ---
    tensor_args = TensorDeviceType()
    ground_z_local = 0.0 - ROBOT_ROOT_HEIGHT - 0.05
    
    world_cfg = WorldConfig(
        cuboid=[
            Cuboid(
                name="ground",
                pose=[0.0, 0.0, ground_z_local, 1.0, 0.0, 0.0, 0.0],
                dims=[5.0, 5.0, 0.1],
            ),
        ],
        sphere=[
            Sphere(
                name="apple",
                pose=[
                    apple_pos_local[0].item(), apple_pos_local[1].item(), apple_pos_local[2].item(),
                    1.0, 0.0, 0.0, 0.0,
                ],
                radius=APPLE_RADIUS,
            ),
        ],
    )
    
    # ✅ FIX 1: Explicitly disable self_collision and CUDA graph compilation
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        ROBOT_CFG_PATH,
        world_model=world_cfg,
        tensor_args=tensor_args,
        ee_link_name=EE_LINK_NAME,
        self_collision=False,          # Prevents INVALID_START_STATE_SELF_COLLISION
        use_cuda_graph=False,          # Prevents indefinite hanging during warmup/planning
        collision_activation_distance=torch.tensor([0.01], device='cuda', dtype=torch.float32),
    )
    
    motion_gen = MotionGen(motion_gen_config)
    
    print("Warming up cuRobo (this should take 2-5 seconds now)...")
    motion_gen.warmup()
    
    # ✅ FIX 2: Pass ONLY the 7 active right-arm joints, NOT all 43 joints
    # We use a known collision-free "tucked" pose from the YAML as the start state
    retract_config = torch.tensor([0.3, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0], device=device)
    
    start_state = JointState.from_position(
        retract_config.unsqueeze(0).to(tensor_args.device),
        joint_names=RIGHT_ARM_JOINT_NAMES,
    )
    
    goal_pose = Pose(
        position=target_pos_local.to(tensor_args.device),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=tensor_args.device),
    )
    
    print("Planning right-arm reach above the apple...")
    
    # ✅ FIX 3: Use SIMPLE's proven planning config (disable finetune for speed, limit attempts)
    result = motion_gen.plan_single(
        start_state=start_state,
        goal_pose=goal_pose,
        plan_config=MotionGenPlanConfig(
            enable_finetune_trajopt=False, 
            max_attempts=10,
            num_trajopt_seeds=12
        ),
    )
    
    if result.success.item():
        print("Plan successful!")
        trajectory = result.interpolated_plan.position.squeeze(0).to(device)  # [T, 7]
    else:
        print(f"cuRobo failed to find a path. status: {result.status}")
        print("Holding default pose instead.")
        trajectory = default_arm_q.unsqueeze(0)
    
    # Execute trajectory
    hand_action = torch.zeros(len(hand_joint_ids), device=device)
    print("Executing trajectory...")
    
    for q in trajectory:
        arm_action = q.to(device) - default_arm_q
        actions = torch.cat([arm_action, hand_action]).unsqueeze(0)
        env.step(actions)
        
    print("Reach complete. Holding final pose -- Ctrl+C to exit.")
    final_actions = torch.cat([trajectory[-1].to(device) - default_arm_q, hand_action]).unsqueeze(0)
    
    while simulation_app.is_running():
        env.step(final_actions)
        
    env.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()