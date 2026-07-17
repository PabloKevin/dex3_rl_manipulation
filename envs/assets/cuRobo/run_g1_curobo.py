# 1. Initialize Isaac Sim (Updated for Pip/Conda Package Style)
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import torch
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.robots import Robot

# 2. cuRobo Imports
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import get_robot_configs_path, join_path

def main():
    # --- Configuration ---
    # Update these paths to where your assets actually live
    usd_path = "/home/pablo_kevin/dex3_rl_manipulation/envs/assets/cuRobo/g1_29dof_with_hand_rev_1_0/g1_29dof_with_hand_rev_1_0.usd"
    robot_cfg_path = "/home/pablo_kevin/dex3_rl_manipulation/envs/assets/cuRobo/unitree_g1.yml" # The YAML you provided
    
    # We are targeting the right hand based on your provided configuration
    ee_link_name = "right_hand_palm_link" 
    
    # --- Isaac Sim World Setup ---
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Load the Unitree G1 into Isaac Sim
    g1_robot = world.scene.add(
        Robot(
            prim_path="/World/UnitreeG1",
            name="unitree_g1",
            usd_path=usd_path,
            position=np.array([0.0, 0.0, 0.8]), # Adjust spawn height as needed
        )
    )

    # Spawn a target cube in space
    target_position = np.array([0.5, -0.3, 1.0]) # 0.5m forward, 0.3m right, 1.0m up
    cube = world.scene.add(
        VisualCuboid(
            prim_path="/World/TargetCube",
            name="target_cube",
            position=target_position,
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.0, 0.0]) # Red cube
        )
    )

    world.reset()

    # --- cuRobo Initialization ---
    tensor_args = TensorDeviceType()
    
    # Configure MotionGen using your unitree_g1.yml
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_path,
        world_config=WorldConfig(), # Add obstacle config here if needed later
        tensor_args=tensor_args,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    # --- Motion Planning ---
    # Get current robot state from Isaac Sim
    articulation_controller = g1_robot.get_articulation_controller()
    current_joint_positions = g1_robot.get_joint_positions()
    
    # Define the goal pose for the right hand (matching the cube's position)
    # Using a neutral orientation (identity quaternion) for the hand grab
    goal_pose = Pose(
        position=torch.tensor(target_position, device=tensor_args.device, dtype=torch.float32),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=tensor_args.device, dtype=torch.float32)
    )

    print(f"Planning motion for {ee_link_name} to reach the cube...")
    
    # Generate the trajectory
    plan_config = MotionGenPlanConfig(enable_graph=True)
    result = motion_gen.plan_single(
        start_state=torch.tensor(current_joint_positions, device=tensor_args.device, dtype=torch.float32).unsqueeze(0),
        goal_pose=goal_pose,
        plan_config=plan_config
    )

    if result.success.item():
        print("Plan successful! Executing trajectory in Isaac Sim...")
        trajectory = result.interpolated_plan.position.squeeze(0).cpu().numpy()
        
        # --- Execution Loop ---
        for i in range(trajectory.shape[0]):
            # Apply the joint positions frame by frame
            articulation_controller.apply_action(
                omni.isaac.core.utils.types.ArticulationAction(joint_positions=trajectory[i])
            )
            world.step(render=True)
    else:
        print("cuRobo failed to find a valid path to the cube.")

    # Keep simulation running to view the result
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()