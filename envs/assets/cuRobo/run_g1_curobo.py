# 1. Initialize Isaac Sim (Pip/Conda Package Style)
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import torch
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction

# 2. cuRobo Imports
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, join_path


def main():
    # --- Configuration ---
    usd_path = "/home/pablo_kevin/dex3_rl_manipulation/envs/assets/cuRobo/g1_29dof_with_hand_rev_1_0/g1_29dof_with_hand_rev_1_0.usd"

    # This must be the RIGHT-ARM-ONLY config (7 active DOF, everything else
    # locked via `kinematics.lock_joints`), not the 49-DOF whole-body config.
    # Copy the generated g1_dex3_right_arm.yml to:
    #   ~/src/nvidia-curobo/src/curobo/content/configs/robot/g1_dex3_right_arm.yml
    robot_cfg_path = join_path(get_robot_configs_path(), "g1_dex3_right_arm.yml")

    # We are targeting the right hand based on your provided configuration
    ee_link_name = "right_hand_palm_link"

    # Fixed-base robot spawn position/orientation in world frame.
    # This MUST match kinematics.lock_joints base_j_x/y/z (and rotation, if any)
    # in the robot yaml, since cuRobo plans in the robot's base_link frame.
    robot_base_position = np.array([0.0, 0.0, 0.8])

    # The 7 active joints, in the exact order the yaml's cspace.joint_names uses.
    right_arm_joint_names = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # --- Isaac Sim World Setup ---
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Load the Unitree G1 into Isaac Sim
    # Robot() does not accept usd_path directly in Isaac Sim 4.5 -- the USD
    # must be referenced onto the stage first, then wrapped.
    add_reference_to_stage(usd_path=usd_path, prim_path="/World/UnitreeG1")

    g1_robot = world.scene.add(
        Robot(
            prim_path="/World/UnitreeG1",
            name="unitree_g1",
            position=robot_base_position,
        )
    )

    # Spawn a target cube in space (world frame)
    target_position_world = np.array([0.5, -0.3, 1.0])  # 0.5m forward, 0.3m right, 1.0m up
    cube = world.scene.add(
        VisualCuboid(
            prim_path="/World/TargetCube",
            name="target_cube",
            position=target_position_world,
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.0, 0.0]),  # Red cube
        )
    )

    world.reset()

    # --- cuRobo Initialization ---
    tensor_args = TensorDeviceType()

    # cuRobo's primitive collision checker errors out on a truly empty world
    # ("Primitive Collision has no obstacles"), so give it at least a ground
    # plane. Pose format is [x, y, z, qw, qx, qy, qz]. This slab sits just
    # under z=0 (the Isaac Sim ground plane), well below the robot's fixed
    # base at z=0.8, so it won't interfere with arm planning.
    world_cfg = WorldConfig(
        cuboid=[
            Cuboid(
                name="ground",
                pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
                dims=[5.0, 5.0, 0.1],
            )
        ]
    )

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_path,
        world_model=world_cfg,  # Add more obstacles here later (e.g. the tree/branch)
        tensor_args=tensor_args,
        ee_link_name=ee_link_name,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    # --- Resolve joint indices for the active (right-arm) DOF ---
    # get_joint_positions() returns ALL of the articulation's DOF, in the
    # order Isaac Sim discovered them in the USD -- which is NOT guaranteed
    # to match cuRobo's cspace.joint_names order. Always index by name.
    all_dof_names = g1_robot.dof_names
    try:
        arm_dof_indices = [all_dof_names.index(n) for n in right_arm_joint_names]
    except ValueError as e:
        raise RuntimeError(
            f"Joint name from cuRobo cspace not found on the USD articulation: {e}\n"
            f"Available DOF names: {all_dof_names}"
        )

    articulation_controller = g1_robot.get_articulation_controller()
    full_joint_positions = g1_robot.get_joint_positions()
    current_arm_positions = full_joint_positions[arm_dof_indices]

    # --- Convert cube target from world frame to robot base frame ---
    # cuRobo plans relative to the robot's base_link (kinematics origin),
    # not the Isaac Sim world origin.
    target_position_local = target_position_world - robot_base_position

    goal_pose = Pose(
        position=torch.tensor(target_position_local, device=tensor_args.device, dtype=torch.float32),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=tensor_args.device, dtype=torch.float32),
    )

    print(f"Planning motion for {ee_link_name} to reach the cube...")
    print(f"  Cube (world):  {target_position_world}")
    print(f"  Cube (robot-local): {target_position_local}")
    print(f"  Current right-arm joints ({right_arm_joint_names}): {current_arm_positions}")

    plan_config = MotionGenPlanConfig(enable_graph=True)
    start_state = JointState.from_position(
        torch.tensor(
            current_arm_positions, device=tensor_args.device, dtype=torch.float32
        ).unsqueeze(0),
        joint_names=right_arm_joint_names,
    )
    result = motion_gen.plan_single(
        start_state=start_state,
        goal_pose=goal_pose,
        plan_config=plan_config,
    )

    if result.success.item():
        print("Plan successful! Executing trajectory in Isaac Sim...")
        trajectory = result.interpolated_plan.position.squeeze(0).cpu().numpy()  # [T, 7]

        for i in range(trajectory.shape[0]):
            articulation_controller.apply_action(
                ArticulationAction(
                    joint_positions=trajectory[i],
                    joint_indices=arm_dof_indices,  # only drive the 7 right-arm joints
                )
            )
            world.step(render=True)
    else:
        print("cuRobo failed to find a valid path to the cube.")
        print(f"  status: {result.status}")

    # Keep simulation running to view the result
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()