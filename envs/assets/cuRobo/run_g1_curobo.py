# 1. Initialize Isaac Sim (Pip/Conda Package Style)
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import torch
import numpy as np
import yaml
from pxr import UsdPhysics, Gf
import omni.usd
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

    # Load the robot yaml once up front -- used both by cuRobo and to build
    # the nominal "hold" pose for Isaac Sim (single source of truth).
    with open(robot_cfg_path) as f:
        _robot_yaml = yaml.safe_load(f)
    _kin = _robot_yaml["robot_cfg"]["kinematics"]
    lock_joints = _kin["lock_joints"]  # {joint_name: value} for the 36 locked joints
    arm_retract = dict(zip(_kin["cspace"]["joint_names"], _kin["cspace"]["retract_config"]))

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

    # This USD is a floating-base humanoid (the pelvis is a free rigid body,
    # not fixed to the world) -- unlike IsaacLab's ArticulationCfg, the raw
    # Robot()/add_reference_to_stage path here has no fix_root_link flag.
    # Without this, gravity pulls the whole body down and it sinks through
    # the floor. Weld the root to the world with an explicit fixed joint,
    # the same mechanism fix_root_link=True uses under the hood.
    stage = omni.usd.get_context().get_stage()
    root_fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/UnitreeG1RootFixedJoint")
    root_fixed_joint.GetBody0Rel().SetTargets([])  # world frame
    root_fixed_joint.GetBody1Rel().SetTargets(["/World/UnitreeG1"])
    root_fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    root_fixed_joint.GetLocalPos0Attr().Set(
        Gf.Vec3f(float(robot_base_position[0]), float(robot_base_position[1]), float(robot_base_position[2]))
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

    # First reset: just to initialize the articulation so dof_names etc.
    # become available. The pose after this reset is whatever the USD
    # authored as default -- often an unnatural rest pose, ignore it.
    world.reset()

    all_dof_names = g1_robot.dof_names
    try:
        arm_dof_indices = [all_dof_names.index(n) for n in right_arm_joint_names]
    except ValueError as e:
        raise RuntimeError(
            f"Joint name from cuRobo cspace not found on the USD articulation: {e}\n"
            f"Available DOF names: {all_dof_names}"
        )

    # Build the full-DOF nominal ("standing") pose from the yaml.
    nominal_positions = np.array(g1_robot.get_joint_positions(), dtype=np.float32, copy=True)
    for name, val in lock_joints.items():
        if name in all_dof_names:
            nominal_positions[all_dof_names.index(name)] = val
    for name, val in arm_retract.items():
        nominal_positions[all_dof_names.index(name)] = val

    # Place the robot AT the nominal pose kinematically (via default state +
    # a second reset), instead of physically snapping it there under PD
    # control. Jumping straight to high stiffness from a very different
    # starting pose creates a huge instantaneous position error -> massive
    # corrective torque -> the robot gets flung/explodes and PhysX either
    # destabilizes or the app becomes unresponsive. This sidesteps that.
    g1_robot.set_joints_default_state(positions=nominal_positions)
    world.reset()

    # Now that the robot is already sitting at the nominal pose, gains only
    # need to resist small drift/gravity, not correct a big jump.
    articulation_controller = g1_robot.get_articulation_controller()
    kps = np.full(len(all_dof_names), 100.0, dtype=np.float32)
    kds = np.full(len(all_dof_names), 10.0, dtype=np.float32)
    for i, name in enumerate(all_dof_names):
        if "hand" in name:
            kps[i], kds[i] = 3.0, 0.3
    articulation_controller.set_gains(kps=kps, kds=kds)
    articulation_controller.apply_action(ArticulationAction(joint_positions=nominal_positions))
    for _ in range(10):
        world.step(render=True)

    # --- cuRobo Initialization ---
    tensor_args = TensorDeviceType()

    # cuRobo's primitive collision checker errors out on a truly empty world
    # ("Primitive Collision has no obstacles"), so give it at least a ground
    # plane. Pose format is [x, y, z, qw, qx, qy, qz].
    #
    # IMPORTANT: cuRobo obstacles are expressed in the robot's base_link
    # frame (pelvis here), NOT the Isaac Sim world frame -- same as the goal
    # pose below. The floor sits at world z=0, and the pelvis is spawned at
    # world z=robot_base_position[2], so in pelvis-local coordinates the
    # floor is at z = 0 - robot_base_position[2].
    ground_z_local = 0.0 - robot_base_position[2] - 0.05  # slab center, 0.1m thick
    world_cfg = WorldConfig(
        cuboid=[
            Cuboid(
                name="ground",
                pose=[0.0, 0.0, float(ground_z_local), 1.0, 0.0, 0.0, 0.0],
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
    # (all_dof_names / arm_dof_indices / nominal_positions were already
    # resolved above, before the robot was placed in its nominal pose.)
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
            full_target = nominal_positions.copy()
            full_target[arm_dof_indices] = trajectory[i]
            articulation_controller.apply_action(ArticulationAction(joint_positions=full_target))
            world.step(render=True)
    else:
        print("cuRobo failed to find a valid path to the cube.")
        print(f"  status: {result.status}")

    # Keep simulation running to view the result. Re-issue the hold command
    # each frame so the locked joints (legs/waist/left arm/hands) don't
    # slowly drift under gravity over a long-running session.
    while simulation_app.is_running():
        articulation_controller.apply_action(ArticulationAction(joint_positions=nominal_positions))
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()