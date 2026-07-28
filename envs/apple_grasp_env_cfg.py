# Copyright 2025 - G1 Dex3 Apple Grasp Project
# Isaac Lab 4.5 / IsaacSim 4.5, Ubuntu 22.04
"""
Apple Grasp Environment Configuration.

Scene:
  - G1-29dof + Dex3 right hand, FIXED BASE (fix_root_link=True)
  - Red sphere "apple" hanging from invisible anchor
  - Ground plane + distant light

Action space:
  - Right arm:  7 joints (shoulder x3, elbow, wrist x3)
  - Right Dex3: 7 joints
  - Total: 14-dim joint position targets (relative to default pose)

Observation space (Phase 1 — no camera yet, add in Phase 3):
  - Right arm joint positions (7)
  - Right arm joint velocities (7)
  - Right arm joint torques (7)
  - Right hand joint positions (7)
  - Right hand joint velocities (7)
  - Right hand fingertip contact forces (7 × 3 = 21, flattened)
  - Apple position relative to wrist (3)
  - Apple position in world (3)
  Total: ~62-dim proprioceptive vector
  (Camera obs added in Phase 3 via AppleGraspCameraEnvCfg)
"""

import math
import os
from dataclasses import MISSING
import random

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from . import mdp as task_mdp  # our custom mdp functions
from .mdp import domain_randomization as dr

# ─────────────────────────────────────────────
#  PATHS  ← set these before running
# ─────────────────────────────────────────────

# TODO(A): Set this to your preferred G1+Dex3 USD path.
# Example: "/path/to/unitree_sim_isaaclab/usd/g1_29dof_dex3/g1.usd"
# Use the URDF that has NO hand cameras but full body + Dex3.
G1_DEX3_USD_PATH = os.environ.get(
    "G1_DEX3_USD_PATH",
    "/home/pablo_kevin/dex3_rl_manipulation/envs/assets/robots/g1_29dof_with_hand_rev_1_0.usd",
)

# TODO: Set the head camera link name as it appears in YOUR urdf/usd.
# Run scripts/run_env.py --debug_links to print all link names.
HEAD_CAMERA_LINK = "head_link"   # ← verify with your URDF

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
RAND_PCT = 0.20  # ±20%

def _jitter(nominal: float, pct: float = RAND_PCT) -> float:
    return nominal * random.uniform(1.0 - pct, 1.0 + pct)

# ── Launch-time domain randomization (once per training run) ─────────────────
# Seed however you like — os.environ, cfg.seed, etc. Left un-seeded here so
# every process launch gets a different layout.
APPLE_RADIUS   = _jitter(0.04)
APPLE_MASS     = _jitter(0.15)
TREE_POS       = (_jitter(0.55), _jitter(-0.55), 0.0)
TREE_HEIGHT    = _jitter(1.25)
BRANCH_LENGTH  = _jitter(0.50)
BRANCH_RADIUS  = _jitter(0.04)
BRANCH_ROTATION = _jitter(110.0)
APPLE_STEM_BREAK_FORCE  = _jitter(2.5)   # nominal only — real per-episode value comes from EventTerm below
APPLE_STEM_BREAK_TORQUE = _jitter(0.5)

# Robot root height (metres). G1 pelvis in standing pose ≈ 0.79 m.
# With fix_root_link the pelvis is fixed here in world frame.
# Branch pivot calculations — rotates around trunk junction
branch_mid_x = TREE_POS[0] + (BRANCH_LENGTH / 2.0) * math.cos(math.radians(BRANCH_ROTATION))
branch_mid_y = TREE_POS[1] + (BRANCH_LENGTH / 2.0) * math.sin(math.radians(BRANCH_ROTATION))

# Apple hangs below the branch tip
# Branch tip X = TREE_POS[0] + BRANCH_LENGTH = 0.55 + 0.50 = 1.05
# But we want apple at ~0.55 in front of robot, so set TREE_POS[0] = 0.05
# and BRANCH_LENGTH = 0.50  →  branch tip at 0.55, apple directly below.
#APPLE_INIT_POS = (0.55, 0.00, TREE_HEIGHT - 0.15)  # 15 cm below branch

branch_tip_x = TREE_POS[0] + BRANCH_LENGTH * math.cos(math.radians(BRANCH_ROTATION))
branch_tip_y = TREE_POS[1] + BRANCH_LENGTH * math.sin(math.radians(BRANCH_ROTATION))
APPLE_INIT_POS = (branch_tip_x, branch_tip_y, TREE_HEIGHT - BRANCH_RADIUS - APPLE_RADIUS)

ROBOT_ROOT_HEIGHT = 0.793  # G1 pelvis height in standing pose (metres)


# ─────────────────────────────────────────────
#  ROBOT ARTICULATION CONFIG
# ─────────────────────────────────────────────

G1_DEX3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_DEX3_USD_PATH,
        activate_contact_sensors=True,  # needed for ContactSensor
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,          # ← fixed base; no locomotion needed
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ROBOT_ROOT_HEIGHT),
        # Standing-like pose so the robot looks correct visually
        # even though legs are irrelevant for this task.
        joint_pos={
            # ── legs (visual only, not in action space) ──────────────
            ".*hip_pitch.*":   -0.10,
            ".*hip_roll.*":     0.00,
            ".*hip_yaw.*":      0.00,
            ".*knee.*":         0.30,
            ".*ankle_pitch.*": -0.20,
            ".*ankle_roll.*":   0.00,
            # ── waist ────────────────────────────────────────────────
            "waist_yaw_joint":   0.0,
            "waist_roll_joint":  0.0,
            "waist_pitch_joint": 0.1,
            # ── left arm (parked, not in action space) ────────────────
            "left_shoulder_pitch_joint":  0.3,
            "left_shoulder_roll_joint":   0.2,
            "left_shoulder_yaw_joint":    0.0,
            "left_elbow_joint":           0.8,
            "left_wrist_roll_joint":      0.0,
            "left_wrist_pitch_joint":     0.0,
            "left_wrist_yaw_joint":       0.0,
            # ── right arm (controlled) ────────────────────────────────
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint":   0.0,
            "right_elbow_joint":          0.5,
            "right_wrist_roll_joint":     0.0,
            "right_wrist_pitch_joint":    0.0,
            "right_wrist_yaw_joint":      0.0,
            # ── right Dex3 hand (open) ────────────────────────────────
            "right_hand_thumb_1_joint":  0.55,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # ── Right arm ─────────────────────────────────────────────────
        # G1 arm joints are GearboxS type: kp=40, kd=1 (from SDK example)
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=40.0,   # GearboxS kp
            damping=1.0,      # GearboxS kd
        ),
        # ── Right Dex3 hand ───────────────────────────────────────────
        # From SDK: kp=0.5 (rotate mode), kp=1.5 (grip mode)
        # We use a moderate value; tune after first visual check.
        # TODO: replace regex with your actual Dex3 joint name pattern.
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_hand.*"],
            effort_limit=2.0,
            velocity_limit=3.14,
            stiffness=1.0,
            damping=0.05,
        ),
        # ── Left arm + hand (parked — position-held, not trained) ─────
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=40.0,
            damping=1.0,
        ),
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=["left_hand.*"],
            effort_limit=2.0,
            velocity_limit=3.14,
            stiffness=1.0,
            damping=0.05,
        ),
        # ── Legs + waist (position-held, not trained) ─────────────────
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*hip.*", ".*knee.*", ".*ankle.*",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            ],
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness=100.0,
            damping=5.0,
        ),
    },
)


# ─────────────────────────────────────────────
#  SCENE CONFIG
# ─────────────────────────────────────────────

@configclass
class AppleGraspSceneCfg(InteractiveSceneCfg):
    """Minimal scene: G1+Dex3 fixed base, red sphere apple, ground, light."""
    # ── Ground ──────────────────────────────────────────────────────────
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # ── Lighting ────────────────────────────────────────────────────────
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(1.0, 0.98, 0.92),   # warm white
            angle=0.5,
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=500.0,
            color=(0.75, 0.75, 1.0),   # cool ambient fill
        ),
    )

    # ── Robot ────────────────────────────────────────────────────────────
    robot: ArticulationCfg = G1_DEX3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ── Tree: base ───────────────────────────────────────────────────────
    # Flat box base anchoring the trunk to the ground.
    tree_base: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TreeBase",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.30, 0.06),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.25, 0.14, 0.05),
                roughness=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TREE_POS[0], TREE_POS[1], 0.03),   # sit on ground
        ),
    )

    # ── Tree: vertical trunk ─────────────────────────────────────────────
    tree_trunk: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TreeTrunk",
        spawn=sim_utils.CylinderCfg(
            radius=0.06,
            height=TREE_HEIGHT,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.20, 0.08),   # brown bark
                roughness=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TREE_POS[0], TREE_POS[1], TREE_HEIGHT / 2.0),
        ),
    )

    # ── Tree: horizontal branch (green) ──────────────────────────────────
    # Extends from trunk top toward the apple position along X axis.
    tree_branch: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TreeBranch",
        spawn=sim_utils.CylinderCfg(
            radius=BRANCH_RADIUS,
            height=BRANCH_LENGTH,
            axis="X",               # extends along X toward the robot
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.13, 0.42, 0.08),   # green branch
                roughness=0.9,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # Centre of branch: halfway between trunk top and apple X position
            pos=(branch_mid_x, branch_mid_y, TREE_HEIGHT),
            rot=(math.cos(math.radians(BRANCH_ROTATION)/2), 0.0, 0.0, math.sin(math.radians(BRANCH_ROTATION)/2)),  # rotation around Z axis (quaternion)
        ),
    )

    # ── Apple (red sphere, gravity enabled, held by breakable stem joint) ─
    # The apple hangs below the branch tip via a thin stem RigidObject.
    # A PhysX D6 joint with break force connects stem → apple.
    # When the robot applies enough downward force the joint breaks and
    # the apple falls free (gravity takes over). This is created at runtime
    # in the post_reset hook in scripts/run_env.py using the helper function
    # create_apple_stem_joint() defined in envs/mdp/tree_utils.py.
    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.SphereCfg(
            radius=APPLE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=50.0,
                max_linear_velocity=20.0,
                enable_gyroscopic_forces=True,
                disable_gravity=False,   # gravity on — stem joint holds apple up
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=APPLE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.06, 0.06),
                roughness=0.5,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=APPLE_INIT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ── Contact sensors on right Dex3 fingertips ─────────────────────────
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand.*_link",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Apple"],
    )

    # ── Head camera (128×128, RGB+Depth) ─────────────────────────────────
    # Disabled by default in Phase 1 (too slow to train).
    # Enable by using AppleGraspCameraEnvCfg below.
    # TODO: set correct prim_path using your head link name (HEAD_CAMERA_LINK).
    head_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{HEAD_CAMERA_LINK}/camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.0),     # slightly forward from head link origin
            rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention: x-forward, z-up
            convention="ros",
        ),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.5,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        width=128,
        height=128,
    )


# ─────────────────────────────────────────────
#  MANAGER CONFIGS
# ─────────────────────────────────────────────

@configclass
class ActionsCfg:
    """14-dim joint position targets: right arm (7) + right Dex3 (7)."""
    right_arm: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        scale=1.0,              # scale actions; range [-1,1] → [-0.5, 0.5] rad delta
        use_default_offset=True,
    )
    right_hand: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_hand.*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups. Policy uses 'policy' group only."""

    @configclass
    class PolicyObs(ObsGroup):
        """Proprioceptive observations for the policy (no camera)."""

        # Right arm
        right_arm_pos = ObsTerm(
            func=task_mdp.joint_pos_selected,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint", "right_elbow_joint",
                    "right_wrist_roll_joint", "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
            )},
        )
        right_arm_vel = ObsTerm(
            func=task_mdp.joint_vel_selected,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint", "right_elbow_joint",
                    "right_wrist_roll_joint", "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
            )},
        )
        right_arm_torque = ObsTerm(
            func=task_mdp.joint_effort_selected,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint", "right_elbow_joint",
                    "right_wrist_roll_joint", "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
            )},
        )

        # Right hand
        right_hand_pos = ObsTerm(
            func=task_mdp.joint_pos_selected,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["right_hand.*"]
            )},
        )
        right_hand_vel = ObsTerm(
            func=task_mdp.joint_vel_selected,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["right_hand.*"]
            )},
        )

        # Contact forces (right fingertips × apple)
        fingertip_contacts = ObsTerm(
            func=task_mdp.fingertip_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )

        # Apple position in world frame
        apple_pos_world = ObsTerm(
            func=task_mdp.object_pos_world,
            params={"object_cfg": SceneEntityCfg("apple")},
        )

        # Apple position relative to right wrist
        apple_pos_wrist = ObsTerm(
            func=task_mdp.object_pos_relative_to_body,
            params={
                "object_cfg": SceneEntityCfg("apple"),
                "robot_cfg": SceneEntityCfg(
                    "robot", body_names=["right_wrist_yaw_link"]
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False  # add noise in Phase 3

    policy: PolicyObs = PolicyObs()


@configclass
class RewardsCfg:
    """Reward terms — curriculum from reach → grasp → lift → place."""

    # ── Reach ────────────────────────────────────────────────────────────
    reaching = RewTerm(
        func=task_mdp.reward_reaching,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "std": 0.15,
        },
    )

    # ── Contact: reward having fingertips touch the apple ─────────────────
    contact = RewTerm(
        func=task_mdp.reward_fingertip_contact,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 0.1,  # N
        },
    )

    # ── Lift: reward apple height above initial position ──────────────────
    lift = RewTerm(
        func=task_mdp.reward_lift,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "init_height": APPLE_INIT_POS[2],
            "target_height": APPLE_INIT_POS[2] + 0.10,  # 10 cm lift
        },
    )

    # ── Action penalty: discourage large jerky actions ────────────────────
    action_penalty = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot"), "soft_ratio": 0.9},
    )

    # Reaching the apple pos
    palm_position = RewTerm(
        func=task_mdp.reward_palm_below_apple,
        weight=1.5,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "offset_z": 0.08,
            "std": 0.05,
        },
    )

    palm_orientation = RewTerm(
        func=task_mdp.reward_palm_orientation,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "torso_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "target_roll_deg": 90.0,   # confirm with orient_tune.py first
            "target_pitch_deg": 0.0,
            "target_yaw_deg": 0.0,
            "std_rad": 0.3,
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    # Always terminate at episode time limit
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if apple falls below ground
    apple_dropped = DoneTerm(
        func=task_mdp.apple_dropped,
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "min_height": -0.1,   # 10 cm below ground = dropped
        },
    )

    # (Phase 2+) Success: apple lifted and held
    # Uncomment after grasp is working:
    # success = DoneTerm(
    #     func=task_mdp.apple_lifted_success,
    #     params={"object_cfg": SceneEntityCfg("apple"),
    #             "target_height": APPLE_INIT_POS[2] + 0.10,
    #             "hold_time": 1.0},
    # )

@configclass
class EventsCfg:
    # Reset robot to default joint positions (with small noise)
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),   # ±0.05 rad noise
            "velocity_range": (-0.01, 0.01),
        },
    )

    """reset_apple = EventTerm(
        func=task_mdp.reset_apple_pose,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "pos_range_x": (-0.06, 0.06),
            "pos_range_y": (-0.06, 0.06),
            "pos_range_z": (-0.03, 0.03),
            "base_pos": APPLE_INIT_POS,
        },
    )"""

    randomize_tree = EventTerm(
        func=dr.randomize_tree_and_apple,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "tree_pos_xy": TREE_POS[:2],
            "tree_height_nominal": TREE_HEIGHT,
            "branch_length_nominal": BRANCH_LENGTH,
            "branch_radius_nominal": BRANCH_RADIUS,
            "branch_rotation_deg_nominal": BRANCH_ROTATION,
            "apple_radius_nominal": APPLE_RADIUS,
        },
    )

    # ── NEW: cheap per-episode randomization ──────────────────────────────
    # (randomize_apple_scale below is the exception: it's a raw USD write, so
    # it runs once at "prestartup" instead — see its EventTerm for why.)
    randomize_apple_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,   # isaaclab.envs.mdp built-in
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("apple"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_apple_scale = EventTerm(
        func=dr.randomize_apple_scale,
        mode="prestartup",   # raw USD scale write — must happen before sim.reset() creates the GPU tensor views
        params={"object_cfg": SceneEntityCfg("apple"), "nominal_radius": APPLE_RADIUS},
    )

    randomize_stem_break = EventTerm(
        func=dr.randomize_stem_break_params,
        mode="reset",
        params={"nominal_force": APPLE_STEM_BREAK_FORCE, "nominal_torque": APPLE_STEM_BREAK_TORQUE},
    )


# ─────────────────────────────────────────────
#  ENVIRONMENT CONFIGS
# ─────────────────────────────────────────────

@configclass
class AppleGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Phase 1-3 env: proprioception only, no camera."""

    scene: AppleGraspSceneCfg = AppleGraspSceneCfg(
        num_envs=32,        # safe for RTX 4060 (8 GB)
        env_spacing=2.5,    # metres between env origins
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005          # 200 Hz physics
        self.decimation = 4          # 50 Hz policy (200 / 4)
        self.episode_length_s = 10.0 # 10-second episodes
        self.sim.render_interval = self.decimation

        # Disable camera scene asset in Phase 1 (saves VRAM)
        self.scene.head_camera = None   # ← remove to enable camera

        # randomize_apple_scale (mode="prestartup") does a raw per-env USD scale
        # write, which IsaacLab refuses to allow under scene replication (cloned
        # envs would otherwise share the same USD-level properties). At our env
        # counts (<=32) the replication speedup is negligible, so this is safe.
        self.scene.replicate_physics = False

        # Viewer camera (for run_env.py, not used during training)
        self.viewer.eye = (1.5, -1.5, 1.8)
        self.viewer.lookat = (0.5, -0.25, 1.0)


@configclass
class AppleGraspEnvCfg_PLAY(AppleGraspEnvCfg):
    """Single-env config for policy evaluation / visualisation."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # No randomisation during play
        self.events.reset_apple = None
        self.events.reset_robot = None


@configclass
class AppleGraspCameraEnvCfg(AppleGraspEnvCfg):
    """Phase 3+ env: adds head camera to observation space."""
    def __post_init__(self):
        super().__post_init__()
        # Re-enable camera (override the None set above)
        self.scene.head_camera = AppleGraspSceneCfg.__dataclass_fields__[
            "head_camera"
        ].default
        # Fewer envs with camera (more VRAM per env)
        self.scene.num_envs = 8