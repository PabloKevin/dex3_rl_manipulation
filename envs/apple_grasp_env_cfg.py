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

Observation space (Phase 1 — reach only, no camera yet, add in Phase 3):
  - Right arm joint positions (7)
  - Right arm joint velocities (7)
  - Right hand joint positions (7)
  - Right hand joint velocities (7)
  - Apple position relative to wrist (3)
  - Palm orientation relative to torso, quaternion wxyz (4)
  Total: 35-dim proprioceptive vector
  (Joint torques and fingertip contact forces are dropped for this reach-only
  phase — not needed to reach, and just extra dimensions for the network to
  learn through. Re-add them (task_mdp.joint_effort_selected /
  task_mdp.fingertip_contact_forces) once grasp/manipulation training starts.)
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
from .mdp import reach_rewards as rr

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
RAND_PCT = 0.20  # +-20%
 
# Seeded so the scene is reproducible across processes. train.py and play.py
# MUST be launched with the same SCENE_SEED, otherwise you evaluate a policy
# on a tree it never trained on.
_SCENE_SEED = int(os.environ.get("SCENE_SEED", "12345"))
_scene_rng = random.Random(_SCENE_SEED)
 
 
def _jitter(nominal: float, pct: float = RAND_PCT) -> float:
    return nominal * _scene_rng.uniform(1.0 - pct, 1.0 + pct)
 
 
APPLE_RADIUS    = _jitter(0.04)
APPLE_MASS      = _jitter(0.15)
TREE_POS        = (_jitter(0.55), _jitter(-0.55), 0.0)
TREE_HEIGHT     = _jitter(1.25)
BRANCH_LENGTH   = _jitter(0.50)
BRANCH_RADIUS   = _jitter(0.04)
BRANCH_ROTATION = _jitter(140.0)
 
# Reach envelope: apple xy is now sampled directly in the right shoulder's
# own polar coordinates (see domain_randomization.randomize_tree_and_apple),
# not derived incidentally from TREE_POS/BRANCH_LENGTH jitter — the old
# scheme routinely placed the apple out of reach (or behind the robot).
# Measured via probe_arm_reach.py: right_shoulder_pitch_link sits at
# env-local (0.0, -0.10, 1.085) and the rigid chain shoulder->elbow->wrist->
# palm sums to ~0.42 m (hard upper bound, before PALM_LOCAL_OFFSET's extra
# ~0.14 m out toward the fingers). If you change the USD/arm or push
# PALM_LOCAL_OFFSET further, re-run probe_arm_reach.py and update REACH_MAX
# to match. REACH_MIN was raised 0.15 -> 0.25: below that the target sits
# close enough to the shoulder that the elbow has to fold in toward the
# torso to reach it, which is what was causing self-collision — reported
# empirically, not derived from a measured self-collision boundary, so
# tighten/loosen it based on what you actually see in play.py.
# REACH_AZIMUTH_CENTER_DEG=-45 keeps the nominal direction this task always
# used (front-right of the robot, matching the old TREE_POS); widen
# REACH_AZIMUTH_SPAN_DEG for a policy that should generalise over more of
# the workspace, narrow it (or watch play.py) if the arm still saturates
# near the arc's edges.
REACH_MIN = 0.25
REACH_MAX = 0.42
REACH_AZIMUTH_CENTER_DEG = -45.0
REACH_AZIMUTH_SPAN_DEG = 70.0

APPLE_STEM_BREAK_FORCE  = _jitter(2.5)
APPLE_STEM_BREAK_TORQUE = _jitter(0.5)
 
# Shared SceneEntityCfgs. preserve_order=True is MANDATORY on _ARM_CFG:
# every reward that takes a per-joint weight list indexes joint_ids in the
# order given here, and without preserve_order SceneEntityCfg silently
# reorders them into articulation order.
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
 
# Palm normal in the palm link's LOCAL frame. Confirmed via run_env.py
# --debug_pose's axis probe: local Z has world-Z component 1.000 at the
# default pose, and stays the largest of the three (0.91 vs 0.4/0.12 for
# X/Y) even under the (0,1,0)-trained policy's converged pose — local Z is
# the true palm normal. The earlier (0,1,0) setting is why the trained palm
# normal pointed sideways (toward the left arm) instead of up: the reward
# was rewarding the wrong local axis for facing the sky.
PALM_NORMAL_AXIS = (0.0, 1.0, 0.0)

# Grasp reference point, in right_hand_palm_link's OWN local frame. The link
# origin sits back near the wrist; probed empirically (probe_palm_offset.py,
# via right_hand_middle_1_link / right_hand_index_1_link relative to the
# palm link at the default pose) that local +X points toward the fingers,
# with the first knuckle ~0.123 m out. Pushed a bit further (0.14) toward
# where the finger pads / contact sensors actually are. Retune by re-running
# the probe against right_hand_*_2_link (fingertip segments) once you have a
# converged policy and want the exact contact point instead of an estimate.
PALM_LOCAL_OFFSET = (0.14, 0.0, 0.0)

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
    soft_joint_pos_limit_factor=0.95,   # top-level ArticulationCfg field, not articulation_props
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
            stiffness=100.0,   # was 40.0
            damping=10.0,      # was 1.0  -> zeta ~0.85 instead of ~0.1
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

    # ── Apple (red sphere, kinematic — reach-only phase) ───────────────────
    # Reach-only phase: the apple is kinematic (tensor-API pose-driven, not
    # affected by gravity/collisions/joints) and simply teleported to the
    # branch-tip target every reset by randomize_tree_and_apple. It stays
    # exactly there for the whole episode — no breakable stem joint needed.
    #
    # We tried a real dynamic apple held by a PhysX FixedJoint to a kinematic
    # anchor body instead (matching tree_utils.StemManager's design, used by
    # run_env.py's manual demo loop). That anchor body correctly held a
    # distinct per-env position throughout (confirmed via direct testing),
    # but the *joint* itself did not — apples across differently-randomized
    # envs still converged to one shared height, pointing at some kind of
    # per-env joint-cooking/replication issue in the GPU physics pipeline
    # that wasn't worth chasing further just to hold the apple still. Revisit
    # this (dynamic apple + working joint, or StemManager's breakable stem)
    # when grasp/manipulation training needs the apple to be a real dynamic
    # body — see tree_utils.py's module docstring for that mechanism.
    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.SphereCfg(
            radius=APPLE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
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
    """7-dim joint position targets: right arm only (reach phase).

    scale=0.5 + clip caps the commanded target at +-1.5 rad from default.
    The previous scale=1.0 with no clip meant the raw Gaussian action sample
    (in radians) was added directly to default_q with nothing bounding it —
    with init_noise_std=1.0 that was +-1 rad jumps (+-3 rad in the tails) at
    50 Hz, which is what drove joints into saturated, contorted poses.

    Dex3 hand action removed: with the apple kinematic_enabled=True, the
    contact/lift rewards are dead, so the 7 hand DOF were pure exploration
    noise for this reach phase. Restore for grasp training:
        right_hand: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["right_hand.*"],
            scale=0.5, use_default_offset=True, clip={".*": (-3.0, 3.0)},
        )
    """
    right_arm: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-3.0, 3.0)},
    )


@configclass
class ObservationsCfg:
    """Observation groups. Policy uses 'policy' group only."""

    @configclass
    class PolicyObs(ObsGroup):
        """Proprioception for the reach phase (no camera, no hand).

        apple_pos_palm and palm_orientation_torso are now both measured from
        right_hand_palm_link, matching the body the reach/orientation rewards
        score against — previously the reaching reward used the wrist while
        palm_position/palm_orientation used the palm, i.e. the reward and part
        of the observation disagreed about which body mattered.

        last_action is required for the action_rate/action_jerk penalties in
        RewardsCfg to be learnable — otherwise the policy is charged for a
        quantity (its own previous command) that it cannot observe.
        """

        right_arm_pos = ObsTerm(
            func=task_mdp.joint_pos_selected,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
        )
        right_arm_vel = ObsTerm(
            func=task_mdp.joint_vel_selected,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
        )

        # Reach-error vector, measured from the palm (same body the reach
        # rewards use — see reach_rewards._palm_target_dist).
        apple_pos_palm = ObsTerm(
            func=task_mdp.object_pos_relative_to_body,
            params={
                "object_cfg": SceneEntityCfg("apple"),
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "local_offset": PALM_LOCAL_OFFSET,
            },
        )

        # Palm orientation relative to torso — same frame reward_palm_normal_up
        # scores against (via world-Z, but torso-relative gives the policy a
        # richer signal than the reward alone).
        palm_orientation_torso = ObsTerm(
            func=task_mdp.body_quat_relative_to_body,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "reference_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        )

        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm"})

        # NOTE: right_hand_pos/vel and fingertip_contacts removed with the
        # hand action for the reach-only phase. Re-add both when grasp
        # training starts (task_mdp.fingertip_contact_forces for the latter).

        def __post_init__(self):
            self.enable_corruption = False  # add noise in Phase 3

    policy: PolicyObs = PolicyObs()


@configclass
class RewardsCfg:
    """Reach-phase reward curriculum.

    Replaces the old 6-term set (reaching / contact / lift / action_penalty /
    palm_position / palm_orientation). That set had two dead terms (contact,
    lift — both require a dynamic apple, but AppleGraspSceneCfg.apple is
    kinematic_enabled=True in this phase), one term that essentially never
    fired (action_penalty = joint_vel_limits at soft_ratio 0.9, only nonzero
    above 1.96 rad/s), and a position reward with no gradient beyond ~15 cm
    (palm_position std=0.05 -> exp(-18) =~ 1.5e-8 at 0.3 m). Orientation was
    the only term with a usable early gradient, and it also scored a FULL
    quaternion target (roll/pitch/yaw all pinned), collectable from anywhere
    in the workspace — so PPO farmed orientation and stopped exploring
    position. That combination is why position never learned.
    """

    # ── Position: three scales, so there is gradient at every distance ───
    reach_progress = RewTerm(
        func=rr.reward_reach_progress,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "offset_z": 0.08,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )
    reach_residual = RewTerm(
        func=rr.penalty_reach_residual,
        weight=-0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "offset_z": 0.08,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )
    reach_coarse = RewTerm(
        func=rr.reward_reach_gaussian,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "offset_z": 0.08,
            "std": 0.25,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )
    reach_fine = RewTerm(
        func=rr.reward_reach_gaussian,
        weight=4.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "offset_z": 0.08,
            "std": 0.04,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # ── Orientation: palm normal -> world up, yaw and roll of the palm
    #    around that normal are left FREE (per your spec: fingers can point
    #    at the horizon or follow the arm's own z-rotation). Gated to zero
    #    beyond 25 cm so it cannot be farmed from anywhere in the workspace.
    palm_up = RewTerm(
        func=rr.reward_palm_normal_up,
        weight=1.5,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "palm_normal_axis": PALM_NORMAL_AXIS,   # verify with play_debug_patch probe
            "std_rad": 0.5,      # ~29 deg basin
            "gate_dist": 0.25,   # metres
            "offset_z": 0.08,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # ── Reason to stop moving once at the target ──────────────────────────
    hold = RewTerm(
        func=rr.reward_reach_hold,
        weight=3.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "pos_tol": 0.04,
            "ori_tol_rad": 0.30,
            "offset_z": 0.08,
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # ── g1_arm_env's R_success (its largest term by far, 700 vs O(1-40)
    #    elsewhere) — one-time bonus paired with the reach_success termination
    #    in TerminationsCfg below (same pos_tol/ori_tol_rad/hold_time there).
    #    hold_time=0.5s at 50 Hz control = 25 steps, so a policy sitting in the
    #    hold neighbourhood the whole time earns 25*3.0=75 from `hold` alone
    #    before this fires — weight=75.0 roughly doubles that on success,
    #    mirroring g1_arm_env's ratio. Tune from the training curve.
    reach_success_bonus = RewTerm(
        func=rr.reward_reach_success,
        weight=75.0,
    )

    # ── Smoothness / anti-vibration ───────────────────────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    action_jerk = RewTerm(func=rr.penalty_action_jerk, weight=-0.05)
    arm_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                            preserve_order=True)},
    )
    brake = RewTerm(
        func=rr.penalty_near_goal_velocity,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                        preserve_order=True),
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "brake_dist": 0.15,
            "offset_z": 0.08,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # ── Posture ─────────────────────────────────────────────────────────
    # Sign-agnostic pull toward home on shoulders + elbow, wrists free —
    # discourages both "elbow to the sky" and "shoulder rolled across the
    # chest" without needing to know which sign of which joint produces
    # them. Weights lifted from g1_arm_policies/config_right.yaml's
    # excursion_weights.
    home_excursion = RewTerm(
        func=rr.penalty_joint_deviation_weighted,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                        preserve_order=True),
            "weights": [1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0],
        },
    )

    # DISABLED (weight=0.0) until you have read the bad pose off
    # play_debug_patch.py's readout and know which joint to box in. Then set
    # the offending entry in `upper` (or `lower`) just inside the angle you
    # want to forbid and raise weight to about -2.0. Joint order:
    #   [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    #    wrist_roll, wrist_pitch, wrist_yaw]
    # Current bounds are just the hardware soft limits (no-op).
    posture_box = RewTerm(
        func=rr.penalty_joint_range_box,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                        preserve_order=True),
            "lower": [-3.09, -2.25, -2.62, -1.05, -1.97, -1.61, -1.61],
            "upper": [ 2.67,  1.59,  2.62,  2.09,  1.97,  1.61,  1.61],
        },
    )

    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                            preserve_order=True)},
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

    # Reach success: palm held in the success neighbourhood for hold_time.
    # Params MUST match reach_rewards.reward_reach_success's RewardsCfg entry
    # (reach_success_bonus below) — same pos_tol/ori_tol_rad/offset_z/
    # palm_normal_axis/hold_time — since that reward reads the bool this
    # termination caches on env._reach_success rather than recomputing it.
    # (apple_lifted_success below is unrelated — a grasp/lift-phase criterion,
    # kept for later, not currently wired to anything.)
    reach_success = DoneTerm(
        func=task_mdp.reach_success,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "pos_tol": 0.04,
            "ori_tol_rad": 0.30,
            "offset_z": 0.08,
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "hold_time": 0.5,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # (Phase 2+) Success: apple lifted and held — grasp/lift phase, unused now.
    # success = DoneTerm(
    #     func=task_mdp.apple_lifted_success,
    #     params={"object_cfg": SceneEntityCfg("apple"),
    #             "target_height": APPLE_INIT_POS[2] + 0.10,
    #             "hold_time": 1.0},
    # )

@configclass
class EventsCfg:
    """Event terms run in DECLARATION ORDER within a mode.

    Split from the old single reset_robot term into two, both mode="reset":
      1. reset_all_exact  — every joint (legs/waist/left-arm/right-arm/hand)
         to EXACT default, EXACT zero velocity. Runs first.
      2. reset_arm_jitter — then scatter ONLY the trained right arm by
         +-0.20 rad (was +-0.05 on ALL joints, and velocity noise was
         +-0.01 instead of exactly 0).

    The old single term was not actually the source of the "arm always
    collapses into the same bad pose" symptom — reset_joints_by_offset with
    no joint_names DOES write every joint back to default+-0.05 every
    episode. That pose is a LEARNED ATTRACTOR: the policy slams to the same
    saturated command within a few hundred ms of every episode start,
    because it never had to learn otherwise (see the reward rewrite above).
    Verify this with play_debug_patch.py's reset printout rather than
    trusting the comment — expect q ~= default +-0.20 rad and |dq| < 0.05
    immediately after every reset.

    The wider +-0.20 rad start scatter (vs the old +-0.05) also matters on
    its own: a narrow start distribution let the policy memorise one
    trajectory from one starting state instead of learning a mapping.
    """

    reset_all_exact = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_arm_jitter = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS),
            "position_range": (-0.20, 0.20),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ── Fixed once per env, at prestartup (before sim.reset() creates the GPU
    #    tensor views — raw USD collider resizes are illegal after that point) ──
    randomize_tree_geometry = EventTerm(
        func=dr.randomize_tree_geometry,
        mode="prestartup",
        params={
            "tree_height_nominal": TREE_HEIGHT,
            "branch_length_nominal": BRANCH_LENGTH,
        },
    )

    randomize_apple_scale = EventTerm(
        func=dr.randomize_apple_scale,
        mode="prestartup",   # raw USD scale write — must happen before sim.reset() creates the GPU tensor views
        params={"object_cfg": SceneEntityCfg("apple"), "nominal_radius": APPLE_RADIUS},
    )

    # Mass randomization uses the tensor API (root_physx_view), so it's safe at
    # any mode — but IsaacLab's own docstring recommends "startup" (once, right
    # after the physics view is created) over "reset" for anything meant to be
    # fixed for the whole run rather than resampled every episode.
    randomize_apple_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,   # isaaclab.envs.mdp built-in
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("apple"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # ── Per-episode variation: whole tree translates + branch rotates, apple
    #    stays attached at the branch tip (see randomize_tree_and_apple) ──────
    randomize_tree = EventTerm(
        func=dr.randomize_tree_and_apple,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "branch_radius_nominal": BRANCH_RADIUS,
            "branch_rotation_deg_nominal": BRANCH_ROTATION,
            "apple_radius_nominal": APPLE_RADIUS,
            "tree_height_nominal": TREE_HEIGHT,
            "branch_length_nominal": BRANCH_LENGTH,
            # Reach envelope around right_shoulder_pitch_link — see
            # probe_arm_reach.py / randomize_tree_and_apple's docstring for
            # how these were derived (measured max chain length ~0.42 m).
            "reach_min": REACH_MIN,
            "reach_max": REACH_MAX,
            "reach_azimuth_center_deg": REACH_AZIMUTH_CENTER_DEG,
            "reach_azimuth_span_deg": REACH_AZIMUTH_SPAN_DEG,
            "offset_z": 0.08,
        },
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
        num_envs=64,        # was 32 — measured <3 GB at 32, 64 is safe on
                            # an 8 GB 4060 and roughly halves wall clock.
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
        self.episode_length_s = 5.0  # was 10.0 — reach ~1.5s + hold ~2s is
                                      # plenty; halving doubles episodes/hour
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
        # No randomisation during play. reset_robot was split into
        # reset_all_exact + reset_arm_jitter in EventsCfg above (see block 7
        # notes) — disable both here, same as the old single reset_robot=None.
        # reset_apple was already dead (never an assigned EventTerm, just an
        # inert triple-quoted string in EventsCfg), removed rather than kept
        # as a no-op assignment.
        #
        # NOTE: with both reset terms off AND episode_length_s forced to
        # 1e6 below, `dones` essentially never fires in play mode (apple is
        # kinematic so apple_dropped can't trigger either) — so
        # play_debug_patch.py's reset-verification block ("RESET -> q=...")
        # will not print here. To actually exercise and verify the reset
        # path, run play.py against AppleGraspEnvCfg (or AppleGrasp-v0)
        # instead of the -Play variant, or temporarily comment out these two
        # lines.
        self.events.reset_all_exact = None
        self.events.reset_arm_jitter = None

        # AppleGraspEnvCfg.__post_init__ sets episode_length_s = 10.0, meant for
        # short RL training rollouts. run_env.py (the interactive viewer used with
        # reach_apple_curobo.py / reach_apple_ik.py / calibrate_reach_apple_curobo.py)
        # uses this PLAY config too, and a silent env reset every 10 s — moving the
        # tree/apple via randomize_tree and teleporting the robot back to its default
        # joint state — was firing in the middle of manual reach testing and
        # multi-second calibration runs, fighting whatever external controller was
        # still streaming targets over DDS. Effectively disable the time_out for
        # interactive use; apple_dropped remains as the only termination.
        self.episode_length_s = 1.0e6


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