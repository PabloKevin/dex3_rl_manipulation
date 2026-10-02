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
 
 
APPLE_RADIUS    = 0.04 # _jitter(0.04)
APPLE_MASS      = _jitter(0.15)

# ── Tree: commercial fruiting wall (tall spindle) ──────────────────────────
# A tall vertical trunk (leader) with SHORT, near-horizontal fruiting
# branches; the apple sits at a branch tip on a horizontal stem, facing the
# row, so nothing sits between the hand and the fruit. Replaces the earlier
# model (long branch at the trunk top, apple hanging BELOW it), which is not
# how commercial apple trees present fruit.
#
# The branch points from the trunk (in the wall) back toward the robot, yawed
# by up to +-BRANCH_YAW_SPREAD_DEG to emulate real-tree imperfections, so the
# trunk is BEHIND the apple. Tree, trunk and stem are visual-only. Placement is
# apple-first (domain_randomization.randomize_tree_and_apple): the trunk
# position is derived from the sampled apple position, so it moves around
# much more than with the old long branch.
TRUNK_HEIGHT_RANGE    = (2.0, 2.5)     # m, per env (prestartup)
TRUNK_RADIUS          = 0.06
BRANCH_LENGTH         = 0.12           # m, +-20% per env (prestartup)
BRANCH_RADIUS         = 0.015
# Fruiting wall in front of the robot: fruit sticks out of the wall toward
# it, so the stem (apple -> branch tip) points INTO the wall, along +x, give
# or take BRANCH_YAW_SPREAD_DEG. (An earlier version pointed every branch at
# the shoulder, so stems ran forward-RIGHT for the usual front-right apples;
# the palm-toward-stem reward then asked for an awkward outward-twisted
# right hand, which the policy refused.)
STEM_HEADING_DEG      = 0.0
BRANCH_YAW_SPREAD_DEG = 30.0           # branch/stem yaw spread around STEM_HEADING_DEG
STEM_LENGTH_RANGE     = (0.02, 0.04)   # m, apple surface -> branch tip (matches harvest.StemParams.length)
STEM_RADIUS           = 0.003
 
# Reach envelope of the GRASP TARGET (reach_rewards._grasp_target_pos),
# sampled in the right shoulder's own spherical coordinates around
# right_shoulder_pitch_link (env-local (0.0, -0.10, 1.085)); see
# domain_randomization.randomize_tree_and_apple.
#
# Sized with scripts/probe_side_reach.py (multi-seed kinematic IK on the real
# USD, logs/probe_side_reach_stem.npz) for the side-on pose: palm normal
# parallel to the stem, finger direction free, every non-driven joint held
# at its default. Freeing the fingers matters: with fingers forced up the
# pose only existed for targets at/above shoulder height; with them free the
# arm reaches low apples fingers-down and high ones fingers-up, and the box
# below is 92% reachable (85% with no torso intrusion; a lower bound, IK can
# stall in local minima). Beyond ~0.34 m the folded arm runs out of length.
# Re-run the probe if the arm, USD, PALM_LOCAL_OFFSET or the pose change.
#
# Rule kept from the palm-up task: never sample targets the arm cannot
# reach. Unreachable episodes are guaranteed-zero-contact data points that
# dilute the touch rewards and push the arm to saturate chasing them.
REACH_MIN = 0.24
REACH_MAX = 0.34
# Height of the grasp target above the shoulder. The apple centre sits at the
# same height (horizontal stem), i.e. 0.79-1.37 m above the ground.
REACH_TARGET_DZ_RANGE = (-0.30, 0.28)
# Front-right of the robot, unchanged from the palm-up task (the probe found
# no strong azimuth dependence across -90..+15 deg).
REACH_AZIMUTH_CENTER_DEG = -45.0
REACH_AZIMUTH_SPAN_DEG = 65.0
# Minimum forward (x) and horizontal distance of the grasp target from the
# shoulder (resampled until met). The 3D distance alone lets a high or low
# target sit almost on top of / beside the shoulder: reachable, but only with
# the hand cramped against the body. Kept generous on purpose: on the real
# robot an apple that close is handled by stepping back. With REACH_MIN
# 0.24 the map still accepts ~86% of the box, every height band covered.
REACH_MIN_FORWARD = 0.15
REACH_MIN_HORIZONTAL = 0.18
# Safe reach workspace (scripts/probe_side_reach.py): 5 cm cells of
# grasp-target positions around the shoulder that the arm reaches with the
# palm along the stem and NO self-collision (full-body sphere model, 1.5 cm
# clearance). Placement only samples feasible cells; the box above is just
# the bounding region. Regenerate whenever the arm, hand pose, USD,
# PALM_LOCAL_OFFSET or the collision model change.
REACH_WORKSPACE_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "reach_workspace.npz")

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
 
# Palm normal in the palm link's LOCAL frame (points out of the palm face).
# The reach drives it along the (horizontal) stem direction: palm facing the
# fruit from the side opposite the stem. Where the fingers point (rotation
# about the normal) is left free -- up suits most apples, down or sideways
# suits low or very lateral ones.
PALM_NORMAL_AXIS = (0.0, 1.0, 0.0)

# Grasp reference point, in right_hand_palm_link's OWN local frame. The link
# origin sits back near the wrist; probed empirically (probe_palm_offset.py,
# via right_hand_middle_1_link / right_hand_index_1_link relative to the
# palm link at the default pose) that local +X points toward the fingers,
# with the first knuckle ~0.123 m out. Pushed a bit further (0.14) toward
# where the finger pads / contact sensors actually are. Retune by re-running
# the probe against right_hand_*_2_link (fingertip segments) once you have a
# converged policy and want the exact contact point instead of an estimate.
PALM_LOCAL_OFFSET = (0.08, 0.0, 0.0)

# Where the apple starts touching the palm, measured in PhysX (the palm
# collider is a convex hull of the mesh, so the visual mesh heights don't
# apply): moving a 40 mm apple centred at PALM_LOCAL_OFFSET along the palm
# normal, palm contact begins with its surface 16 mm from the reference
# point (0.2 N); 13 mm gives ~1.2 N, 10 mm ~3 N, and nothing but the palm
# touches. (The previous 20 mm target left the apple 4 mm off the palm: no
# contact even at a perfect pose.) Centred at x = 0.06 instead, the thumb
# base touches as soon as the palm is pressed, so x = 0.08 stays.
_PALM_TOUCH_HEIGHT = 0.016
# How far the target presses past first contact: ~1.2 N on the compliant
# reach-phase apple (AppleGraspSceneCfg.apple).
_TOUCH_PRESS = 0.003

# Gap between the apple SURFACE and the palm reference point along the palm
# normal at the target (reach_rewards._palm_target_dist): apple resting on
# the palm with a light press -- the reach scores contact, not proximity.
GRASP_STANDOFF = _PALM_TOUCH_HEIGHT - _TOUCH_PRESS

# Palm contact force that counts as a gentle touch (hold, success, touch
# reward); above the top a quadratic penalty takes over.
TOUCH_FORCE_MIN = 0.3
TOUCH_FORCE_MAX = 6.0

# The apple is touched from a LATERAL so the harvest motion can be
# horizontal: palm normal within this of horizontal (any heading -- which
# side of the apple is free). Hold and success require it; reward_palm_lateral
# shapes it.
PALM_TILT_TOL_RAD = math.radians(20.0)
# ...and roughly opposite the stem (palm normal toward the stem direction),
# so the hand stays off the stem/branch and out of neighbouring fruit. The
# stem is not observed and the branch yaw varies +-BRANCH_YAW_SPREAD_DEG, so
# this has to be wide: "similar", not equal. Hold and success require it;
# reward_palm_toward_stem shapes it.
PALM_STEM_TOL_RAD = math.radians(45.0)

# Compliant contact of the reach-phase (kinematic) apple: a real apple on its
# stem yields when pushed. Without it a position-controlled arm touching an
# immovable sphere spikes to ~20 N on contact (measured with the previous
# policy), so "gentle" was nearly unreachable. With it the gentle band
# (TOUCH_FORCE_MIN..MAX) spans ~1 cm of press (see _PALM_TOUCH_HEIGHT).
APPLE_CONTACT_STIFFNESS = 1000.0
APPLE_CONTACT_DAMPING = 30.0

# Initial (pre-randomization) placement: nominal geometry at the centre of
# the reach envelope. randomize_tree_and_apple overwrites all of it at reset.
_SHOULDER_LOCAL = (0.0, -0.10, 1.085)
_az0 = math.radians(REACH_AZIMUTH_CENTER_DEG)
_r0 = 0.5 * (REACH_MIN + REACH_MAX) + APPLE_RADIUS + GRASP_STANDOFF
_stem0 = 0.5 * sum(STEM_LENGTH_RANGE)
APPLE_INIT_POS = (_SHOULDER_LOCAL[0] + _r0 * math.cos(_az0), _SHOULDER_LOCAL[1] + _r0 * math.sin(_az0), _SHOULDER_LOCAL[2])
_branch_yaw0 = math.radians(STEM_HEADING_DEG) + math.pi
_bx0, _by0 = math.cos(_branch_yaw0), math.sin(_branch_yaw0)
_tip0 = (APPLE_INIT_POS[0] - _bx0 * (APPLE_RADIUS + _stem0), APPLE_INIT_POS[1] - _by0 * (APPLE_RADIUS + _stem0))
TREE_POS = (_tip0[0] - _bx0 * BRANCH_LENGTH, _tip0[1] - _by0 * BRANCH_LENGTH, 0.0)
_TRUNK_HEIGHT0 = 0.5 * sum(TRUNK_HEIGHT_RANGE)
_YAW_QUAT0 = (math.cos(_branch_yaw0 / 2), 0.0, 0.0, math.sin(_branch_yaw0 / 2))

# Robot root height (metres). G1 pelvis in standing pose ≈ 0.79 m.
# With fix_root_link the pelvis is fixed here in world frame.
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
            "right_hand_thumb_1_joint":  0.6,
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

    # ── Tree: vertical trunk (leader) ────────────────────────────────────
    tree_trunk: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TreeTrunk",
        spawn=sim_utils.CylinderCfg(
            radius=TRUNK_RADIUS,
            height=_TRUNK_HEIGHT0,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.20, 0.08),   # brown bark
                roughness=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TREE_POS[0], TREE_POS[1], _TRUNK_HEIGHT0 / 2.0),
        ),
    )

    # ── Tree: short horizontal fruiting branch (green) ───────────────────
    # From the trunk back toward the robot, at the apple's height.
    tree_branch: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TreeBranch",
        spawn=sim_utils.CylinderCfg(
            radius=BRANCH_RADIUS,
            height=BRANCH_LENGTH,
            axis="X",               # yawed about Z to the branch direction
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.13, 0.42, 0.08),   # green branch
                roughness=0.9,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(_tip0[0] - _bx0 * BRANCH_LENGTH / 2.0, _tip0[1] - _by0 * BRANCH_LENGTH / 2.0, APPLE_INIT_POS[2]),
            rot=_YAW_QUAT0,
        ),
    )

    # ── Stem: horizontal, from the branch tip to the apple surface ───────
    apple_stem: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AppleStem",
        spawn=sim_utils.CylinderCfg(
            radius=STEM_RADIUS,
            height=_stem0,
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.30, 0.22, 0.10),
                roughness=0.9,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(APPLE_INIT_POS[0] - _bx0 * (APPLE_RADIUS + _stem0 / 2.0),
                 APPLE_INIT_POS[1] - _by0 * (APPLE_RADIUS + _stem0 / 2.0), APPLE_INIT_POS[2]),
            rot=_YAW_QUAT0,
        ),
    )

    # ── Apple (red sphere, kinematic — reach-only phase) ───────────────────
    # Reach-only phase: the apple is kinematic (tensor-API pose-driven, not
    # affected by gravity/collisions/joints) and simply teleported to its
    # place at the branch tip every reset by randomize_tree_and_apple. It stays
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
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                compliant_contact_stiffness=APPLE_CONTACT_STIFFNESS,
                compliant_contact_damping=APPLE_CONTACT_DAMPING,
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

    # ── Contact sensors on the right Dex3 hand ───────────────────────────
    #
    # TWO groups, deliberately, because IsaacLab's ContactSensor cannot do
    # both jobs with one sensor:
    #
    #  1. `contact_forces` — one multi-body sensor over the whole hand,
    #     UNFILTERED. Only `net_forces_w` is ever read from it (total force
    #     on each hand link against anything): scripts/run_env.py's tactile
    #     bridge, scripts/tactile_monitor.py, mdp/tree_utils.py and
    #     observations.fingertip_contact_* all use it that way.
    #
    #  2. `contact_forces_<link>` — one SINGLE-body sensor per hand link,
    #     each filtered to the Apple. `force_matrix_w` is read from these.
    #
    # The split is forced by an IsaacLab constraint, and getting it wrong is
    # silent: filtered contact reporting "is only possible as one-to-many
    # [...] only one sensor body in an environment can be filtered against
    # multiple bodies in that environment" (isaaclab/sensors/contact_sensor/
    # contact_sensor.py, and the same warning on ContactSensorCfg.
    # filter_prim_paths_expr). A multi-body prim_path plus a filter — which
    # is exactly what this used to be, `right_hand.*_link` matching all 8
    # links — does NOT raise; `force_matrix_w` is simply allocated at the
    # right shape and stays ZERO forever.
    #
    # That is what killed the first reach training run: every force-based
    # term (reach_rewards.reward_gentle_touch / penalty_excess_palm_force /
    # penalty_apple_contact_force, reward_reach_hold's touch gate, and
    # terminations_and_events.reach_success's matching gate) read a constant
    # 0.0 N, so `hold`, `gentle_touch` and `reach_success_bonus` — the three
    # largest positive terms — were flat zero in TensorBoard for the whole
    # run. The tactile monitor showed real contact the entire time because
    # it reads `net_forces_w`, which is unaffected by this constraint; that
    # disagreement is the signature of this bug, not a calibration problem.
    #
    # Read these through observations.apple_force_per_link /
    # apple_force_vec_per_link rather than indexing them directly — those
    # resolve `contact_forces` + link name -> sensor and keep the naming
    # convention in one place.
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand.*_link",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
        # NO filter_prim_paths_expr: see above. This sensor is net_forces_w only.
    )

    # Per-link, Apple-filtered. Names must stay in sync with
    # observations.link_sensor_name("contact_forces", <link>).
    contact_forces_palm: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Apple"],
    )
    contact_forces_thumb_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_thumb_0_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Apple"],
    )
    contact_forces_thumb_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_thumb_1_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Apple"],
    )
    contact_forces_thumb_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_thumb_2_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Apple"],
    )
    contact_forces_index_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_index_0_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Apple"],
    )
    contact_forces_index_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Apple"],
    )
    contact_forces_middle_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_middle_0_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Apple"],
    )
    contact_forces_middle_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
        update_period=0.0,
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

    scale=0.5 + clip caps the RAW target at +-1.5 rad from default before
    rate-limiting. The previous scale=1.0 with no clip meant the raw
    Gaussian action sample (in radians) was added directly to default_q with
    nothing bounding it — with init_noise_std=1.0 that was +-1 rad jumps
    (+-3 rad in the tails) at 50 Hz, which is what drove joints into
    saturated, contorted poses.

    task_mdp.RateLimitedJointPositionActionCfg (not the stock
    mdp.JointPositionActionCfg) additionally hard-caps the COMMANDED
    trajectory's velocity/acceleration every control step — see
    envs/mdp/actions.py. Added after the arm was observed moving too fast /
    generating large accelerations for safe sim-to-real transfer: the static
    clip above only bounds where the target CAN be, not how fast it's
    allowed to get there, so a policy could still slam between two points
    inside that range every 20 ms.

    Dex3 hand action removed: with the apple kinematic_enabled=True, the
    contact/lift rewards are dead, so the 7 hand DOF were pure exploration
    noise for this reach phase. Restore for grasp training:
        right_hand: task_mdp.RateLimitedJointPositionActionCfg = task_mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot", joint_names=["right_hand.*"],
            scale=0.5, use_default_offset=True, clip={".*": (-3.0, 3.0)},
        )
    """
    right_arm: task_mdp.RateLimitedJointPositionActionCfg = task_mdp.RateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-3.0, 3.0)},
        max_velocity=2.0,
        # 15 -> 8 rad/s^2: the reach policy always asks for targets further
        # than one step can reach, so the limiter runs bang-bang at this cap
        # (measured: commanded |acc| pinned at the cap the whole reach, 3-7
        # direction reversals per joint per second) -- visibly jerky far
        # from the apple. 8 keeps the frozen v1 reach as accurate (75% vs
        # 69% within 2 cm at hand-off, ~0.15 s slower) with half the jerk;
        # 5 broke it (31%). Keep the reach and grasp tasks identical.
        max_acceleration=8.0,
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

        # Palm orientation relative to torso (the orientation rewards score
        # it in world frame against the approach direction / world up; the
        # torso-relative quaternion gives the policy a richer signal).
        palm_orientation_torso = ObsTerm(
            func=task_mdp.body_quat_relative_to_body,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "reference_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        )

        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm"})

        # Palm tactile sensing: "reach" now means arrive AND gently touch
        # (see RewardsCfg.reach_touch_bonus / palm_excess_force below and
        # TerminationsCfg.reach_success's touch gate), so the policy needs
        # to actually sense contact force to learn a controlled final
        # approach rather than a blind position target. Raw/un-gained (see
        # observations.py's own note on why this differs from the grasp
        # phase's calibration-robust tactile terms).
        #
        # IMPORTANT: this changes obs dim 28 -> 29. envs/mdp/grasp_autoplay.
        # py's _reach_obs (used by the grasp task's reach-autoplay hand-off)
        # reassembles this SAME vector by hand and has been updated to match
        # -- if you ever add/reorder a PolicyObs term here again, update
        # _reach_obs's assembly order too, and scripts/reach_apple_hybrid.py's
        # OBS_POLICY_DIM guard.
        palm_touch_force = ObsTerm(
            func=task_mdp.palm_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )

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
            "standoff": GRASP_STANDOFF,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )
    reach_residual = RewTerm(
        func=rr.penalty_reach_residual,
        weight=-0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "standoff": GRASP_STANDOFF,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )
    reach_coarse = RewTerm(
        func=rr.reward_reach_gaussian,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "standoff": GRASP_STANDOFF,
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
            "standoff": GRASP_STANDOFF,
            "std": 0.04,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # ── Orientation: palm facing SIDEWAYS (normal horizontal) so the apple
    #    is touched from a lateral and the harvest motion can be horizontal;
    #    never from above or below. Where the fingers point is free. The
    #    position terms already put the apple centred ON the palm face
    #    (reach_rewards._palm_target_dist), so the normal points at the
    #    apple. Gated to zero beyond gate_dist.
    palm_lateral = RewTerm(
        func=rr.reward_palm_lateral,
        weight=6.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "std_rad": 0.3,      # ~17 deg
            "gate_dist": 0.55,
            "standoff": GRASP_STANDOFF,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # ...and coming from roughly opposite the stem (soft: std ~34 deg; the
    #    stem is not observed, so only the typical direction -- forward, into
    #    the wall, STEM_HEADING_DEG -- is learnable).
    #    Without it the policy touched the apple from the side/trunk side
    #    (palm 120 deg off the stem direction, median) and swept the hand into
    #    the stem/branch in 84% of episodes.
    palm_toward_stem = RewTerm(
        func=rr.reward_palm_toward_stem,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "std_rad": 0.6,
            "gate_dist": 0.55,
            "standoff": GRASP_STANDOFF,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # ── Reason to stop moving once at the target ──────────────────────────
    # sensor_cfg/pressure_min/pressure_max added: "hold" now requires actual
    # gentle palm+thumb-base contact (0.5 < force <= 5.0 N), not just
    # position/orientation -- kept in lockstep with TerminationsCfg.
    # reach_success's own touch gate below (see reward_reach_hold's
    # docstring for why: a mismatch here would let the policy collect this
    # per-step reward without ever bothering to close the last few cm to
    # touch). Band widened from an earlier 0.05-3.0 N attempt: an empirical
    # trace showed contact forces of 15-58 N when the policy did make
    # contact, so a wider, more forgiving "good enough" target (up to 5 N,
    # not 3) gives a decelerating-but-imperfect approach a real chance to
    # land inside it, rather than demanding near-perfect braking.
    hold = RewTerm(
        func=rr.reward_reach_hold,
        weight=5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "pos_tol": 0.04,
            "tilt_tol_rad": PALM_TILT_TOL_RAD,
            # No stem gate here (success has it): hold keeps rewarding a
            # good touch while the heading is still being learned.
            "standoff": GRASP_STANDOFF,
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "palm_local_offset": PALM_LOCAL_OFFSET,
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "pressure_min": TOUCH_FORCE_MIN,
            "pressure_max": TOUCH_FORCE_MAX,
        },
    )

    # ── g1_arm_env's R_success (its largest term by far, 700 vs O(1-40)
    #    elsewhere) — one-time bonus paired with the reach_success termination
    #    in TerminationsCfg below (same pos_tol/tilt_tol_rad/hold_time there).
    #    hold_time=0.5s at 50 Hz control = 25 steps, so a policy sitting in the
    #    hold neighbourhood the whole time earns 25*3.0=75 from `hold` alone
    #    before this fires — weight=75.0 roughly doubles that on success,
    #    mirroring g1_arm_env's ratio. Tune from the training curve.
    reach_success_bonus = RewTerm(
        func=rr.reward_reach_success,
        weight=95.0,
    )

    # ── Smoothness / anti-vibration ───────────────────────────────────────
    # Soft companions to the hard rate limiter on ActionsCfg.right_arm
    # (envs/mdp/actions.py) — that limiter caps the WORST case; these still
    # matter to make the policy actually prefer using less of the available
    # velocity/accel budget rather than habitually pinning against it.
    # arm_vel bumped 10x (-0.002 -> -0.02): at the old weight it was
    # negligible next to reach_progress=10.0 etc., essentially free to move
    # at any speed the whole episode, not just near the goal (that's brake's
    # job, gated to the last 15 cm).
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    action_jerk = RewTerm(func=rr.penalty_action_jerk, weight=-0.08)
    arm_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                            preserve_order=True)},
    )
    # weight raised 15x (-0.02 -> -0.3) then again 4x (-> -2.0): reported
    # symptom was a smooth (non-vibrating) but never-settling drift near the
    # target -- not buzzing, just failing to commit to zero velocity once
    # inside the hold neighbourhood. reward_reach_hold pays out the SAME
    # 3.0/step whether the arm sits dead still or wanders continuously in
    # and out of the tol box, so at the old brake weight there was nothing
    # to actually prefer stillness over drifting -- this is that
    # preference. Raised again after the palm-touch goal was added: an
    # empirical trace showed the arm carrying enough residual velocity into
    # contact to register 30-58 N palm impacts (see penalty_excess_palm_
    # force's own note) despite brake_dist=0.15 already covering this
    # region -- -0.5 wasn't a strong enough deterrent against carrying
    # speed all the way to contact. Watch for over-correction (arm stalling
    # short of the target instead of arriving and stopping) and back this
    # off if that shows up instead.
    brake = RewTerm(
        func=rr.penalty_near_goal_velocity,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                        preserve_order=True),
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "brake_dist": 0.15,
            "standoff": GRASP_STANDOFF,
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

    # Elbow-into-torso guard. Measured via probe_shoulder_roll.py (varied
    # right_shoulder_roll_joint, read right_elbow_link's world-Y relative to
    # the shoulder): negative shoulder_roll swings the elbow AWAY from the
    # torso (elbow_y more negative than shoulder_y=-0.10), positive swings
    # it IN, crossing past the shoulder line (self-collision territory)
    # around +0.5 rad. This is the redundant "swivel" DOF of the arm's IK
    # (fixed palm position+orientation still leaves shoulder_roll free to
    # pick which side the elbow swings to) — PPO was landing on the
    # colliding branch since nothing told it not to. upper[1]=0.35 boxes
    # shoulder_roll well before the ~0.5 rad crossing point, zero cost
    # below that (negative/away-from-torso stays completely free). Only
    # shoulder_roll is boxed -- elbow_joint itself is flexion (bend amount),
    # it doesn't control which side the arm swings to. Joint order:
    #   [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    #    wrist_roll, wrist_pitch, wrist_yaw]
    posture_box = RewTerm(
        func=rr.penalty_joint_range_box,
        weight=-2.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                        preserve_order=True),
            "lower": [-3.09, -2.25, -2.62, -1.05, -1.97, -1.61, -1.61],
            "upper": [ 2.67,  0.35,  2.62,  2.09,  1.97,  1.61,  1.61],
        },
    )

    # Task-space self-collision guard (envs/mdp/self_collision.py): right
    # arm + Dex3 hand spheres vs. the rest of the body (pelvis, legs,
    # torso/head, left arm), from the project's cuRobo spherization plus
    # mesh-sized hand spheres. Replaces the torso-only capsule: low apples
    # bring the hand down by the hips/thighs, high ones up by the head.
    # posture_box above is a cheap joint-space fence on the one DOF that
    # usually picks the colliding IK branch; this catches everything else.
    # Needed because enabled_self_collisions=False means PhysX never stops
    # the arm passing through the body. The reach envelope is restricted to
    # placements reachable with NO intrusion (REACH_WORKSPACE_MAP), so this
    # can always be driven to 0.
    #
    # Weight: `ref`=0.05 m means one sphere buried 5 cm past the required
    # clearance costs 1.0 before the weight, and depth grows smoothly from
    # zero, so -8.0 makes a real collision clearly worse than any posture
    # gain while costing exactly nothing in the clear. The curve should
    # fall to 0 and stay there.
    self_collision = RewTerm(
        func=task_mdp.penalty_self_collision,
        weight=-8.0,
        params={
            "margin": 0.015,  # metres of clearance on top of both radii (= probe)
            "ref": 0.05,      # metres of intrusion that costs 1.0
        },
    )

    # The tree has no colliders (visual prims), so nothing stops the hand
    # passing through the stem/branch/trunk -- one episode ended with the
    # hand between the apple and the trunk. Arm + hand spheres vs. the
    # per-episode stem, branch and trunk capsules (self_collision.py),
    # 1 cm clearance; 2 cm inside costs 1.0 before the weight.
    tree_contact = RewTerm(
        func=task_mdp.penalty_tree_contact,
        weight=-6.0,
        params={"margin": 0.01, "ref": 0.02},
    )

    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS,
                                            preserve_order=True)},
    )

    # Penalises pressing hard into the apple (e.g. the thumb-spearing issue),
    # WITHOUT requiring a contact-force observation -- reward functions see
    # full sim state regardless of ObservationsCfg, so this stays reward-
    # only and the obs vector doesn't grow. Apple-filtered via the per-link
    # contact_forces_* sensors (see their block above), not net_forces_w, so
    # self-collision/tree contact can't trigger it.
    # ref=5.0 N: a light touch costs little (squared penalty), pressing
    # harder costs much more than linearly. Tune weight from the training
    # curve -- start small since this stacks with GRASP_STANDOFF already
    # keeping the hand clear of the apple in the nominal converged pose.
    apple_contact_force = RewTerm(
        func=rr.penalty_apple_contact_force,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "ref": 5.0,
        },
    )

    # Direct, position-independent stepping stone toward touching gently --
    # separate from hold's combined position+orientation+force gate (see
    # reward_gentle_touch's own docstring for why an isolated version of
    # this signal matters). Same 0.5-5.0 N band as hold's own touch gate.
    #
    # Split into TWO terms over two parts of the hand, because not every
    # gentle touch is equally good: the apple settling into the palm / against
    # the thumb is the hand-off pose the grasp phase wants, while catching it
    # on an index or middle finger is a pose the grasp phase then has to undo.
    # The single summed term this replaces paid exactly the same for either,
    # so the policy had no reason to prefer one, and it was drifting to
    # finger contact. The ratio below (4.0 vs 1.0) is the preference; the
    # finger term stays clearly POSITIVE on purpose -- finger contact is
    # still much better than no contact at all, and zeroing it would throw
    # away the stepping stone that got contact working in the first place.
    #
    # Two terms also means two TensorBoard curves, so which part of the hand
    # the policy is settling on is directly readable. Expect palm to climb
    # and fingers to fall; if fingers stays on top, widen the ratio.
    #
    # BOTH terms count only the PALMAR component of the contact force. A
    # ContactSensor fires on a whole rigid body, so the magnitude-only
    # version of this reward paid identically for "apple cupped in the palm"
    # and "apple bumped with the back of the hand" -- and the backhand needs
    # no wrist rotation, so it is far easier to find. That is exactly what
    # the previous run converged to, with palm_up flat at ~0 all the way
    # through, because once the touch reward was collectable without facing
    # the apple nothing else required the palm to point anywhere.
    gentle_touch_palm = RewTerm(
        func=rr.reward_gentle_touch,
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "good_min": TOUCH_FORCE_MIN,
            "good_max": TOUCH_FORCE_MAX,
            # Palm face only: thumb_0 used to be in this group, and the side
            # of the thumb pressed against the side of the apple was paid as
            # a palm touch (see thumb_contact below).
            "link_names": ["right_hand_palm_link"],
        },
    )

    gentle_touch_fingers = RewTerm(
        func=rr.reward_gentle_touch,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "good_min": TOUCH_FORCE_MIN,
            "good_max": TOUCH_FORCE_MAX,
            "link_names": [
                "right_hand_index_0_link",
                "right_hand_middle_0_link",
            ],
        },
    )

    # "Collided on the way in" -- the back-of-the-thumb clipping observed
    # during the approach. See penalty_premature_contact's docstring for why
    # this is gated on DISTANCE rather than on trying to identify the dorsal
    # surface geometrically. Covers all 8 hand links, so it catches the clip
    # whichever surface it lands on.
    #
    # gate_dist WIDENED 0.05 -> 0.10 after the first run with this term.
    # 0.05 was too tight to be safe: the palm reference point is ~0.08 m
    # behind the fingertips (PALM_LOCAL_OFFSET), so during a CORRECT final
    # approach the digits are already brushing the apple while the reference
    # point still reads 0.05-0.08 m from target. The term was therefore
    # taxing the legitimate approach, not just the bad one -- a plausible
    # contributor to the policy abandoning the cupping approach entirely.
    # At 0.10 m only contact made genuinely early, before the hand is
    # anywhere near the cup pose, is charged.
    #
    # Dorsal contact specifically is now handled by back_of_hand_contact
    # below, WITHOUT a distance gate, because it is wrong whenever it
    # happens. This term is left as the weaker "you bumped it on the way in
    # from a long way out" catch-all over both sides of the hand.
    #
    # Weight -3.0: clearly worse than the 4.0 a clean palm touch pays, so
    # bumping your way in never beats going around, while staying well under
    # reach_success's 95 so a single unlucky graze doesn't erase a
    # successful episode.
    #
    # Watch this curve together with gentle_touch_palm: premature_contact
    # should fall toward 0 while gentle_touch_palm rises. If BOTH go to zero
    # the policy has gone timid and is refusing to approach -- lower the
    # weight rather than widening the gate.
    premature_contact = RewTerm(
        func=rr.penalty_premature_contact,
        weight=-3.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "gate_dist": 0.10,
            "ramp": 0.03,
            "cap": 5.0,
            "standoff": GRASP_STANDOFF,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )

    # The apple pressing on the BACK of the hand -- the observed "touches it
    # with the back of the hand / back of the thumb" failure. No distance
    # gate: there is no pose in this task where dorsal contact is correct,
    # so unlike premature_contact this is wrong on its own terms rather than
    # wrong because of when it happened, and gating it would leave a region
    # where backhanding is free again.
    #
    # Pairs with the palmar-only gentle_touch terms above: those remove the
    # INCENTIVE to backhand the apple, this adds an active cost so sweeping
    # the back of the hand through it on the way in is worse than going
    # around rather than merely no better. Weight -4.0 matches what a clean
    # palm touch pays, so the two can't be traded off against each other.
    back_of_hand_contact = RewTerm(
        func=rr.penalty_back_of_hand_contact,
        weight=-4.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "cap": 5.0,
        },
    )

    # Any thumb contact during the reach. The reach ends with the apple on
    # the palm face and the thumb open (thumb_1 at its open limit, held by
    # EventsCfg.hold_default_targets) below it on the wrist side, so a
    # thumb touch always means a misplaced hand -- typically the side of
    # the thumb against the side of the apple, which the old palm+digit-base
    # touch group even rewarded. Weight -3.0: clearly worse than the 4.0 a
    # palm touch pays, so the two cannot be traded.
    thumb_contact = RewTerm(
        func=rr.penalty_apple_contact_links,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "link_names": [
                "right_hand_thumb_0_link",
                "right_hand_thumb_1_link",
                "right_hand_thumb_2_link",
            ],
            "cap": 5.0,
        },
    )

    # Distinct from the soft shaping penalty above (apple_contact_force):
    # gentle contact (0.5-5 N at the palm+thumb-base) is now the GOAL (see
    # gentle_touch/reach_success's touch gate), so a term that costs a
    # little for ANY contact can't be the only force-related signal -- this
    # specifically punishes crossing 5 N, growing quadratically and capped
    # at 20 N (see the function's own docstring for why the cap matters --
    # an uncapped version let one 58 N slam, observed empirically, produce
    # a reward outlier that likely taught outright avoidance instead of
    # gentler contact). Weight intentionally much larger than
    # apple_contact_force's -0.05 -- tune from the training curve, but
    # don't shrink it toward that scale or it stops being a real safety
    # signal.
    palm_excess_force = RewTerm(
        func=rr.penalty_excess_palm_force,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "start": TOUCH_FORCE_MAX,
            "cap": 20.0,
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

    # Reach success: palm held in the success neighbourhood for hold_time.
    # Params MUST match reach_rewards.reward_reach_success's RewardsCfg entry
    # (reach_success_bonus below) — same pos_tol/tilt_tol_rad/standoff/
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
            "tilt_tol_rad": PALM_TILT_TOL_RAD,
            "stem_tol_rad": PALM_STEM_TOL_RAD,
            "standoff": GRASP_STANDOFF,
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "hold_time": 0.5,
            "palm_local_offset": PALM_LOCAL_OFFSET,
            # Success = arrived AND gently touching with the palm, same gate
            # as the per-step hold reward.
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "pressure_min": TOUCH_FORCE_MIN,
            "pressure_max": TOUCH_FORCE_MAX,
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
    # Isaac Lab starts every PD target at 0; hold all joints that no action
    # term drives (hand, waist, left arm, legs) at their defaults.
    hold_default_targets = EventTerm(
        func=task_mdp.hold_default_joint_targets,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
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
            "trunk_height_range": TRUNK_HEIGHT_RANGE,
            "branch_length_nominal": BRANCH_LENGTH,
            "stem_length_range": STEM_LENGTH_RANGE,
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

    # ── Per-episode variation: apple position/height and branch yaw are
    #    sampled, the tree follows (see randomize_tree_and_apple) ───────────
    randomize_tree = EventTerm(
        func=dr.randomize_tree_and_apple,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "apple_radius_nominal": APPLE_RADIUS,
            # Reach envelope of the GRASP TARGET around
            # right_shoulder_pitch_link — see REACH_MAX / REACH_TARGET_DZ_RANGE
            # and randomize_tree_and_apple's docstring.
            "reach_min": REACH_MIN,
            "reach_max": REACH_MAX,
            "reach_azimuth_center_deg": REACH_AZIMUTH_CENTER_DEG,
            "reach_azimuth_span_deg": REACH_AZIMUTH_SPAN_DEG,
            "target_dz_range": REACH_TARGET_DZ_RANGE,
            "reach_min_forward": REACH_MIN_FORWARD,
            "reach_min_horizontal": REACH_MIN_HORIZONTAL,
            "workspace_map": REACH_WORKSPACE_MAP,
            "branch_yaw_spread_deg": BRANCH_YAW_SPREAD_DEG,
            "stem_heading_deg": STEM_HEADING_DEG,
            "standoff": GRASP_STANDOFF,
            "stem_radius": STEM_RADIUS,
            "branch_radius": BRANCH_RADIUS,
            "trunk_radius": TRUNK_RADIUS,
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
        # path, run play.py against AppleGraspEnvCfg (or AppleReach-v1)
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