# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach with the waist, facing the apple, on a TALL-SPINDLE tree
(AppleReachSpindle-v1): AppleReachTreeWaistFace-v1
(apple_reach_tree_waist_env_cfg.py -- subclassed) on the tree shape of the
real orchard, plus the fixes for the hover before the touch.

Tree (apple_tree.sample_spindle_tree): one ~6 cm trunk, vertical up to a
kink at 0.5-0.9 m, then straight, leaning 3-15 deg in a random direction up
to its 2.0-2.5 m top; every apple on a 1-5 cm branch plus its stem, on the
aisle side, stems into the row within +-30 deg. The trunk now stands
~10-15 cm behind every fruit (a real obstacle for the fingers), and the 7
apples stack in clusters along it. The old fruiting wall had 10-30 cm
branches running back to a trunk 0.65 m from the shoulder.

Hover fixes (the face run reached the apple by ~1.3 s and succeeded by
~2.6 s; 14% hovered 2-4 cm off for seconds):
  - brake (near-goal joint speed) on shoulder + elbow only: it charged the
    wrists too, so the last palm-heading correction crept;
  - gentle_touch_palm scaled by palm heading (floor 0.5): a side-on touch
    paid in full, so touching first and turning the palm slowly in contact
    was the comfortable path;
  - reach_touch: a narrow (1.2 cm) position term -- the touch rewards are
    on/off and the 4 cm reach_fine is nearly flat at 2 cm, so nothing pulled
    the last 2 cm.
"""

from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from .apple_grasp_env_cfg import (
    GRASP_STANDOFF,
    PALM_LOCAL_OFFSET,
    PALM_NORMAL_AXIS,
    PALM_STEM_TOL_RAD,
    RIGHT_ARM_JOINTS,
    RewardsCfg as ReachRewardsCfg,
)
from .apple_reach_tree_waist_env_cfg import (
    AppleReachTreeWaistFaceEnvCfg,
    EventsCfg,
    HOLD_HEADING_ZERO_RAD,
    RewardsFaceCfg,
)
from .mdp import reach_rewards as rr
from .mdp import terminations_and_events as te
from .mdp import tree_reach as trr
from .mdp import waist_reach as wr

#: Trunk radius (~6 cm trunk).
SPINDLE_TRUNK_RADIUS = 0.03
SPINDLE = {
    "kink_range": (0.5, 0.9),       # m: vertical up to here (about half the robot's height)
    "lean_deg": (3.0, 15.0),        # lean of the upper segment, random direction
    "branch_range": (0.01, 0.05),   # m: branch, bark -> stem tip
    "trunk_radius": SPINDLE_TRUNK_RADIUS,
    "top_prim": "TreeTrunkTop",
}

_REACH = ReachRewardsCfg()


@configclass
class RewardsSpindleCfg(RewardsFaceCfg):
    # shoulder + elbow only (was all 7 arm joints): the wrists finish the
    # palm heading near the apple without being slowed
    brake = RewTerm(
        func=rr.penalty_near_goal_velocity,
        weight=_REACH.brake.weight,
        params={**_REACH.brake.params,
                "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS[:4], preserve_order=True)},
    )
    gentle_touch_palm = RewTerm(
        func=wr.reward_gentle_touch_heading,
        weight=_REACH.gentle_touch_palm.weight,
        params={**_REACH.gentle_touch_palm.params, "object_cfg": SceneEntityCfg("apple"),
                "heading_full_rad": PALM_STEM_TOL_RAD, "heading_zero_rad": HOLD_HEADING_ZERO_RAD,
                "floor": 0.5},
    )
    # pulls the last 2 cm onto the palm (target = touching, ~1.2 N)
    reach_touch = RewTerm(
        func=rr.reward_reach_gaussian,
        weight=3.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "standoff": GRASP_STANDOFF,
            "std": 0.012,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )


@configclass
class AppleReachSpindleEnvCfg(AppleReachTreeWaistFaceEnvCfg):
    rewards: RewardsSpindleCfg = RewardsSpindleCfg()

    def __post_init__(self):
        super().__post_init__()
        sc = self.scene
        # lower trunk segment: the scene's trunk, thinner; upper: its own prim
        sc.tree_trunk = sc.tree_trunk.replace(spawn=sc.tree_trunk.spawn.replace(radius=SPINDLE_TRUNK_RADIUS))
        sc.tree_trunk_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TreeTrunkTop",
            spawn=sim_utils.CylinderCfg(
                radius=SPINDLE_TRUNK_RADIUS,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.20, 0.08), roughness=1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 1.5)),
        )
        self.events.randomize_tree.params.update({"spindle": dict(SPINDLE), "trunk_radius": SPINDLE_TRUNK_RADIUS})


@configclass
class AppleReachSpindleEnvCfg_PLAY(AppleReachSpindleEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None


# ─────────────────────────────────────────────
#  v2: stop pushing the neighbours on the way in (AppleReachSpindle-v2)
# ─────────────────────────────────────────────
# The first spindle reach (2026-10-11_02-03-47) pushed a neighbour > 5 mm in
# 83% of episodes (> 2 cm in 57%): thumb tip nearest in 53% of first pushes,
# 69% of the pushed fruit above/below the target (82% of those below), the
# palm a median 4 cm off the target's height when it happened; 19% of the
# pushed fruit hung beyond the ring's 10 cm.
#: Clearance ring range: 10 -> 20 cm (fingers reach ~10 cm past the palm
#: point, and the hand sweeps up/down past the fruit on its way in).
SPINDLE_V2_RING_RANGE = 0.20


@configclass
class RewardsSpindleV2Cfg(RewardsSpindleCfg):
    # -2 -> -5: the one term that charges an actual push
    neighbour_push = RewTerm(func=trr.penalty_neighbour_push, weight=-5.0,
                             params={"deadband": 0.005, "ref": 0.02})
    # hand spheres within 1 cm of a neighbour, before it moves
    hand_neighbour_clearance = RewTerm(func=trr.penalty_hand_neighbour_clearance, weight=-1.0,
                                       params={"margin": 0.01, "ref": 0.02})
    # hand's long axis (fingers / forearm) lined up with close neighbours:
    # fingers sideways in a vertical cluster, free when alone
    fingers_toward_neighbours = RewTerm(
        func=trr.penalty_fingers_toward_neighbours,
        weight=-1.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "object_cfg": SceneEntityCfg("apple"), "standoff": GRASP_STANDOFF,
                "palm_local_offset": PALM_LOCAL_OFFSET, "palm_normal_axis": PALM_NORMAL_AXIS,
                "near": 0.15, "gate_near": 0.05, "gate_far": 0.20},
    )
    # the elbow / forearm / wrist pushing fruit on the way in (seen in the
    # v2 run; the hand-only terms do not see the arm)
    arm_fruit_clearance = RewTerm(func=trr.penalty_arm_fruit_clearance, weight=-3.0,
                                  params={"margin": 0.015, "ref": 0.02})
    # level with the fruit before closing in. Only within 12 cm horizontally:
    # with 20 cm the start pose already paid for near targets (palm at home
    # 30-50 cm below, < 20 cm out) -- -2.4 per episode, more than every push
    # term together.
    height_first = RewTerm(
        func=trr.penalty_height_first,
        weight=-1.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "object_cfg": SceneEntityCfg("apple"), "standoff": GRASP_STANDOFF,
                "palm_local_offset": PALM_LOCAL_OFFSET, "tol": 0.02, "ref": 0.08, "near": 0.04, "far": 0.12},
    )


#: Hand pre-shape held during the reach (user, 2026-10-11): index + middle
#: base joints straight, their middle joints curled to 80 deg; thumb base a
#: little more closed than the grasp's pre-shape (thumb_1 0.15 = 26 deg from
#: open), thumb_2 20% (-0.34), thumb_0 0. The grasp opens the middle joints
#: again before closing the bases. Measured on the real collision meshes
#: (apples 32-48 mm landing at first palm touch): every size clear with up to
#: 1 cm reach error toward the fingertips (1.5 cm: apples >= 46 mm touch);
#: middle 90 deg already touched apples >= 41 mm at 1 cm, and with the base
#: curled 13 deg a centred apple >= 40 mm. Thumb_1 0.0 touches centred apples
#: >= 46 mm; a fruit landing 1.5 cm toward the wrist touches the thumb (>= 39
#: mm) in every thumb pose (it sits there, standing 57 mm out even open).
REACH_HAND_PRESHAPE = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.15,
    "right_hand_thumb_2_joint": -0.340,
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 1.396,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_middle_1_joint": 1.396,
}


@configclass
class EventsSpindleV2Cfg(EventsCfg):
    # after hold_default_targets (declaration order): the hand closes into
    # the cup at the start of every reach
    hand_preshape = EventTerm(func=te.hold_joint_targets, mode="reset",
                              params={"joint_targets": REACH_HAND_PRESHAPE})


@configclass
class AppleReachSpindleV2EnvCfg(AppleReachSpindleEnvCfg):
    rewards: RewardsSpindleV2Cfg = RewardsSpindleV2Cfg()
    events: EventsSpindleV2Cfg = EventsSpindleV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        ring = self.observations.policy.neighbour_ring
        ring.params = {**ring.params, "max_range": SPINDLE_V2_RING_RANGE}


@configclass
class AppleReachSpindleV2EnvCfg_PLAY(AppleReachSpindleV2EnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None


# ─────────────────────────────────────────────
#  v3: + motor load / energy (AppleReachSpindle-v3)
# ─────────────────────────────────────────────
#: Rated torque (N*m, URDF effort) of RIGHT_ARM_JOINTS, in that order, and of
#: the waist yaw.
ARM_RATED_TORQUE = [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0]
WAIST_RATED_TORQUE = [88.0]


@configclass
class RewardsSpindleV3Cfg(RewardsSpindleV2Cfg):
    # Energy: (torque / rated)^2 per motor, arm and waist as separate terms
    # (same weight) so their costs can be compared in TensorBoard. Measured
    # on the v2 policy (targets clamped): arm ~0.17 per step (shoulder pitch
    # 0.039, wrist pitch / yaw 0.04 / 0.07 -- small motors, most loaded
    # relative to their rating), waist 0.0007 (no gravity load). -3: about
    # -0.5 per episode, felt without outweighing the push / touch terms.
    motor_load_arm = RewTerm(
        func=wr.penalty_motor_load, weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True),
                "rated_torque": ARM_RATED_TORQUE})
    motor_load_waist = RewTerm(
        func=wr.penalty_motor_load, weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"], preserve_order=True),
                "rated_torque": WAIST_RATED_TORQUE})


#: Arm targets clamped to the soft joint limits (Isaac Lab's: 95% of the
#: URDF range about its middle). The v2 policy commanded the wrist yaw past
#: its limit in 46% of steps, pressing it into the stop at the actuator cap
#: (87 N*m in sim; the real motor is rated 5 N*m) -- the old +-3 rad clip
#: around default never stopped that.
ARM_TARGET_CLIP = {
    "right_shoulder_pitch_joint": (-2.945, 2.526),
    "right_shoulder_roll_joint": (-2.156, 1.492),
    "right_shoulder_yaw_joint": (-2.487, 2.487),
    "right_elbow_joint": (-0.969, 2.016),
    "right_wrist_roll_joint": (-1.874, 1.874),
    "right_wrist_pitch_joint": (-1.534, 1.534),
    "right_wrist_yaw_joint": (-1.534, 1.534),
}


@configclass
class AppleReachSpindleV3EnvCfg(AppleReachSpindleV2EnvCfg):
    rewards: RewardsSpindleV3Cfg = RewardsSpindleV3Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.right_arm.clip = dict(ARM_TARGET_CLIP)


@configclass
class AppleReachSpindleV3EnvCfg_PLAY(AppleReachSpindleV3EnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None
