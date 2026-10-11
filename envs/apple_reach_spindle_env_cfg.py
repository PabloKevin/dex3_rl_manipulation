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

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
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
    HOLD_HEADING_ZERO_RAD,
    RewardsFaceCfg,
)
from .mdp import reach_rewards as rr
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
