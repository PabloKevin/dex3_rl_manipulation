# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Two-stage reach on the spindle tree, both stages subclassed from
AppleReachSpindle-v3 (apple_reach_spindle_env_cfg.py) -- same scene, arm +
waist action, observation layout (40 inputs / 8 outputs), so both
warm-start from the v3 reach and run through sim2real/reach_io.py's
"waist" layout.

Why: on the robot the v3 reach knocked the lab's replica apples (magnet
stems) off with its curled fingers before the palm got there. In training
a finger contact inside the last 10 cm cost nothing and the sim stem only
broke at 15-25 N pull / ~5 N sideways.

AppleReachPregrasp-v1 (part 1): home -> PREGRASP: palm point
pregrasp.PREGRASP_STANDOFF (6 cm) from the apple surface along the palm
normal -- at the apple's height, centre ~10 cm away -- palm facing it,
fingers OPEN (the default hand), held still. No touch input (the slot is
kept, fed zero) and any hand contact with the target is charged; success
needs no contact.

AppleReachFinal-v1 (part 2): from where part 1 stops (pregrasp start
library, envs/mdp/pregrasp.py) with the hand in a wider cup
(pregrasp.FINAL_HAND_PRESHAPE, index / middle curled 60 deg instead of 80)
to the v3 goal: the gentle palm touch, with the palm tactile input. Any
index / middle contact with the target is charged at any distance (v3
only charged the thumb, and paid 1.0 for a finger touch).

Both: every fruit on a magnet-like stem (pregrasp.MAGNET_STEM_PARAMS), so
pushing the target or a neighbour off is what it is on the real test tree:
knocked off (target: episode over, -100 x dt).

On the robot: part 1 with the hand open, stop, close the hand to the
part-2 cup, part 2 with the palm sensor (toy_reach.py / reach_io.py need
the hand-off added; not done here).
"""

from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .apple_reach_spindle_env_cfg import (
    AppleReachSpindleV3EnvCfg,
    EventsSpindleV2Cfg,
    RewardsSpindleV3Cfg,
)
import os

import isaaclab.envs.mdp as mdp

from .apple_grasp_env_cfg import PALM_LOCAL_OFFSET, PALM_NORMAL_AXIS, RIGHT_ARM_JOINTS
from .chained_reach_env import stage_touch_obs, staged_reach_success
from .mdp import pregrasp as pg
from .mdp import pregrasp_rewards as pgr
from .mdp import reach_rewards as rr
from .mdp import terminations_and_events as te
from .mdp import tree_reach as trr

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HAND_LINKS = [
    "right_hand_palm_link",
    "right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link",
    "right_hand_index_0_link", "right_hand_index_1_link",
    "right_hand_middle_0_link", "right_hand_middle_1_link",
]
_FINGER_LINKS = [
    "right_hand_index_0_link", "right_hand_index_1_link",
    "right_hand_middle_0_link", "right_hand_middle_1_link",
]
#: per-link target-contact penalties saturate at this force: any real touch
#: (> 0.5 N) costs the full weight, a graze proportionally less
_CONTACT_CAP_N = 0.5


def _set_params(terms, **values) -> None:
    """params[key] = value on every term (reward / termination) that has key.
    A new dict per term, as the spindle cfgs do: never edit a params dict in
    place (it may be shared with the parent task's defaults)."""
    for name in list(vars(terms)):
        t = getattr(terms, name)
        params = getattr(t, "params", None)
        if isinstance(params, dict) and any(k in params for k in values):
            t.params = {**params, **{k: v for k, v in values.items() if k in params}}


# ─────────────────────────────────────────────
#  Part 1: pregrasp (AppleReachPregrasp-v1)
# ─────────────────────────────────────────────

#: palm point within this of the pregrasp point for hold / success (v3: 4 cm)
PREGRASP_POS_TOL = 0.03


@configclass
class RewardsPregraspCfg(RewardsSpindleV3Cfg):
    # touching is not the goal any more
    gentle_touch_palm = None
    gentle_touch_fingers = None
    # replaced by target_contact (any link, any distance)
    premature_contact = None
    thumb_contact = None
    back_of_hand_contact = None
    target_contact = RewTerm(
        func=rr.penalty_apple_contact_links,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "link_names": _HAND_LINKS, "cap": _CONTACT_CAP_N},
    )


@configclass
class EventsPregraspCfg(EventsSpindleV2Cfg):
    # fingers open: the default hand (thumb_1 0.6, the rest 0), no cup
    hand_preshape = None


@configclass
class AppleReachPregraspEnvCfg(AppleReachSpindleV3EnvCfg):
    rewards: RewardsPregraspCfg = RewardsPregraspCfg()
    events: EventsPregraspCfg = EventsPregraspCfg()

    def __post_init__(self):
        super().__post_init__()
        # no touch input (slot kept for the v3 layout)
        self.observations.policy.palm_touch_force = ObsTerm(func=pg.zero_obs, params={"dim": 1})
        # the goal: PREGRASP_STANDOFF in front of the apple, no touch gate
        for terms in (self.rewards, self.terminations):
            _set_params(terms, standoff=pg.PREGRASP_STANDOFF)
        for t in (self.rewards.hold, self.rewards.face_apple_hold, self.terminations.reach_success):
            t.params = {**t.params, "sensor_cfg": None, "pos_tol": PREGRASP_POS_TOL}
        tree = self.events.randomize_tree
        tree.params = {**tree.params, "stem_params": dict(pg.MAGNET_STEM_PARAMS)}


@configclass
class AppleReachPregraspEnvCfg_PLAY(AppleReachPregraspEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None


# ─────────────────────────────────────────────
#  Part 2: final approach (AppleReachFinal-v1)
# ─────────────────────────────────────────────

@configclass
class RewardsFinalCfg(RewardsSpindleV3Cfg):
    # a finger touch is not a stepping stone here: it is what knocks the fruit off
    gentle_touch_fingers = None
    finger_contact = RewTerm(
        func=rr.penalty_apple_contact_links,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "link_names": _FINGER_LINKS,
                "cap": _CONTACT_CAP_N},
    )


@configclass
class EventsFinalCfg(EventsSpindleV2Cfg):
    # the wider cup (targets; set_pregrasp_joints also puts the joints there)
    hand_preshape = EventTerm(func=te.hold_joint_targets, mode="reset",
                              params={"joint_targets": pg.FINAL_HAND_PRESHAPE})
    # LAST reset event: arm, waist, hand to the start place_tree_at_pregrasp chose
    pregrasp_joints = EventTerm(func=pg.set_pregrasp_joints, mode="reset")


@configclass
class AppleReachFinalEnvCfg(AppleReachSpindleV3EnvCfg):
    rewards: RewardsFinalCfg = RewardsFinalCfg()
    events: EventsFinalCfg = EventsFinalCfg()

    def __post_init__(self):
        super().__post_init__()
        # ~5 cm to go: v3's 6 s is mostly idle here
        self.episode_length_s = 4.0
        ev = self.events
        # the library start replaces the home pose + jitter
        ev.reset_arm_jitter = None
        ev.reset_waist_jitter = None
        tree = dict(ev.randomize_tree.params)
        tree["stem_params"] = dict(pg.MAGNET_STEM_PARAMS)
        tree["order"] = "random"
        ev.randomize_tree = EventTerm(func=pg.place_tree_at_pregrasp, mode="reset", params={"tree": tree})


@configclass
class AppleReachFinalEnvCfg_PLAY(AppleReachFinalEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


# ─────────────────────────────────────────────
#  Part 1, v2: arrive smoothly, stop, STAY (AppleReachPregrasp-v2)
# ─────────────────────────────────────────────
# Seen on the v1 policy (model_599, sim, 3 seeds): jerky motion and no settling --
# after 6 s the palm was still 12-14 cm from the apple centre (goal ~10 cm,
# 3 cm tolerance), so the hold never completed. What v1 paid for:
#   hold 5/step only inside the 3 cm ball, brake on shoulder + elbow only
#   (-2, last 15 cm), commanded_accel -3, limiter_lead -1, action_rate -0.2.
# v2 (the goal, the observation and the action layout are unchanged, so the
# v1 weights warm-start it):
#   smoother commanded motion: commanded_accel -3 -> -6, limiter_lead -1 -> -2,
#     action_rate -0.2 -> -0.5, arm_vel -0.02 -> -0.06
#   brake on ALL 7 arm joints (wrists carried the residual motion), -2 -> -6,
#     starting 20 cm out instead of 15
#   palm_speed_near_goal (new, task space): (v / 0.15 m/s)^2 over the last 20 cm
#   settled (new, dense): exp(-(d/3cm)^2) * exp(-(v/5cm/s)^2) per step -- a
#     gradient on arriving AND being still, which the binary hold lacks
#   success needs the palm held 1.0 s (was 0.5 s), reward and termination alike
_PALM = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
_ARM7 = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)
_GOAL = {
    "robot_cfg": _PALM,
    "object_cfg": SceneEntityCfg("apple"),
    "standoff": pg.PREGRASP_STANDOFF,
    "palm_local_offset": PALM_LOCAL_OFFSET,
    "palm_normal_axis": PALM_NORMAL_AXIS,
}
PREGRASP_HOLD_TIME_S = 1.0


@configclass
class RewardsPregraspV2Cfg(RewardsPregraspCfg):
    commanded_accel = RewTerm(func=trr.penalty_commanded_accel, weight=-6.0)
    limiter_lead = RewTerm(func=trr.penalty_limiter_lead, weight=-2.0, params={"ref": 0.5})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    arm_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.06, params={"asset_cfg": _ARM7})
    brake = RewTerm(
        func=rr.penalty_near_goal_velocity,
        weight=-6.0,
        params={"asset_cfg": _ARM7, "brake_dist": 0.20,
                **{k: v for k, v in _GOAL.items() if k != "palm_normal_axis"}},
    )
    palm_speed_near_goal = RewTerm(
        func=pgr.penalty_palm_speed_near_goal,
        weight=-2.0,
        params={"near": 0.20, "v_ref": 0.15, **_GOAL},
    )
    settled = RewTerm(
        func=pgr.reward_settled_at_goal,
        weight=6.0,
        params={"pos_std": 0.03, "v_std": 0.05, **_GOAL},
    )


@configclass
class AppleReachPregraspV2EnvCfg(AppleReachPregraspEnvCfg):
    rewards: RewardsPregraspV2Cfg = RewardsPregraspV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        for terms in (self.rewards, self.terminations):
            _set_params(terms, hold_time=PREGRASP_HOLD_TIME_S)


@configclass
class AppleReachPregraspV2EnvCfg_PLAY(AppleReachPregraspV2EnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None


# ─────────────────────────────────────────────
#  Part 2 trained behind part 1 (AppleReachFinalChained-v1)
# ─────────────────────────────────────────────
# envs/chained_reach_env.py: home pose, hand open, the frozen part-1 policy
# drives until its success (or pregrasp_s), then the hand goes to the cup and
# the learner drives with AppleReachFinal's rewards / success / observation.

FINAL_STAGE_S = 4.0  # part 2's own time budget (AppleReachFinal: episode_length_s)


@configclass
class AppleReachFinalChainedEnvCfg(AppleReachFinalEnvCfg):
    #: "" = newest checkpoint under logs/rsl_rl/apple_reach_pregrasp_v2
    pregrasp_ckpt: str = ""
    #: stage-1 cap (s); the episode is pregrasp_s + FINAL_STAGE_S long
    pregrasp_s: float = 6.0
    #: part 1's success criterion, observed in stage 1 (see ChainedReachEnv)
    pregrasp_success_params: dict = {}

    def __post_init__(self):
        super().__post_init__()
        pre = AppleReachPregraspV2EnvCfg()  # fully post-initialised
        # reset like part 1: home pose + jitter, hand open, tree with magnet stems
        self.events = pre.events
        self.episode_length_s = self.pregrasp_s + FINAL_STAGE_S
        self.pregrasp_success_params = dict(pre.terminations.reach_success.params)
        # tactile input: zero in stage 1, real in stage 2
        t = self.observations.policy.palm_touch_force
        self.observations.policy.palm_touch_force = ObsTerm(
            func=stage_touch_obs, params={"sensor_cfg": t.params["sensor_cfg"]}
        )
        # success: part 2's (touch) in stage 2, part 1's observed in stage 1
        rs = self.terminations.reach_success
        rs.func = staged_reach_success
        rs.params = dict(rs.params)


@configclass
class AppleReachFinalChainedEnvCfg_PLAY(AppleReachFinalChainedEnvCfg):
    """Single env, for watching (play.py / play_two_stage.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None


# ─────────────────────────────────────────────
#  Part 1, v3: rise near the body, then go out level (AppleReachPregrasp-v3)
# ─────────────────────────────────────────────
# User, 2026-10-14: part 1 has no tactile input (the slot is removed: 39
# inputs, trained from scratch); pushing the target or a
# neighbour is charged from the env (fruit positions), and the hand should
# rise close to the body to the fruit's height and only then go out to it,
# through less of the cluster. On top of v2:
#   target_push (new): the target pushed off where it hangs, like
#     neighbour_push (-5): (m beyond 5 mm) / 2 cm per step
#   advance_before_height (new, -3): palm held out beyond BODY_REACH (0.35 m) from the
#     torso while still > 4 cm off the goal's height
#     (full weight from 12 cm); self_collision keeps the hugging honest
#: horizontal palm-to-torso distance allowed before the goal height is reached (m).
#: Measured at the home pose (16 envs, with jitter): 0.27-0.33 m, so 0.35 leaves
#: the start free and charges holding the hand out further while still low/high.
BODY_REACH = 0.35


@configclass
class RewardsPregraspV3Cfg(RewardsPregraspV2Cfg):
    target_push = RewTerm(func=pgr.penalty_target_push, weight=-5.0, params={"deadband": 0.005, "ref": 0.02})
    advance_before_height = RewTerm(
        func=pgr.penalty_advance_before_height,
        weight=-3.0,
        params={"robot_cfg": _PALM, "object_cfg": SceneEntityCfg("apple"), "standoff": pg.PREGRASP_STANDOFF,
                "palm_local_offset": PALM_LOCAL_OFFSET, "body_reach": BODY_REACH, "reach_ref": 0.10,
                "tol": 0.04, "z_ref": 0.08},
    )


@configclass
class AppleReachPregraspV3EnvCfg(AppleReachPregraspV2EnvCfg):
    rewards: RewardsPregraspV3Cfg = RewardsPregraspV3Cfg()

    def __post_init__(self):
        super().__post_init__()
        # no tactile input at all (v1/v2 kept the slot, fed zero): 39 inputs,
        # so v3 trains from scratch -- no warm start from the 40-input weights.
        # sim2real/reach_io.py's "waist" layout (40) needs a 39-input variant
        # before this runs on the robot.
        self.observations.policy.palm_touch_force = None


@configclass
class AppleReachPregraspV3EnvCfg_PLAY(AppleReachPregraspV3EnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None


# ─────────────────────────────────────────────
#  Part 2, v2: tiered touch (AppleReachFinalChained-v2)
# ─────────────────────────────────────────────
# User, 2026-10-14: the end of the reach may touch with the BASE phalanx of
# any finger (fruit near the middle of the closed hand is a fine grasp
# start); palm alone pays most, palm + base phalanx less, base phalanx alone
# less again; the tip phalanges pay nothing. Replaces gentle_touch_palm (4.0)
# with touch_tiered (4.0 x 1.0 / 0.7 / 0.4). The finger / thumb contact
# penalties keep charging the TIPS only (a tip touch is what knocked the
# fruit off on the robot); base phalanges are no longer charged.
# Hold / success still need the palm touch (unchanged).
_BASE_PHALANX = ["right_hand_index_0_link", "right_hand_middle_0_link",
                 "right_hand_thumb_0_link", "right_hand_thumb_1_link"]
_TIP_PHALANX = ["right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"]


@configclass
class RewardsFinalV2Cfg(RewardsFinalCfg):
    gentle_touch_palm = None
    touch_tiered = RewTerm(
        func=pgr.reward_contact_tiered,
        weight=4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "palm_links": ["right_hand_palm_link"],
                "base_links": _BASE_PHALANX, "good_min": 0.5, "good_max": 5.0,
                "palm_only": 1.0, "palm_and_base": 0.7, "base_only": 0.4},
    )
    finger_contact = RewTerm(
        func=rr.penalty_apple_contact_links,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "link_names": _TIP_PHALANX[:2],
                "cap": _CONTACT_CAP_N},
    )
    thumb_contact = RewTerm(
        func=rr.penalty_apple_contact_links,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "link_names": _TIP_PHALANX[2:], "cap": 5.0},
    )


@configclass
class AppleReachFinalChainedV2EnvCfg(AppleReachFinalChainedEnvCfg):
    rewards: RewardsFinalV2Cfg = RewardsFinalV2Cfg()
    #: part 1 = the newest AppleReachPregrasp-v3 checkpoint
    pregrasp_log_root: str = os.path.join(_REPO, "logs", "rsl_rl", "apple_reach_pregrasp_v3")

    def __post_init__(self):
        super().__post_init__()
        pre = AppleReachPregraspV3EnvCfg()
        self.events = pre.events
        self.pregrasp_success_params = dict(pre.terminations.reach_success.params)


@configclass
class AppleReachFinalChainedV2EnvCfg_PLAY(AppleReachFinalChainedV2EnvCfg):
    """Single env, for watching (scripts/play_chained.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None
