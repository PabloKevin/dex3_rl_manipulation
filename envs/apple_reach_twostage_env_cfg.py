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
from .mdp import pregrasp as pg
from .mdp import reach_rewards as rr
from .mdp import terminations_and_events as te

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
