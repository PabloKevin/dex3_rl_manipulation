# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
AppleReachSingle-v2 WITH the palm touch input and the touch rewards of the
reaches that worked best (AppleReachTree-v1 -> AppleReachSpindle-v1..v3):
AppleReachSingleTactile-v1.

User, 2026-10-16: the no-touch v2 reach parks short and buzzes; the tree /
spindle reaches -- which felt the apple -- were precise and held. Train
this in parallel as a possible TEACHER: if it is clearly better, distil it
into a no-touch student for the robot (the real palm reading is not
trusted). Same scene, goal and action space as v2, so a student can copy it:

  - kept from v2: the tree / pendulum stems / placement map, the perceived
    apple (held per episode), the waist, the scripted hand (open -> partial
    cup at 10 cm), the braking arm limiter, the enclosure goal + tilted palm
    (enclosed, palm tilt band, fingertips at one depth, fingers vertical),
    the knock-off / push costs, the elbow terms, low arm torque weight.
  - from AppleReachSpindle-v3 (copied from its config, so they ARE those
    terms): the palm_touch_force input (any contact on the palm, in the
    slot the no-touch layouts removed -> 41 inputs = the spindle's 40 +
    hand_phase), gentle_touch_palm (heading-scaled) / gentle_touch_fingers,
    premature / back-of-hand / thumb contact, hold + success + the facing
    terms gated on a gentle palm touch (0.3-6 N), success after 0.5 s, and
    the spindle's smoothing weights (action_rate, arm_vel, brake,
    commanded_accel, limiter_lead).
  - dropped (pregrasp-era additions the spindle never had): settled,
    palm_speed_near_goal, advance_before_height, and v2's target_contact
    (the spindle's thumb / premature contact terms replace it).

Warm start: scripts/warmstart_reach_tree.py from AppleReachSpindle-v3
(40 -> 41 inputs, hand_phase appended with zero weights).
"""

import copy

from isaaclab.utils import configclass

from .apple_reach_single_v2_env_cfg import (
    SINGLE_V2_STANDOFF,
    AppleReachSingleV2EnvCfg,
    RewardsSingleV2Cfg,
)
from .apple_reach_spindle_env_cfg import AppleReachSpindleV3EnvCfg
from .apple_reach_twostage_env_cfg import _set_params

#: spindle terms copied whole (function, weight, params)
_TOUCH_TERMS = ("gentle_touch_palm", "gentle_touch_fingers", "premature_contact", "back_of_hand_contact",
                "thumb_contact")
_SMOOTH_TERMS = ("action_rate", "arm_vel", "brake", "commanded_accel", "limiter_lead")
#: hold / success terms that take the spindle's palm-touch gate
_GATED_TERMS = ("hold", "face_apple_hold")
_GATE_KEYS = ("sensor_cfg", "pressure_min", "pressure_max")


@configclass
class RewardsSingleTactileCfg(RewardsSingleV2Cfg):
    # filled from AppleReachSpindleV3EnvCfg in __post_init__
    gentle_touch_palm = None
    gentle_touch_fingers = None
    premature_contact = None
    back_of_hand_contact = None
    thumb_contact = None
    # not in the spindle reward set
    settled = None
    palm_speed_near_goal = None
    advance_before_height = None
    target_contact = None


@configclass
class AppleReachSingleTactileEnvCfg(AppleReachSingleV2EnvCfg):
    rewards: RewardsSingleTactileCfg = RewardsSingleTactileCfg()

    def __post_init__(self):
        super().__post_init__()
        sp = AppleReachSpindleV3EnvCfg()  # fully post-initialised
        for name in _TOUCH_TERMS + _SMOOTH_TERMS:
            setattr(self.rewards, name, copy.deepcopy(getattr(sp.rewards, name)))
        _set_params(self.rewards, standoff=SINGLE_V2_STANDOFF)
        # the touch gate on hold / success; the position test stays v2's enclosure
        for name in _GATED_TERMS:
            src, dst = getattr(sp.rewards, name), getattr(self.rewards, name)
            dst.params = {**dst.params, **{k: copy.deepcopy(src.params[k]) for k in _GATE_KEYS}}
        rs, src = self.terminations.reach_success, sp.terminations.reach_success
        rs.params = {**rs.params, **{k: copy.deepcopy(src.params[k]) for k in _GATE_KEYS},
                     "hold_time": src.params["hold_time"]}
        # the palm touch input, in its old slot (before neighbour_ring)
        self.observations.policy.palm_touch_force = copy.deepcopy(sp.observations.policy.palm_touch_force)


@configclass
class AppleReachSingleTactileEnvCfg_PLAY(AppleReachSingleTactileEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None
