# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach that ends with the apple ENCLOSED in the hand (AppleReachSingle-v2).

User, 2026-10-15: AppleReachSingle-v1 (apple_reach_single_env_cfg.py --
subclassed: no tactile input, waist, perceived apple, scripted hand) stopped
2 cm short of the apple with any touch charged, so the grasp had to finish
the reach (AppleGraspSingle-v2: 6-21 cm). Now the reach puts the apple in
the hand and the grasp only closes (AppleGraspSingle-v3). Same 40 inputs /
8 outputs: warm-starts from the v1 weights.

  - goal: the apple centre in the pocket -- the palm touching it
    (single_reach.enclosure_state: radius + 16 mm out along the palm normal
    = GRASP_CENTER_LOCAL), a push of up to 1.5 cm accepted; hold / success
    / the grasp hand-off all test single_reach.enclosed (set_reach_goal_v2
    switches reach_rewards.reach_neighbourhood). + reward_enclosed (dense):
    1 cm short pays 0.61, 3 cm short ~0.
  - contact: fingertips charged at any force; palm / base phalanx free up to
    2 N (a light touch, even a ~1.5 cm push on the swinging stem);
    target_push only beyond 1.5 cm.
  - stems: every fruit hangs like a pendulum (k_lateral 15-40 N/m, ~mg/L for
    a 7 cm stem): a 1.5 cm push is 0.2-0.6 N, under the magnet's 0.5-1.5 N
    shear limit; harder pushes knock it off. GUESSES: measure the lab's
    magnet apples (spring scale) and put the numbers in PENDULUM_STEM_PARAMS.
  - knocked off: the target -30 (ends the episode), a neighbour -10, each
    about a whole good episode's return (v1: -2 each, -100 x dt).
  - palm tilt: all three fingertips at ONE depth along the approach
    (reward_fingertip_depth_equal; measured on the URDF: with the palm normal
    level the thumb tip stands 6.9 cm ahead of the curled index / middle
    tips, the first thing to hit the trunk; ~30 deg forward lean brings
    them level). For that: the tilt is free up to 40 deg (palm_lateral ->
    reward_palm_tilt_band, hold / success tilt gate 20 -> 40 deg), the
    heading terms measure the horizontal heading only, fingers_vertical
    only the roll about the approach line.
  - hand: open, then the partial cup once the palm point is within 10 cm
    (3D) of the perceived apple (v1: once level with it).
  - energy: motor_load_arm -3 -> -8, elbow_out -1 -> -2.
  - placement: the re-measured workspace for THIS end pose
    (scripts/probe_side_reach.py --single_v2 -> REACH_SINGLE_V2_WORKSPACE_MAP,
    apple centres, not grasp targets), and STRETCH_FRAC of the episodes put
    the target in a cell just beyond it (logged apart:
    Curriculum/reach_metrics/stretch_*).
"""

import math
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .apple_grasp_env_cfg import PALM_LOCAL_OFFSET, PALM_NORMAL_AXIS, RIGHT_ARM_JOINTS
from .apple_reach_single_env_cfg import (
    AppleReachSingleEnvCfg,
    EventsSingleCfg,
    FINGER_SWITCH_DZ,
    HAND_CLOSED,
    HAND_OPEN,
    RewardsSingleCfg,
)
from .apple_reach_spindle_env_cfg import ARM_RATED_TORQUE
from .apple_reach_twostage_env_cfg import _set_params
from .mdp import pregrasp as pg
from .mdp import pregrasp_rewards as pgr
from .mdp import single_reach as sr
from .mdp import tree_reach as trr
from .mdp import waist_reach as wr

#: palm point -> apple surface at the touch (the palm collider starts 16 mm
#: out, apple_grasp_env_cfg GRASP_STANDOFF notes); every inherited position
#: term (reach_fine / reach_touch / settled / brake ...) aims here
SINGLE_V2_STANDOFF = 0.016
#: enclosure test (single_reach.enclosed): pressed in at most PUSH_IN, short
#: at most SHORT_TOL, off the palm normal less than LATERAL_TOL, the target
#: dragged less than MAX_DISP from where it hangs
SINGLE_V2_PUSH_IN = 0.015
SINGLE_V2_SHORT_TOL = 0.008
SINGLE_V2_LATERAL_TOL = 0.02
SINGLE_V2_MAX_DISP = 0.02
#: palm tilt allowed (hold / success gate, and free band of palm_lateral)
SINGLE_V2_TILT_TOL = math.radians(40.0)
#: hand closes once the palm point is this close (3D) to the perceived apple
SINGLE_V2_HAND_NEAR = 0.10
#: one-off penalties, in return units (the knock functions pay 1 per event and
#: every term is multiplied by dt: weight = value / dt)
SINGLE_V2_STEP_DT = 0.02
TARGET_KNOCKED_COST = 30.0
NEIGHBOUR_KNOCKED_COST = 10.0

#: magnet attachment (pregrasp.MAGNET_STEM_PARAMS) on a stem that swings: the
#: default lateral 100-300 N/m knocked the fruit off at a 0.5 cm push
PENDULUM_STEM_PARAMS = {**pg.MAGNET_STEM_PARAMS, "k_lateral": (15.0, 40.0)}

#: REACH-only retune (2026-10-16, model_1999 of the first run held badly and
#: ended far from the apple, even before closing the hand): pushing up to
#: 5 cm is cheap, so the hand can finish IN the pocket instead of hovering
#: short of it; the stem moves as easily (same k_lateral) but needs a much
#: harder pull / shear to come off, so a push does not end the episode. The
#: grasp v3 task keeps the SINGLE_V2_* values / PENDULUM_STEM_PARAMS above.
REACH_PUSH_IN = 0.05
REACH_MAX_DISP = 0.06
REACH_STEM_PARAMS = {**PENDULUM_STEM_PARAMS, "break_pull": (3.0, 6.0), "break_shear": (2.0, 4.0),
                     "break_bend": (0.15, 0.30)}

#: arm command limiter (2026-10-16). model_1999 ran at the 2.0 rad/s cap
#: (palm peak ~1.2 m/s) and, without brake_to_target, the bang-bang command
#: overshoots and rings: the hold buzzed at ~4.5 Hz (commanded_accel the
#: largest penalty while parked). Braking + slower caps, as the waist has.
#: sim2real/reach_io.SINGLE_V2 must carry the same three numbers.
REACH_ARM_MAX_VELOCITY = 1.2
REACH_ARM_MAX_ACCELERATION = 5.0
REACH_ARM_BRAKE = True

#: placement: APPLE CENTRES relative to the right shoulder (waist yaw 0) on the
#: map scripts/probe_side_reach.py --single_v2 measured; the box is wide, the
#: map decides
REACH_SINGLE_V2_WORKSPACE_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                             "reach_workspace_single_v2.npz")
SINGLE_V2_ENVELOPE = dict(
    reach_min=0.20, reach_max=0.70, az_center_deg=-25.0, az_span_deg=200.0, dz_range=(-0.45, 0.45),
    min_forward=0.05, min_horizontal=0.15, apple_ref=True)
#: share of episodes whose target sits just beyond the map (1 cell = 5 cm)
STRETCH_FRAC = 0.10

_PALM = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
_APPLE = SceneEntityCfg("apple")
_GOAL = {"robot_cfg": _PALM, "object_cfg": _APPLE, "palm_local_offset": PALM_LOCAL_OFFSET,
         "palm_normal_axis": PALM_NORMAL_AXIS}
_TIP_LINKS = ["right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"]
_SOFT_LINKS = ["right_hand_palm_link", "right_hand_index_0_link", "right_hand_middle_0_link",
               "right_hand_thumb_0_link", "right_hand_thumb_1_link"]


@configclass
class RewardsSingleV2Cfg(RewardsSingleCfg):
    # the apple in the hand
    enclosed = RewTerm(func=sr.reward_enclosed, weight=12.0,
                       params={**_GOAL, "touch_gap": SINGLE_V2_STANDOFF, "push_in": REACH_PUSH_IN,
                               "short_std": 0.01, "lateral_std": 0.015})
    target_contact = RewTerm(
        func=sr.penalty_target_contact_v2, weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "tip_links": _TIP_LINKS, "soft_links": _SOFT_LINKS,
                "tip_cap": 0.5, "soft_free": 2.0, "soft_ref": 2.0})
    target_push = RewTerm(func=pgr.penalty_target_push, weight=-3.0,
                          params={"deadband": REACH_PUSH_IN, "ref": 0.03})
    # dropping fruit
    target_knocked = RewTerm(func=trr.penalty_target_knocked, weight=-TARGET_KNOCKED_COST / SINGLE_V2_STEP_DT)
    neighbour_knocked = RewTerm(func=trr.penalty_neighbour_knocked, weight=-NEIGHBOUR_KNOCKED_COST / SINGLE_V2_STEP_DT)
    # the tilted palm: tips at one depth, tilt free up to 40 deg, finger roll only
    palm_lateral = RewTerm(func=sr.reward_palm_tilt_band, weight=6.0,
                           params={**_GOAL, "standoff": SINGLE_V2_STANDOFF, "free_rad": SINGLE_V2_TILT_TOL,
                                   "std_rad": 0.2, "gate_dist": 0.55})
    fingertip_depth_equal = RewTerm(func=sr.reward_fingertip_depth_equal, weight=6.0,
                                    params={**_GOAL, "standoff": SINGLE_V2_STANDOFF, "std": 0.015,
                                            "gate_dist": 0.25})
    fingers_vertical = RewTerm(
        func=sr.reward_fingers_vertical, weight=8.0,
        params={"robot_cfg": _PALM, "object_cfg": _APPLE, "standoff": SINGLE_V2_STANDOFF,
                "palm_local_offset": PALM_LOCAL_OFFSET, "palm_normal_axis": PALM_NORMAL_AXIS,
                "switch_dz": FINGER_SWITCH_DZ, "band": 0.03, "std_rad": 0.4, "gate_dist": 0.55, "roll_only": True})
    # dense centring: enclosed's 1.5 cm lateral gaussian gave ~0 at the 5 cm
    # off-centre model_1999 parked at, where the orientation terms (gated
    # linearly over 55 cm) already paid ~90%
    enclosed_coarse = RewTerm(func=sr.reward_enclosed, weight=6.0,
                              params={**_GOAL, "touch_gap": SINGLE_V2_STANDOFF, "push_in": REACH_PUSH_IN,
                                      "short_std": 0.03, "lateral_std": 0.04})
    # the elbow OUT of the torso, not pointing at the left arm (user,
    # 2026-10-16); the old elbow_out only stops an extreme chicken wing now
    elbow_in = RewTerm(func=sr.penalty_elbow_in, weight=-3.0, params={"margin": 0.0, "ref": 0.05})
    # energy barely charged for now: the -8 made it take odd paths and stop
    # short; fine-tune the torque once it reaches well
    elbow_out = RewTerm(func=sr.penalty_elbow_out, weight=-1.0, params={"margin": 0.20, "ref": 0.10})
    motor_load_arm = RewTerm(
        func=wr.penalty_motor_load, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True),
                "rated_torque": ARM_RATED_TORQUE})


@configclass
class EventsSingleV2Cfg(EventsSingleCfg):
    reach_goal_v2 = EventTerm(func=sr.set_reach_goal_v2, mode="startup",
                              params={"touch_gap": SINGLE_V2_STANDOFF, "push_in": REACH_PUSH_IN,
                                      "short_tol": SINGLE_V2_SHORT_TOL, "lateral_tol": SINGLE_V2_LATERAL_TOL,
                                      "max_disp": REACH_MAX_DISP, "heading_horizontal": True})
    hand_schedule = EventTerm(
        func=sr.step_hand_schedule, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        params={"open_targets": HAND_OPEN, "closed_targets": HAND_CLOSED, "robot_cfg": _PALM,
                "object_cfg": _APPLE, "palm_local_offset": PALM_LOCAL_OFFSET,
                "level_tol": 0.03, "ramp_s": 0.3, "trigger": "distance", "near": SINGLE_V2_HAND_NEAR},
    )


@configclass
class CurriculumSingleV2Cfg:
    #: logs only (Curriculum/reach_metrics/<key>), changes nothing
    reach_metrics = CurrTerm(func=sr.reach_v2_metrics, params=dict(_GOAL))


def apply_single_v2_placement(randomize_tree: EventTerm) -> None:
    """The v2 tree: pendulum stems, the re-measured map (apple centres), the
    stretch share. Shared with AppleGraspSingle-v3."""
    env_seed = {**SINGLE_V2_ENVELOPE, "stretch_frac": STRETCH_FRAC, "stretch_cells": 1}
    env_member = {**SINGLE_V2_ENVELOPE}
    randomize_tree.params.update({
        "seed_envelope": env_seed, "member_envelope": env_member,
        "workspace_map": REACH_SINGLE_V2_WORKSPACE_MAP,
        "stem_params": dict(PENDULUM_STEM_PARAMS),
    })


@configclass
class AppleReachSingleV2EnvCfg(AppleReachSingleEnvCfg):
    rewards: RewardsSingleV2Cfg = RewardsSingleV2Cfg()
    events: EventsSingleV2Cfg = EventsSingleV2Cfg()
    curriculum: CurriculumSingleV2Cfg = CurriculumSingleV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        for terms in (self.rewards, self.terminations):
            _set_params(terms, standoff=SINGLE_V2_STANDOFF)
        # palm tilt allowed in hold / success; the heading gate (45 deg) is now horizontal
        for t in (self.rewards.hold, self.rewards.face_apple_hold, self.terminations.reach_success):
            t.params = {**t.params, "tilt_tol_rad": SINGLE_V2_TILT_TOL}
        apply_single_v2_placement(self.events.randomize_tree)
        self.events.randomize_tree.params["stem_params"] = dict(REACH_STEM_PARAMS)
        arm = self.actions.right_arm
        arm.max_velocity = REACH_ARM_MAX_VELOCITY
        arm.max_acceleration = REACH_ARM_MAX_ACCELERATION
        arm.brake_to_target = REACH_ARM_BRAKE


@configclass
class AppleReachSingleV2EnvCfg_PLAY(AppleReachSingleV2EnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None
