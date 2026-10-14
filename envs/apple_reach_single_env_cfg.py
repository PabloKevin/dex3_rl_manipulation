# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
One reach policy, no touch input (AppleReachSingle-v1), on the spindle tree.

User, 2026-10-14: back to ONE policy for the whole reach (the two-stage
chain is dropped). On the real tree the hanging apple is too light for the
palm tactile to see the touch, so the reach runs on perception only; touch
is for the grasp and after. Built on AppleReachPregrasp-v3
(apple_reach_twostage_env_cfg.py): no tactile input, magnet stems,
target_push, rise near the body first (advance_before_height), smooth
arrival (settled / brake / palm speed). Changes:

  - nothing from the stem/branch: the approach direction of every heading /
    goal term is the horizontal line waist axis -> apple
    (single_reach.set_approach_origin), so with face_apple the palm normal
    ends along torso +X -- away from the robot, i.e. away from the trunk
    for a fruit in front of it and side-on for one on its flank.
  - fingers vertical (fingers_vertical): fingers DOWN, thumb above, for
    fruit below shoulder + 15 cm; fingers UP, thumb below, above it.
    fingers_toward_neighbours (it charged vertical fingers in a vertical
    cluster) is removed.
  - arm near the body: elbow_out, the elbow beyond the shoulder sideways.
  - goal: palm point SINGLE_STANDOFF (2 cm) from the apple surface, held
    still 1 s, no contact (target_contact still charges any touch). The
    grasp (with tactile) takes over from there.
  - hand: open at the start, closed to the partial cup
    (pregrasp.FINAL_HAND_PRESHAPE) once the palm point is level with the
    fruit (3 cm), ramped over 0.3 s; the phase is an input (40 inputs:
    Pregrasp-v3's 39 + hand phase LAST). NOT the "waist" 40-input layout
    of sim2real/reach_io.py, which picks its layout by input size.
  - perception noise on the apple position input (and the hand trigger,
    which reads the same estimate on the robot): per-episode bias, larger
    along the viewing depth, + per-step jitter. Rewards / success use the
    true position. The sizes are GUESSES (the only measured number is the
    head-camera calibration residual, 3 mm mean over 37 points; a wrong
    radius in the surface -> centre step adds up to ~1 cm along the depth):
    measure the real error on fixed apples and put it in APPLE_MEAS_NOISE.
"""

from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .apple_grasp_env_cfg import PALM_LOCAL_OFFSET, PALM_NORMAL_AXIS
from .apple_reach_twostage_env_cfg import (
    AppleReachPregraspV3EnvCfg,
    EventsPregraspCfg,
    RewardsPregraspV3Cfg,
    _set_params,
)
from .mdp import pregrasp as pg
from .mdp import single_reach as sr

#: goal: palm reference point this far from the apple SURFACE along the palm
#: normal (pregrasp: 6 cm, touch reach: 1.3 cm)
SINGLE_STANDOFF = 0.02
#: palm point within this of the goal for hold / success
SINGLE_POS_TOL = 0.02
#: apple centre this far above the right shoulder: fingers up / thumb down
FINGER_SWITCH_DZ = 0.15

#: perceived apple centre = true + per-episode bias + per-step jitter (m)
APPLE_MEAS_NOISE = {
    "depth_std": 0.010, "depth_clip": 0.025,   # along the horizontal pelvis -> apple line
    "cross_std": 0.005, "cross_clip": 0.015,   # sideways and vertically
}
APPLE_MEAS_JITTER_STD = 0.002

HAND_OPEN = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.6,
    "right_hand_thumb_2_joint": 0.0,
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_middle_1_joint": 0.0,
}
HAND_CLOSED = dict(pg.FINAL_HAND_PRESHAPE)

_PALM = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
_APPLE = SceneEntityCfg("apple")


@configclass
class RewardsSingleCfg(RewardsPregraspV3Cfg):
    fingers_toward_neighbours = None
    fingers_vertical = RewTerm(
        func=sr.reward_fingers_vertical,
        weight=8.0,
        params={"robot_cfg": _PALM, "object_cfg": _APPLE, "standoff": SINGLE_STANDOFF,
                "palm_local_offset": PALM_LOCAL_OFFSET, "palm_normal_axis": PALM_NORMAL_AXIS,
                "switch_dz": FINGER_SWITCH_DZ, "band": 0.03, "std_rad": 0.4, "gate_dist": 0.55},
    )
    elbow_out = RewTerm(func=sr.penalty_elbow_out, weight=-1.0, params={"margin": 0.05, "ref": 0.10})


@configclass
class EventsSingleCfg(EventsPregraspCfg):
    approach_origin = EventTerm(func=sr.set_approach_origin, mode="startup", params={"body": "pelvis"})
    # after the tree / apple placement (declaration order)
    apple_measurement_reset = EventTerm(func=sr.reset_apple_measurement, mode="reset",
                                        params={"object_cfg": _APPLE, "origin_body": "pelvis", **APPLE_MEAS_NOISE})
    # every step, before hand_schedule (it reads this step's measurement)
    apple_measurement = EventTerm(func=sr.step_apple_measurement, mode="interval", interval_range_s=(0.0, 0.0),
                                  is_global_time=True,
                                  params={"object_cfg": _APPLE, "jitter_std": APPLE_MEAS_JITTER_STD})
    # after hold_default_targets (declaration order)
    hand_schedule_reset = EventTerm(func=sr.reset_hand_schedule, mode="reset",
                                    params={"open_targets": HAND_OPEN, "closed_targets": HAND_CLOSED})
    hand_schedule = EventTerm(
        func=sr.step_hand_schedule, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        params={"open_targets": HAND_OPEN, "closed_targets": HAND_CLOSED, "robot_cfg": _PALM,
                "object_cfg": _APPLE, "palm_local_offset": PALM_LOCAL_OFFSET,
                "level_tol": 0.03, "ramp_s": 0.3},
    )


@configclass
class AppleReachSingleEnvCfg(AppleReachPregraspV3EnvCfg):
    rewards: RewardsSingleCfg = RewardsSingleCfg()
    events: EventsSingleCfg = EventsSingleCfg()

    def __post_init__(self):
        super().__post_init__()
        for terms in (self.rewards, self.terminations):
            _set_params(terms, standoff=SINGLE_STANDOFF)
        for t in (self.rewards.hold, self.rewards.face_apple_hold, self.terminations.reach_success):
            t.params = {**t.params, "pos_tol": SINGLE_POS_TOL}
        # same slot, perceived apple instead of the true one
        ap = self.observations.policy.apple_pos_palm
        self.observations.policy.apple_pos_palm = ObsTerm(func=sr.apple_pos_palm_measured, params=dict(ap.params))
        self.observations.policy.hand_phase = ObsTerm(func=sr.hand_phase_obs)


@configclass
class AppleReachSingleEnvCfg_PLAY(AppleReachSingleEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None
