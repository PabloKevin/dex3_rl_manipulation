# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp after the single no-touch reach (AppleGraspSingle-v1):
AppleGraspSpindle-v3 (apple_grasp_spindle_env_cfg.py -- subclassed: spindle
tree, hand-only 44-input grasp with tactile, stem release + twist / pull
load) with its reach phase replaced by AppleReachSingle-v1
(apple_reach_single_env_cfg.py).

  - reach phase: the frozen AppleReachSingle-v1 policy (layout "single": 40
    inputs, perceived apple, hand phase), exported to
    best_weights/grasp_handoff_reach_single.pt (or $REACH_SINGLE_POLICY);
    the hand goes open -> partial cup once level with the fruit, exactly as
    in the reach training (single_reach.step_hand_schedule, phase only; the
    hand action term turns it into targets);
  - the same perception model as the reach: one perceived apple point per
    episode (start position + bias, APPLE_MEAS_NOISE), held for the reach
    input AND the grasp's fruit-in-pocket input (toy_reach.py keeps one
    detected point for both);
  - hand-off when the palm has held the reach goal (2 cm from the surface,
    2 cm tolerance) for 0.5 s, SINGLE_HANDOFF_WATCHDOG_S at the latest;
  - the reach stops 2 cm short (+ the perception bias), so the grasp has to
    finish the approach: the WHOLE arm is free before the grip
    (pre_grip_free_dims 7; the spindle grasp froze shoulder + elbow until
    the grip and left only the wrist). Still the grasp's small arm scale
    (0.05 rad per unit, an offset from the hand-off pose): up to ~0.15 rad,
    a few cm at the hand.
"""

import os

from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .apple_grasp_env_cfg import PALM_LOCAL_OFFSET
from .apple_grasp_spindle_env_cfg import AppleGraspSpindleV3EnvCfg
from .apple_grasp_tree_train_env_cfg import TREE_GRASP_WINDOW_S
from .apple_reach_twostage_env_cfg import SAMPLER_WORKERS
from .apple_reach_single_env_cfg import (
    APPLE_MEAS_NOISE,
    HAND_CLOSED,
    HAND_OPEN,
    SINGLE_POS_TOL,
    SINGLE_STANDOFF,
)
from .mdp import single_reach as sr

SINGLE_REACH_POLICY = os.environ.get(
    "REACH_SINGLE_POLICY",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "best_weights", "grasp_handoff_reach_single.pt"))
#: hand-off at the latest this long after the start (the reach episode is 6 s)
SINGLE_HANDOFF_WATCHDOG_S = 5.0

_PALM = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
_APPLE = SceneEntityCfg("apple")


@configclass
class AppleGraspSingleEnvCfg(AppleGraspSpindleV3EnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = SINGLE_HANDOFF_WATCHDOG_S + TREE_GRASP_WINDOW_S

        # reach phase: the single reach
        arm = self.actions.right_arm
        arm.reach_policy_path = SINGLE_REACH_POLICY
        arm.reach_obs_layout = "single"
        arm.pre_grip_free_dims = 7
        hand = self.actions.right_hand
        hand.reach_offset = None
        hand.reach_hold_schedule = (dict(HAND_OPEN), dict(HAND_CLOSED))

        ev = self.events
        # trees built in worker processes ahead of the resets, as in the reach
        # ($TREE_SAMPLER_WORKERS; inline, ~100 ms each, stalls large runs)
        ev.randomize_tree.params.update(sampler_workers=SAMPLER_WORKERS)
        ev.approach_origin = EventTerm(func=sr.set_approach_origin, mode="startup", params={"body": "pelvis"})
        # after the tree placement (declaration order: appended last)
        ev.apple_measurement_reset = EventTerm(func=sr.reset_apple_measurement, mode="reset",
                                               params={"object_cfg": _APPLE, "origin_body": "pelvis",
                                                       **APPLE_MEAS_NOISE})
        ev.hand_schedule_reset = EventTerm(func=sr.reset_hand_schedule, mode="reset",
                                           params={"open_targets": HAND_OPEN, "closed_targets": HAND_CLOSED})
        ev.hand_schedule = EventTerm(
            func=sr.step_hand_schedule, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
            params={"open_targets": HAND_OPEN, "closed_targets": HAND_CLOSED, "robot_cfg": _PALM,
                    "object_cfg": _APPLE, "palm_local_offset": PALM_LOCAL_OFFSET,
                    "level_tol": 0.03, "ramp_s": 0.3, "write_targets": False},
        )
        # hand-off on the single reach's goal
        ev.update_reach_grasp_phase.params.update(
            handoff_mode="converged", handoff_time_s=SINGLE_HANDOFF_WATCHDOG_S,
            standoff=SINGLE_STANDOFF, pos_tol=SINGLE_POS_TOL, palm_local_offset=PALM_LOCAL_OFFSET,
            hold_time=0.5)

        # fruit in the pocket, as perceived
        pk = self.observations.policy.apple_pos_pocket
        self.observations.policy.apple_pos_pocket = ObsTerm(func=sr.apple_pos_pocket_measured,
                                                            params=dict(pk.params))


@configclass
class AppleGraspSingleEnvCfg_PLAY(AppleGraspSingleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
