# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
AppleReachVisible-v2: AppleReachVisible-v1 with the camera in the loop, wide
starts, a late command path and the robot's real shoulder-roll limit.

User, 2026-10-18 (no wrist camera; train before moving the policy onto the
Jetson): use the camera more for closed-loop control, approach from any
starting pose (so the policy can be restarted after unexpected behaviour,
less reliance on "home"), and train with lag in the loop.

Same 37 inputs / 7 outputs as v1, so it warm-starts from a v1 run (same log
folder, logs/rsl_rl/apple_reach_visible: --resume --load_run <v1 run>) and the
AppleGraspVisible hand-off keeps its reach layout. What the inputs MEAN
changes only in one place: the perceived apple is refreshed during the
episode instead of held.

  - camera tracking (mdp/visible_tracking.py): a detection every
    CAMERA_PERIOD_S of where the apple was CAMERA_LATENCY_S ago, only when the
    head camera sees it whole and the arm / hand is not in the way; a hand
    nearly in the way shifts it along the viewing ray; CAMERA_MISS_PROB of the
    frames are lost; per-detection noise on top of the episode's bias.
    FROZEN_PROB of the episodes never update (v1, today's toys).
  - wide starts: WIDE_START_PROB of the episodes start at a clear pose of
    envs/assets/arm_pose_library.npz (body-clear at the episode's waist yaw,
    pocket >= 12 cm from every apple and nearer the waist axis than the
    target), the rest at home + jitter as v1.
  - command delay: the joint PD gets the target DELAY_STEPS policy steps late
    (per episode, 0-80 ms) and keeps the previous one at HICCUP_PROB of the
    steps (robot telemetry 2026-10-16: late daemon threads).
  - right shoulder roll target <= +0.0524 rad (3 deg): the robot's limit
    (onboard/arm_mpc_acados.py THETA_LIMITS: a screw holding the hand wiring,
    in no model, fouls the torso beyond it); v1 allowed 1.49.

Episode length +2 s for the longer approaches from wide starts.
"""

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .apple_grasp_env_cfg import RIGHT_ARM_JOINTS
from .apple_reach_visible_env_cfg import (
    CAMERA_D435I,
    AppleReachVisibleEnvCfg,
    CurriculumVisibleCfg,
    EventsVisibleCfg,
)
from .mdp import visible_tracking as vt

#: camera frames: period (s, per frame, ~10 Hz throttled colour), latency (s, per episode)
CAMERA_PERIOD_S = (0.08, 0.14)
CAMERA_LATENCY_S = (0.08, 0.30)
CAMERA_MISS_PROB = 0.15
#: per-detection noise (m), and the largest shift along the ray with the hand nearly in the way
CAMERA_NOISE_STD = 0.004
CAMERA_PARTIAL_SHIFT = 0.02
APPLE_RADIUS = 0.035
#: share of the episodes with no updates at all (the v1 / toy behaviour)
FROZEN_PROB = 0.25

WIDE_START_PROB = 0.5
ARM_POSE_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "arm_pose_library.npz")

#: policy steps (20 ms) the PD gets the target late, per episode; per-step hold probability
DELAY_STEPS = (0, 4)
HICCUP_PROB = 0.03

#: the robot's right-arm limits where they are tighter than the sim's (rad)
SHOULDER_ROLL_MAX = 0.0524
ARM_CLIP = {j: ((-3.0, SHOULDER_ROLL_MAX) if j == "right_shoulder_roll_joint" else (-3.0, 3.0))
            for j in RIGHT_ARM_JOINTS}
WIDE_Q_LO = [-3.0] * 7
WIDE_Q_HI = [3.0 if j != "right_shoulder_roll_joint" else SHOULDER_ROLL_MAX for j in RIGHT_ARM_JOINTS]

EXTRA_EPISODE_S = 2.0


@configclass
class EventsVisibleV2Cfg(EventsVisibleCfg):
    # declared after v1's terms: after the tree, the arm jitter and the apple measurement
    reset_arm_wide = EventTerm(
        func=vt.reset_arm_wide, mode="reset",
        params={"library_file": ARM_POSE_LIBRARY, "prob": WIDE_START_PROB, "q_lo": WIDE_Q_LO, "q_hi": WIDE_Q_HI})
    camera_tracking_reset = EventTerm(
        func=vt.reset_camera_tracking, mode="reset",
        params={"latency_s": CAMERA_LATENCY_S, "detect_period_s": CAMERA_PERIOD_S, "frozen_prob": FROZEN_PROB})
    camera_tracking = EventTerm(
        func=vt.step_camera_tracking, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        params={"camera": dict(CAMERA_D435I), "detect_period_s": CAMERA_PERIOD_S, "noise_std": CAMERA_NOISE_STD,
                "miss_prob": CAMERA_MISS_PROB, "partial_shift": CAMERA_PARTIAL_SHIFT,
                "apple_radius": APPLE_RADIUS})


@configclass
class CurriculumVisibleV2Cfg(CurriculumVisibleCfg):
    #: logs only (Curriculum/camera_tracking/<key>)
    camera_tracking = CurrTerm(func=vt.camera_tracking_metrics)


@configclass
class AppleReachVisibleV2EnvCfg(AppleReachVisibleEnvCfg):
    events: EventsVisibleV2Cfg = EventsVisibleV2Cfg()
    curriculum: CurriculumVisibleV2Cfg = CurriculumVisibleV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        arm = self.actions.right_arm
        arm.clip = dict(ARM_CLIP)
        arm.delay_steps_range = DELAY_STEPS
        arm.hiccup_prob = HICCUP_PROB
        self.episode_length_s = self.episode_length_s + EXTRA_EPISODE_S


@configclass
class AppleReachVisibleV2EnvCfg_PLAY(AppleReachVisibleV2EnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
