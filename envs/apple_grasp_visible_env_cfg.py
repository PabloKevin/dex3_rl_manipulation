# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Tactile grasp behind the visible-apple reach (AppleGraspVisible-v1).

User, 2026-10-18: a new grasp, with the tactile input, after the
AppleReachVisible-v1 policy (apple_reach_visible_env_cfg.py), that also
finishes positioning the arm for the best grasp. AppleGraspSingle-v2
(apple_grasp_single_v2_env_cfg.py -- subclassed: 65 inputs = hand-only grasp
+ per-place tactile + tactile memory + the arm's last action, tactile
realism, the WHOLE arm free before the grip at 0.15 rad per unit, pocket /
contact-depth rewards, stem release curriculum) with the reach phase and the
world of the visible reach:

  - reach phase: the frozen AppleReachVisible-v1 policy (pinned
    best_weights/grasp_handoff_reach_visible.pt, or $REACH_VISIBLE_POLICY;
    layout "visible", 37 in / 7 out, no waist output), driven through its
    OWN rate limiter (1.2 rad/s, 5 rad/s^2, braking: REACH_ARM_*), the
    grasp's own (2.0 / 8.0, no braking) only after the hand-off;
  - the waist held at a random fixed yaw per episode, the tree turned with
    it, the first apple seen whole by the head D435i and reachable, the
    target a seen apple, unseen neighbours masked out of the ring, the
    perception bias along the camera ray (visible_reach.place_tree_fixed_
    waist / mask_unseen_neighbours, as in the reach); torso upright (waist
    pitch 0, the robot's);
  - hand-off on the reach's own goal: the apple enclosed (pushed in up to
    5 cm, set_reach_goal_v2 with the reach's REACH_* values) held 0.5 s,
    VISIBLE_HANDOFF_WATCHDOG_S at the latest (the reach's episode length);
  - the hand: open, then the partial cup within 10 cm of the perceived
    apple (the reach's distance trigger);
  - stems: the real tree's break limits (the grasp twists / pulls the fruit
    off), swinging like the reach's (SINGLE_V3_STEM_PARAMS).
Warm start: logs/rsl_rl/apple_grasp_visible/warmstart_spindle_v3/model_0.pt
(the AppleGraspSpindle-v3 tactile grasp in the 65-input layout, the same
file AppleGraspSingle-v2 starts from).
"""

import os

from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .apple_grasp_grasp_env_cfg import GRASP_PPO_STEPS_PER_ITER
from .apple_grasp_single_v2_env_cfg import SINGLE_V2_GRASP_WINDOW_S, SINGLE_V2_RELEASE_S, AppleGraspSingleV2EnvCfg
from .apple_grasp_single_v3_env_cfg import SINGLE_V3_STEM_PARAMS
from .apple_reach_single_v2_env_cfg import (
    REACH_ARM_BRAKE,
    REACH_ARM_MAX_ACCELERATION,
    REACH_ARM_MAX_VELOCITY,
    REACH_MAX_DISP,
    REACH_PUSH_IN,
    SINGLE_V2_HAND_NEAR,
    SINGLE_V2_LATERAL_TOL,
    SINGLE_V2_SHORT_TOL,
    SINGLE_V2_STANDOFF,
    SINGLE_V2_TILT_TOL,
)
from .apple_reach_visible_env_cfg import (
    CAMERA_D435I,
    FIXED_WAIST_YAW_DEG,
    REACH_VISIBLE_WORKSPACE_MAP,
    STRETCH_FRAC,
    VISIBLE_MEMBER_ENVELOPE,
    VISIBLE_SEED_ENVELOPE,
    WAIST_PITCH,
)
from .mdp import single_reach as sr
from .mdp import visible_reach as vr

VISIBLE_REACH_POLICY = os.environ.get(
    "REACH_VISIBLE_POLICY",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "best_weights",
                 "grasp_handoff_reach_visible.pt"))
#: hand-off at the latest this long after the start (the reach's episode: 6 s)
VISIBLE_HANDOFF_WATCHDOG_S = 6.0
#: the reach policy's own action scale and clearance-ring range
VISIBLE_REACH_SCALE = 0.5
VISIBLE_RING_RANGE = 0.20


@configclass
class AppleGraspVisibleEnvCfg(AppleGraspSingleV2EnvCfg):
    #: the camera model, also read by scripts/play.py (frustum, seen apples, its image)
    head_camera: dict = CAMERA_D435I

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = VISIBLE_HANDOFF_WATCHDOG_S + SINGLE_V2_GRASP_WINDOW_S

        # torso upright, as on the robot and in the reach
        robot = self.scene.robot
        jp = dict(robot.init_state.joint_pos)
        jp["waist_pitch_joint"] = WAIST_PITCH
        self.scene.robot = robot.replace(init_state=robot.init_state.replace(joint_pos=jp))

        # reach phase: the frozen visible reach, with its own limiter; no waist output
        arm = self.actions.right_arm
        arm.reach_policy_path = VISIBLE_REACH_POLICY
        arm.reach_obs_layout = "visible"
        arm.reach_waist_joint = None
        arm.reach_scale = VISIBLE_REACH_SCALE
        arm.reach_ring_range = VISIBLE_RING_RANGE
        arm.reach_max_velocity = REACH_ARM_MAX_VELOCITY
        arm.reach_max_acceleration = REACH_ARM_MAX_ACCELERATION
        arm.reach_brake_to_target = REACH_ARM_BRAKE

        ev = self.events
        if getattr(ev, "reset_waist_jitter", None) is not None:
            ev.reset_waist_jitter = None
        # the visible reach's world: fixed waist yaw, seen + reachable apples
        tree = ev.randomize_tree
        tree.func = vr.place_tree_fixed_waist
        tree.params.update({
            "yaw_range_deg": FIXED_WAIST_YAW_DEG,
            "camera": dict(CAMERA_D435I),
            "seed_envelope": {**VISIBLE_SEED_ENVELOPE, "stretch_frac": STRETCH_FRAC, "stretch_cells": 1},
            "member_envelope": dict(VISIBLE_MEMBER_ENVELOPE),
            "workspace_map": REACH_VISIBLE_WORKSPACE_MAP,
            "stem_params": dict(SINGLE_V3_STEM_PARAMS),
        })
        ev.apple_measurement_reset.params.update({"origin_body": "d435_link", "horizontal": False})
        # after apple_measurement_reset (set later in __post_init__ = later in the reset order)
        ev.mask_unseen_neighbours = EventTerm(func=vr.mask_unseen_neighbours, mode="reset")
        # the reach's goal (apple enclosed, pushed in up to 5 cm) is the hand-off test
        ev.reach_goal_v2 = EventTerm(func=sr.set_reach_goal_v2, mode="startup",
                                     params={"touch_gap": SINGLE_V2_STANDOFF, "push_in": REACH_PUSH_IN,
                                             "short_tol": SINGLE_V2_SHORT_TOL, "lateral_tol": SINGLE_V2_LATERAL_TOL,
                                             "max_disp": REACH_MAX_DISP, "heading_horizontal": True})
        ev.hand_schedule.params.update(trigger="distance", near=SINGLE_V2_HAND_NEAR)
        ev.update_reach_grasp_phase.params.update(standoff=SINGLE_V2_STANDOFF, tilt_tol_rad=SINGLE_V2_TILT_TOL,
                                                  handoff_time_s=VISIBLE_HANDOFF_WATCHDOG_S)


@configclass
class AppleGraspVisibleEnvCfg_PLAY(AppleGraspVisibleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None


# ─────────────────────────────────────────────
#  v2: small arm adjustments, no arm through the body, later stem release
# ─────────────────────────────────────────────
# User, 2026-10-18: v1 after ~1000 iterations was very bad (run
# 2026-10-18_19-13-07): 97.5% of the episodes ended with the fruit lost, the
# grip terms peaked at it ~260 and died, the noise std grew 0.68 -> 0.88,
# and the arm swept through the torso although the reach had ended well.
# Measured with the warm start (deterministic, 64 envs): its raw arm outputs
# are large (median 1.5-2.4, p95 3-4: harmless at the 0.05 rad/unit it was
# trained with), so v1's 0.15 rad/unit moved the arm off the hand-off pose
# by 0.2-0.35 rad median (0.5-0.65 p95) on shoulder pitch / elbow / wrist
# pitch -- the hand 10-20 cm off the enclosed apple -- with no penalty for
# entering the body (no physical self-collision in the sim). The grasp is
# to adjust a little and close:
#   - arm 0.15 -> 0.05 rad per unit (the warm start's own), still all 7
#     joints free before the grip; posture_from_handoff on shoulder / elbow
#     2 -> 10 (as AppleGraspSingle-v3);
#   - + self_collision in the grasp phase (-8, as the reach);
#   - stem release curriculum 300..1000 -> 800..2000 iterations, twist / pull
#     load 400..1000 -> 900..2000: hold the fruit before the stem lets go;
#   - PPO entropy 0.005 -> 0.002 (AppleGraspVisibleV2PPORunnerCfg).
VISIBLE_V2_ARM_SCALE = 0.05
VISIBLE_V2_RELEASE_CURRICULUM = (800, 2000)
VISIBLE_V2_LOAD_CURRICULUM = (900, 2000)


@configclass
class AppleGraspVisibleV2EnvCfg(AppleGraspVisibleEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.actions.right_arm.scale = VISIBLE_V2_ARM_SCALE

        ev = self.events
        it0, it1 = VISIBLE_V2_RELEASE_CURRICULUM
        ev.reset_reach_grasp_phase.params["stem_release_schedule"] = (
            SINGLE_V2_GRASP_WINDOW_S + 1.0, SINGLE_V2_RELEASE_S[0],
            it0 * GRASP_PPO_STEPS_PER_ITER, it1 * GRASP_PPO_STEPS_PER_ITER,
            SINGLE_V2_RELEASE_S[1] - SINGLE_V2_RELEASE_S[0])
        ev.release_stem_and_load.params["curriculum"] = VISIBLE_V2_LOAD_CURRICULUM

        rw = self.rewards
        rw.posture_from_handoff.params["weights"] = [10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
        rw.self_collision = RewTerm(func=vr.penalty_self_collision_grasp, weight=-8.0,
                                    params={"margin": 0.015, "ref": 0.05})


@configclass
class AppleGraspVisibleV2EnvCfg_PLAY(AppleGraspVisibleV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
