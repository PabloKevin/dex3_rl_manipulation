# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Tactile grasp behind the point-to-point arm policy (AppleGraspArmPose-v1).

User, 2026-10-19: a grasp policy that starts where the AppleArmPose reach
ends, trained FROM SCRATCH (the last grasps warm-started from earlier grasp
checkpoints did not learn well).

AppleGraspVisible-v2 (apple_grasp_visible_env_cfg.py -- subclassed: the
65-input tactile grasp of AppleGraspSingle-v2, the visible reach's world --
tree, fixed random waist yaw, apples the head D435i sees, perception bias
along the camera ray, waist pitch 0 -- small arm adjustments 0.05 rad/unit,
self-collision in the grasp, stem release curriculum 800..2000 its) with the
reach phase replaced by what the robot runs (sim2real/toy_arm_pose.py):

  - reach: the frozen AppleArmPose policy (pinned
    best_weights/grasp_handoff_arm_pose.pt = apple_arm_pose/2026-10-19_
    13-31-55, the robot's; or $ARM_POSE_POLICY), layout "arm_pose" (35 in /
    7 out), its own limiter (1.2 rad/s, 5 rad/s^2, braking) and the
    soft-limit target clamp, steered by the robot's waypoints
    (arm_pose.waypoint_reach_step): STAGING at the apple's height 0.22-0.28
    m from the torso axis (robot 0.24), left at 5 cm / 30 deg or after
    2.4 s, then the APPLE -- the pocket on the PERCEIVED apple centre moved
    by a calibration-like offset (-1.5..+1.5 cm along the approach, +-1 cm
    sideways / vertically: the dashboard's offset fields);
  - the hand in the reach's cup during the reach (REACH_HAND_PRESHAPE: the
    arm-pose policy trained with it, the robot holds it), no open -> close
    schedule;
  - start: the robot's tucked HOME (sim2real/harvest_config.json home_q: the
    hand by the hip, 0.30 m from the waist axis). The reach knocks fruit on
    its way UP to the staging point (it does not know the tree): measured
    with zero grasp actions, 128 envs, seed 7, the target moved > 5 cm during
    the reach in 62 / 60 / 83 envs and was still > 10 cm off at the hand-off
    in 3% / 5% / 16% from HOME / the default arm pose / READY (reach_io.
    HOME_Q, the hand 0.46 m out, level with the low fruit); with no reach,
    2 of 128. Overlap with the body (sphere model) during the reach: 11% /
    20% / 2% -- harmless here (frozen reach, no self-collision in PhysX);
    pocket within 3.4 / 5.8 / 12 cm of the true apple (p90) at an arrival
    hand-off. HOME gives the grasp the cleanest hand-offs;
  - hand-off: the apple goal reached (the arm-pose policy's own test: 2 cm /
    15 deg, slower than 8 cm/s, 0.2 s) and kept ARM_POSE_HANDOFF_HOLD_S, at
    the latest ARM_POSE_HANDOFF_WATCHDOG_S after the start.
Trained from scratch: no warm start (PPO's fresh policy, init noise 0.8).
"""

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .apple_arm_pose_env_cfg import POCKET
from .apple_grasp_single_v2_env_cfg import SINGLE_V2_GRASP_WINDOW_S
from .apple_grasp_visible_env_cfg import AppleGraspVisibleV2EnvCfg
from .apple_grasp_env_cfg import RIGHT_ARM_JOINTS
from .apple_reach_spindle_env_cfg import ARM_TARGET_CLIP, REACH_HAND_PRESHAPE
from .mdp import arm_pose as ap

ARM_POSE_POLICY = os.environ.get(
    "ARM_POSE_POLICY",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "best_weights", "grasp_handoff_arm_pose.pt"))
#: hand-off: the apple goal kept this long once reached, or this long after the start at the latest
#: (measured in the visible world: staging 3.5-3.8 s, the apple 1-2.4 s more)
ARM_POSE_HANDOFF_HOLD_S = 0.3
ARM_POSE_HANDOFF_WATCHDOG_S = 8.0
#: the episode's arm start, RIGHT_ARM_JOINTS order: the robot's tucked HOME (sim2real/harvest_config.json
#: home_q). (READY, reach_io.HOME_Q, would be [-0.15, -0.15, -0.15, 0.5, 0, 0, 0].)
ARM_START_Q = [0.3, -0.6, 0.8, 1.0, 0.0, 0.0, 0.0]
#: the robot's waypoints (sim2real/arm_pose_io.py) with some spread
ARM_POSE_REACH = dict(
    pocket_local=POCKET,
    stage_radius=(0.22, 0.28),
    min_stage_back=0.08,
    stage_advance_d=0.05,
    stage_advance_deg=30.0,
    stage_advance_s=2.4,
    goal_offset_axial=(-0.015, 0.015),
    goal_offset_cross=0.01,
    track=dict(pos_tol=0.02, ang_tol_deg=15.0, finger_tol_deg=25.0, speed_tol=0.08, hold_time=0.2,
               via_pos_tol=0.03, via_ang_tol_deg=25.0),
)


@configclass
class AppleGraspArmPoseEnvCfg(AppleGraspVisibleV2EnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = ARM_POSE_HANDOFF_WATCHDOG_S + SINGLE_V2_GRASP_WINDOW_S

        arm = self.actions.right_arm
        arm.reach_policy_path = ARM_POSE_POLICY
        arm.reach_obs_layout = "arm_pose"
        arm.reach_arm_pose = dict(ARM_POSE_REACH)
        arm.reach_scale = 0.5
        arm.clip = dict(ARM_TARGET_CLIP)

        hand = self.actions.right_hand
        hand.reach_hold_schedule = None
        hand.reach_offset = dict(REACH_HAND_PRESHAPE)

        # after every joint reset (set here: appended last in the reset order)
        self.events.arm_ready_start = EventTerm(func=ap.set_arm_start, mode="reset",
                                                params={"joint_pos": dict(zip(RIGHT_ARM_JOINTS, ARM_START_Q))})
        self.events.update_reach_grasp_phase.params.update(
            handoff_mode="arm_pose", handoff_time_s=ARM_POSE_HANDOFF_WATCHDOG_S, hold_time=ARM_POSE_HANDOFF_HOLD_S)


@configclass
class AppleGraspArmPoseEnvCfg_PLAY(AppleGraspArmPoseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
