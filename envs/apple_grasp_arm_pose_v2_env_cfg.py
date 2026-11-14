# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Tactile grasp behind the fruit-aware arm policy (AppleGraspArmPose-v2), from scratch: FINGERS ONLY (the arm
holds the reach's end pose; the 7 arm outputs are unused).

User, 2026-11-13: the harvest fails in SIMULATION too -- retrain the reach and the grasp, with nothing the real
robot cannot know. AppleGraspArmPose-v1 (apple_grasp_arm_pose_env_cfg.py -- subclassed: the 65-input tactile grasp,
realistic pads, the tree the head camera sees, fixed waist yaw, perception bias, the arm free at 0.05 rad / unit,
stem release then a harvest load) with:

  - reach phase: the frozen AppleArmPose-v5 policy (pinned best_weights/grasp_handoff_arm_pose_v5.pt, or
    $ARM_POSE_V5_POLICY; layout "arm_pose_v5", 37 in / 7 out) on ONE fruit goal (arm_pose_fruit.fruit_reach_step):
    the pocket on the perceived apple + a calibration-like offset, the palm direction chosen by the aim map
    (envs/assets/aim_map_v5.pt, at the episode's waist yaw; 15%: the torso -> apple line turned up to +-20 deg and
    tilted up to +-15 deg), no staging waypoint (the policy comes in along the palm normal by itself), within the
    robot's shoulder-roll limit;
  - the head camera keeps tracking the apple (visible_tracking: ~10 Hz detections, 0.08-0.30 s old, noise, missed
    frames, hidden by the arm / hand, a quarter of the episodes with no update at all): the reach's goal and the
    grasp's fruit-in-pocket input follow it -- a fruit bumped on the way in is aimed at again;
  - the fruit-in-pocket input is ZERO once the stem has let go: from then on the fruit is in the hand or lost, and
    on the robot / the bench the camera no longer tells (the hand hides it), so the fingers must hold by feel while
    the arm pulls, carries and turns (the v1 input kept pointing at where the fruit had hung);
  - start: the robot's READY or tucked HOME pose (half each), jittered;
  - the harvest load after the release: the bench's -- mostly the straight pull (stems 6-14 N), a twist in 40%;
  - 1024 envs x 32 steps (v1: 32 x 64), so the curricula count fewer iterations.
Training (2026-11-13, laptop): from scratch the fingers had not started to close into a hold after 120 iterations
(3.9M samples; kept as run ..._scratch_stopped_it120), so the run is WARM-STARTED from the AppleGraspSpindle-v3
tactile grasp in this 65-input layout (logs/rsl_rl/apple_grasp_arm_pose_v2/warmstart_spindle_v3/model_0.pt = the
file AppleGraspSingle-v2 / AppleGraspVisible started from, noise std set to 0.2):
    train.py --task AppleGraspArmPose-v2 --resume --load_run warmstart_spindle_v3 --checkpoint model_0.pt
with a short curriculum (release 20..150 its, load 30..150) and entropy 0.001.
Result (run 2026-11-13_13-42-31, stopped at model_400): it closes on the fruit from the first iterations; under
the full harvest load ~70% of the training episodes still lose the fruit (its 150-400, flat). In the harvest bench
(scripts/play_harvest_robot.py, v5 reach + aim map, 5 seeds x 16 trees) model_300 gives 52/80 deposited when the
fingers are FROZEN once two pads hold (--learned_hold freeze_after_grasp), the scripted close 62% on the same
world; left to regulate the fingers through the pull and the carry it drops most fruit (4/16: the moving arm is
outside its training, where the arm stands still). Pinned: best_weights/grasp_arm_pose_v2_300.pt.
"""

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .apple_arm_pose_env_cfg import POCKET
from .apple_grasp_arm_pose_env_cfg import (
    ARM_POSE_HANDOFF_HOLD_S,
    ARM_POSE_HANDOFF_WATCHDOG_S,
    ARM_START_Q,
    AppleGraspArmPoseEnvCfg,
)
from .apple_grasp_env_cfg import RIGHT_ARM_JOINTS
from .apple_grasp_single_v2_env_cfg import SINGLE_V2_GRASP_WINDOW_S, SINGLE_V2_RELEASE_S
from .apple_reach_spindle_env_cfg import ARM_TARGET_CLIP
from .apple_reach_visible_env_cfg import CAMERA_D435I
from .apple_reach_visible_v2_env_cfg import (
    APPLE_RADIUS,
    CAMERA_LATENCY_S,
    CAMERA_MISS_PROB,
    CAMERA_NOISE_STD,
    CAMERA_PARTIAL_SHIFT,
    CAMERA_PERIOD_S,
    FROZEN_PROB,
    SHOULDER_ROLL_MAX,
)
from .mdp import arm_pose_fruit as apf
from .mdp import visible_tracking as vt

ARM_POSE_V5_POLICY = os.environ.get(
    "ARM_POSE_V5_POLICY",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "best_weights", "grasp_handoff_arm_pose_v5.pt"))
#: the aim map of the pinned reach (scripts/build_aim_map.py): the palm direction per apple, as the harvest chooses it
AIM_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "aim_map_v5.pt")
#: PPO rollout length of AppleGraspArmPoseV2PPORunnerCfg (the curricula below count env steps)
V2_STEPS_PER_ITER = 32
#: (from scratch these were 150..500 / 200..500; the warm start already closes and holds)
V2_RELEASE_CURRICULUM = (20, 150)
V2_LOAD_CURRICULUM = (30, 150)
#: the bench's READY (reach_io.HOME_Q); the tucked HOME is ARM_START_Q
READY_Q = [-0.15, -0.15, -0.15, 0.5, 0.0, 0.0, 0.0]
ARM_STARTS = [READY_Q, ARM_START_Q]
ARM_START_JITTER = 0.08
#: the fruit goal (arm_pose_fruit.fruit_reach_step)
FRUIT_REACH = dict(
    pocket_local=POCKET,
    goal_offset_axial=(-0.015, 0.015),
    goal_offset_cross=0.01,
    heading_deg=(-20.0, 20.0),
    tilt_deg=(-15.0, 15.0),
    follow=True,
    aim_map=AIM_MAP,
    aim_prob=0.85,
    track=dict(pos_tol=0.02, ang_tol_deg=15.0, finger_tol_deg=25.0, speed_tol=0.08, hold_time=0.2,
               via_pos_tol=0.03, via_ang_tol_deg=25.0),
)
#: straight-pull break force of the stem in the harvest load, N (the bench's held target: 6-10)
V2_PULL_BREAK = (6.0, 14.0)
V2_TWIST_FAIL_PROB = 0.6


@configclass
class CurriculumV2Cfg:
    #: logs only (Curriculum/<term>/<key>)
    camera_tracking = CurrTerm(func=vt.camera_tracking_metrics)
    handoff = CurrTerm(func=apf.handoff_metrics, params={"pocket_local": POCKET})


@configclass
class AppleGraspArmPoseV2EnvCfg(AppleGraspArmPoseEnvCfg):
    curriculum: CurriculumV2Cfg = CurriculumV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.episode_length_s = ARM_POSE_HANDOFF_WATCHDOG_S + SINGLE_V2_GRASP_WINDOW_S

        arm = self.actions.right_arm
        arm.reach_policy_path = ARM_POSE_V5_POLICY
        arm.reach_obs_layout = "arm_pose_v5"
        arm.reach_arm_pose = dict(FRUIT_REACH)
        clip = dict(ARM_TARGET_CLIP)
        clip["right_shoulder_roll_joint"] = (clip["right_shoulder_roll_joint"][0], SHOULDER_ROLL_MAX)
        arm.clip = clip

        ev = self.events
        ev.arm_ready_start = EventTerm(func=apf.set_arm_start_choice, mode="reset",
                                       params={"joint_names": list(RIGHT_ARM_JOINTS), "poses": ARM_STARTS,
                                               "jitter": ARM_START_JITTER, "clip": clip})
        # after apple_measurement_reset (appended last): the camera keeps tracking the apple
        ev.camera_tracking_reset = EventTerm(
            func=vt.reset_camera_tracking, mode="reset",
            params={"latency_s": CAMERA_LATENCY_S, "detect_period_s": CAMERA_PERIOD_S, "frozen_prob": FROZEN_PROB})
        ev.camera_tracking = EventTerm(
            func=vt.step_camera_tracking, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
            params={"camera": dict(CAMERA_D435I), "detect_period_s": CAMERA_PERIOD_S, "noise_std": CAMERA_NOISE_STD,
                    "miss_prob": CAMERA_MISS_PROB, "partial_shift": CAMERA_PARTIAL_SHIFT,
                    "apple_radius": APPLE_RADIUS})
        ev.update_reach_grasp_phase.params.update(
            handoff_mode="arm_pose", handoff_time_s=ARM_POSE_HANDOFF_WATCHDOG_S, hold_time=ARM_POSE_HANDOFF_HOLD_S)
        ev.record_handoff = EventTerm(func=apf.record_handoff, mode="interval", interval_range_s=(0.0, 0.0),
                                      is_global_time=True, params={"pocket_local": POCKET})
        it0, it1 = V2_RELEASE_CURRICULUM
        # start: later than the episode can last after ANY hand-off (v1's window + 1 = 9 s was reached here: the v5
        # reach hands off after ~4 s of the 16 s episode, and 78% of the first run's episodes ended with the fruit
        # falling at 13 s, before anything was learned)
        ev.reset_reach_grasp_phase.params["stem_release_schedule"] = (
            self.episode_length_s, SINGLE_V2_RELEASE_S[0], it0 * V2_STEPS_PER_ITER, it1 * V2_STEPS_PER_ITER,
            SINGLE_V2_RELEASE_S[1] - SINGLE_V2_RELEASE_S[0])
        ev.release_stem_and_load.params.update(
            curriculum=V2_LOAD_CURRICULUM, steps_per_iter=V2_STEPS_PER_ITER, pull_break=V2_PULL_BREAK,
            twist_fail_prob=V2_TWIST_FAIL_PROB)

        # FINGERS ONLY: the arm stays where the reach left it (its 7 action dims do nothing). First run (arm free at
        # 0.05 rad / unit, 176 its): nothing learned, noise 0.70 -> 0.89 -- the arm's exploration shoved the fruit
        # about, and v5 hands off with the fruit within 3 cm of the pocket in ~80% of the episodes anyway
        arm.scale = 0.0
        rw = self.rewards
        old = rw.apple_displacement
        rw.apple_displacement = RewTerm(func=apf.beyond_handoff, weight=old.weight, params={
            "module": "probe_grasp", "term": old.func.__name__, "inner": dict(old.params)})
        old = rw.neighbour_push
        rw.neighbour_push = RewTerm(func=apf.beyond_handoff, weight=old.weight, params={
            "module": "tree_reach", "term": old.func.__name__, "inner": dict(old.params)})

        pk = self.observations.policy.apple_pos_pocket
        self.observations.policy.apple_pos_pocket = ObsTerm(func=apf.apple_pos_pocket_until_release,
                                                            params=dict(pk.params))


@configclass
class AppleGraspArmPoseV2EnvCfg_PLAY(AppleGraspArmPoseV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
