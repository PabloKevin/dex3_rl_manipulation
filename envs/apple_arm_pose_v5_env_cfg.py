# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Fruit-aware point-to-point arm policy (AppleArmPose-v5), from scratch.

User, 2026-11-13: the harvest fails most of the time in SIMULATION too (37%
deposited over 2604 tries, ml_project/datos) -- retrain the reach (and the
grasp), keeping to what the real robot can know; AppleArmPose is a good
start. What the sim harvest showed about the v1 reach (play_harvest_robot,
16 trees, seed 11: 5 deposited): it arrives within 2 cm of where the apple
WAS (15 of 16), but swept the hanging fruit 6-16 cm away on its way in 4,
and never settled on the goal in 6 (timeout at ~2 cm, buzzing).

v5 = the v1 task (arm_pose.py: pose library, chained goals, starts anywhere,
self-collision) with
  - FRUIT goals (arm_pose_fruit.py): most apple-like stop goals carry a
    virtual fruit at the goal point; no part of the hand / forearm may pass
    through it except the fruit entering the cupped hand -- so the policy
    comes in along the palm normal by itself (no staging waypoint needed),
    told by one more flag input; half of them are TRACKED (the goal jumps
    0.5-3 cm now and then: a fresh detection of a fruit that moved);
  - the v2 HOLD phase (stay, still, also under a push) -- the grasp closes
    while this policy holds the arm;
  - the ROBOT's shoulder-roll limit (+0.0524 rad: v4's library, clamp and
    posture box) -- v1 used rolls up to +0.35 the robot cannot do;
  - the waist yaw as an input (v3: where the legs are under the arm);
  - small sensor noise on the joint inputs.
Observation (37): arm joint positions / velocities (14), goal - pocket (3),
goal - right shoulder (3), goal palm normal in the palm frame (3), goal
finger direction in the palm frame (3), flags (finger counts, via, FRUIT)
(3), last action (7), waist yaw (1). Action (7): as v1 (1.2 rad/s, 5
rad/s^2, braking, joint-limit clamp).
"""

from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .apple_arm_pose_env_cfg import (
    ARM_POSE_ROLL_LIBRARY,
    POCKET,
    POSTURE_UPPER_V4,
    AppleArmPoseV2EnvCfg,
    CurriculumCfg,
    EventsV2Cfg,
    ObservationsCfg,
    RewardsV2Cfg,
)
from .apple_reach_visible_v2_env_cfg import SHOULDER_ROLL_MAX
from .mdp import arm_pose as ap
from .mdp import arm_pose_fruit as apf

#: a goal's time budget (v1: 4 s; a fruit goal is flown round to its front)
V5_GOAL_TIMEOUT_S = 5.0
#: fruit goals (arm_pose_fruit.assign_fruit)
V5_FRUIT = dict(fruit_prob=0.7, tracked_prob=0.5, radius_range=(0.032, 0.048), max_tilt_deg=35.0,
                min_forward=0.10, finger_free_prob=0.7, jump_every_s=2.5, jump_range=(0.005, 0.03))
#: push while holding, N (v2: 3): the fruit and its stem pressing on the cup, the fingers closing on it
V5_PUSH_MAX_N = 5.0


@configclass
class ObservationsV5Cfg(ObservationsCfg):
    @configclass
    class PolicyObs(ObservationsCfg.PolicyObs):
        goal_flags = ObsTerm(func=apf.obs_goal_flags)                 # 2 -> 3: + the goal is a fruit
        waist_yaw = ObsTerm(func=ap.obs_waist_yaw)                    # appended

        def __post_init__(self):
            self.enable_corruption = True
            self.arm_pos.noise = Unoise(n_min=-0.003, n_max=0.003)
            self.arm_vel.noise = Unoise(n_min=-0.03, n_max=0.03)

    policy: PolicyObs = PolicyObs()


@configclass
class RewardsV5Cfg(RewardsV2Cfg):
    # an arm / hand sphere 2 cm deeper in the fruit than the cup allows: 1 per sphere per second x 3
    fruit_sweep = RewTerm(func=apf.penalty_fruit_sweep, weight=-3.0,
                          params={"pocket_local": POCKET, "deadband": 0.01, "ref": 0.02, "cap": 4.0})


@configclass
class EventsV5Cfg(EventsV2Cfg):
    reset_fruit = EventTerm(func=apf.reset_fruit, mode="reset")
    # after advance_goals / hold_push (declaration order)
    assign_fruit = EventTerm(func=apf.assign_fruit, mode="interval", interval_range_s=(0.0, 0.0),
                             params={"pocket_local": POCKET, **V5_FRUIT})


@configclass
class CurriculumV5Cfg(CurriculumCfg):
    fruit = CurrTerm(func=apf.fruit_metrics)


@configclass
class AppleArmPoseV5EnvCfg(AppleArmPoseV2EnvCfg):
    observations: ObservationsV5Cfg = ObservationsV5Cfg()
    rewards: RewardsV5Cfg = RewardsV5Cfg()
    events: EventsV5Cfg = EventsV5Cfg()
    curriculum: CurriculumV5Cfg = CurriculumV5Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        # the robot's shoulder-roll limit (v4)
        clip = dict(self.actions.right_arm.clip)
        lo, _ = clip["right_shoulder_roll_joint"]
        clip["right_shoulder_roll_joint"] = (lo, SHOULDER_ROLL_MAX)
        self.actions.right_arm.clip = clip
        self.rewards.posture_box.params["upper"] = POSTURE_UPPER_V4
        self.events.reset_arm_pose.params["library_file"] = ARM_POSE_ROLL_LIBRARY
        self.events.advance_goals.params.update({"goal_timeout_s": V5_GOAL_TIMEOUT_S, "push_max_n": V5_PUSH_MAX_N})


@configclass
class AppleArmPoseV5EnvCfg_PLAY(AppleArmPoseV5EnvCfg):
    """A few envs, the goal drawn (env 0), for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.3, -1.0, 1.45)
        self.viewer.lookat = (0.15, -0.15, 1.0)
        self.events.draw_goal = EventTerm(func=ap.draw_goal, mode="interval", interval_range_s=(0.0, 0.0))
