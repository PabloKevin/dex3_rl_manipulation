# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Point-to-point arm policy, no tree, no home pose (AppleArmPose-v1).

User, 2026-10-18: a reaching policy like AppleReachVisible-v1, but starting
every episode somewhere else, so it learns to move from a point A to a point
B; maybe without the tree at all -- given a target point and a palm normal,
move the arm there, never colliding with itself, as close and as well
oriented as possible. A caller can then give it waypoints (an L path into the
fruit wall that does not push the neighbours) and finally the apple.

envs/mdp/arm_pose.py has the goals, terms and the pose library; the library
is built by scripts/build_arm_pose_library.py (run it first).

Scene: the robot alone (as AppleArmGoal-v1), torso upright (waist pitch 0,
as AppleReachVisible-v1 and the robot), waist held at a random fixed yaw per
episode in the library's yaw range (-45..+15 deg, the visible reach's), the
hand held in the reach's cup (REACH_HAND_PRESHAPE), so the collision shape
and the hand-off match the visible reach.

Episodes: start = a library pose clear of the body at the episode's yaw;
goals chained every time one is reached or after 4 s (see arm_pose.py).
Goal = the GRASP POCKET (where an enclosed apple's centre sits) at a point,
the palm normal along a direction; the finger direction counts in half of
the goals (free otherwise); a quarter are pass-through waypoints (looser,
no stop).

Observation (35): arm joint positions / velocities (14), goal - pocket
(torso frame, 3), goal - right shoulder (torso frame, 3), goal palm normal
in the palm frame (3), goal finger direction in the palm frame (3, zero when
free), flags (finger counts, via) (2), last action (7).
Action (7): the visible reach's arm action -- rate-limited joint position
targets, 1.2 rad/s, 5 rad/s^2, braking, clamped to the soft joint limits --
so the motion budget and sim2real limiter are the same.

Self-collision: a 3 cm danger zone (steep) on the sphere model, and a real
overlap ends the episode at a cost of 50 (return units); goals keep >= 3 cm
from the body at their yaw, starts >= 1.5 cm.
"""

import os

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from . import mdp as task_mdp
from .apple_arm_goal_env_cfg import ArmGoalSceneCfg
from .apple_grasp_env_cfg import RIGHT_ARM_JOINTS, ActionsCfg as ReachActionsCfg
from .apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL
from .apple_reach_single_v2_env_cfg import REACH_ARM_BRAKE, REACH_ARM_MAX_ACCELERATION, REACH_ARM_MAX_VELOCITY
from .apple_reach_spindle_env_cfg import ARM_RATED_TORQUE, ARM_TARGET_CLIP, REACH_HAND_PRESHAPE
from .apple_reach_visible_env_cfg import WAIST_PITCH
from .mdp import arm_pose as ap
from .mdp import reach_rewards as rr
from .mdp import terminations_and_events as te
from .mdp import tree_reach as trr
from .mdp import waist_reach as wr

LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "arm_pose_library.npz")
POCKET = GRASP_CENTER_LOCAL
ARM = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)
#: posture box (AppleArmGoal-v1's; shoulder roll < 0.35 keeps the upper arm
#: off the chest, which the sphere model does not check). The library only
#: samples inside it.
POSTURE_LOWER = [-3.09, -2.25, -2.62, -1.05, -1.97, -1.61, -1.61]
POSTURE_UPPER = [2.67, 0.35, 2.62, 2.09, 1.97, 1.61, 1.61]
#: clearance from the body (sphere model) a start / a goal needs at the episode's yaw
START_CLEARANCE = 0.015
GOAL_CLEARANCE = 0.03
#: "task-like" goals (60%): where reach targets and their waypoints are --
#: pocket in front of the right shoulder (torso frame), palm within 45 deg of
#: horizontal and facing within 75 deg of forward
TASK_FILTER = dict(min_forward=0.10, max_reach=0.60, dz_range=(-0.45, 0.30), max_tilt_deg=45.0,
                   max_heading_deg=75.0)
#: one goal's time budget; then the next comes anyway
GOAL_TIMEOUT_S = 4.0
#: one-off amounts in return units; every term is multiplied by dt
STEP_DT = 0.02
GOAL_BONUS = 8.0
COLLISION_COST = 50.0


@configclass
class ObservationsCfg:
    @configclass
    class PolicyObs(ObsGroup):
        arm_pos = ObsTerm(func=task_mdp.joint_pos_selected, params={"asset_cfg": ARM})
        arm_vel = ObsTerm(func=task_mdp.joint_vel_selected, params={"asset_cfg": ARM})
        goal_delta = ObsTerm(func=ap.obs_goal_delta, params={"pocket_local": POCKET})
        goal_position = ObsTerm(func=ap.obs_goal_position, params={"pocket_local": POCKET})
        goal_normal = ObsTerm(func=ap.obs_goal_normal, params={"pocket_local": POCKET})
        goal_finger = ObsTerm(func=ap.obs_goal_finger, params={"pocket_local": POCKET})
        goal_flags = ObsTerm(func=ap.obs_goal_flags)
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm"})

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyObs = PolicyObs()


@configclass
class RewardsCfg:
    # ── getting there: position, palm normal, fingers (when they count) ──
    pos_coarse = RewTerm(func=ap.reward_pos, weight=1.0, params={"pocket_local": POCKET, "std": 0.25})
    pos_fine = RewTerm(func=ap.reward_pos, weight=4.0, params={"pocket_local": POCKET, "std": 0.04})
    pos_precise = RewTerm(func=ap.reward_pos, weight=4.0, params={"pocket_local": POCKET, "std": 0.01})
    normal_coarse = RewTerm(func=ap.reward_normal, weight=1.0, params={"pocket_local": POCKET, "std_rad": 0.6})
    normal_fine = RewTerm(func=ap.reward_normal, weight=4.0,
                          params={"pocket_local": POCKET, "std_rad": 0.15, "gate": 0.06})
    finger_coarse = RewTerm(func=ap.reward_finger, weight=0.5, params={"pocket_local": POCKET, "std_rad": 0.6})
    finger_fine = RewTerm(func=ap.reward_finger, weight=2.0,
                          params={"pocket_local": POCKET, "std_rad": 0.25, "gate": 0.06})
    # metres (and 0.2 x radians) closed per second: a 0.4 m move pays ~2
    progress = RewTerm(func=ap.reward_progress, weight=5.0, params={"pocket_local": POCKET, "ang_scale": 0.2})
    goal_reached = RewTerm(func=ap.reward_goal_reached, weight=GOAL_BONUS / STEP_DT, params={"via_value": 0.5})
    # ── how: straight where the straight line is clear, smooth, stopping at stop goals ──
    path_deviation = RewTerm(func=ap.penalty_path_deviation, weight=-1.0,
                             params={"pocket_local": POCKET, "deadband": 0.03, "ref": 0.05, "cap": 4.0})
    brake = RewTerm(func=ap.penalty_near_goal_speed, weight=-2.0, params={"pocket_local": POCKET, "dist": 0.10})
    # -1, not the reach's -3: at -3 it outweighed every positive term together
    # in the first 300 its (-2.8 vs +1.6 per second, mostly exploration noise)
    commanded_accel = RewTerm(func=trr.penalty_commanded_accel, weight=-1.0)
    limiter_lead = RewTerm(func=trr.penalty_limiter_lead, weight=-1.0, params={"ref": 0.5})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    arm_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.02, params={"asset_cfg": ARM})
    motor_load_arm = RewTerm(func=wr.penalty_motor_load, weight=-3.0,
                             params={"asset_cfg": ARM, "rated_torque": ARM_RATED_TORQUE})
    # ── safety ──
    # 3 cm danger zone, steep (2 cm into it costs 1.0 per sphere per step
    # before the weight); a real overlap ends the episode
    self_collision = RewTerm(func=task_mdp.penalty_self_collision, weight=-10.0, params={"margin": 0.03, "ref": 0.02})
    collision = RewTerm(func=ap.reward_collision, weight=-COLLISION_COST / STEP_DT)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": ARM})
    posture_box = RewTerm(func=rr.penalty_joint_range_box, weight=-2.4,
                          params={"asset_cfg": ARM, "lower": POSTURE_LOWER, "upper": POSTURE_UPPER})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(func=ap.self_collision)
    # never ends the episode: this step's goal success, before the rewards
    goal_tracker = DoneTerm(func=ap.track_goal, params={
        "pocket_local": POCKET, "pos_tol": 0.02, "ang_tol_deg": 15.0, "finger_tol_deg": 25.0, "speed_tol": 0.08,
        "hold_time": 0.2, "via_pos_tol": 0.03, "via_ang_tol_deg": 25.0})


@configclass
class EventsCfg:
    reset_all_exact = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                                params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.0, 0.0),
                                        "velocity_range": (0.0, 0.0)})
    hold_default_targets = EventTerm(func=task_mdp.hold_default_joint_targets, mode="reset",
                                     params={"asset_cfg": SceneEntityCfg("robot")})
    hand_preshape = EventTerm(func=te.hold_joint_targets, mode="reset",
                              params={"joint_targets": REACH_HAND_PRESHAPE})
    # after the three above (declaration order): waist yaw, arm start, hand at the cup
    reset_arm_pose = EventTerm(func=ap.reset_arm_pose, mode="reset", params={
        "library_file": LIBRARY, "start_clearance": START_CLEARANCE, "goal_clearance": GOAL_CLEARANCE,
        "task_filter": TASK_FILTER, "hand_targets": REACH_HAND_PRESHAPE})
    advance_goals = EventTerm(func=ap.advance_goals, mode="interval", interval_range_s=(0.0, 0.0), params={
        "pocket_local": POCKET, "task_prob": 0.6, "finger_prob": 0.5, "via_prob": 0.25,
        "goal_timeout_s": GOAL_TIMEOUT_S})


@configclass
class CurriculumCfg:
    #: logs only (Curriculum/metrics/<key>)
    metrics = CurrTerm(func=ap.arm_pose_metrics)


@configclass
class AppleArmPoseEnvCfg(ManagerBasedRLEnvCfg):
    scene: ArmGoalSceneCfg = ArmGoalSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ReachActionsCfg = ReachActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 12.0         # ~3-5 goals
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.5, -1.5, 1.8)
        self.viewer.lookat = (0.1, -0.3, 1.0)
        # torso upright, as on the robot (AppleReachVisible-v1)
        robot = self.scene.robot
        jp = dict(robot.init_state.joint_pos)
        jp["waist_pitch_joint"] = WAIST_PITCH
        self.scene.robot = robot.replace(init_state=robot.init_state.replace(joint_pos=jp))
        # the visible reach's arm limiter and target clamp
        arm = self.actions.right_arm
        arm.max_velocity = REACH_ARM_MAX_VELOCITY
        arm.max_acceleration = REACH_ARM_MAX_ACCELERATION
        arm.brake_to_target = REACH_ARM_BRAKE
        arm.clip = dict(ARM_TARGET_CLIP)


@configclass
class AppleArmPoseEnvCfg_PLAY(AppleArmPoseEnvCfg):
    """A few envs, the goal drawn (env 0), for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        # close on env 0, whose goal is drawn
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.3, -1.0, 1.45)
        self.viewer.lookat = (0.15, -0.15, 1.0)
        self.events.draw_goal = EventTerm(func=ap.draw_goal, mode="interval", interval_range_s=(0.0, 0.0))
