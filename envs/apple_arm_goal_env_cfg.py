# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Goal-reaching arm policy for the harvest path (AppleArmGoal-v1): move the
grasp pocket to a goal point with the palm normal along a goal direction,
from any arm pose, carrying an apple's weight or not, out of the body and
the basket. The harvest bench gives it the goals in turn: the safe zone
straight back from the pick, the drop pose over the basket, home
(envs/mdp/arm_goal.py for the goals, load and terms).

Scene: the robot alone -- the basket is an analytic obstacle at a random
place (+-15 cm in x, y and height around its nominal position), the load is
the apple's weight (0.10-0.35 kg) applied at the pocket. Thousands of cheap
envs; the harvest bench then tests it with real held fruit.

Observation (31): arm joint positions/velocities (14), goal - pocket in the
torso frame (3), goal palm normal in the palm frame (3), basket (axis at rim
height) - pocket in the torso frame (3), carrying flag (1), last action (7).
Action: the reach task's arm action (rate-limited joint position targets,
same limits), so the motion budget matches everything else on the arm.
"""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp as task_mdp
from .apple_grasp_env_cfg import G1_DEX3_CFG, RIGHT_ARM_JOINTS, ActionsCfg as ReachActionsCfg
from .apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL
from .mdp import arm_goal as ag
from .mdp import reach_rewards as rr
from .mdp import tree_reach as trr

POSE_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "harvest_pose_library.npz")
POCKET = GRASP_CENTER_LOCAL
ARM = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)


@configclass
class ArmGoalSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    distant_light = AssetBaseCfg(prim_path="/World/DistantLight",
                                 spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.92), angle=0.5))
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight",
                              spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.75, 0.75, 1.0)))
    robot: ArticulationCfg = G1_DEX3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyObs(ObsGroup):
        arm_pos = ObsTerm(func=task_mdp.joint_pos_selected, params={"asset_cfg": ARM})
        arm_vel = ObsTerm(func=task_mdp.joint_vel_selected, params={"asset_cfg": ARM})
        goal_pocket = ObsTerm(func=ag.obs_goal_pocket, params={"pocket_local": POCKET})
        goal_normal = ObsTerm(func=ag.obs_goal_normal, params={"pocket_local": POCKET})
        basket = ObsTerm(func=ag.obs_basket, params={"pocket_local": POCKET})
        carrying = ObsTerm(func=ag.obs_carrying)
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm"})

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyObs = PolicyObs()


@configclass
class RewardsCfg:
    # ── getting there ──
    goal_coarse = RewTerm(func=ag.reward_goal_pos, weight=1.0, params={"pocket_local": POCKET, "std": 0.25})
    goal_fine = RewTerm(func=ag.reward_goal_pos, weight=4.0, params={"pocket_local": POCKET, "std": 0.04})
    normal_coarse = RewTerm(func=ag.reward_goal_normal, weight=1.0, params={"pocket_local": POCKET, "std_rad": 0.6})
    normal_fine = RewTerm(func=ag.reward_goal_normal, weight=3.0,
                          params={"pocket_local": POCKET, "std_rad": 0.25, "gate": 0.06})
    success = RewTerm(func=ag.reward_success, weight=95.0)
    # ── how: smooth, straight, slowing down at the goal ──
    commanded_accel = RewTerm(func=trr.penalty_commanded_accel, weight=-3.0)
    limiter_lead = RewTerm(func=trr.penalty_limiter_lead, weight=-1.0, params={"ref": 0.5})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    arm_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.02, params={"asset_cfg": ARM})
    path_deviation = RewTerm(func=ag.penalty_path_deviation, weight=-1.0,
                             params={"pocket_local": POCKET, "deadband": 0.03, "ref": 0.05})
    brake = RewTerm(func=ag.penalty_near_goal_speed, weight=-2.0, params={"pocket_local": POCKET, "dist": 0.10})
    # ── safety: body, held apple, basket, joints ──
    # Body and basket: a 4 cm DANGER ZONE, steep (2 cm into it costs 1.0 per
    # step before the weight), and a real overlap ends the episode with
    # `collision` -- the first policy still cut through the body now and
    # then (1.5 cm margin, gentle penalty: a brief pass cost less than a detour).
    self_collision = RewTerm(func=task_mdp.penalty_self_collision, weight=-10.0, params={"margin": 0.04, "ref": 0.02})
    apple_body = RewTerm(func=ag.penalty_apple_body, weight=-10.0,
                         params={"pocket_local": POCKET, "margin": 0.04, "ref": 0.02})
    basket = RewTerm(func=ag.penalty_basket, weight=-10.0, params={"pocket_local": POCKET, "margin": 0.04, "ref": 0.02})
    collision = RewTerm(func=ag.reward_collision, weight=-150.0)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": ARM})
    posture_box = RewTerm(
        func=rr.penalty_joint_range_box, weight=-2.4,
        params={"asset_cfg": ARM,
                "lower": [-3.09, -2.25, -2.62, -1.05, -1.97, -1.61, -1.61],
                "upper": [2.67, 0.35, 2.62, 2.09, 1.97, 1.61, 1.61]},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(func=ag.collision, params={"pocket_local": POCKET})
    goal_reached = DoneTerm(func=ag.goal_reached, params={"pocket_local": POCKET, "pos_tol": 0.02,
                                                          "ang_tol_deg": 20.0, "speed_tol": 0.15, "hold_time": 0.3})


@configclass
class EventsCfg:
    reset_all_exact = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                                params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.0, 0.0),
                                        "velocity_range": (0.0, 0.0)})
    hold_default_targets = EventTerm(func=task_mdp.hold_default_joint_targets, mode="reset",
                                     params={"asset_cfg": SceneEntityCfg("robot")})
    reset_goal = EventTerm(func=ag.reset_arm_goal, mode="reset",
                           params={"library": POSE_LIBRARY, "basket_range": 0.15, "mass_range": (0.10, 0.35)})
    resolve_goals = EventTerm(func=ag.resolve_goals, mode="interval", interval_range_s=(0.0, 0.0),
                              params={"pocket_local": POCKET, "retreat_range": (0.10, 0.20)})
    payload = EventTerm(func=ag.apply_payload, mode="interval", interval_range_s=(0.0, 0.0),
                        params={"pocket_local": POCKET})


@configclass
class AppleArmGoalEnvCfg(ManagerBasedRLEnvCfg):
    scene: ArmGoalSceneCfg = ArmGoalSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ReachActionsCfg = ReachActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.5, -1.5, 1.8)
        self.viewer.lookat = (0.1, -0.3, 1.0)


@configclass
class AppleArmGoalEnvCfg_PLAY(AppleArmGoalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
