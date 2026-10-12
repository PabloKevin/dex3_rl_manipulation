# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Goal-reaching arm policy for the harvest path WITH the waist yaw
(AppleArmGoalWaist-v1): AppleArmGoal-v1 (apple_arm_goal_env_cfg.py --
subclassed) for the waist reach. The reach (AppleReachSpindle-v3) leaves
the torso turned (|yaw| median ~12 deg, p90 ~28 deg); the harvest path has
to start from there and bring the waist back to 0 at home, where every
reach starts.

Differences from AppleArmGoal-v1:
  - action (7 -> 8): + the waist yaw, the reach's rate-limited, braking
    term (+-45 deg); arm targets clamped to the soft joint limits, as v3;
    waist actuator with the deployed gains;
  - observation (34 -> 37): goal - pocket and basket - pocket in the torso
    frame WITHOUT the waist yaw (identical to the old inputs at yaw 0: the
    arm-only policy warm-starts exactly), + waist yaw position, velocity
    and last action, appended;
  - starts: 70% of the pick-like ones from the v3 reach's real end poses on
    the spindle tree, waist included (scripts/record_reach_starts.py
    --spindle -> envs/assets/reach_end_poses_waist.npz); the others with a
    waist yaw in +-0.4 rad;
  - HOME succeeds only with the waist back within 5 deg of 0 (+ a reward
    for it); SAFE keeps its straight-back path term (a waist turn would
    sweep the hand sideways through the fruit wall); DROP leaves the waist
    free (the basket is to the right, behind the shoulder);
  - + waist commanded smoothness and speed, as in the reach.
"""

import os

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp as task_mdp
from .apple_arm_goal_env_cfg import (
    AppleArmGoalEnvCfg,
    ObservationsCfg as ArmGoalObservationsCfg,
    POCKET,
    POSE_LIBRARY,
    RewardsCfg as ArmGoalRewardsCfg,
    TerminationsCfg as ArmGoalTerminationsCfg,
)
from .apple_reach_spindle_env_cfg import ARM_TARGET_CLIP
from .apple_reach_tree_waist_env_cfg import ActionsCfg as WaistActionsCfg, WAIST_YAW_JOINT
from .mdp import arm_goal as ag
from .mdp import tree_reach as trr

WAIST_BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "reach_end_poses_waist.npz")
_WAIST = SceneEntityCfg("robot", joint_names=[WAIST_YAW_JOINT], preserve_order=True)
#: HOME = waist within this of 0.
HOME_WAIST_TOL_DEG = 5.0


@configclass
class ObservationsCfg(ArmGoalObservationsCfg):
    @configclass
    class PolicyObs(ArmGoalObservationsCfg.PolicyObs):
        # same slots, the torso frame without the waist yaw
        goal_pocket = ObsTerm(func=ag.obs_goal_pocket, params={"pocket_local": POCKET, "unyaw": True})
        basket = ObsTerm(func=ag.obs_basket, params={"pocket_local": POCKET, "unyaw": True})
        # appended (34 -> 37)
        waist_pos = ObsTerm(func=task_mdp.joint_pos_selected, params={"asset_cfg": _WAIST})
        waist_vel = ObsTerm(func=task_mdp.joint_vel_selected, params={"asset_cfg": _WAIST})
        waist_last_action = ObsTerm(func=mdp.last_action, params={"action_name": "waist_yaw"})

    policy: PolicyObs = PolicyObs()


@configclass
class RewardsCfg(ArmGoalRewardsCfg):
    waist_home = RewTerm(func=ag.reward_waist_home, weight=3.0, params={"std_rad": 0.15})
    commanded_accel_waist = RewTerm(func=trr.penalty_commanded_accel, weight=-0.5,
                                    params={"action_name": "waist_yaw"})
    limiter_lead_waist = RewTerm(func=trr.penalty_limiter_lead, weight=-1.0,
                                 params={"action_name": "waist_yaw", "ref": 0.5})
    waist_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.1, params={"asset_cfg": _WAIST})


@configclass
class TerminationsCfg(ArmGoalTerminationsCfg):
    goal_reached = DoneTerm(func=ag.goal_reached, params={"pocket_local": POCKET, "pos_tol": 0.02,
                                                          "ang_tol_deg": 20.0, "speed_tol": 0.15, "hold_time": 0.3,
                                                          "home_waist_tol_deg": HOME_WAIST_TOL_DEG})


@configclass
class AppleArmGoalWaistEnvCfg(AppleArmGoalEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: WaistActionsCfg = WaistActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 6.0          # 5 s + the waist's way home
        acts = dict(self.scene.robot.actuators)
        acts["legs"] = acts["legs"].replace(joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"])
        acts["waist_yaw"] = ImplicitActuatorCfg(joint_names_expr=[WAIST_YAW_JOINT], effort_limit=88.0,
                                                velocity_limit=32.0, stiffness=200.0, damping=5.0)
        self.scene.robot = self.scene.robot.replace(actuators=acts)
        self.actions.right_arm.clip = dict(ARM_TARGET_CLIP)
        self.events.reset_goal.params.update({"bank_file": WAIST_BANK, "waist_range": 0.4,
                                              "library": POSE_LIBRARY})


@configclass
class AppleArmGoalWaistEnvCfg_PLAY(AppleArmGoalWaistEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
