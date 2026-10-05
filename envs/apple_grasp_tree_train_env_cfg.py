# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp TRAINING on the multi-apple tree (AppleGraspTree-v1): the grasp task
(AppleGrasp-v1) retrained with neighbours around the fruit, the fruit on a
real stem, and a hold-under-load stage.

Built on the tree bench's env (apple_grasp_tree_env_cfg.AppleGraspTreeEnvCfg:
7 apples in clusters, every fruit on a breakable stem, no rigid pin), with
the training-time choices back:
  - exact joint reset + start-pose jitter every episode, a NEW tree every
    episode, the target any of its apples;
  - the reach phase driven by the tree-trained reach policy
    (best_weights/grasp_handoff_reach_tree.pt, 37-dim observation);
  - observation 66 -> 41, HAND ONLY (apple_grasp_tree_env_cfg.
    HandOnlyObservationsCfg): no arm joint states, no palm orientation, no
    arm half of the last action -- so the same policy can keep driving the
    fingers through the harvest twist and carry, where the arm moves as it
    never did in training; tactile terms read any apple, pad face only (no
    neighbour ring: after the reach the grasp can barely move the hand, the
    squeeze should not depend on the neighbours, and the hand hides them
    from the camera exactly then);
  - stem release 2.0-2.5 s after the hand-off (no curriculum: the
    warm-started policy already grasps), then the HOLD-UNDER-LOAD stage
    (envs/mdp/tree_grasp.py): the stem lets go and a twist torque about
    the palm normal + a force ramp up on the fruit, scaled by a curriculum
    over iterations 100 -> 800;
  - rewards: + holding the fruit in the pocket while loaded, - its slip,
    - neighbours pushed / knocked off.
Grasp-phase window 6 s (5 s in AppleGrasp-v1): release ~2.3 s + 0.4 s delay
+ 1 s ramp leaves ~2.3 s of full load.
"""

import os

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from .apple_grasp_grasp_env_cfg import (
    GRASP_CENTER_LOCAL,
    GRASP_PPO_STEPS_PER_ITER,
    REACH_HANDOFF_TIME_S,
    RewardsCfg as GraspRewardsCfg,
    RIGHT_ARM_JOINTS,
)
from .apple_grasp_tree_env_cfg import AppleGraspTreeEnvCfg, HandOnlyObservationsCfg
from .mdp import tree_grasp as tg
from .mdp import tree_reach as trr
from .mdp.terminations_and_events import hold_default_joint_targets

TREE_REACH_POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "best_weights",
                                 "grasp_handoff_reach_tree.pt")
TREE_GRASP_WINDOW_S = 6.0
#: Hold-under-load stage (tree_grasp.release_stem_and_load).
LOAD_FORCE = 6.0              # N, random direction
LOAD_TORQUE = 0.4             # N*m about the palm normal (tree stems break at 0.25-0.40 in twist)
LOAD_DELAY = 0.4              # s after the stem lets go
LOAD_RAMP = 1.0               # s to full load
LOAD_CURRICULUM = (100, 800)  # iterations: no load -> full load


@configclass
class RewardsCfg(GraspRewardsCfg):
    # Holding against the load: per step, ~the size of lifted_hold (10) at
    # full load, so a grip that survives the harvest roughly doubles the
    # grasp's reward rate.
    hold_under_load = RewTerm(func=tg.reward_hold_under_load, weight=10.0,
                              params={"center_local": GRASP_CENTER_LOCAL, "tol": 0.02,
                                      "delay": LOAD_DELAY, "ramp": LOAD_RAMP})
    slip_under_load = RewTerm(func=tg.penalty_slip_under_load, weight=-1.0, params={"ref": 0.02})
    # Neighbours (same terms as the tree reach).
    neighbour_push = RewTerm(func=trr.penalty_neighbour_push, weight=-2.0, params={"deadband": 0.005, "ref": 0.02})
    neighbour_knocked = RewTerm(func=trr.penalty_neighbour_knocked, weight=-100.0)


@configclass
class AppleGraspTreeTrainEnvCfg(AppleGraspTreeEnvCfg):
    observations: HandOnlyObservationsCfg = HandOnlyObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = REACH_HANDOFF_TIME_S + TREE_GRASP_WINDOW_S

        # the tree-trained reach drives the reach phase
        self.actions.right_arm.reach_policy_path = TREE_REACH_POLICY
        self.actions.right_arm.reach_obs_layout = "tree"

        ev = self.events
        # training resets (the bench keeps the joints between apples; here
        # every episode starts from home, jittered like the reach training)
        ev.reset_all_exact = EventTerm(func=base_mdp.reset_joints_by_offset, mode="reset",
                                       params={"asset_cfg": SceneEntityCfg("robot"),
                                               "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})
        ev.hold_default_targets = EventTerm(func=hold_default_joint_targets, mode="reset",
                                            params={"asset_cfg": SceneEntityCfg("robot")})
        ev.reset_arm_jitter = EventTerm(func=base_mdp.reset_joints_by_offset, mode="reset",
                                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS),
                                                "position_range": (-0.20, 0.20), "velocity_range": (0.0, 0.0)})
        ev.randomize_tree.params.update(order="random", new_tree_each_reset=True)
        ev.reset_reach_grasp_phase.params.update(
            release_on_latch=False, stem_release_schedule=None, stem_release_time_range=(2.0, 2.5),
            shaping_grace_s=None)
        ev.reset_load = EventTerm(func=tg.reset_load, mode="reset")
        ev.release_stem_and_load = EventTerm(
            func=tg.release_stem_and_load, mode="interval", interval_range_s=(0.0, 0.0),
            params={"load_force": LOAD_FORCE, "load_torque": LOAD_TORQUE, "delay": LOAD_DELAY,
                    "ramp": LOAD_RAMP, "curriculum": LOAD_CURRICULUM,
                    "steps_per_iter": GRASP_PPO_STEPS_PER_ITER},
        )


@configclass
class AppleGraspTreeTrainEnvCfg_PLAY(AppleGraspTreeTrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
