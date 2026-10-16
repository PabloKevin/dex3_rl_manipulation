# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp that ONLY grasps (AppleGraspSingle-v3), behind AppleReachSingle-v2.

User, 2026-10-15: the reach now ends with the apple enclosed in the hand
(apple_reach_single_v2_env_cfg.py), so the grasp must close, not finish the
approach -- arm adjustment is being worked on elsewhere. AppleGraspSingle-v2
(apple_grasp_single_v2_env_cfg.py -- subclassed: 65 inputs with per-place
tactile + memory + the arm's last action, tactile realism, stem release
curriculum, centred grasp-quality terms, fruit lost only once off its stem)
with:

  - reach phase: the frozen AppleReachSingle-v2 policy
    (best_weights/grasp_handoff_reach_single_v2.pt, or
    $REACH_SINGLE_V2_POLICY; same "single" 40-input layout), its hand
    trigger (cup once within 10 cm of the perceived apple), and its hand-off
    test: single_reach.enclosed held 0.5 s (set_reach_goal_v2, the same
    switch as in the reach), SINGLE_HANDOFF_WATCHDOG_S at the latest;
  - its tree: the re-measured map, stretch apples, stems swinging like a
    pendulum (k_lateral 15-40 N/m) -- with the real-tree break limits the
    grasp has to twist against (TREE_STEM_PARAMS), not the magnet's;
  - the arm back to a grasp's: 0.05 rad per unit (v2: 0.15), wrist only
    until the grip (pre_grip_free_dims 3; v2: whole arm), posture penalty
    from the hand-off back to 10 on shoulder / elbow (v2: 2);
  - pocket_progress 20 -> 8: the fruit arrives in the pocket.
"""

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .apple_grasp_single_v2_env_cfg import AppleGraspSingleV2EnvCfg
from .apple_grasp_tree_env_cfg import TREE_STEM_PARAMS
from .apple_reach_single_v2_env_cfg import (
    PENDULUM_STEM_PARAMS,
    SINGLE_V2_HAND_NEAR,
    SINGLE_V2_LATERAL_TOL,
    SINGLE_V2_MAX_DISP,
    SINGLE_V2_PUSH_IN,
    SINGLE_V2_SHORT_TOL,
    SINGLE_V2_STANDOFF,
    SINGLE_V2_TILT_TOL,
    apply_single_v2_placement,
)
from .mdp import single_reach as sr

SINGLE_V2_REACH_POLICY = os.environ.get(
    "REACH_SINGLE_V2_POLICY",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "best_weights",
                 "grasp_handoff_reach_single_v2.pt"))
#: trainee arm action, rad per unit (as the spindle grasp; v2: 0.15)
SINGLE_V3_ARM_SCALE = 0.05
#: real-tree stem limits (the grasp twists the fruit off), swinging like the reach's
SINGLE_V3_STEM_PARAMS = {**TREE_STEM_PARAMS, "k_lateral": PENDULUM_STEM_PARAMS["k_lateral"], "k_bend": (0.3, 0.7)}


@configclass
class AppleGraspSingleV3EnvCfg(AppleGraspSingleV2EnvCfg):

    def __post_init__(self):
        super().__post_init__()
        arm = self.actions.right_arm
        arm.reach_policy_path = SINGLE_V2_REACH_POLICY
        arm.reach_obs_layout = "single"
        arm.scale = SINGLE_V3_ARM_SCALE
        arm.unlock_with_grip = True
        arm.pre_grip_free_dims = 3

        ev = self.events
        ev.reach_goal_v2 = EventTerm(func=sr.set_reach_goal_v2, mode="startup",
                                     params={"touch_gap": SINGLE_V2_STANDOFF, "push_in": SINGLE_V2_PUSH_IN,
                                             "short_tol": SINGLE_V2_SHORT_TOL, "lateral_tol": SINGLE_V2_LATERAL_TOL,
                                             "max_disp": SINGLE_V2_MAX_DISP, "heading_horizontal": True})
        ev.hand_schedule.params.update(trigger="distance", near=SINGLE_V2_HAND_NEAR)
        ev.update_reach_grasp_phase.params.update(standoff=SINGLE_V2_STANDOFF, tilt_tol_rad=SINGLE_V2_TILT_TOL)
        apply_single_v2_placement(ev.randomize_tree)
        ev.randomize_tree.params["stem_params"] = dict(SINGLE_V3_STEM_PARAMS)

        rw = self.rewards
        rw.posture_from_handoff.params["weights"] = [10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
        rw.pocket_progress.weight = 8.0


@configclass
class AppleGraspSingleV3EnvCfg_PLAY(AppleGraspSingleV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
