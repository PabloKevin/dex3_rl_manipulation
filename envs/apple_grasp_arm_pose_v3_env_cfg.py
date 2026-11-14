# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Tactile grasp shaped like its use in the harvest (AppleGraspArmPose-v3): AppleGraspArmPose-v2's world (the frozen
AppleArmPose-v5 fruit reach + aim map, camera tracking, fingers only, 65 in / 14 out) with

  - only REACHABLE targets (user, 2026-11-14: "some low apples seem to be outside the reach ... bad for training
    the grasp"): the tree sampler's map is envs/assets/reach_workspace_aim_v5.npz (scripts/build_reach_map_aim.py:
    the old map AND the aim map's predicted reach success >= 0.5 at every waist yaw of the range) and no
    "stretch" targets (v2 inherited STRETCH_FRAC = 0.10 from the reach training: 10% of the targets drawn just
    OUTSIDE the map on purpose). v2's hand-off: the reach arrived in only 75-84% of the episodes;
  - the episode the harvest bench runs (grasp_v3.py): the policy closes the hand; once two finger groups read
    > 0.5 N for 0.3 + 0.5 s (or 4.1 s after the hand-off) the fingers FREEZE at that command, the target stem
    gets the bench's break limits (6-10 N pull, 0.15-0.25 N m bend) and the ARM goes back along its own reach
    path (the bench's --detach_path monotone, then home): the real stem, the fruit's inertia, gravity turning
    with the hand. SUCCESS (+100, as large as the loss) = the stem has broken and the fruit is still in the hand
    1.5 s later (within 6 cm of the pocket: the bench's LEAVE). v2: the stem let go on a timer, the policy kept regulating the fingers under a
    synthetic twist + pull with the arm still (the bench freezes them: holding on in the policy's way dropped 12
    of 16), and its success term (a lift) could never fire with the arm locked. (A first v3 smoke test with v2's
    synthetic pull -- force along the palm normal up to 6-14 N + 2 N carry -- held 15-23% where the same
    policy deposits 65% on the bench: harsher than the real detach.);
  - warm start from v2 model_300 (the best grasp so far on the bench, 65% with frozen fingers): it already
    closes, so the run starts at the bench's level instead of from nothing (two from-scratch v2 runs never closed);
  - PPO: 48 steps per env (v2: 32, for ~800-step episodes), entropy 0.0005, gamma 0.995 (the outcome comes
    ~2.5 s = 125 steps after the close that decides it).
Training (2026-11-14): with success tested 5 s after the freeze and gamma 0.99 the held share stayed flat at ~0.39
for 110 iterations (runs ..._tol35_stopped_it64, 2026-11-14_06-11-20): 0.99^250 = 8% of the outcome reached the
close. Then success at break + 1.5 s, gamma 0.995, warm start from that run's model_100.
Result (run 2026-11-14_06-42-19, stopped at model_300): held ~0.40-0.44, lost after a grip ~0.30, damaged ~0.10,
FLAT over 300 iterations (value loss ~95 vs +-100 outcomes: the outcome is barely predictable from the policy's
inputs). Bench (v5 + aim + --skip_q 0.5, seeds 11-15 x 16 trees, fingers frozen at the grip): model_300 52/72
attempted = 72%, model_100 51/72, v2 model_300 (the warm start) 52/72 -- no gain. Remaining misses: slipped in the
detach 8, nothing between the fingers after the close 6, stem held 4, dropped in the carry 1, reach 1.
"""

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .apple_grasp_arm_pose_v2_env_cfg import (
    AppleGraspArmPoseV2EnvCfg,
    CurriculumV2Cfg,
)
from .apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL
from .mdp import grasp_v3 as g3
from .mdp import probe_grasp as pg

#: apple centres the v5 reach gets to (scripts/build_reach_map_aim.py)
V3_WORKSPACE_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "reach_workspace_aim_v5.npz")
V3_STEPS_PER_ITER = 48
#: the bench's grip rule (play_harvest_robot.py, S_GRASP with --learned_grasp) and its stem weakening
V3_FREEZE = dict(level_n=0.5, groups=2, grip_s=0.3, hold_s=0.5, deadline_s=4.1, pull_break=(6.0, 10.0),
                 bend_break=(0.15, 0.25))
#: success: the stem broken and the fruit still in the hand this long after the break (first try: 5 s after the
#: freeze -- too late for the credit to reach the close, see grasp_v3.harvest_held)
V3_CARRY_S = 1.5
#: hand-off watchdog 8 s + grasp <= 4.1 s + the detach (v2: 16 s)
V3_EPISODE_S = 18.0


@configclass
class CurriculumV3Cfg(CurriculumV2Cfg):
    v3 = CurrTerm(func=g3.v3_metrics)


@configclass
class AppleGraspArmPoseV3EnvCfg(AppleGraspArmPoseV2EnvCfg):
    curriculum: CurriculumV3Cfg = CurriculumV3Cfg()

    def __post_init__(self):
        super().__post_init__()
        ev = self.events

        # reachable targets only
        tree = ev.randomize_tree
        tree.params["workspace_map"] = V3_WORKSPACE_MAP
        tree.params["seed_envelope"] = {**tree.params["seed_envelope"], "stretch_frac": 0.0}

        # the hand: frozen at the bench's grip; the arm: back along its own reach path once frozen
        def _as(cls, old, **kw):
            return cls(**{k: getattr(old, k) for k in old.__dataclass_fields__ if k != "class_type"}, **kw)

        self.actions.right_hand = _as(g3.FreezableHandActionCfg, self.actions.right_hand)
        self.actions.right_arm = _as(g3.PathBackArmActionCfg, self.actions.right_arm)
        self.episode_length_s = V3_EPISODE_S

        # the stem is never released by a timer or a synthetic load: the freeze weakens it to the bench's break
        # limits and the arm going back pulls the fruit off (or not)
        ev.reset_reach_grasp_phase.params["stem_release_schedule"] = (1.0e3, 1.0e3, 0, 1, 0.0)
        ev.reset_reach_grasp_phase.params["release_on_latch"] = False
        ev.release_stem_and_load = None
        ev.reset_v3 = EventTerm(func=g3.reset_v3, mode="reset")
        ev.freeze_on_bench_grip = EventTerm(func=g3.freeze_on_bench_grip, mode="interval",
                                            interval_range_s=(0.0, 0.0), is_global_time=True,
                                            params={"sensor_cfg": SceneEntityCfg("contact_forces"), **V3_FREEZE})

        # success: held through the pull and the carry
        self.terminations.harvest_held = DoneTerm(func=g3.harvest_held, params={
            "object_cfg": SceneEntityCfg("apple"), "robot_cfg": SceneEntityCfg("robot"),
            "center_local": GRASP_CENTER_LOCAL, "carry_s": V3_CARRY_S})
        self.rewards.harvest_success = RewTerm(func=pg.penalty_terminated, weight=100.0,
                                               params={"term_keys": ["harvest_held"]})


@configclass
class AppleGraspArmPoseV3EnvCfg_PLAY(AppleGraspArmPoseV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
