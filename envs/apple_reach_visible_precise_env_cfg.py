# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
"v1.5": the visible reach, more precise at the goal (AppleReachVisiblePrecise-v1).

User, 2026-10-18: raise the reach's enclosure rate (the grasp bench showed
every apple the reach hands off enclosed ends in the basket with the scripted
grasp, and looser hand-off gates do not help: the extra apples fall out in
the twist). AppleReachVisible-v1 (apple_reach_visible_env_cfg.py --
subclassed: same 37 inputs / 7 outputs, same world) with the goal rewards
sharpened where the v1 policy fails. Independent of AppleReachVisible-v2
(camera tracking / wide starts / delay, another session): own file, own log
folder logs/rsl_rl/apple_reach_visible_v15.

Diagnosis of the v1 policy (2819 deterministic episodes in its own env):
success 27%; the failures end with the apple pushed in or touching, but the
palm TILT past the 40 deg gate (median end tilt 39 deg; tilt-only = a third
of the failures) or the apple OFF-CENTRE (lateral median 2.4 cm vs the 2 cm
gate). Success by apple height vs the shoulder: -45..-30 cm 43%, -30..-20
33%, -20..-10 10%, -10..0 0% (tilt there 58 deg). The dense terms pay
"almost" nearly in full -- palm_lateral was free to 40 deg and still paid
~70% at 50; enclosed_coarse paid ~85% at 2.4 cm -- so only the sparse
success bonus told the right pose apart. Changes:

  - tilt: palm_lateral free only to 30 deg (the fingertip-equalising lean the
    probe measured), falling off with std 0.1 rad (was 40 deg / 0.2), weight
    6 -> 8;
  - centring: enclosed's lateral std 1.5 -> 1.0 cm, enclosed_coarse 6 -> 4,
    + enclosed_centre (lateral std 0.7 cm, weight 6, level-gated like the
    others);
  - hold 5 -> 8 (paid only inside every gate: settle there; 4% of v1's
    failures had every gate at the end but not for 0.5 s);
  - placement: 35% of the first apples (the target is among them) take
    their height from the hard band, -22..+5 cm vs the shoulder
    (Envelope dz_focus).
Success, hold and hand-off gates are unchanged (tilt 40 deg, lateral 2 cm),
so the curves compare with v1 and the AppleGraspVisible hand-off is the same.
Warm start: logs/rsl_rl/apple_reach_visible_v15/warmstart_v1/model_0.pt
(scripts/warmstart_from_jit.py: the v1 actor + its input normalizer from
best_weights/grasp_handoff_reach_visible.pt, a fresh critic).
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import AppleReachVisiblePPORunnerCfg
from .apple_reach_visible_env_cfg import AppleReachVisibleEnvCfg, _GATE, _GOAL
from .mdp import single_reach as sr
from .mdp import visible_reach as vr

PRECISE_TILT_FREE_RAD = math.radians(30.0)
PRECISE_TILT_STD_RAD = 0.1
PRECISE_LATERAL_STD = 0.010
PRECISE_CENTRE_STD = 0.007
#: share of first apples drawn in the hard height band (shoulder-relative, m)
PRECISE_DZ_FOCUS = (-0.22, 0.05)
PRECISE_DZ_FOCUS_PROB = 0.35


@configclass
class AppleReachVisiblePreciseEnvCfg(AppleReachVisibleEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        # tilt: free to 30 deg, then a steep fall-off (the term is level-gated: its own params are inside)
        tp = rw.palm_lateral.params["term_params"]
        tp["free_rad"] = PRECISE_TILT_FREE_RAD
        tp["std_rad"] = PRECISE_TILT_STD_RAD
        rw.palm_lateral.weight = 8.0
        # centring
        rw.enclosed.params["term_params"]["lateral_std"] = PRECISE_LATERAL_STD
        rw.enclosed_coarse.weight = 4.0
        centre = dict(rw.enclosed.params["term_params"])
        centre["lateral_std"] = PRECISE_CENTRE_STD
        rw.enclosed_centre = RewTerm(func=vr.level_gated, weight=6.0,
                                     params={"term": sr.reward_enclosed, "term_params": centre, **_GOAL, **_GATE})
        # settle inside the gates
        rw.hold.weight = 8.0
        # placement: more of the hard band near shoulder height
        self.events.randomize_tree.params["seed_envelope"].update(
            dz_focus=PRECISE_DZ_FOCUS, dz_focus_prob=PRECISE_DZ_FOCUS_PROB)


@configclass
class AppleReachVisiblePreciseEnvCfg_PLAY(AppleReachVisiblePreciseEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None


@configclass
class AppleReachVisiblePrecisePPORunnerCfg(AppleReachVisiblePPORunnerCfg):
    """Own log folder (logs/rsl_rl/apple_reach_visible_v15): v1 and v2 runs untouched."""
    experiment_name = "apple_reach_visible_v15"
    max_iterations = 2000
    save_interval = 50
