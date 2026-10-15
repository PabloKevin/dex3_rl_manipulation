# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp that finishes the approach by feel (AppleGraspSingle-v2): the
AppleGraspSingle-v1 grasp (apple_grasp_single_env_cfg.py -- subclassed,
unchanged: frozen single reach, spindle tree, stem release + twist / pull)
with what it needs to FIND the fruit after a reach that stops 2 cm short of a
perceived point that is itself off by up to 2.5 cm (envs/mdp/probe_grasp.py):

  - inputs 44 -> 65, appended (the v3 warm start keeps its columns):
    tactile_taxels (7 places: thumb / middle / index prox + dist, palm),
    tactile_memory (their decaying peak), the arm's last action (the offset
    it has commanded since the hand-off). v1 saw the fingers only as a
    per-finger mean (base and tip together), the palm only through a
    derivative, and the arm not at all -- so "fruit on the tips" and "fruit
    at the finger bases" looked the same, and the arm was blind;
  - tactile realism: dead band per taxel (palm 1.5-3.5 N: the real palm
    answered 2 of 9 elements at 2.45 N), gain 0.6-1.6x, dead taxels,
    per-step noise -- for every sensed reader, the grip detector included;
  - a softer branch: the fruit sways under a palm push (lateral 40-150 N/m,
    bend 0.3-0.7 N*m/rad, break limits unchanged), so pushing it before
    feeling it is what happens, as on a real tree;
  - the reach hands off with the fruit 6-21 cm from the pocket (measured
    with policy_1600 in v1, where 20 of the episodes seen ended right there:
    apple_fell_from_grip fired on the first grasp step). Here the fruit
    only counts as lost once it is off its stem (probe_grasp.
    apple_left_hand), and the arm action is 0.15 rad / unit (v1 0.05:
    ~6 cm of travel), so the grasp can finish the approach;
  - time to probe: stem release 3.5-4.5 s after the hand-off (was 2.0-2.5),
    grasp window 8 s (was 6);
  - rewards: + pocket progress, + contact depth (palm > base > tip),
    - fruit pushed off its spot, - stem strained, - stem broken early,
    - fruit lost, - fruit damaged, - trunk too close / hit; the
    grasp-quality terms scaled by how centred the fruit is (25% on the
    tips), hand_still only after the release (re-opening to reposition is
    free), finger_approach 4 -> 2, the posture penalty on shoulder / elbow
    10 -> 2 (moving away from the hand-off is the job now).

The perceived apple point is unchanged (APPLE_MEAS_NOISE: one point per
episode, held): the frozen reach and the grasp share it, as on the robot.
"""

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL
from .apple_grasp_single_env_cfg import SINGLE_HANDOFF_WATCHDOG_S, AppleGraspSingleEnvCfg
from .apple_grasp_tree_env_cfg import HandOnlyObservationsCfg, TREE_STEM_PARAMS
from .mdp import probe_grasp as pg

#: grasp-phase window after the hand-off (v1: 6 s)
SINGLE_V2_GRASP_WINDOW_S = 8.0
#: stem release, s after the hand-off (v1: 2.0-2.5)
SINGLE_V2_RELEASE_S = (3.5, 4.5)
#: the branch the fruit hangs from: softer than harvest.StemParams (lateral
#: 100-300 N/m, bend 0.6-1.0), so a push moves the fruit. Guesses.
SINGLE_V2_STEM_PARAMS = {**TREE_STEM_PARAMS, "k_lateral": (40.0, 150.0), "k_bend": (0.3, 0.7)}
#: tactile model (probe_grasp.reset_tactile_realism). Palm from the 2026-10-14
#: force calibration (2 of 9 palm elements at 2.45 N); the rest guesses.
TACTILE_REALISM = dict(finger_deadband=(0.2, 0.6), palm_deadband=(1.5, 3.5), gain_log_range=(0.6, 1.6),
                       dead_taxel_prob=0.03, noise_rel=0.10, noise_abs=0.15)
#: trainee arm action, rad per unit (offset from the hand-off pose; v1: 0.05,
#: ~6 cm of hand travel at most). The single reach hands off with the fruit
#: 6-21 cm from the pocket (measured, policy_1600 in AppleGraspSingle-v1), so
#: the grasp must be able to finish that approach. Rate limits unchanged.
SINGLE_V2_ARM_SCALE = 0.15
#: grasp-quality terms scaled by how centred the fruit is (probe_grasp.centred)
CENTRED_TERMS = ("grasp_enclosure", "gentle_contact_groups", "grasp_balance", "finger_wrap", "palm_support",
                 "finger_curl", "grasp_setpoint_tracking", "grasp_hold")

_CONTACT = SceneEntityCfg("contact_forces")


@configclass
class SingleV2ObservationsCfg(HandOnlyObservationsCfg):
    """The 44 hand-only inputs, then 21 new ones (appended: warm start)."""

    @configclass
    class PolicyObs(HandOnlyObservationsCfg.PolicyObs):
        tactile_taxels = ObsTerm(func=pg.tactile_taxels, params={"sensor_cfg": _CONTACT})
        tactile_memory = ObsTerm(func=pg.tactile_memory, params={"sensor_cfg": _CONTACT, "tau_s": 1.0})
        last_action_arm = ObsTerm(func=base_mdp.last_action, params={"action_name": "right_arm"})

    policy: PolicyObs = PolicyObs()


@configclass
class AppleGraspSingleV2EnvCfg(AppleGraspSingleEnvCfg):
    observations: SingleV2ObservationsCfg = SingleV2ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = SINGLE_HANDOFF_WATCHDOG_S + SINGLE_V2_GRASP_WINDOW_S

        self.actions.right_arm.scale = SINGLE_V2_ARM_SCALE
        # "fell from the grip" only once the fruit is off its stem: before
        # that, a fruit far from the pocket is one still to be reached
        old = self.terminations.apple_fell_from_grip
        self.terminations.apple_fell_from_grip = DoneTerm(func=pg.apple_left_hand, params=dict(old.params))

        ev = self.events
        ev.randomize_tree.params["stem_params"] = dict(SINGLE_V2_STEM_PARAMS)
        ev.reset_reach_grasp_phase.params.update(stem_release_time_range=SINGLE_V2_RELEASE_S,
                                                 stem_release_schedule=None)
        # the realism model owns the gain now
        ev.randomize_tactile_gain.params["gain_range"] = (1.0, 1.0)
        ev.tactile_realism = EventTerm(func=pg.reset_tactile_realism, mode="reset", params=dict(TACTILE_REALISM))
        ev.probe_state_reset = EventTerm(func=pg.reset_probe_state, mode="reset")

        rw = self.rewards
        for name in CENTRED_TERMS:
            old = getattr(rw, name)
            setattr(rw, name, RewTerm(func=pg.centred, weight=old.weight, params={
                "term": old.func.__name__, "inner": dict(old.params), "center_local": GRASP_CENTER_LOCAL,
                "std": 0.02, "floor": 0.25}))
        old = rw.hand_still
        rw.hand_still = RewTerm(func=pg.after_release, weight=old.weight,
                                params={"term": old.func.__name__, "inner": dict(old.params)})
        rw.finger_approach.weight = 2.0
        rw.posture_from_handoff.params["weights"] = [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]

        # find the fruit, bring it into the pocket
        rw.pocket_progress = RewTerm(func=pg.reward_pocket_progress, weight=20.0,
                                     params={"center_local": GRASP_CENTER_LOCAL, "scale_far": 0.10,
                                             "scale_near": 0.02})
        rw.contact_depth = RewTerm(func=pg.reward_contact_depth, weight=2.0, params={"sensor_cfg": _CONTACT})
        # do not push it away, do not detach it
        rw.apple_displacement = RewTerm(func=pg.penalty_apple_displacement, weight=-4.0)
        rw.stem_stress = RewTerm(func=pg.penalty_stem_stress, weight=-3.0)
        # one-off events: weight = reward per event (the functions divide by dt)
        rw.stem_broken_early = RewTerm(func=pg.penalty_stem_broken_early, weight=-50.0)
        rw.apple_lost = RewTerm(func=pg.penalty_terminated, weight=-100.0,
                                params={"term_keys": ["apple_fell_from_grip", "apple_dropped"]})
        rw.fruit_damaged_term = RewTerm(func=pg.penalty_terminated, weight=-50.0,
                                        params={"term_keys": ["fruit_damaged"]})
        # keep off the trunk
        rw.trunk_contact = RewTerm(func=pg.penalty_trunk_contact, weight=-5.0, params={"margin": 0.02, "ref": 0.02})
        rw.trunk_hit = RewTerm(func=pg.penalty_trunk_hit, weight=-30.0)


@configclass
class AppleGraspSingleV2EnvCfg_PLAY(AppleGraspSingleV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
