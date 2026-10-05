# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach task on a multi-apple tree (AppleReachTree-v1): the side-on reach
(AppleReach-v1, apple_grasp_env_cfg.py -- subclassed, unchanged) retrained
with the target's neighbours in the way.

Differences from AppleReach-v1:
  - scene: the 7-apple tree of apple_grasp_tree_env_cfg (clusters of 1-4),
    a fresh tree every episode, the target any of its apples; EVERY fruit,
    the target included, is a dynamic sphere on a breakable stem (the
    single-apple reach pushed against an immovable, kinematic apple);
  - observation (29 -> 37): palm_touch_force reads ANY contact, as the real
    palm sensor does, and an 8-sector clearance ring around the target
    (tree_reach.neighbour_clearance_ring) is appended;
  - rewards: + neighbours pushed / knocked off, + target knocked off, +
    smoothness of the COMMANDED trajectory (its acceleration, and how far the
    policy asks beyond the rate limiter -- the frozen v1 reach kept the
    limiter saturated, accelerating and braking flat out: visible jerk);
  - termination: the target knocked off its stem.
The position / orientation / gentle-palm-touch rewards and the success gate
are the v1 ones; they read contact with the TARGET only (privileged), so
touching a neighbour never pays.

New observation size: the grasp task's reach hand-off (grasp_autoplay.
_reach_obs) assembles the v1 29-dim vector, so a policy trained here needs
that assembly extended before the grasp task / harvest benches can use it.
"""

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from .apple_grasp_env_cfg import (
    APPLE_RADIUS,
    AppleGraspEnvCfg,
    BRANCH_RADIUS,
    BRANCH_YAW_SPREAD_DEG,
    GRASP_STANDOFF,
    ObservationsCfg as ReachObservationsCfg,
    REACH_AZIMUTH_CENTER_DEG,
    REACH_AZIMUTH_SPAN_DEG,
    REACH_MAX,
    REACH_MIN,
    REACH_MIN_FORWARD,
    REACH_MIN_HORIZONTAL,
    REACH_TARGET_DZ_RANGE,
    REACH_WORKSPACE_MAP,
    RewardsCfg as ReachRewardsCfg,
    STEM_LENGTH_RANGE,
    STEM_RADIUS,
    TerminationsCfg as ReachTerminationsCfg,
    TRUNK_HEIGHT_RANGE,
    TRUNK_RADIUS,
)
from .apple_grasp_tree_env_cfg import (
    APPLES_PER_TREE,
    CLUSTER_GAP,
    CLUSTER_SIZES,
    CLUSTER_WEIGHTS,
    TREE_APPLE_RADIUS_RANGE,
    TREE_SEPARATION,
    TREE_STEM_PARAMS,
    TRUNK_DISTANCE,
    TRUNK_JITTER,
    add_tree_apples,
)
from .mdp import apple_tree
from .mdp import tree_reach as trr

#: Clearance ring: sectors and the distance at which a neighbour stops
#: counting (beyond it the sector reads 1 = free).
RING_SECTORS = 8
RING_RANGE = 0.10


@configclass
class ObservationsCfg(ReachObservationsCfg):
    @configclass
    class PolicyObs(ReachObservationsCfg.PolicyObs):
        # same slot as v1, but from any contact (the real palm sensor)
        palm_touch_force = ObsTerm(
            func=trr.palm_contact_force_any,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        neighbour_ring = ObsTerm(
            func=trr.neighbour_clearance_ring,
            params={"n_sectors": RING_SECTORS, "max_range": RING_RANGE},
        )

    policy: PolicyObs = PolicyObs()


@configclass
class RewardsCfg(ReachRewardsCfg):
    # Neighbours: pushed off where they hang (per step, 2 cm beyond 5 mm costs
    # 1.0 before the weight) and knocked off (once per fruit; one-off terms
    # are multiplied by dt like every term, so -100 -> -2, about the success
    # bonus's size: knocking one off costs as much as succeeding pays).
    neighbour_push = RewTerm(func=trr.penalty_neighbour_push, weight=-2.0,
                             params={"deadband": 0.005, "ref": 0.02})
    neighbour_knocked = RewTerm(func=trr.penalty_neighbour_knocked, weight=-100.0)
    # The target knocked off its stem (also ends the episode).
    target_knocked = RewTerm(func=trr.penalty_target_knocked, weight=-100.0)
    # Smooth COMMANDED motion: 1.0 per step = every joint at the limiter's
    # max acceleration / asking 0.5 rad beyond reach, on every joint.
    # -1 -> -3 at iteration 700: the first run smoothed its targets
    # (limiter_lead halved) but kept accelerating as hard (flat ~0.2 of the
    # cap, i.e. ~3.6 rad/s^2 RMS).
    commanded_accel = RewTerm(func=trr.penalty_commanded_accel, weight=-3.0)
    limiter_lead = RewTerm(func=trr.penalty_limiter_lead, weight=-1.0, params={"ref": 0.5})
    # Slow down near the neighbours (added at iteration 700: the reach still
    # swept past fruit at full speed and pushed it). Passing 2 cm from a
    # neighbour at 0.6 m/s costs ~3/step per hand point before the weight.
    hand_speed_near_fruit = RewTerm(func=trr.penalty_hand_speed_near_fruit, weight=-1.0,
                                    params={"near": 0.10, "v_ref": 0.3})


@configclass
class TerminationsCfg(ReachTerminationsCfg):
    target_knocked_off = DoneTerm(func=trr.target_knocked_off)


@configclass
class AppleReachTreeEnvCfg(AppleGraspEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        sc = self.scene
        # the target: a real fruit on a stem (v1: kinematic, immovable)
        sc.apple = sc.apple.replace(spawn=sc.apple.spawn.replace(
            rigid_props=sc.apple.spawn.rigid_props.replace(kinematic_enabled=False, disable_gravity=False)))
        add_tree_apples(sc, APPLES_PER_TREE)

        ev = self.events
        ev.randomize_apple_scale = None
        ev.randomize_tree_geometry = EventTerm(
            func=apple_tree.setup_tree_prims,
            mode="prestartup",
            params={"n_apples": APPLES_PER_TREE, "trunk_height_range": TRUNK_HEIGHT_RANGE,
                    "radius_range": TREE_APPLE_RADIUS_RANGE, "spawn_radius": APPLE_RADIUS},
        )
        ev.randomize_tree = EventTerm(
            func=apple_tree.place_apple_tree,
            mode="reset",
            params={
                "n_apples": APPLES_PER_TREE,
                # cluster seeds AND targets cover the reach task's own envelope
                "seed_envelope": dict(
                    reach_min=REACH_MIN, reach_max=REACH_MAX,
                    az_center_deg=REACH_AZIMUTH_CENTER_DEG, az_span_deg=REACH_AZIMUTH_SPAN_DEG,
                    dz_range=REACH_TARGET_DZ_RANGE, min_forward=REACH_MIN_FORWARD,
                    min_horizontal=REACH_MIN_HORIZONTAL),
                "member_envelope": dict(
                    reach_min=REACH_MIN, reach_max=REACH_MAX,
                    az_center_deg=REACH_AZIMUTH_CENTER_DEG, az_span_deg=REACH_AZIMUTH_SPAN_DEG,
                    dz_range=REACH_TARGET_DZ_RANGE, min_forward=REACH_MIN_FORWARD,
                    min_horizontal=REACH_MIN_HORIZONTAL),
                "workspace_map": REACH_WORKSPACE_MAP,
                "standoff": GRASP_STANDOFF,
                "stem_heading_max_deg": BRANCH_YAW_SPREAD_DEG,
                "stem_length_range": STEM_LENGTH_RANGE,
                "trunk_dist": TRUNK_DISTANCE,
                "trunk_jitter": TRUNK_JITTER,
                "cluster_sizes": CLUSTER_SIZES,
                "cluster_weights": CLUSTER_WEIGHTS,
                "cluster_gap": CLUSTER_GAP,
                "sep_min": TREE_SEPARATION,
                "order": "random",
                "stem_params": TREE_STEM_PARAMS,
                "new_tree_each_reset": True,
                "stem_radius": STEM_RADIUS,
                "branch_radius": BRANCH_RADIUS,
                "trunk_radius": TRUNK_RADIUS,
            },
        )


@configclass
class AppleReachTreeEnvCfg_PLAY(AppleReachTreeEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
