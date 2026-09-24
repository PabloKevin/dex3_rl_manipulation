# Copyright 2025 - G1 Dex3 Apple Grasp Project
# Isaac Lab 4.5 / IsaacSim 4.5, Ubuntu 22.04
"""
Apple Grasp — GRASP phase environment configuration (phase 2 of the
reach -> grasp -> harvest pipeline; harvest is explicitly out of scope here).

Builds on top of apple_grasp_env_cfg.py's already-trained, already-deployed
reach task WITHOUT modifying it — that file/scene/task stays byte-for-byte
as-is, so its exported policy.pt and its whole deployment pipeline
(scripts/reach_apple_hybrid.py, scripts/run_env.py, scripts/play.py) are
unaffected by anything added here. This module imports shared building
blocks (robot articulation, arm joint list, palm geometry constants, apple/
tree constants) rather than duplicating them, and overrides only what the
grasp phase genuinely needs to differ:

  - Apple: a REAL dynamic RigidObject (kinematic_enabled=False) instead of
    the reach phase's kinematic/teleported apple -- grasp physics (lift,
    drop, contact force) require it. Pinned in place during the reach phase
    and until the trainee's first force closure (see envs/mdp/
    grasp_autoplay.py) via per-substep tensor-API pose/velocity writes,
    NOT a runtime kinematic-flag toggle (Isaac Lab's kinematic_
    enabled is a USD/PhysX schema attribute with no tensor-API setter) and
    NOT a PhysX FixedJoint (apple_grasp_env_cfg.py already documents a known
    GPU-multi-env-replication bug with that approach) -- sidestepped
    entirely since harvest's stem-detachment physics isn't needed yet.

  - Action space: 14-dim (arm 7 + hand 7), not arm-only -- if the reach
    hand-off isn't perfect, fingers alone may not reach a good grasp. A
    per-joint-weighted action-rate penalty (see RewardsCfg.action_rate_
    weighted) makes proximal (shoulder/elbow) motion much more expensive
    than distal (wrist/hand) motion, so the policy learns to mostly use
    fingers/wrist and only lightly correct with the arm.

  - Each episode starts with a REACH phase: the FROZEN, already-trained
    reach policy (pinned copy, see grasp_autoplay.PINNED_REACH_POLICY)
    drives the arm from its own reset pose to the apple inside the episode
    (real chained rollout, not a fabricated "post-reach" start pose), the
    trainee's actions are ignored and every grasp reward/termination is
    gated off; then control hands off to the trainee (REACH_HANDOFF_* below,
    envs/mdp/grasp_autoplay.py's module docstring for why this is in-episode
    rather than a reset event). Reliability mitigation: this task's own
    apple-placement randomization (GRASP_TRAIN_REACH_*) is NARROWER than the reach task's
    own training range, so hand-off is reliable rather than a stress test of
    reach robustness.

  - Tactile observations (slip + cross-finger balance + a scoped absolute
    force-level term) instead of raw pressure -- see envs/mdp/observations.py
    -- deliberately robust to tactile_layout.SIM_FORCE_TO_RAW_GAIN being an
    unverified placeholder, reinforced by training-time independent per-
    taxel gain randomization (envs/mdp/domain_randomization.py::
    randomize_tactile_gain).

  - A command-conditioned grip-force setpoint (env._target_grip_level,
    randomized per episode between a friction-based minimum-to-not-slip
    estimate and a literature-grounded, per-episode randomized bio-yield/
    damage-force threshold -- see domain_randomization.
    randomize_fruit_damage_force) that the policy tracks, rather than a
    single fixed "don't crush it" heuristic -- dialable at deployment
    without retraining, same pattern as a commanded velocity in locomotion.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from . import mdp as task_mdp
from .mdp import domain_randomization as dr
from .mdp import grasp_rewards as gr
from .mdp import grasp_autoplay as ga

from .apple_grasp_env_cfg import (
    G1_DEX3_CFG,
    AppleGraspSceneCfg,
    RIGHT_ARM_JOINTS,
    PALM_NORMAL_AXIS,
    PALM_LOCAL_OFFSET,
    GRASP_OFFSET_Z,
    APPLE_RADIUS,
    TREE_HEIGHT,
    BRANCH_LENGTH,
    BRANCH_RADIUS,
    BRANCH_ROTATION,
    APPLE_STEM_BREAK_FORCE,
    APPLE_STEM_BREAK_TORQUE,
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

# Explicit, ordered list (NOT the "right_hand.*" regex left as a comment in
# apple_grasp_env_cfg.py) -- the proximal-vs-distal action-rate weighting
# below needs a known, verified concatenation order. preserve_order=True on
# the SceneEntityCfg makes THIS list's order authoritative regardless of the
# articulation's own internal joint ordering; still worth confirming once
# interactively (e.g. print env.action_manager.get_term("right_hand").
# _joint_names) before trusting the weights vector on a new USD/URDF.
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

# Narrower than the reach task's own REACH_MIN/REACH_MAX/REACH_AZIMUTH_SPAN_DEG
# (0.30-0.62 m, 65 deg) -- deliberately: this range only needs to be
# comfortably WELL INSIDE the arm's measured hard kinematic ceiling (~0.42 m,
# see apple_grasp_env_cfg.py's REACH_MAX comment for the probe_arm_reach.py
# measurement and its own caveat that 0.62 already exceeds that ceiling), so
# the frozen reach policy reliably converges during reach-autoplay instead of
# this task's own placement randomization stress-testing reach robustness.
GRASP_TRAIN_REACH_MIN = 0.30
GRASP_TRAIN_REACH_MAX = 0.40
GRASP_TRAIN_REACH_AZIMUTH_CENTER_DEG = -45.0
GRASP_TRAIN_REACH_AZIMUTH_SPAN_DEG = 40.0

# Reach -> grasp hand-off (see grasp_autoplay.update_reach_grasp_phase).
# "fixed": the reach policy gets exactly REACH_HANDOFF_TIME_S, then the
# trainee takes over wherever the arm is -- doesn't depend on the reach
# policy settling into a stable hold (it was trained to terminate at
# reach_success, never to hold past it). "converged": hand off as soon as
# the palm holds the reach-success neighbourhood for 0.5 s, with
# REACH_HANDOFF_TIME_S as the watchdog.
REACH_HANDOFF_MODE = "fixed"
REACH_HANDOFF_TIME_S = 2.5
# Trainee's own control window after the hand-off (in "fixed" mode; longer
# in "converged" mode whenever the reach phase ends early).
GRASP_WINDOW_S = 5.0

# Literature-grounded per-episode bruising-force range (Li et al., 2026: Red
# Fuji apple 24.33-26.38 N; broader pome-fruit literature 20-40 N) -- see
# domain_randomization.randomize_fruit_damage_force's own docstring for the
# full reasoning and caveats.
FRUIT_DAMAGE_LOW = 20.0
FRUIT_DAMAGE_HIGH = 40.0
FRUIT_DAMAGE_APPLE_LOW = 22.0
FRUIT_DAMAGE_APPLE_HIGH = 28.0
FRUIT_DAMAGE_APPLE_WEIGHT_PCT = 0.6
APPLE_MASS_NOMINAL = 0.15   # matches apple_grasp_env_cfg.py's _jitter(0.15) nominal
GRIP_FRICTION_MU = 0.5      # assumed -- no PhysX material explicitly configured on the apple

# The hand's GRASP POCKET: where the fruit centre sits, in right_hand_palm_
# link's own frame, when the hand is actually holding it. MEASURED, not
# chosen: across every scripted curl combination that both latched a grip
# and still held the apple 1 s after the stem released, the centre landed
# here to within 4-6 mm (n=45). For reference the reach hand-off leaves the
# apple at roughly (0.080, 0.055, 0.000) -- about 3.3 cm off along the
# palm's lateral axis, which is the gap RewardsCfg.apple_centered exists to
# close. Re-measure (scripted power grasps, apple pose in palm frame 1 s
# after latch) if the hand, the fruit radius or PALM_LOCAL_OFFSET change.
GRASP_CENTER_LOCAL = (0.066, 0.054, 0.033)


# ─────────────────────────────────────────────
#  SCENE: dynamic apple override
# ─────────────────────────────────────────────

@configclass
class AppleGraspGraspSceneCfg(AppleGraspSceneCfg):
    """Same scene as the reach phase, apple made a REAL dynamic body.

    Everything else (robot, tree, contact sensor, lighting) is inherited
    unchanged. Overriding only `apple` here (rather than editing
    AppleGraspSceneCfg directly) keeps the reach task's own scene, and
    therefore its trained policy's validity, completely untouched.
    """
    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.SphereCfg(
            radius=APPLE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=APPLE_MASS_NOMINAL),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.06, 0.06),
                roughness=0.5,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.3, TREE_HEIGHT - BRANCH_RADIUS - APPLE_RADIUS),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# ─────────────────────────────────────────────
#  MANAGER CONFIGS
# ─────────────────────────────────────────────

@configclass
class ActionsCfg:
    """14-dim joint position targets: right arm (7) + right Dex3 hand (7).

    Both are phase-aware (grasp_autoplay.ReachHandoffJointPositionAction):
    during the reach phase the arm is driven by the frozen reach policy and
    the hand held at its default open pose, whatever the trainee outputs.
    """

    # clip/max_velocity/max_acceleration (and reach_scale) apply to the
    # reach policy's output during the reach phase too, so they MUST keep
    # matching apple_grasp_env_cfg.ActionsCfg.right_arm. After the hand-off
    # the trainee's actions are offsets from the hand-off target
    # (anchor_to_handoff), so a zero action holds the pose the reach policy
    # left -- with a default-pose offset, a fresh policy's ~zero actions
    # pulled the arm back toward home, away from the apple.
    #
    # scale (the TRAINEE's, 0.5 -> 0.05 rad per unit action): at 0.5, PPO's
    # exploration noise alone moved arm targets by ~0.5 rad, driving the
    # palm ~2.5 cm into the pinned apple within ~10 steps of every hand-off.
    # A 4-group ablation of a trained policy showed the arm, not the hand,
    # caused every fruit_damaged termination (arm zeroed: 0/16 damaged,
    # hand zeroed: 16/16), and pure N(0, 0.8) exploration noise damaged
    # 10/16 at 0.5 vs 0/16 at <= 0.1. 0.05 keeps the arm a light correction,
    # as intended; the reach phase is unaffected (it uses reach_scale).
    right_arm: ga.ReachHandoffJointPositionActionCfg = ga.ReachHandoffJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        scale=0.05,
        use_default_offset=True,
        clip={".*": (-3.0, 3.0)},
        max_velocity=2.0,
        max_acceleration=15.0,
        drive_with_reach_policy=True,
        anchor_to_handoff=True,
        reach_scale=0.5,
        palm_local_offset=PALM_LOCAL_OFFSET,
        pin_object="apple",
        # Arm frozen at the hand-off pose until the fingers have a grip --
        # see the action class for the measurements behind this. Until then
        # this task is "close the fingers on an apple the stem is holding";
        # after, the arm is free for the lift grasp_success requires.
        unlock_with_grip=True,
        # Wrist (last 3 of RIGHT_ARM_JOINTS) stays free even before a
        # grip exists -- it is the only thing that can correct a badly
        # aligned hand-off before any finger touches the apple; see the
        # cfg field's own docstring for the measurement behind this.
        pre_grip_free_dims=3,
    )
    # Weaker actuator than the arm (effort_limit=2.0 vs 87.0), so a smaller
    # scale and lower rate limits -- starting guesses, not measured/validated
    # like the arm's own (see RateLimitedJointPositionActionCfg's docstring
    # for the arm's real-hardware-grounded numbers); retune from training.
    right_hand: ga.ReachHandoffJointPositionActionCfg = ga.ReachHandoffJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_HAND_JOINTS,
        scale=0.3,
        use_default_offset=True,
        clip={".*": (-3.0, 3.0)},
        max_velocity=3.0,
        max_acceleration=20.0,
    )


@configclass
class ObservationsCfg:
    """Observation groups. Policy uses 'policy' group only."""

    @configclass
    class PolicyObs(ObsGroup):
        """Everything the reach phase observes, plus hand joint state,
        tactile slip/balance/force-level, and the commanded grip setpoint.
        """

        right_arm_pos = ObsTerm(
            func=task_mdp.joint_pos_selected,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)},
        )
        right_arm_vel = ObsTerm(
            func=task_mdp.joint_vel_selected,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)},
        )
        right_hand_pos = ObsTerm(
            func=task_mdp.joint_pos_selected,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS, preserve_order=True)},
        )
        right_hand_vel = ObsTerm(
            func=task_mdp.joint_vel_selected,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS, preserve_order=True)},
        )

        palm_orientation_torso = ObsTerm(
            func=task_mdp.body_quat_relative_to_body,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "reference_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        )

        # Calibration-agnostic tactile features (see observations.py's own
        # module docstring for the full reasoning) -- NOT raw pressure.
        tactile_slip = ObsTerm(
            func=task_mdp.tactile_slip,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        tactile_finger_balance = ObsTerm(
            func=task_mdp.tactile_finger_balance,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        tactile_force_level = ObsTerm(
            func=task_mdp.tactile_force_level,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )

        # Commanded grip-force setpoint -- a COMMAND, not a measurement, so
        # no calibration gain applies (same as a commanded velocity input to
        # a locomotion policy).
        target_grip_level = ObsTerm(func=task_mdp.target_grip_level)

        # 0 while the frozen reach policy drives (trainee actions ignored,
        # all rewards gated to 0), 1 after the hand-off -- without it the
        # critic can't tell the two apart from the rest of the state.
        grasp_phase = ObsTerm(func=ga.grasp_phase_obs)

        # 1.0 once the fingers have a grip: the apple is off the stem and
        # the arm is unlocked. Both of those change what the policy's own
        # actions do, so it has to be observable.
        grip_established = ObsTerm(func=ga.grip_established_obs)

        # Fruit position in the PALM's own rotating frame, relative to the
        # grasp pocket -- exactly what RewardsCfg.apple_centered scores, so
        # the policy can see the error it is paid to close.
        #
        # This REPLACES the reach task's apple_pos_palm (kept there, and in
        # grasp_autoplay._reach_obs, because that policy is frozen), which
        # is the same difference vector expressed in WORLD axes. That form
        # is only reproducible off-line because this task pins the base with
        # identity rotation (fix_root_link, no init rot), which makes world
        # axes and pelvis axes coincide -- a real robot would have to know
        # its base orientation and would break the moment the pelvis yawed
        # or tilted. Anchoring to the palm needs no such assumption: FK from
        # the joint encoders plus the fruit pose from perception, both of
        # which the robot has. Same dimensionality, so this costs nothing.
        apple_pos_pocket = ObsTerm(
            func=task_mdp.object_pos_in_body_frame,
            params={
                "object_cfg": SceneEntityCfg("apple"),
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "ref_local": GRASP_CENTER_LOCAL,
            },
        )

        last_action = ObsTerm(func=mdp.last_action)  # action_name=None -> full 14-dim vector

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyObs = PolicyObs()


@configclass
class RewardsCfg:
    """Grasp-phase reward set. Every term is gated to exactly 0 during the
    reach phase (see grasp_rewards.py's module docstring).

    Deliberately NOT the reach phase's position/orientation-reaching terms
    (reach_progress, reach_coarse/fine, palm_up, hold, reach_success_bonus)
    -- the arm is no longer driving toward a target here, the reach policy
    already did that during autoplay; this phase is about the hand closing
    firmly and evenly without crushing or dropping the apple, with the arm
    only lightly correcting.
    """

    # ── Primary grasp-quality objectives ──────────────────────────────────
    # Dense shaping toward the apple even before any contact -- without
    # this, every other grasp-quality term below is exactly zero (or, for
    # setpoint tracking, favors near-zero force) until contact happens at
    # all, giving a purely-exploring policy nothing pulling it toward the
    # apple in the first place. See its own docstring for the reasoning
    # (same pattern as reach_rewards.reward_reach_gaussian).
    finger_approach = RewTerm(
        func=gr.reward_finger_approach,
        weight=2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("apple"),
            "std": 0.05,
        },
    )
    # "Is the apple in the middle of the hand." The reward set had nothing
    # asking for this: finger_approach averages fingertip distances (one
    # close finger hides two far ones) and grasp_balance only pays once all
    # three fingers already touch -- both satisfied by the observed failure
    # pose, the apple caught on the index alone while middle and thumb close
    # on nothing. See reward_apple_centered for the measurement behind
    # GRASP_CENTER_LOCAL.
    apple_centered = RewTerm(
        func=gr.reward_apple_centered,
        weight=4.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "center_local": GRASP_CENTER_LOCAL,
            "normal_axis_local": PALM_NORMAL_AXIS,
            # std_perp=0.02 validated in the isotropic version: the failure
            # poses (apple caught on the index alone, or resting against
            # fully open fingers) scored 0.46-0.66 against 0.81-0.85 for a
            # properly curled grasp. std_along loosened to 0.03 -- how deep
            # the apple sits matters less than how far off to one side it
            # is, which is what actually determines whether the fingers
            # close on it symmetrically.
            "std_perp": 0.02,
            "std_along": 0.03,
        },
    )

    grasp_setpoint_tracking = RewTerm(
        func=gr.reward_grasp_setpoint_tracking,
        weight=4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )
    # Per-contact-group stepping stones toward the closure criterion, added
    # after a run where episodes survived and the fingers learned to press
    # (fruit_damaged 0 -> 72%) but grasp_hold stayed at exactly 0: the
    # policy was pressing HARD from ONE side, which the old term set paid
    # for (setpoint_tracking/balance both read force magnitude only) and
    # nothing pulled it toward opposing contacts or gentleness.
    #
    # gentle_contact_groups: pays per group (palm/thumb/index/middle) for
    # force inside the gentle band, and DECAYS above it -- the dense
    # anti-crush gradient fruit_damaged's hard cutoff can't give.
    # grasp_enclosure: continuous version of _force_closure_ok (groups
    # engaged x how much their pushes oppose), so "2 groups on" or "3
    # groups all pushing one way" finally has a gradient out.
    # Weights are in the same range as the reach phase's gentle_touch_palm
    # (4.0); retune from the curves.
    gentle_contact_groups = RewTerm(
        func=gr.reward_gentle_contact_groups,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "good_min": 0.5,
            "good_max": 5.0,
            "decay": 5.0,
        },
    )
    # Weight raised 3.0 -> 5.0: this is the term that actually separates a
    # real grasp from the apple merely resting in the hand, and at equal
    # weights a measured comparison had the degenerate pose scoring 2.21
    # against 2.51 for a properly curled grasp -- too thin a margin to
    # expect PPO to prefer the latter.
    grasp_enclosure = RewTerm(
        func=gr.reward_grasp_enclosure,
        weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    # Bridges "finger open, not touching" -> "finger curled enough that
    # touching pays": now that gentle_contact_groups scales by flexion, an
    # open finger earns nothing for contact, and curling a finger that
    # isn't touching yet earns nothing either, so without this there is a
    # flat region between the two. Gated on palm contact (apple actually in
    # the hand) so it can't be farmed by curling in mid-air. Weight kept
    # UNDER gentle_contact_groups so curling is a stepping stone toward
    # gentle contact, never a substitute for it.
    finger_curl = RewTerm(
        func=gr.reward_finger_curl,
        weight=1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    grasp_balance = RewTerm(
        func=gr.reward_grasp_balance,
        weight=2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )
    grasp_hold = RewTerm(
        func=gr.reward_grasp_hold,
        weight=3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )
    grasp_success_bonus = RewTerm(func=gr.reward_grasp_success, weight=95.0)

    # Dense, within-episode gradient toward backing off BEFORE
    # terminations_and_events.fruit_damaged's hard, instantaneous cutoff --
    # added after the first real training run showed 21-76 N spikes on
    # first contact and fruit_damaged firing in 93.75% of episodes at a
    # median of just 11 control steps (0.22s) into the grasp phase, so
    # close to hand-off that grasp_hold/grasp_success never once fired
    # across 1407 iterations. See penalty_excess_hand_force's own docstring
    # for the full diagnosis and why this scales to each episode's own
    # randomized damage threshold instead of a fixed N band. Weight -2.0
    # matches reach_rewards.penalty_excess_palm_force's precedent for the
    # same shape of penalty; retune from the training curve.
    excess_hand_force = RewTerm(
        func=gr.penalty_excess_hand_force,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    # ── Action-space discipline: mostly fingers/wrist, arm as a light
    #    correction only. Weight order MUST match ActionsCfg's concatenation
    #    order (right_arm then right_hand):
    #    [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,   <- proximal, heavy
    #     wrist_roll, wrist_pitch, wrist_yaw,                    <- distal, light
    #     thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1]  <- distal, light
    # Halved from the original -0.3/-0.1 (see training-log diagnosis in
    # commit history / conversation): at those weights, combined with the
    # grasp-quality terms all being exactly zero until first contact, "don't
    # move" was a strictly dominant, trivially-reachable local optimum --
    # the policy converged to near-total inaction by iteration 500, never
    # once achieving contact. Loosening this (alongside finger_approach
    # above and higher exploration in the PPO cfg) gives the policy more
    # room to actually explore hand-closing motions before these penalties
    # dominate. Revisit upward once grasping is actually being learned.
    action_rate_weighted = RewTerm(
        func=gr.penalty_action_rate_weighted,
        weight=-0.25,
        params={"weights": [10.0, 10.0, 10.0, 10.0, 3.0, 3.0, 3.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "arm_dims": 7,
                # Wrist (the trailing 3 of the 7 arm dims) matches
                # ActionsCfg.right_arm's own pre_grip_free_dims=3 -- it is
                # never locked, so its jitter always counts.
                "arm_free_dims": 3},
    )
    posture_from_handoff = RewTerm(
        func=gr.penalty_joint_deviation_from_handoff,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True),
            "weights": [10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
        },
    )

    # ── Joint-limit safety, both arm and hand ─────────────────────────────
    arm_joint_limits = RewTerm(
        func=gr.penalty_joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)},
    )
    hand_joint_limits = RewTerm(
        func=gr.penalty_joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS, preserve_order=True)},
    )

    # Diagnostic only -- see hand_motion_monitor's own docstring for why the
    # weight can't be exactly 0. Watch Episode_Reward/hand_motion_monitor in
    # the training printout/TensorBoard to answer "are the fingers actually
    # moving" without stopping training to run play.py.
    hand_motion_monitor = RewTerm(
        func=gr.hand_motion_monitor,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Coarse backstop only (e.g. the apple ending up far below the scene
    # entirely) -- NOT the real "failed to hold it" signal for this phase,
    # since a failed grasp just lets the apple rest on the ground at ~0.04 m,
    # comfortably above this -0.1 m absolute threshold. See apple_fell_from_
    # grip below for the actual per-episode-relative failure signal.
    apple_dropped = DoneTerm(
        func=task_mdp.apple_dropped,
        params={"object_cfg": SceneEntityCfg("apple"), "min_height": -0.1},
    )

    # The real "failed to hold it" signal: fires as soon as the apple falls
    # meaningfully below where the reach->grasp hand-off left it, whether it
    # ends up on the ground, rolled aside, or anywhere else -- catches a
    # failed/slipping grasp promptly instead of idling out to time_out.
    apple_fell_from_grip = DoneTerm(
        func=task_mdp.apple_fell_from_grip,
        params={"object_cfg": SceneEntityCfg("apple"), "drop_margin": 0.08},
    )

    # Hard, immediate failure -- bio-yield is a near-instantaneous threshold
    # effect, on par with dropping it. Checked BEFORE grasp_success below in
    # declaration order isn't load-bearing (both are independent DoneTerms,
    # IsaacLab ORs all termination results together), but keeping the
    # "can this episode still succeed" failure modes grouped first reads
    # more naturally.
    fruit_damaged = DoneTerm(
        func=task_mdp.fruit_damaged,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    grasp_success = DoneTerm(
        func=task_mdp.grasp_success,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "object_cfg": SceneEntityCfg("apple"),
            "hold_time": 0.5,
            "lift_margin": 0.03,
        },
    )


@configclass
class EventsCfg:
    """Event terms run in DECLARATION ORDER within a mode.

    reset_reach_grasp_phase MUST come after randomize_tree among the
    "reset" events -- it records the freshly placed apple pose as this
    episode's pin pose. Kept last so all per-episode sampling happens first.
    """

    # ── Fixed once per env, before the GPU tensor views exist (raw USD
    #    writes are illegal after that point) -- same as the reach task ──
    randomize_tree_geometry = EventTerm(
        func=dr.randomize_tree_geometry,
        mode="prestartup",
        params={"tree_height_nominal": TREE_HEIGHT, "branch_length_nominal": BRANCH_LENGTH},
    )
    randomize_apple_scale = EventTerm(
        func=dr.randomize_apple_scale,
        mode="prestartup",
        params={"object_cfg": SceneEntityCfg("apple"), "nominal_radius": APPLE_RADIUS},
    )
    randomize_apple_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("apple"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # ── Per-episode ────────────────────────────────────────────────────────
    reset_all_exact = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_arm_jitter = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS),
            "position_range": (-0.20, 0.20),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_tree = EventTerm(
        func=dr.randomize_tree_and_apple,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "branch_radius_nominal": BRANCH_RADIUS,
            "branch_rotation_deg_nominal": BRANCH_ROTATION,
            "apple_radius_nominal": APPLE_RADIUS,
            "tree_height_nominal": TREE_HEIGHT,
            "branch_length_nominal": BRANCH_LENGTH,
            "reach_min": GRASP_TRAIN_REACH_MIN,
            "reach_max": GRASP_TRAIN_REACH_MAX,
            "reach_azimuth_center_deg": GRASP_TRAIN_REACH_AZIMUTH_CENTER_DEG,
            "reach_azimuth_span_deg": GRASP_TRAIN_REACH_AZIMUTH_SPAN_DEG,
            "offset_z": GRASP_OFFSET_Z,
        },
    )
    randomize_stem_break = EventTerm(
        func=dr.randomize_stem_break_params,
        mode="reset",
        params={"nominal_force": APPLE_STEM_BREAK_FORCE, "nominal_torque": APPLE_STEM_BREAK_TORQUE},
    )
    randomize_tactile_gain = EventTerm(
        func=dr.randomize_tactile_gain,
        mode="reset",
        params={"gain_range": (0.7, 1.3)},
    )
    randomize_fruit_damage_force = EventTerm(
        func=dr.randomize_fruit_damage_force,
        mode="reset",
        params={
            "low": FRUIT_DAMAGE_LOW,
            "high": FRUIT_DAMAGE_HIGH,
            "apple_low": FRUIT_DAMAGE_APPLE_LOW,
            "apple_high": FRUIT_DAMAGE_APPLE_HIGH,
            "apple_weight_pct": FRUIT_DAMAGE_APPLE_WEIGHT_PCT,
            "apple_mass_nominal": APPLE_MASS_NOMINAL,
            "mu": GRIP_FRICTION_MU,
        },
    )

    # The one real uncertainty left in the grip detector now that it reads
    # only pads + encoders: the pressure->newton calibration it thresholds
    # against (see grasp_rewards._pad_state).
    randomize_closure_threshold = EventTerm(
        func=dr.randomize_closure_threshold,
        mode="reset",
        params={},
    )

    # ── Reach -> grasp phase bookkeeping (MUST be last among "reset") ────
    reset_reach_grasp_phase = EventTerm(
        func=ga.reset_reach_grasp_phase,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # ── Every control step (interval_range_s=(0.0, 0.0) makes Isaac Lab's
    #    EventManager fire these for every env, every step, after resets and
    #    before the observation). Declaration order matters: first-closure
    #    is latched before the phase switch, so a closure the reach policy
    #    happens to cause on its last step never counts for the trainee. ──
    update_first_closure = EventTerm(
        func=ga.update_first_closure,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "min_steps": 5},
    )
    update_reach_grasp_phase = EventTerm(
        func=ga.update_reach_grasp_phase,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "object_cfg": SceneEntityCfg("apple"),
            "handoff_mode": REACH_HANDOFF_MODE,
            "handoff_time_s": REACH_HANDOFF_TIME_S,
            # Only used in "converged" mode -- same neighbourhood as the
            # reach task's own reach_success termination.
            "offset_z": GRASP_OFFSET_Z,
            "palm_normal_axis": PALM_NORMAL_AXIS,
            "palm_local_offset": PALM_LOCAL_OFFSET,
            "pos_tol": 0.04,
            "ori_tol_rad": 0.30,
            "hold_time": 0.5,
        },
    )


# ─────────────────────────────────────────────
#  ENVIRONMENT CONFIGS
# ─────────────────────────────────────────────

@configclass
class AppleGraspGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Grasp-phase env: dynamic apple, 14-dim action, tactile observations,
    in-episode reach phase + hand-off every episode."""

    scene: AppleGraspGraspSceneCfg = AppleGraspGraspSceneCfg(
        num_envs=32,
        env_spacing=2.5,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.decimation = 4
        # The reach phase is now part of the episode, so the time limit
        # covers both it and the trainee's own window.
        self.episode_length_s = REACH_HANDOFF_TIME_S + GRASP_WINDOW_S
        self.sim.render_interval = self.decimation

        self.scene.head_camera = None
        self.scene.replicate_physics = False

        self.viewer.eye = (1.5, -1.5, 1.8)
        self.viewer.lookat = (0.5, -0.25, 1.0)


@configclass
class AppleGraspGraspEnvCfg_PLAY(AppleGraspGraspEnvCfg):
    """Single-env config for policy evaluation / visualisation."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.events.reset_all_exact = None
        self.events.reset_arm_jitter = None
        self.episode_length_s = 1.0e6
