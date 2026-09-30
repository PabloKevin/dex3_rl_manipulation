# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Phase 3 -- HARVEST: detach the fruit from the stem with a lift-and-roll
of the arm and wrist while the fingers keep the grasp.

Episodes start from a ready grasp out of the start bank
(scripts/collect_harvest_starts.py, from a trained grasp policy), so every
simulated second is harvest -- see envs/mdp/harvest_task.py for the MDP and
envs/mdp/harvest.py for the breakable stem.

Same scene, robot, sensors and physics settings as the grasp task
(AppleGraspGraspEnvCfg); only actions, observations, rewards, terminations
and the reset differ. The reach and grasp tasks are untouched.

  train:  python scripts/train.py --task AppleGraspHarvest-v0 --num_envs 32 --headless
  view:   python scripts/run_harvest_env.py --checkpoint logs/rsl_rl/apple_grasp_harvest/<run>/model_XXX.pt
"""

import os

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
)
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from . import mdp as task_mdp
from .mdp import harvest_task as ht
from .apple_grasp_grasp_env_cfg import (
    AppleGraspGraspEnvCfg,
    EventsCfg as GraspEventsCfg,
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
    GRASP_CENTER_LOCAL,
    BRANCH_RADIUS,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Written by scripts/collect_harvest_starts.py (it always refreshes latest.pt).
HARVEST_START_BANK = os.path.join(_REPO, "logs", "harvest_starts", "latest.pt")

# The reference motion: the scripted lift-and-roll that worked best
# (scripts/harvest_scripted.py, "side-": 4/4 broke the stem with the fruit
# held). Swing about the horizontal robot->fruit line, through the stem pivot.
HARVEST_SWING_MODE = "side"
HARVEST_SWING_SIGN = -1.0
HARVEST_REF_DURATION_S = 2.0
HARVEST_REF_SWING_DEG = 90.0
HARVEST_REF_TWIST_DEG = 90.0
HARVEST_EPISODE_S = 4.0          # 2 s motion + 2 s to show the fruit stays in the hand
IN_HAND_MARGIN = 0.06            # fruit centre this far from the palm's grasp point = not in the hand

_ARM = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)
_HAND = SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS, preserve_order=True)


@configclass
class ActionsCfg:
    right_arm = ht.HarvestArmActionCfg(
        asset_name="robot",
        arm_joint_names=RIGHT_ARM_JOINTS,
        hand_joint_names=RIGHT_HAND_JOINTS,
        delta_scale=0.025,
        clip=3.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyObs(ObsGroup):
        right_arm_pos = ObsTerm(func=task_mdp.joint_pos_selected, params={"asset_cfg": _ARM})
        right_arm_vel = ObsTerm(func=task_mdp.joint_vel_selected, params={"asset_cfg": _ARM})
        right_hand_pos = ObsTerm(func=task_mdp.joint_pos_selected, params={"asset_cfg": _HAND})
        palm_orientation_torso = ObsTerm(
            func=task_mdp.body_quat_relative_to_body,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "reference_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        )
        palm_motion = ObsTerm(func=ht.palm_motion_since_start)
        tactile_slip = ObsTerm(func=task_mdp.tactile_slip, params={"sensor_cfg": SceneEntityCfg("contact_forces")})
        tactile_finger_balance = ObsTerm(
            func=task_mdp.tactile_finger_balance, params={"sensor_cfg": SceneEntityCfg("contact_forces")}
        )
        tactile_force_level = ObsTerm(
            func=task_mdp.tactile_force_level, params={"sensor_cfg": SceneEntityCfg("contact_forces")}
        )
        apple_pos_pocket = ObsTerm(
            func=task_mdp.object_pos_in_body_frame,
            params={
                "object_cfg": SceneEntityCfg("apple"),
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "ref_local": GRASP_CENTER_LOCAL,
            },
        )
        harvest_time = ObsTerm(func=ht.harvest_time, params={"ref_duration": HARVEST_REF_DURATION_S})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyObs = PolicyObs()


_HAND_REF = {"margin": IN_HAND_MARGIN, "center_local": GRASP_CENTER_LOCAL}


@configclass
class RewardsCfg:
    # Guide: follow the reference lift-and-roll while the stem is intact.
    # ~0.04/step at full tracking (x dt), ~4 over the 2 s motion.
    track_reference = RewTerm(
        func=ht.reward_track_reference,
        weight=2.0,
        params={
            "duration": HARVEST_REF_DURATION_S,
            "swing_deg": HARVEST_REF_SWING_DEG,
            "twist_deg": HARVEST_REF_TWIST_DEG,
        },
    )
    # Goal: the stem breaks with the fruit in the hand (one-off, 250 * dt = 5)...
    stem_break = RewTerm(func=ht.reward_stem_break, weight=250.0, params=_HAND_REF)
    # ...and the fruit stays in the hand afterwards (0.1/step): holding it
    # for the rest of the episode is worth more than the whole guide.
    detached_hold = RewTerm(func=ht.reward_detached_hold, weight=5.0, params=_HAND_REF)
    # Bend and twist the stem, do not yank it.
    stem_pull = RewTerm(func=ht.penalty_stem_pull, weight=-2.0)
    # Keep the fruit unbruised and unslipped.
    fruit_force = RewTerm(
        func=ht.penalty_fruit_force, weight=-3.0, params={"sensor_cfg": SceneEntityCfg("contact_forces")}
    )
    fruit_slip = RewTerm(func=ht.penalty_fruit_slip, weight=-1.0)
    # Smoothness and limits.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    arm_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.002, params={"asset_cfg": _ARM})
    arm_joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": _ARM})
    # Losing or bruising the fruit ends the episode: -250 * dt = -5, on top
    # of the hold reward it forfeits.
    failure = RewTerm(
        func=mdp.is_terminated_term, weight=-250.0, params={"term_keys": ["fruit_left_hand", "fruit_damaged"]}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fruit_left_hand = DoneTerm(
        func=task_mdp.apple_fell_from_grip,
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "center_local": GRASP_CENTER_LOCAL,
            "leave_margin": IN_HAND_MARGIN,
        },
    )
    fruit_damaged = DoneTerm(func=task_mdp.fruit_damaged, params={"sensor_cfg": SceneEntityCfg("contact_forces")})


@configclass
class EventsCfg(GraspEventsCfg):
    """The grasp task's randomization (tree geometry, apple size and mass,
    tactile gain/footprint, bruise threshold), minus everything about the
    reach and grasp phases; plus the reset from the start bank (declared
    last, so it runs after the randomizers)."""
    reset_all_exact = None
    reset_arm_jitter = None
    randomize_tree = None
    randomize_stem_break = None
    randomize_closure_threshold = None
    reset_reach_grasp_phase = None
    update_first_closure = None
    update_reach_grasp_phase = None
    reset_from_bank = EventTerm(
        func=ht.reset_harvest_from_bank,
        mode="reset",
        params={
            "bank_path": HARVEST_START_BANK,
            "swing_mode": HARVEST_SWING_MODE,
            "swing_sign": HARVEST_SWING_SIGN,
            "branch_radius": BRANCH_RADIUS,
        },
    )


@configclass
class AppleGraspHarvestEnvCfg(AppleGraspGraspEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HARVEST_EPISODE_S


@configclass
class AppleGraspHarvestEnvCfg_PLAY(AppleGraspHarvestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
