# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach on the multi-apple tree WITH the waist yaw (AppleReachTreeWaist-v1):
AppleReachTree-v1 (apple_reach_tree_env_cfg.py -- subclassed) where the
policy also turns the torso about the vertical, so the right arm reaches
further and across the front of the body, and the tree moves toward the
centre of the robot.

Differences from AppleReachTree-v1:
  - action (7 -> 8): + waist_yaw_joint, its own rate-limited term, slower
    than the arm (the torso and both arms swing with it, and on the real
    robot the standing controller absorbs the reaction), within
    +-WAIST_YAW_LIMIT -- the travel g1_fruits_picker's grasp planner already
    uses on the robot;
  - actuator: waist_yaw_joint leaves the "legs" group (kp 100) for the
    deployed waist gains (kp 200, kd 5, as g1_squat/robot_cfg.py);
  - observation (37 -> 40): palm orientation relative to the torso with the
    waist yaw taken out (waist_reach.palm_quat_torso_unyawed: identical to
    the old input at yaw 0, so the reach-tree policy warm-starts exactly),
    + waist yaw position, velocity and last action, appended;
  - placement: the waist-aware safe workspace (scripts/probe_side_reach.py
    --waist -> envs/assets/reach_workspace_waist.npz, still relative to the
    shoulder at yaw 0) and a wider, more central, further envelope; the
    trunk moves out with it;
  - rewards: + waist excursion / velocity / near-goal velocity, + commanded
    smoothness of the waist term, + torso / left arm entering the tree,
    + a waist-turn hint; hold weighted by palm heading;
  - reset: waist yaw scattered +-WAIST_RESET_JITTER;
  - episode 5 -> 6 s (the further, slower-turning reaches).

Downstream (grasp task hand-off, arm-goal policy, benches, sim2real) all
still assume the waist at 0: none of them can use this policy as is.
"""

import math
import os

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from . import mdp as task_mdp
from .apple_grasp_env_cfg import (
    ActionsCfg as ReachActionsCfg,
    EventsCfg as ReachEventsCfg,
    GRASP_STANDOFF,
    PALM_LOCAL_OFFSET,
    PALM_STEM_TOL_RAD,
    RewardsCfg as ReachRewardsCfg,
)
from .apple_reach_tree_env_cfg import (
    AppleReachTreeEnvCfg,
    ObservationsCfg as TreeObservationsCfg,
    RewardsCfg as TreeRewardsCfg,
)
from .mdp import reach_rewards as rr
from .mdp import tree_reach as trr
from .mdp import waist_reach as wr

WAIST_YAW_JOINT = "waist_yaw_joint"
#: +-45 deg: g1_fruits_picker/apple_picker/grasp_orientation_opt.WAIST_LIMIT,
#: the travel the robot's own planner already turns the torso through.
WAIST_YAW_LIMIT = math.radians(45.0)
#: Commanded waist trajectory caps. The robot's scripted waist ramp does
#: 45 deg in 1.2 s (~1.2 rad/s peak, min-jerk); the arm runs 2.0 / 8.0.
WAIST_MAX_VELOCITY = 1.0
WAIST_MAX_ACCELERATION = 4.0
WAIST_RESET_JITTER = 0.10

# Placement of the GRASP TARGET, relative to the right shoulder at waist yaw
# 0 (as AppleReachTree-v1: 0.24-0.34 m, azimuth -45 +- 32.5 deg). From the
# waist probe (logs/probe_ws_waist2.log; 1276 feasible 5 cm cells vs 556
# arm-only): 0.40-0.48 m out goes from ~10% to 60-95% reachable, and the
# front-left (+10..+30 deg) from ~50% to 85-95%. The map accepts 95% of this
# box (every height band >= 93%).
REACH_WAIST_MIN = 0.24
REACH_WAIST_MAX = 0.44
REACH_WAIST_DZ_RANGE = (-0.30, 0.28)
REACH_WAIST_AZIMUTH_CENTER_DEG = -25.0
REACH_WAIST_AZIMUTH_SPAN_DEG = 110.0
REACH_WAIST_MIN_FORWARD = 0.15
REACH_WAIST_MIN_HORIZONTAL = 0.18
REACH_WAIST_WORKSPACE_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                         "reach_workspace_waist.npz")
#: Trunk this far beyond REACH_WAIST_MAX (AppleReachTree-v1: 0.55 vs 0.34).
TRUNK_BEYOND_REACH = 0.21

#: hold fades out between the success gate's palm-to-stem tolerance (45 deg)
#: and this: the first run touched central apples 51-67 deg off the stem
#: and hold paid for it (see RewardsCfg.hold).
HOLD_HEADING_ZERO_RAD = math.radians(75.0)
#: Waist hint (RewardsCfg.waist_toward_target): turn the torso so the grasp
#: target sits this far right of straight ahead of the shoulder, in the
#: torso frame -- inside the band where the first run (arm doing the work)
#: succeeded 65% of the time (-50..-30 deg).
WAIST_HINT_TARGET_AZ_DEG = -35.0

_WAIST = SceneEntityCfg("robot", joint_names=[WAIST_YAW_JOINT], preserve_order=True)
_PALM = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])


@configclass
class ActionsCfg(ReachActionsCfg):
    waist_yaw: task_mdp.RateLimitedJointPositionActionCfg = task_mdp.RateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[WAIST_YAW_JOINT],
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-WAIST_YAW_LIMIT, WAIST_YAW_LIMIT)},
        max_velocity=WAIST_MAX_VELOCITY,
        max_acceleration=WAIST_MAX_ACCELERATION,
        # the torso never overshoots its target (rings otherwise: 0.125 rad
        # stopping distance at these caps)
        brake_to_target=True,
    )


@configclass
class ObservationsCfg(TreeObservationsCfg):
    @configclass
    class PolicyObs(TreeObservationsCfg.PolicyObs):
        # same slot as before; the torso frame without the waist yaw
        palm_orientation_torso = ObsTerm(
            func=wr.palm_quat_torso_unyawed,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
                "reference_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
                "pelvis_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
                "waist_cfg": _WAIST,
            },
        )
        # appended (37 -> 40)
        waist_yaw_pos = ObsTerm(func=task_mdp.joint_pos_selected, params={"asset_cfg": _WAIST})
        waist_yaw_vel = ObsTerm(func=task_mdp.joint_vel_selected, params={"asset_cfg": _WAIST})
        waist_last_action = ObsTerm(func=mdp.last_action, params={"action_name": "waist_yaw"})

    policy: PolicyObs = PolicyObs()


_HOLD = ReachRewardsCfg().hold

@configclass
class RewardsCfg(TreeRewardsCfg):
    # Same hold, weighted by palm heading: full within the success gate's
    # 45 deg of the stem, nothing beyond HOLD_HEADING_ZERO_RAD. (The v1 hold
    # deliberately has no stem gate, so the heading could be learned; here
    # that left a side-on touch of central apples paying every step.)
    hold = RewTerm(
        func=wr.reward_reach_hold_heading,
        weight=_HOLD.weight,
        params={**_HOLD.params, "heading_full_rad": PALM_STEM_TOL_RAD,
                "heading_zero_rad": HOLD_HEADING_ZERO_RAD},
    )
    # Exploration hint for the torso: present the right shoulder to the
    # fruit (waist_reach.waist_yaw_hint). Weight 1 (palm_toward_stem 10,
    # hold 5): when the hint and the palm heading disagree, the heading wins.
    waist_toward_target = RewTerm(
        func=wr.reward_waist_toward_target,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("apple"),
            "waist_cfg": _WAIST,
            "standoff": GRASP_STANDOFF,
            "target_az_rad": math.radians(WAIST_HINT_TARGET_AZ_DEG),
            "limit_rad": WAIST_YAW_LIMIT,
            "std_rad": 0.25,
        },
    )
    # Turn the torso only when it buys a better arm pose: 45 deg costs 0.12
    # before the weight, about what home_excursion charges for a typical
    # reach (the robot planner's WAIST_COST is as mild).
    waist_excursion = RewTerm(func=rr.penalty_joint_deviation_weighted, weight=-0.2,
                              params={"asset_cfg": _WAIST, "weights": [1.0]})
    # The torso carries both arms: 5x arm_vel per joint.
    waist_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.1, params={"asset_cfg": _WAIST})
    # Settle the torso before the touch: the palm sits ~0.4 m from the waist
    # axis, so turning moves it faster than one arm joint does (brake: -2).
    waist_brake = RewTerm(
        func=rr.penalty_near_goal_velocity,
        weight=-4.0,
        params={
            "asset_cfg": _WAIST,
            "robot_cfg": _PALM,
            "object_cfg": SceneEntityCfg("apple"),
            "brake_dist": 0.15,
            "standoff": GRASP_STANDOFF,
            "palm_local_offset": PALM_LOCAL_OFFSET,
        },
    )
    # Smooth commanded waist trajectory. -3 (the arm's) -> -0.5: at the
    # waist's 4 rad/s^2 cap, exploration noise alone cost ~2.2 per episode
    # and the policy shrank its waist noise (0.50 -> 0.39) instead of
    # exploring turns; brake_to_target already keeps the command smooth.
    commanded_accel_waist = RewTerm(func=trr.penalty_commanded_accel, weight=-0.5,
                                    params={"action_name": "waist_yaw"})
    limiter_lead_waist = RewTerm(func=trr.penalty_limiter_lead, weight=-1.0,
                                 params={"action_name": "waist_yaw", "ref": 0.5})
    # Turning swings the torso and the parked left arm: keep them out of the
    # stems/branches/trunk too (the fruit itself pushes back physically and
    # is charged by neighbour_push / neighbour_knocked).
    body_tree_contact = RewTerm(func=task_mdp.penalty_body_tree_contact, weight=-6.0,
                                params={"margin": 0.01, "ref": 0.02})


@configclass
class EventsCfg(ReachEventsCfg):
    # After the tree placement (declaration order): placement records the
    # shoulder once, at the default (yaw 0) pose.
    reset_waist_jitter = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[WAIST_YAW_JOINT]),
            "position_range": (-WAIST_RESET_JITTER, WAIST_RESET_JITTER),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class AppleReachTreeWaistEnvCfg(AppleReachTreeEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 6.0

        # waist yaw: out of the legs group, deployed gains
        robot = self.scene.robot
        acts = dict(robot.actuators)
        acts["legs"] = acts["legs"].replace(joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"])
        acts["waist_yaw"] = ImplicitActuatorCfg(
            joint_names_expr=[WAIST_YAW_JOINT],
            effort_limit=88.0,
            velocity_limit=32.0,
            stiffness=200.0,
            damping=5.0,
        )
        self.scene.robot = robot.replace(actuators=acts)

        # wider, more central, further placement on the waist-aware map
        envelope = dict(
            reach_min=REACH_WAIST_MIN, reach_max=REACH_WAIST_MAX,
            az_center_deg=REACH_WAIST_AZIMUTH_CENTER_DEG, az_span_deg=REACH_WAIST_AZIMUTH_SPAN_DEG,
            dz_range=REACH_WAIST_DZ_RANGE, min_forward=REACH_WAIST_MIN_FORWARD,
            min_horizontal=REACH_WAIST_MIN_HORIZONTAL)
        self.events.randomize_tree.params.update({
            "seed_envelope": dict(envelope),
            "member_envelope": dict(envelope),
            "workspace_map": REACH_WAIST_WORKSPACE_MAP,
            "trunk_dist": REACH_WAIST_MAX + TRUNK_BEYOND_REACH,
        })


@configclass
class AppleReachTreeWaistEnvCfg_PLAY(AppleReachTreeWaistEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None


# ─────────────────────────────────────────────
#  Variant: face the apple (AppleReachTreeWaistFace-v1)
# ─────────────────────────────────────────────
# The torso turns so the head camera looks at the apple, as a person would,
# keeping it in view for the robot's own checks -- but only where the reach
# still succeeds that way. Facing is paid mostly ON TOP of a good touch
# (hold) and a success, never on its own, so the order of preference is:
# succeed facing > succeed turned elsewhere > fail facing. For a right-arm
# reach, facing turns the torso RIGHT for most fruit and puts it in front
# of the chest (8-16 deg left of the shoulder): the policy has to find where
# that still works and turn the other way where it does not.
FACE_STD_RAD = 0.25     # ~14 deg: half the bonus at ~17 deg off


@configclass
class RewardsFaceCfg(RewardsCfg):
    # replaces the shoulder-presenting hint (it pulls the other way)
    waist_toward_target = None
    # Holding the torso turned costs no gravity torque (vertical axis), and
    # facing needs up to the full 45 deg: -0.2 -> -0.05.
    waist_excursion = RewTerm(func=rr.penalty_joint_deviation_weighted, weight=-0.05,
                              params={"asset_cfg": _WAIST, "weights": [1.0]})
    # small dense pull, to explore turning toward the fruit
    face_apple = RewTerm(func=wr.reward_face_apple, weight=0.5,
                         params={"object_cfg": SceneEntityCfg("apple"), "std_rad": FACE_STD_RAD})
    # facing while holding a good touch: up to +40% on hold (5)
    face_apple_hold = RewTerm(
        func=wr.reward_face_apple_hold,
        weight=2.0,
        params={**_HOLD.params, "heading_full_rad": PALM_STEM_TOL_RAD,
                "heading_zero_rad": HOLD_HEADING_ZERO_RAD, "face_std_rad": FACE_STD_RAD},
    )
    # facing at success: up to +30 on the 95 success bonus
    face_apple_success = RewTerm(func=wr.reward_face_apple_success, weight=30.0,
                                 params={"object_cfg": SceneEntityCfg("apple"), "std_rad": FACE_STD_RAD})


@configclass
class AppleReachTreeWaistFaceEnvCfg(AppleReachTreeWaistEnvCfg):
    rewards: RewardsFaceCfg = RewardsFaceCfg()


@configclass
class AppleReachTreeWaistFaceEnvCfg_PLAY(AppleReachTreeWaistFaceEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
        self.events.reset_waist_jitter = None
