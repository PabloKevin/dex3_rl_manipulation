# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp training after the waist reach, on the tall-spindle tree
(AppleGraspSpindle-v1): the tree grasp (AppleGraspTree-v1,
apple_grasp_tree_train_env_cfg.py -- subclassed; hand-only 44-input policy,
hold-under-load stage, any-apple pad tactile) with its reach phase and scene
brought to the reach the robot now runs (AppleReachSpindle-v3):

  - reach phase driven by the pinned v3 reach (best_weights/
    grasp_handoff_reach_spindle.pt, 40 inputs, 8 outputs): the waist yaw
    turns with the reach and is held where the reach left it during the
    grasp (the grasp never drives it); arm targets clamped to the soft
    joint limits and the 20 cm clearance ring, as in v3;
  - the hand pre-shaped during the reach (REACH_HAND_PRESHAPE: index /
    middle bases straight, middle joints 80 deg, thumb_1 0.15): the grasp
    starts from that cup and has to open the middle joints before closing
    the bases;
  - the tall-spindle tree (thin leaning trunk, short branches) on the v3
    reach's placement (waist workspace map, 0.24-0.44 m, -80..+30 deg);
  - waist yaw actuator with the deployed gains, as in the reach;
  - hand-off "converged": once the palm has held the reach-success
    neighbourhood for 0.5 s (v3: 87% of reaches, median 1.84 s, p90 2.57 s),
    3.5 s at the latest -- the fixed 2.5 s hand-off would cut ~10% of the
    reaches short.

The grasp policy's inputs are hand-side only (fingers, tactile, fruit and
gravity in the palm frame), so the turned torso changes nothing it sees.
"""

import os

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .apple_grasp_tree_train_env_cfg import AppleGraspTreeTrainEnvCfg, TREE_GRASP_WINDOW_S
from .apple_reach_spindle_env_cfg import (
    ARM_TARGET_CLIP,
    REACH_HAND_PRESHAPE,
    SPINDLE,
    SPINDLE_TRUNK_RADIUS,
    SPINDLE_V2_RING_RANGE,
)
from .apple_reach_tree_waist_env_cfg import (
    REACH_WAIST_AZIMUTH_CENTER_DEG,
    REACH_WAIST_AZIMUTH_SPAN_DEG,
    REACH_WAIST_DZ_RANGE,
    REACH_WAIST_MAX,
    REACH_WAIST_MIN,
    REACH_WAIST_MIN_FORWARD,
    REACH_WAIST_MIN_HORIZONTAL,
    REACH_WAIST_WORKSPACE_MAP,
    WAIST_MAX_ACCELERATION,
    WAIST_MAX_VELOCITY,
    WAIST_YAW_JOINT,
    WAIST_YAW_LIMIT,
)

SPINDLE_REACH_POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "best_weights",
                                    "grasp_handoff_reach_spindle.pt")
#: Hand-off: converged (0.5 s in the success neighbourhood), this at the latest.
SPINDLE_HANDOFF_WATCHDOG_S = 3.5


def apply_spindle_setup(cfg) -> None:
    """Bring a tree grasp env cfg (AppleGraspTreeEnvCfg or a subclass: the
    training env, the harvest bench) to the v3 waist reach on the spindle
    tree: scene trunk, waist actuator, reach phase, hand pre-shape,
    placement, converged hand-off. Shared so the bench runs exactly what the
    grasp trained on."""
    sc = cfg.scene
    # scene: spindle trunk (lower = the scene's trunk, thinner; upper: own prim)
    sc.tree_trunk = sc.tree_trunk.replace(spawn=sc.tree_trunk.spawn.replace(radius=SPINDLE_TRUNK_RADIUS))
    sc.tree_trunk_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TreeTrunkTop",
        spawn=sim_utils.CylinderCfg(
            radius=SPINDLE_TRUNK_RADIUS, height=1.0, axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.20, 0.08), roughness=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 1.5)),
    )
    # waist yaw: out of the legs group, deployed gains (as the reach)
    acts = dict(sc.robot.actuators)
    acts["legs"] = acts["legs"].replace(joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"])
    acts["waist_yaw"] = ImplicitActuatorCfg(joint_names_expr=[WAIST_YAW_JOINT], effort_limit=88.0,
                                            velocity_limit=32.0, stiffness=200.0, damping=5.0)
    sc.robot = sc.robot.replace(actuators=acts)

    # reach phase: the v3 waist reach
    arm = cfg.actions.right_arm
    arm.reach_policy_path = SPINDLE_REACH_POLICY
    arm.reach_obs_layout = "waist"
    arm.reach_waist_joint = WAIST_YAW_JOINT
    arm.reach_waist_limit = WAIST_YAW_LIMIT
    arm.reach_waist_max_velocity = WAIST_MAX_VELOCITY
    arm.reach_waist_max_acceleration = WAIST_MAX_ACCELERATION
    arm.reach_ring_range = SPINDLE_V2_RING_RANGE
    arm.clip = dict(ARM_TARGET_CLIP)
    # hand pre-shaped during the reach
    cfg.actions.right_hand.reach_offset = dict(REACH_HAND_PRESHAPE)

    # tree + placement: the v3 reach's
    envelope = dict(
        reach_min=REACH_WAIST_MIN, reach_max=REACH_WAIST_MAX,
        az_center_deg=REACH_WAIST_AZIMUTH_CENTER_DEG, az_span_deg=REACH_WAIST_AZIMUTH_SPAN_DEG,
        dz_range=REACH_WAIST_DZ_RANGE, min_forward=REACH_WAIST_MIN_FORWARD,
        min_horizontal=REACH_WAIST_MIN_HORIZONTAL)
    cfg.events.randomize_tree.params.update({
        "seed_envelope": dict(envelope),
        "member_envelope": dict(envelope),
        "workspace_map": REACH_WAIST_WORKSPACE_MAP,
        "spindle": dict(SPINDLE),
        "trunk_radius": SPINDLE_TRUNK_RADIUS,
    })

    # hand-off once the reach has settled
    cfg.events.update_reach_grasp_phase.params.update(
        handoff_mode="converged", handoff_time_s=SPINDLE_HANDOFF_WATCHDOG_S)


@configclass
class AppleGraspSpindleEnvCfg(AppleGraspTreeTrainEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = SPINDLE_HANDOFF_WATCHDOG_S + TREE_GRASP_WINDOW_S
        apply_spindle_setup(self)


@configclass
class AppleGraspSpindleEnvCfg_PLAY(AppleGraspSpindleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None


# ─────────────────────────────────────────────
#  v2: a stem-like twist load and a harvest grip command (AppleGraspSpindle-v2)
# ─────────────────────────────────────────────
# AppleGraspSpindle-v1 flattened at full load (it 800-1100: fell ~60% of
# episode ends). Measured on its it-1100 policy (scratchpad grip_capacity /
# active_vs_frozen): its own grip, frozen, holds a slowly rising twist to a
# median 0.38 N*m with no bruising (0.47 with +0.2 rad squeeze), but the
# v1 load HELD up to 0.4 N*m for ~5 s plus up to 6 N (= 40 m/s^2 on the
# fruit): even frozen, 15% survived full twist. And at 0.2 N*m the acting
# policy held 30% vs 60% frozen -- it reacted its grip away. So:
#   - the twist load follows the stem: the bench's 90 deg in 1.5 s against
#     the tree stems' stiffness until their break torque, then nothing;
#     carry force up to 2 N, held (tree_grasp.release_stem_and_twist);
#   - at the release the grip command (an input) goes to 0.45-0.75 of the
#     episode's damage force, as the bench commands it for the harvest;
#   - + a one-off reward for the stem breaking with the fruit in the pocket.
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm  # noqa: E402

from .apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL, GRASP_PPO_STEPS_PER_ITER  # noqa: E402
from .apple_grasp_tree_env_cfg import TREE_STEM_PARAMS  # noqa: E402
from .apple_grasp_tree_train_env_cfg import RewardsCfg as TreeGraspRewardsCfg  # noqa: E402
from .mdp import tree_grasp as tg  # noqa: E402


@configclass
class RewardsSpindleV2Cfg(TreeGraspRewardsCfg):
    # one-off when the stem breaks with the fruit held: ~2 per episode
    # (x dt), about all of hold_under_load's per-episode share
    twist_survived = RewTerm(func=tg.reward_twist_survived, weight=100.0,
                             params={"center_local": GRASP_CENTER_LOCAL, "tol": 0.03})


@configclass
class AppleGraspSpindleV2EnvCfg(AppleGraspSpindleEnvCfg):
    rewards: RewardsSpindleV2Cfg = RewardsSpindleV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.release_stem_and_load = EventTerm(
            func=tg.release_stem_and_twist, mode="interval", interval_range_s=(0.0, 0.0),
            params={"k_twist": TREE_STEM_PARAMS["k_twist"], "break_twist": TREE_STEM_PARAMS["break_twist"],
                    "twist_deg": 90.0, "twist_s": 1.5, "load_force": 2.0, "delay": 0.4, "ramp": 1.0,
                    "grip_cmd": (0.45, 0.75), "curriculum": (0, 300),
                    "steps_per_iter": GRASP_PPO_STEPS_PER_ITER},
        )


@configclass
class AppleGraspSpindleV2EnvCfg_PLAY(AppleGraspSpindleV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None


# ─────────────────────────────────────────────
#  v3: + the bench's pull after an unbroken twist (AppleGraspSpindle-v3)
# ─────────────────────────────────────────────
# First full harvest bench on the spindle chain (2026-10-12, 56 apples,
# v2 grasp it 1499): 24 in the basket. Stems broke by PULL 21 times, by
# twist 10, by bend 12 -- the hand twist often turns the fruit in the hand
# or is stopped by a neighbour / the body (11 times), and the 5 cm pull-back
# then breaks the stem at 15-25 N, through the grip: 10 of 17 drops were in
# the pull, and most of the 7 left on the tree slipped out there. Low
# fruit (crowded bottom clusters, twist blocked) finished 3 of 15.
STEM_PULL_BREAK = (15.0, 25.0)        # harvest.StemParams.break_pull (the bench's stems)


@configclass
class AppleGraspSpindleV3EnvCfg(AppleGraspSpindleV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.release_stem_and_load.params.update(
            twist_fail_prob=0.5, pull_break=STEM_PULL_BREAK, pull_s=0.6, curriculum=(0, 300))


@configclass
class AppleGraspSpindleV3EnvCfg_PLAY(AppleGraspSpindleV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
