# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp task (AppleGrasp-v1) on a procedural tree carrying APPLES_PER_TREE
apples in clusters (envs/mdp/apple_tree.py), for harvesting a whole tree
one apple at a time (scripts/tree_harvest_bench.py).

Same procedural fruiting wall, robot, sensors, observations, actions and
frozen-policy hand-off as AppleGraspGraspEnvCfg (subclassed, nothing there
changes, so training is unaffected unless a task is registered on this
cfg). Added:
  - apple_1 .. apple_{N-1}: the target's neighbours, dynamic spheres on
    breakable stems -- the hand and fingers can push them, swing them and
    knock them off;
  - one branch + stem per apple, all from one trunk behind them;
  - per-env reset that either grows a new tree or moves on to the next
    apple of the same one (apple_tree.place_apple_tree);
  - no start-pose jitter: every apple starts from home, where the last one
    ended -- and no joint teleport between the apples of a tree (only a new
    tree resets the robot), which snapped the arm by its gravity sag;
  - the target hangs on a breakable stem like its neighbours, instead of
    the grasp task's rigid reach-phase pin (a per-step teleport back into
    the hand, which let the hand sink into the fruit).
"""

import math

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .apple_grasp_env_cfg import (
    APPLE_RADIUS,
    GRASP_STANDOFF,
    REACH_AZIMUTH_CENTER_DEG,
    REACH_AZIMUTH_SPAN_DEG,
    REACH_MAX,
    REACH_MIN,
    REACH_MIN_FORWARD,
    REACH_MIN_HORIZONTAL,
    REACH_TARGET_DZ_RANGE,
    REACH_WORKSPACE_MAP,
    BRANCH_RADIUS,
    BRANCH_YAW_SPREAD_DEG,
    STEM_LENGTH_RANGE,
    STEM_RADIUS,
    TRUNK_HEIGHT_RANGE,
    TRUNK_RADIUS,
)
from . import mdp as task_mdp
from .apple_grasp_grasp_env_cfg import (
    AppleGraspGraspEnvCfg,
    AppleGraspGraspSceneCfg,
    ObservationsCfg as GraspObservationsCfg,
    GRASP_TRAIN_REACH_AZIMUTH_CENTER_DEG,
    GRASP_TRAIN_REACH_AZIMUTH_SPAN_DEG,
    GRASP_TRAIN_REACH_MAX,
    GRASP_TRAIN_REACH_MIN,
)
from .mdp import apple_tree
from .mdp import tree_grasp

#: Apples per tree. All of them sit inside the frozen policies' reach
#: envelope (0.24-0.34 m from the shoulder, a ~0.2 x 0.6 m patch of wall):
#: 10 always fits (even at 48 mm radius, offline test of sample_tree) but
#: crowds it; 7 leaves room between the groups.
APPLES_PER_TREE = 7
#: Cluster sizes and their probabilities (per cluster): about a third of
#: the fruit hangs alone (>= TREE_SEPARATION from any other), the rest
#: touch at least one neighbour.
CLUSTER_SIZES = (1, 2, 3, 4)
CLUSTER_WEIGHTS = (0.60, 0.20, 0.12, 0.08)
#: Surface gap between apples of one cluster: negative = pressed together.
CLUSTER_GAP = (-0.003, 0.006)
#: Minimum surface gap between apples of different clusters: room for a
#: finger (~2 cm) to pass with margin. With 7 apples every tree still fits
#: at 48 mm radius (offline: 0/100 failures; 8 cm fails 2%).
TREE_SEPARATION = 0.06
#: Trunk behind the fruit, this far from the shoulder (along the envelope's
#: centre azimuth), +- jitter: beyond the 0.34 m reach, so branches run
#: back toward the robot as in the single-apple tree.
TRUNK_DISTANCE = 0.55
TRUNK_JITTER = 0.05
#: Harvest order: "top_down", "bottom_up", "random" or "clearance" (each
#: time, the hanging fruit the hand can reach with the least contact with
#: the others, apple_tree.access_cost; ties go to the lowest). Bottom-up: every
#: reach starts at home, below the fruit, so going for the top apple first
#: sweeps the hand up through the whole wall (measured, 4 trees each: top-
#: down knocked 7 of 10 apples off while picking the first one; bottom-up
#: 2.25 per tree in all).
HARVEST_ORDER = "bottom_up"
#: Stem model for every fruit of the tree (overrides of harvest.StemParams).
#: Twist: the shared defaults break at ~20-35 deg of twist (chosen so the
#: harvest twist breaks easily), and in this tree that knocked fruit off
#: whenever a palm or finger merely ROLLED it (measured: 19-27 deg twist
#: breaks, targets during the reach and neighbours alike). Picking by
#: twisting takes ~90 deg; here incidental rolls under ~40 deg survive and
#: the bench's 90 deg harvest twist still breaks practically all stems
#: (break angle = break torque / stiffness: 0.25/0.35 = 41 to 0.40/0.25 = 92 deg).
TREE_STEM_PARAMS = {"k_twist": (0.25, 0.35), "break_twist": (0.25, 0.40)}
#: Friction, randomized every episode on every apple AND the hand's links
#: (PhysX averages the two): rubber/silicone pads on apple skin, ~0.8, with
#: no measured value for the Dex3 pads (the spec only says 500 g payload on
#: 5 cm hard objects). The grasp has to adapt its force through the pads.
#: Before, the dynamic apple had NO material: PhysX's default ~0.5.
FRICTION_STATIC = (0.6, 1.0)
FRICTION_DYNAMIC = (0.45, 0.8)
#: One radius per tree (see apple_tree): the grasp policy's 32-48 mm.
TREE_APPLE_RADIUS_RANGE = (0.032, 0.048)


@configclass
class HandOnlyObservationsCfg(GraspObservationsCfg):
    """The grasp policy's inputs WITHOUT the arm (66 -> 41, + gravity in the
    palm frame -> 44): hand joint
    positions/velocities, the three tactile terms (any apple, pad face only),
    the grip-force command, the phase flag, the fruit in the palm frame and
    the last HAND action. No arm joint positions/velocities, no palm
    orientation, no arm half of the last action -- so the policy can keep
    driving the fingers while the arm twists and carries the fruit (moves it
    never saw in training): nothing it observes depends on the arm's pose."""

    @configclass
    class PolicyObs(GraspObservationsCfg.PolicyObs):
        right_arm_pos = None
        right_arm_vel = None
        palm_orientation_torso = None
        tactile_slip = ObsTerm(func=task_mdp.tactile_slip,
                               params={"sensor_cfg": SceneEntityCfg("contact_forces"), "all_apples": True})
        tactile_finger_balance = ObsTerm(func=task_mdp.tactile_finger_balance,
                                         params={"sensor_cfg": SceneEntityCfg("contact_forces"), "all_apples": True})
        tactile_force_level = ObsTerm(func=task_mdp.tactile_force_level,
                                      params={"sensor_cfg": SceneEntityCfg("contact_forces"), "all_apples": True})
        last_action = ObsTerm(func=base_mdp.last_action, params={"action_name": "right_hand"})
        # which way the fruit's weight pulls, in the palm frame (FK + torso IMU)
        gravity_palm = ObsTerm(func=tree_grasp.gravity_in_palm)

    policy: PolicyObs = PolicyObs()


#: Columns of the 66-dim AppleGrasp-v1 observation that the hand-only one
#: keeps, in its order (for the warm start): hand pos/vel 14:28, tactile
#: slip/balance/level + grip command + phase + fruit-in-palm 32:52, hand half
#: of the 14-dim last action 59:66.
HAND_ONLY_FROM_GRASP_COLS = list(range(14, 28)) + list(range(32, 52)) + list(range(59, 66))


def add_tree_apples(scene, n_apples: int) -> None:
    """The target's neighbours (apple_1..), and a branch + stem prim per
    apple, added to a grasp scene cfg."""
    # Pressed-together fruit must separate gently, not jump.
    scene.apple = scene.apple.replace(spawn=scene.apple.spawn.replace(
        rigid_props=scene.apple.spawn.rigid_props.replace(max_depenetration_velocity=0.3)))
    for k in range(1, n_apples):
        setattr(scene, apple_tree.apple_entity(k),
                scene.apple.replace(prim_path=f"{{ENV_REGEX_NS}}/{apple_tree.apple_prim(k)}"))
        setattr(scene, f"tree_branch_{k}",
                scene.tree_branch.replace(prim_path=f"{{ENV_REGEX_NS}}/{apple_tree.branch_prim(k)}"))
        setattr(scene, f"apple_stem_{k}",
                scene.apple_stem.replace(prim_path=f"{{ENV_REGEX_NS}}/{apple_tree.stem_prim(k)}"))
    # The per-link hand sensors feel EVERY apple (filter m = entity
    # apple_tree.apple_entity(m), the target first): a real pad reads any
    # fruit pressing on it. Readers that pass all_apples (the tactile
    # observations) sum them, pad-face weighted; everything else (rewards,
    # fruit damage, the grip latch) keeps reading filter 0, the target.
    from .mdp.observations import RIGHT_HAND_CONTACT_LINKS, link_sensor_name
    filters = [f"{{ENV_REGEX_NS}}/{apple_tree.apple_prim(m)}" for m in range(n_apples)]
    for link in RIGHT_HAND_CONTACT_LINKS:
        sensor = getattr(scene, link_sensor_name("contact_forces", link), None)
        if sensor is not None:
            sensor.filter_prim_paths_expr = list(filters)


@configclass
class AppleGraspTreeEnvCfg(AppleGraspGraspEnvCfg):
    scene: AppleGraspGraspSceneCfg = AppleGraspGraspSceneCfg(num_envs=4, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        add_tree_apples(self.scene, APPLES_PER_TREE)
        ev = self.events
        ev.randomize_apple_scale = None
        ev.reset_arm_jitter = None
        ev.reset_all_exact = EventTerm(func=apple_tree.reset_joints_on_new_tree, mode="reset",
                                       params={"asset_cfg": SceneEntityCfg("robot")})
        ev.hold_default_targets = EventTerm(func=apple_tree.hold_targets_on_new_tree, mode="reset",
                                            params={"asset_cfg": SceneEntityCfg("robot")})
        self.actions.right_arm.pin_object = None          # the stems hold every fruit
        friction = dict(static_friction_range=FRICTION_STATIC, dynamic_friction_range=FRICTION_DYNAMIC,
                        restitution_range=(0.0, 0.0), num_buckets=64, make_consistent=True)
        for k in range(APPLES_PER_TREE):
            setattr(ev, f"friction_{apple_tree.apple_entity(k)}", EventTerm(
                func=base_mdp.randomize_rigid_body_material, mode="reset",
                params={"asset_cfg": SceneEntityCfg(apple_tree.apple_entity(k)), **friction}))
        ev.friction_hand = EventTerm(
            func=base_mdp.randomize_rigid_body_material, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_.*"]), **friction})
        # prestartup slot of the single tree's geometry randomization
        ev.randomize_tree_geometry = EventTerm(
            func=apple_tree.setup_tree_prims,
            mode="prestartup",
            params={"n_apples": APPLES_PER_TREE, "trunk_height_range": TRUNK_HEIGHT_RANGE,
                    "radius_range": TREE_APPLE_RADIUS_RANGE, "spawn_radius": APPLE_RADIUS},
        )
        # reset slot of the single apple's placement (keeps the reset order)
        ev.randomize_tree = EventTerm(
            func=apple_tree.place_apple_tree,
            mode="reset",
            params={
                "n_apples": APPLES_PER_TREE,
                "seed_envelope": dict(
                    reach_min=GRASP_TRAIN_REACH_MIN, reach_max=GRASP_TRAIN_REACH_MAX,
                    az_center_deg=GRASP_TRAIN_REACH_AZIMUTH_CENTER_DEG, az_span_deg=GRASP_TRAIN_REACH_AZIMUTH_SPAN_DEG,
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
                "order": HARVEST_ORDER,
                "stem_params": TREE_STEM_PARAMS,
                "stem_radius": STEM_RADIUS,
                "branch_radius": BRANCH_RADIUS,
                "trunk_radius": TRUNK_RADIUS,
            },
        )
        self.viewer.eye = (1.3, -1.6, 1.7)
        self.viewer.lookat = (0.35, -0.3, 1.05)
