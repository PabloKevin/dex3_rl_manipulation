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

from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
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
from .apple_grasp_grasp_env_cfg import (
    AppleGraspGraspEnvCfg,
    AppleGraspGraspSceneCfg,
    GRASP_TRAIN_REACH_AZIMUTH_CENTER_DEG,
    GRASP_TRAIN_REACH_AZIMUTH_SPAN_DEG,
    GRASP_TRAIN_REACH_MAX,
    GRASP_TRAIN_REACH_MIN,
)
from .mdp import apple_tree

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
#: One radius per tree (see apple_tree): the grasp policy's 32-48 mm.
TREE_APPLE_RADIUS_RANGE = (0.032, 0.048)


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
