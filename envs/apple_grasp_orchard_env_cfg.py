# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp task (AppleGrasp-v1) in the NOVA commercial orchard -- for TESTING the
frozen reach + grasp policies (scripts/harvest_test_bench.py --orchard), not
for training: training stays on the light procedural tree of
apple_grasp_grasp_env_cfg.py, which this file only subclasses (nothing there
changes).

Differences from AppleGraspGraspEnvCfg:
  - scene: the procedural trunk/branch/stem prims are dropped; the orchard
    (envs/mdp/orchard.py: 15 tall-spindle trees, trellis, terrain) is one
    kinematic body, moved around the robot every episode. Its wood colliders
    are OFF by default (ORCHARD_SOLID_WOOD), like the procedural tree the
    policies were trained on (no colliders, intrusion only measured with
    capsules): the frozen reach policy has no obstacle awareness, and with
    solid wood it drives the hand into the branches next to the fruit and
    stays there. Turn them on to test exactly that. The ground plane stays as
    an invisible floor at the robot's feet (the terrain is visual).
  - events: the apple is one of the orchard's hanging fruit
    (orchard.place_orchard_and_apple, same reach-envelope sampling as the
    grasp task), its stem is the real pedicel (vertical), and its radius is
    drawn from the orchard's fruit range.
  - one env (the orchard is 12 m x 10 m).
"""

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .apple_grasp_env_cfg import APPLE_RADIUS, GRASP_STANDOFF, REACH_MIN_FORWARD, REACH_MIN_HORIZONTAL, \
    REACH_TARGET_DZ_RANGE, REACH_WORKSPACE_MAP, BRANCH_YAW_SPREAD_DEG, STEM_RADIUS
from .apple_grasp_grasp_env_cfg import (
    AppleGraspGraspEnvCfg,
    AppleGraspGraspSceneCfg,
    GRASP_TRAIN_REACH_AZIMUTH_CENTER_DEG,
    GRASP_TRAIN_REACH_AZIMUTH_SPAN_DEG,
    GRASP_TRAIN_REACH_MAX,
    GRASP_TRAIN_REACH_MIN,
)
from .mdp import orchard

#: Fruit radius per run (USD scale, fixed at startup): the orchard's apples
#: are 32-42 mm, all inside the 32-48 mm the grasp policy trained on.
ORCHARD_APPLE_RADIUS_RANGE = (0.032, 0.042)
#: Wood (trunks, scaffolds, posts, wires, bins) collides with the robot and
#: the apple. Default off -- see the module docstring.
ORCHARD_SOLID_WOOD = False


@configclass
class AppleGraspOrchardSceneCfg(AppleGraspGraspSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(visible=False))
    tree_base = None
    tree_trunk = None
    tree_branch = None
    apple_stem = None
    orchard: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Orchard",
        spawn=sim_utils.UsdFileCfg(
            usd_path=orchard.USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=ORCHARD_SOLID_WOOD, contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -50.0)),
    )

    def __post_init__(self):
        # A fruit pressed into solid wood while pinned is otherwise shot out
        # at PhysX's default depenetration speed when the pin lets go
        # (measured: several metres).
        self.apple = self.apple.replace(spawn=self.apple.spawn.replace(
            rigid_props=self.apple.spawn.rigid_props.replace(max_depenetration_velocity=0.5)))


@configclass
class AppleGraspOrchardEnvCfg(AppleGraspGraspEnvCfg):
    scene: AppleGraspOrchardSceneCfg = AppleGraspOrchardSceneCfg(num_envs=1, env_spacing=30.0)

    def __post_init__(self):
        super().__post_init__()
        orchard.build_usd()            # (re)builds orchard.usd if the export is newer
        self.scene.num_envs = 1
        ev = self.events
        ev.randomize_tree_geometry = None
        # Same slots as the grasp task's events, so the reset order is kept
        # (reset_reach_grasp_phase must stay last among the resets).
        ev.randomize_apple_scale = EventTerm(
            func=orchard.set_apple_radius,
            mode="prestartup",
            params={"object_cfg": SceneEntityCfg("apple"), "spawn_radius": APPLE_RADIUS,
                    "radius_range": ORCHARD_APPLE_RADIUS_RANGE},
        )
        ev.randomize_tree = EventTerm(
            func=orchard.place_orchard_and_apple,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("apple"),
                "orchard_cfg": SceneEntityCfg("orchard"),
                "reach_min": GRASP_TRAIN_REACH_MIN,
                "reach_max": GRASP_TRAIN_REACH_MAX,
                "reach_azimuth_center_deg": GRASP_TRAIN_REACH_AZIMUTH_CENTER_DEG,
                "reach_azimuth_span_deg": GRASP_TRAIN_REACH_AZIMUTH_SPAN_DEG,
                "target_dz_range": REACH_TARGET_DZ_RANGE,
                "approach_yaw_spread_deg": BRANCH_YAW_SPREAD_DEG,
                "standoff": GRASP_STANDOFF,
                "reach_min_forward": REACH_MIN_FORWARD,
                "reach_min_horizontal": REACH_MIN_HORIZONTAL,
                "workspace_map": REACH_WORKSPACE_MAP,
                "stem_radius": STEM_RADIUS,
                "screen_robot": ORCHARD_SOLID_WOOD,
            },
        )
        self.viewer.eye = (-1.0, -1.3, 1.7)
        self.viewer.lookat = (0.3, -0.15, 1.0)
