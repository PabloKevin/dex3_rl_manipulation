# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach only apples the head camera sees, waist held (AppleReachVisible-v1).

User, 2026-10-16/17: on the robot the head camera (the built-in D435i the
detector uses) sees only fruit in front of and below it -- about 15% of the
apples AppleReachSingle-v2 trained on, nothing above the camera's own height
-- and the waist motion seemed to make the policies worse. So: the reach
works with the torso HELD, on apples the camera can see; a separate script
on the robot (later) turns the waist slowly to the right, like a lidar, until
apples come into view, then runs the reach. And the reach must rise / lower
next to the body to the apple's height first and only then go in, instead of
cutting diagonally through the trunk and the neighbours. Trained from
scratch (v2 had not learned that path). AppleReachSingle-v2
(apple_reach_single_v2_env_cfg.py) subclassed:

  - waist: no action. Each episode holds it at a random fixed yaw in
    FIXED_WAIST_YAW_DEG (joint position + PD target), and the tree --
    sampled for the torso at yaw 0 -- turns with it about the waist axis
    (visible_reach.place_tree_fixed_waist). Waist yaw turns about gravity, so
    in the torso frame the arm sees the same task at every yaw; only the legs
    move relative to it.
  - inputs (40 -> 37): palm orientation relative to the torso (no "unyawed"
    term), the perceived apple in the torso's yaw frame
    (visible_reach.apple_pos_palm_measured_torso), no waist slots.
    37 inputs / 7 outputs: NOT the 37-input tree layout of sim2real/
    reach_io.py, which picks its layout by input size.
  - placement: the tree's first apple is seen whole by the D435i
    (CAMERA_D435I: real intrinsics + calibrated torso pose, the apple's
    image + 8 px inside the 640x480 image, 0.25-1.2 m deep) and reachable
    with the waist held (REACH_VISIBLE_WORKSPACE_MAP: probe_side_reach.py
    --single_v2 --fixed_waist -45 15 --waist_pitch 0 --sweep_visible); the target is an apple the camera
    sees. The other apples (reachable) may hang out of view, above it
    especially, as on a real tree; the ones the camera does not see read as
    absent in the clearance ring (visible_reach.mask_unseen_neighbours).
    The trunk stands behind the first apple: in front of the robot.
  - perception bias along the camera's 3D viewing ray (d435_link -> apple),
    not the horizontal pelvis line.
  - approach: the staging point (visible_reach): reach_progress /
    reach_coarse -> the same on the L path palm -> S -> goal (S at the
    apple's height, 15 cm short of it horizontally, never nearer the waist
    axis than 0.28 m); the near-apple rewards are paid inside the cluster
    zone (past S toward the apple: from 3 cm, full at 8 cm) only when level
    with the goal (4 cm, nothing from 9 cm); cluster_keepout charges being
    off-level (> 5 cm) there. advance_before_height / height_first are dropped (replaced).
  - waist and face-the-apple terms dropped (nothing turns).
  - torso upright (waist pitch 0, as the robot's locked waist; every earlier
    task leaned it 0.1 rad forward -- the toys warn about it).
"""

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp as task_mdp
from .apple_grasp_env_cfg import PALM_LOCAL_OFFSET
from .apple_reach_single_v2_env_cfg import (
    AppleReachSingleV2EnvCfg,
    CurriculumSingleV2Cfg,
    EventsSingleV2Cfg,
    RewardsSingleV2Cfg,
    SINGLE_V2_STANDOFF,
)
from .mdp import visible_reach as vr

#: the robot's detection camera: the head D435i ("unitree" in
#: ~/g1_fruits_picker/apple_picker). Intrinsics from its camera_info at
#: 640x480 (camera_config.yaml: HFOV 55.7, VFOV 43.2 deg); pose = the
#: calibrated camera -> torso_link map (camera_extrinsics.yaml "unitree",
#: hand-guided ArUco, 37 points, 3 mm mean error): optical axis 48.5 deg
#: down, so it sees 27-70 deg below horizontal and nothing above its own
#: height. A: columns = optical x right, y UP, z forward in torso_link.
#: radius / margin_px / z_range: the apple's image (a 4 cm radius covers
#: the +-20% sizes) plus 8 px must be inside the image, 0.25-1.2 m deep
#: (D435i depth minimum ~0.2-0.3 m).
CAMERA_D435I = {
    "A": [[0.006540266461485991, 0.7488958507804779, 0.662655438065965],
          [-0.9999219788636268, -0.002154753281066041, 0.012304195371741364],
          [-0.010642419840478354, 0.6626842096519896, -0.7488233284145578]],
    "t": [0.045963094943757415, 0.032092549991621463, 0.42729155279866415],
    "fx": 606.2100830078125, "fy": 605.9196166992188, "cx": 327.6353454589844, "cy": 246.68521118164062,
    "width": 640, "height": 480,
    "radius": 0.04, "margin_px": 8.0, "z_range": (0.25, 1.2),
    "torso_body": "torso_link",
}

#: waist pitch held (rad): the robot's locked waist (earlier tasks: 0.1)
WAIST_PITCH = 0.0

#: fixed waist yaw per episode (deg, + = left): the scan turns the torso to
#: the right looking for fruit, within the +-45 deg the robot's planner uses
FIXED_WAIST_YAW_DEG = (-45.0, 15.0)

#: apple centres reachable with the waist held (shoulder-relative, torso at yaw 0)
REACH_VISIBLE_WORKSPACE_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                           "reach_workspace_single_v2_fixed_waist.npz")
#: the first apple: front band (the camera test does the rest); its dz range
#: also bounds the tree's other cluster seeds, which need not be visible
VISIBLE_SEED_ENVELOPE = dict(
    reach_min=0.20, reach_max=0.70, az_center_deg=20.0, az_span_deg=140.0, dz_range=(-0.45, 0.30),
    min_forward=0.05, min_horizontal=0.15, apple_ref=True)
#: the rest of the tree: anywhere reachable
VISIBLE_MEMBER_ENVELOPE = dict(
    reach_min=0.20, reach_max=0.70, az_center_deg=-25.0, az_span_deg=200.0, dz_range=(-0.45, 0.45),
    min_forward=0.05, min_horizontal=0.15, apple_ref=True)
STRETCH_FRAC = 0.10

#: staging point S and cluster zone (visible_reach): S 15 cm short of the goal
#: horizontally, never nearer the waist axis than 0.28 m (the hand at rest:
#: 0.27-0.33 m); the zone starts 3 cm past S along the approach, full 5 cm
#: later; level = within 4 cm of the goal's height (the tilted palm's
#: reference point sits up to ~3.6 cm off the apple's height), nothing from 9
STAGE_BACK = 0.15
STAGE_BODY_REACH = 0.28
ZONE_START, ZONE_RAMP = 0.03, 0.05
LEVEL_TOL, LEVEL_BAND = 0.04, 0.05
#: near-apple rewards paid in the cluster zone only when level
GATED_TERMS = ("reach_fine", "reach_touch", "palm_lateral", "palm_toward_stem", "fingers_vertical",
               "fingertip_depth_equal", "enclosed", "enclosed_coarse")

_PALM = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
_TORSO = SceneEntityCfg("robot", body_names=["torso_link"])
_APPLE = SceneEntityCfg("apple")
_GOAL = {"robot_cfg": _PALM, "object_cfg": _APPLE, "standoff": SINGLE_V2_STANDOFF,
         "palm_local_offset": PALM_LOCAL_OFFSET}
_PATH = {"back": STAGE_BACK, "body_reach": STAGE_BODY_REACH, "level_tol": LEVEL_TOL, "level_band": LEVEL_BAND}
_ZONE = {"back": STAGE_BACK, "body_reach": STAGE_BODY_REACH, "zone_start": ZONE_START, "zone_ramp": ZONE_RAMP}
_GATE = {**_ZONE, "level_tol": LEVEL_TOL, "level_band": LEVEL_BAND}


@configclass
class RewardsVisibleCfg(RewardsSingleV2Cfg):
    # the palm off the goal's height inside the cluster zone (rising or
    # dropping through the fruit under / over the target)
    cluster_keepout = RewTerm(
        func=vr.penalty_cluster_keepout, weight=-2.0,
        params={**_GOAL, **_ZONE, "tol": LEVEL_TOL + 0.01, "ref": 0.04, "cap": 3.0})


@configclass
class EventsVisibleCfg(EventsSingleV2Cfg):
    # after apple_measurement_reset (declaration order)
    mask_unseen_neighbours = EventTerm(func=vr.mask_unseen_neighbours, mode="reset")


@configclass
class CurriculumVisibleCfg(CurriculumSingleV2Cfg):
    #: logs only (Curriculum/visible_metrics/<key>)
    visible_metrics = CurrTerm(func=vr.visible_reach_metrics)


@configclass
class AppleReachVisibleEnvCfg(AppleReachSingleV2EnvCfg):
    rewards: RewardsVisibleCfg = RewardsVisibleCfg()
    events: EventsVisibleCfg = EventsVisibleCfg()
    curriculum: CurriculumVisibleCfg = CurriculumVisibleCfg()
    #: the camera model, also read by scripts/play.py (frustum, seen apples, its image)
    head_camera: dict = CAMERA_D435I

    def __post_init__(self):
        super().__post_init__()

        # torso upright on the pelvis, as on the robot (waist roll / pitch locked at 0;
        # sim2real logs: "waist pitch +0.0 deg (training: +5.7)"); every earlier task
        # leaned it 0.1 rad forward. Also what the camera calibration assumes.
        robot = self.scene.robot
        jp = dict(robot.init_state.joint_pos)
        jp["waist_pitch_joint"] = WAIST_PITCH
        self.scene.robot = robot.replace(init_state=robot.init_state.replace(joint_pos=jp))

        # waist: held, no action, no inputs
        self.actions.waist_yaw = None
        obs = self.observations.policy
        obs.waist_yaw_pos = obs.waist_yaw_vel = obs.waist_last_action = None
        obs.palm_orientation_torso = ObsTerm(func=task_mdp.body_quat_relative_to_body,
                                             params={"robot_cfg": _PALM, "reference_cfg": _TORSO})
        obs.apple_pos_palm = ObsTerm(func=vr.apple_pos_palm_measured_torso, params=dict(obs.apple_pos_palm.params))

        # placement: fixed yaw, seen + reachable first apple, a seen target
        ev = self.events
        ev.reset_waist_jitter = None
        tree = ev.randomize_tree
        tree.func = vr.place_tree_fixed_waist
        tree.params.update({
            "yaw_range_deg": FIXED_WAIST_YAW_DEG,
            "camera": dict(CAMERA_D435I),
            "seed_envelope": {**VISIBLE_SEED_ENVELOPE, "stretch_frac": STRETCH_FRAC, "stretch_cells": 1},
            "member_envelope": dict(VISIBLE_MEMBER_ENVELOPE),
            "workspace_map": REACH_VISIBLE_WORKSPACE_MAP,
        })
        ev.apple_measurement_reset.params.update({"origin_body": "d435_link", "horizontal": False})

        rw = self.rewards
        # nothing turns
        for name in ("waist_excursion", "waist_vel", "waist_brake", "commanded_accel_waist", "limiter_lead_waist",
                     "motor_load_waist", "face_apple", "face_apple_hold", "face_apple_success"):
            setattr(rw, name, None)
        # the staging-point approach replaces the straight-line pulls and the old height terms
        rw.reach_progress = RewTerm(func=vr.reward_path_progress, weight=10.0, params={**_GOAL, **_PATH})
        rw.reach_coarse = RewTerm(func=vr.reward_path_gaussian, weight=1.0, params={**_GOAL, **_PATH, "std": 0.25})
        rw.advance_before_height = None
        rw.height_first = None
        for name in GATED_TERMS:
            t = getattr(rw, name)
            setattr(rw, name, RewTerm(func=vr.level_gated, weight=t.weight,
                                      params={"term": t.func, "term_params": dict(t.params), **_GOAL, **_GATE}))


@configclass
class AppleReachVisibleEnvCfg_PLAY(AppleReachVisibleEnvCfg):
    """Single env, for watching a policy (play.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_arm_jitter = None
