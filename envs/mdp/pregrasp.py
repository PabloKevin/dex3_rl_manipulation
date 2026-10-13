# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Two-stage reach (envs/apple_reach_twostage_env_cfg.py): the pieces the
two tasks need beyond the spindle reach.

  Part 1 -- PREGRASP (AppleReachPregrasp-v1): home -> a point in front of
    the apple (its height, ~10 cm from its centre), palm facing it, fingers
    OPEN, no touch input, never touching the target.
  Part 2 -- FINAL (AppleReachFinal-v1): from that point, the fingers in a
    (wider) cup, to the gentle palm touch the grasp starts from, with the
    palm tactile input; no finger may touch the target.

Both run on trees whose fruit hang on MAGNET-like stems (MAGNET_STEM_PARAMS):
the lab's replica apples come off with a light push.

Part 2 does not start from home. Its episodes start from a START LIBRARY of
pregrasp hand poses (scripts/build_pregrasp_starts.py, from the v3 reach's
recorded end poses backed off along the approach with pinocchio IK): at
reset, place_tree_at_pregrasp puts the target apple in front of the palm
(randomized gap / offset / stem heading, standing for part 1's error),
builds the spindle tree around it with no other fruit inside the hand, and
set_pregrasp_joints teleports the arm, the waist and the cupped hand there.

Module-level imports are numpy / torch only, so the library builder can
load the constants below without Isaac Lab.
"""

from __future__ import annotations

import math
import os
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START_LIBRARY = os.path.join(REPO, "envs", "assets", "pregrasp_starts.npz")

#: Part 1's goal: the palm reference point (PALM_LOCAL_OFFSET) this far from
#: the apple SURFACE along the palm normal (the reach rewards' `standoff`;
#: the spindle reach uses GRASP_STANDOFF = 0.013, touching). With the trees'
#: 32-48 mm apples the centre is 9.2-10.8 cm from the palm point.
PREGRASP_STANDOFF = 0.06

#: Part 2's start: apple surface this far from the palm point (part 1's
#: standoff +-2.5 cm), its centre off the palm normal by up to this much in
#: each cross direction, and the stem heading this far off the palm heading
#: (clipped to the trees' +-30 deg).
FINAL_START_GAP = (0.035, 0.085)
FINAL_START_LATERAL = 0.02
FINAL_START_HEADING_DEG = 20.0
#: the target apple itself must clear the hand by this much at the start
FINAL_START_TARGET_CLEAR = 0.005

#: Part 2's hand: the reach cup (apple_reach_spindle_env_cfg.REACH_HAND_PRESHAPE)
#: with the index / middle middle joints curled LESS (80 -> 60 deg): the
#: curled tips stand further out, so the palm can arrive further off-centre
#: toward the fingertips before they touch the fruit. Still closed enough to
#: keep the long straight fingers out of the neighbours.
FINAL_FINGER_CURL = 1.05
FINAL_HAND_PRESHAPE = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.15,
    "right_hand_thumb_2_joint": -0.340,
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": FINAL_FINGER_CURL,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_middle_1_joint": FINAL_FINGER_CURL,
}

#: Magnet-like attachment for EVERY fruit (target and neighbours), as the
#: lab's replica apples: only the load beyond the fruit's own weight counts;
#: pressing toward the branch never breaks it, pulling it away does above
#: break_pull, pushing it sideways above break_shear, rolling it on the
#: stem above break_bend. GUESSES, not measured -- measure the real magnet
#: (spring scale: straight pull, sideways push at the apple's equator) and
#: put the numbers here. Twist as the spindle trees (TREE_STEM_PARAMS).
MAGNET_STEM_PARAMS = {
    "pull_over_rest": True,
    "break_pull": (1.0, 3.0),        # N, tension beyond the weight
    "break_shear": (0.5, 1.5),       # N, sideways
    "break_bend": (0.03, 0.10),      # N*m: ~0.5-1.5 N tangential on the ~7 cm stem lever
    "k_twist": (0.25, 0.35),
    "break_twist": (0.25, 0.40),
}

#: Points of the hand and forearm (link, offset in the link frame, radius)
#: no neighbour may overlap at part 2's start: the Dex3 spheres of
#: self_collision.HAND_SPHERES (copied: that module needs Isaac Lab) plus
#: the wrist and forearm.
KEEP_OUT_SPHERES = [
    ("right_hand_palm_link", (0.020, 0.0, 0.018), 0.028),
    ("right_hand_palm_link", (0.020, 0.0, -0.018), 0.028),
    ("right_hand_palm_link", (0.060, 0.0, 0.018), 0.028),
    ("right_hand_palm_link", (0.060, 0.0, -0.018), 0.028),
    ("right_hand_index_0_link", (0.010, 0.0, 0.0), 0.016),
    ("right_hand_index_0_link", (0.038, 0.0, 0.0), 0.016),
    ("right_hand_middle_0_link", (0.010, 0.0, 0.0), 0.016),
    ("right_hand_middle_0_link", (0.038, 0.0, 0.0), 0.016),
    ("right_hand_index_1_link", (0.010, 0.0, 0.0), 0.015),
    ("right_hand_index_1_link", (0.036, 0.0, 0.0), 0.015),
    ("right_hand_middle_1_link", (0.010, 0.0, 0.0), 0.015),
    ("right_hand_middle_1_link", (0.036, 0.0, 0.0), 0.015),
    ("right_hand_thumb_0_link", (0.0, 0.012, 0.0), 0.016),
    ("right_hand_thumb_1_link", (0.0, 0.010, 0.0), 0.016),
    ("right_hand_thumb_1_link", (0.0, 0.038, 0.0), 0.016),
    ("right_hand_thumb_2_link", (0.0, 0.012, 0.0), 0.016),
    ("right_hand_thumb_2_link", (0.0, 0.038, 0.0), 0.016),
    ("right_wrist_yaw_link", (0.0, 0.0, 0.0), 0.035),
    ("right_wrist_roll_link", (0.0, 0.0, 0.0), 0.035),
    ("right_elbow_link", (0.0, 0.0, 0.0), 0.040),
    ("forearm_mid", None, 0.040),           # halfway elbow -> wrist roll (build_pregrasp_starts.py)
]
#: extra clearance on top of the sphere radii [m]. 1 cm left 7 neighbours
#: of 64 trees knocked off within 2 s of merely HOLDING the start pose (the
#: arm sags under gravity at kp 100, up to 0.08 rad).
KEEP_OUT_MARGIN = 0.02

#: right_shoulder_pitch_joint ... right_wrist_yaw_joint (RIGHT_ARM_JOINTS)
ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
WAIST_YAW = "waist_yaw_joint"


# ─────────────────────────────────────────────
#  Observation
# ─────────────────────────────────────────────

def zero_obs(env: "ManagerBasedRLEnv", dim: int = 1) -> torch.Tensor:
    """[N, dim] zeros: an input slot kept (same layout as the spindle reach,
    so it warm-starts) but never fed -- part 1 has no touch input."""
    return torch.zeros(env.num_envs, dim, device=env.device)


# ─────────────────────────────────────────────
#  Part 2 start: tree around a pregrasp hand pose
# ─────────────────────────────────────────────

def _library(env: "ManagerBasedRLEnv", path: str) -> dict:
    lib = getattr(env, "_pg_lib", None)
    if lib is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: build it first -- python3 scripts/build_pregrasp_starts.py")
        m = np.load(path)
        lib = {k: m[k] for k in m.files}
        robot = env.scene["robot"]
        names = [str(n) for n in lib["joint_names"]]
        if names != ARM_JOINTS:
            raise ValueError(f"{path}: joint order {names}")
        if abs(float(lib["finger_curl"]) - FINAL_FINGER_CURL) > 1e-6:
            raise ValueError(f"{path}: built for a finger curl of {float(lib['finger_curl'])} rad, the task "
                             f"holds {FINAL_FINGER_CURL} -- rebuild it")
        lib["arm_ids"] = torch.tensor([robot.joint_names.index(n) for n in ARM_JOINTS], device=env.device)
        lib["waist_id"] = robot.joint_names.index(WAIST_YAW)
        lib["hand_ids"] = torch.tensor([robot.joint_names.index(n) for n in FINAL_HAND_PRESHAPE], device=env.device)
        lib["hand_q"] = torch.tensor(list(FINAL_HAND_PRESHAPE.values()), device=env.device)
        env._pg_lib = lib
        env._pg_q = torch.zeros(env.num_envs, len(ARM_JOINTS), device=env.device)
        env._pg_waist = torch.zeros(env.num_envs, device=env.device)
    return lib


def sample_target_in_front(p, n, pts, rad, r, gap_range=FINAL_START_GAP, lateral=FINAL_START_LATERAL,
                           clear=FINAL_START_TARGET_CLEAR, rng=random, gap_max=0.12, step=0.005):
    """Target apple centre in front of a start pose, or None. p / n: palm
    reference point and normal; pts / rad: the hand's keep-out spheres; r:
    apple radius. Surface gap uniform in gap_range, centre off the normal by
    up to `lateral` each way -- but never inside the hand: the cupped fingers
    stand out in front of the palm, so for a big apple the short gaps are
    in them. Those are moved OUT along the normal to the first clear gap
    (then uniform up to gap_range's top), instead of rejected: rejecting
    them failed 25% of the 48 mm apples' tries and, after enough resets,
    a whole run. None only when nothing clear exists up to gap_max."""
    u = np.cross(n, [0.0, 0.0, 1.0])
    u = u / max(np.linalg.norm(u), 1e-6)
    v = np.cross(u, n)
    base = p + u * rng.uniform(-lateral, lateral) + v * rng.uniform(-lateral, lateral)
    need = r + rad + clear

    def free(t):
        return bool(np.all(np.linalg.norm(pts - (base + n * (r + t)), axis=1) >= need))

    t = gap_range[0]
    while not free(t):
        t += step
        if t > gap_max:
            return None
    hi = max(gap_range[1], t)
    return base + n * (r + rng.uniform(t, hi) if hi > t else r + t)


def place_tree_at_pregrasp(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    tree: dict,
    library: str = START_LIBRARY,
    gap_range: tuple[float, float] = FINAL_START_GAP,
    lateral: float = FINAL_START_LATERAL,
    heading_noise_deg: float = FINAL_START_HEADING_DEG,
    target_clear: float = FINAL_START_TARGET_CLEAR,
    attempts: int = 100,
) -> None:
    """mode="reset", in place of apple_tree.place_apple_tree (`tree`: its
    parameters; spindle tree, new tree every reset). Per env: a library
    hand pose; the target apple in front of its palm (surface gap_range from
    the palm point along the palm normal, +-lateral across it); its stem
    heading the palm's heading +-heading_noise_deg, clipped to the trees'
    range; the tree built around it with no other fruit inside the hand
    (keep-out spheres + KEEP_OUT_MARGIN). The pose is kept in env._pg_q /
    env._pg_waist for set_pregrasp_joints."""
    from . import apple_tree as at

    lib = _library(env, library)
    M = len(lib["q"])
    ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    lim = math.radians(tree["stem_heading_max_deg"])
    rad_keep = lib["keep_rad"] + KEEP_OUT_MARGIN
    if not hasattr(env, "_forced_seed"):
        env._forced_seed = {}
    for i in ids:
        r = float(env._apple_radius[i])
        for _ in range(attempts):
            k = random.randrange(M)
            p, n = lib["palm_ref"][k].astype(float), lib["palm_normal"][k].astype(float)
            pts = lib["keep_pts"][k].astype(float)
            apple = sample_target_in_front(p, n, pts, lib["keep_rad"], r, gap_range, lateral, target_clear)
            if apple is None:
                continue
            h_palm = math.atan2(n[1], n[0])
            heading = max(-lim, min(lim, h_palm + math.radians(random.uniform(-heading_noise_deg, heading_noise_deg))))
            env._forced_seed[i] = {"apple": apple, "heading": heading, "pts": pts, "rad": rad_keep}
            try:
                at.place_apple_tree(env, torch.tensor([i], device=env.device), **tree)
            except RuntimeError:
                continue                      # no tree fits around this one: another pose
            env._pg_q[i] = torch.tensor(lib["q"][k], device=env.device)
            env._pg_waist[i] = float(lib["waist"][k])
            break
        else:
            raise RuntimeError(f"place_tree_at_pregrasp: env {i}: no start in {attempts} attempts")


def set_pregrasp_joints(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """mode="reset", after every other joint reset: arm and waist to the pose
    place_tree_at_pregrasp chose, the hand already in FINAL_HAND_PRESHAPE
    (state and PD targets), at rest. The arm's rate limiter re-seeds from
    the joint state on the first step."""
    lib = env._pg_lib
    robot = env.scene["robot"]
    q = robot.data.joint_pos[env_ids].clone()
    q[:, lib["arm_ids"]] = env._pg_q[env_ids]
    q[:, lib["waist_id"]] = env._pg_waist[env_ids]
    q[:, lib["hand_ids"]] = lib["hand_q"]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    tgt = robot.data.joint_pos_target[env_ids].clone()
    tgt[:, lib["arm_ids"]] = env._pg_q[env_ids]
    tgt[:, lib["waist_id"]] = env._pg_waist[env_ids]
    tgt[:, lib["hand_ids"]] = lib["hand_q"]
    robot.set_joint_position_target(tgt, env_ids=env_ids)
