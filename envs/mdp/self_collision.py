# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Full-body self-collision check for the right arm + Dex3 hand.

Sphere model of the robot: every body link except the right arm comes from
the project's cuRobo spherization (envs/assets/cuRobo/
g1_29dof_rev_1_0_spherized.yml: pelvis, legs, torso -- whose top sphere also
covers the head -- and the parked left arm). The right arm uses the same
file's upper-arm / forearm / wrist spheres, plus Dex3 hand spheres sized
from the hand's collision meshes (that file models the stock G1 hand; its
big hand sphere on right_wrist_yaw_link is dropped).

PhysX runs with self-collisions disabled, so nothing stops the arm passing
through the body. This is used both by the training penalty
(penalty_self_collision) and by scripts/probe_side_reach.py to decide which
apple placements are reachable WITHOUT any self-collision -- the same
definition in both places.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import yaml
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

SPHERE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "cuRobo", "g1_29dof_rev_1_0_spherized.yml",
)

#: Moving set: right-arm links taken from SPHERE_FILE.
ARM_LINKS = [
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
#: The shoulder itself sits against the torso in every pose; neither a mover
#: nor an obstacle.
IGNORED_LINKS = ["right_shoulder_pitch_link", "right_shoulder_roll_link"]
#: SPHERE_FILE's stock-hand sphere on the wrist (centre x ~0.12) is replaced
#: by HAND_SPHERES.
WRIST_YAW_MAX_X = 0.08
#: Pairs never checked. The upper arm hangs against the torso: under the
#: sphere model it already overlaps the torso spheres in the relaxed default
#: pose, so it would flag every arm-at-the-side posture. Swinging it INTO the
#: chest needs shoulder_roll > 0, which posture_box fences (< 0.35 rad), and
#: the elbow/forearm spheres are still checked against the torso.
IGNORED_PAIRS = {("right_shoulder_yaw_link", "torso_link")}

#: Required clearance between spheres. The spheres are already slightly
#: inflated; at the default (relaxed) arm pose the forearm sits 1.8 cm from
#: the torso, which must count as clear.
DEFAULT_MARGIN = 0.015

#: Dex3 right hand, (link, centre in link frame, radius), from the collision
#: mesh extents (meshes/right_hand_*.STL). Slightly conservative.
HAND_SPHERES = [
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
]


def _build(env: "ManagerBasedRLEnv") -> dict:
    robot = env.scene["robot"]
    names = robot.data.body_names
    with open(SPHERE_FILE) as f:
        spheres = yaml.safe_load(f)["collision_spheres"]

    arm, body = [], []
    for link, items in spheres.items():
        if link in IGNORED_LINKS or link not in names:
            continue
        for it in items:
            c, r = tuple(it["center"]), float(it["radius"])
            if link in ARM_LINKS:
                if link == "right_wrist_yaw_link" and c[0] > WRIST_YAW_MAX_X:
                    continue
                arm.append((link, c, r))
            elif not link.startswith("right_"):
                body.append((link, c, r))
            elif "hip" in link or "knee" in link or "ankle" in link:
                body.append((link, c, r))
    arm += [h for h in HAND_SPHERES if h[0] in names]

    def pack(lst):
        return {
            "ids": torch.tensor([names.index(l) for l, _, _ in lst], device=env.device),
            "c": torch.tensor([c for _, c, _ in lst], device=env.device, dtype=torch.float32),
            "r": torch.tensor([r for _, _, r in lst], device=env.device, dtype=torch.float32),
        }

    mask = torch.tensor([[(la, lb) not in IGNORED_PAIRS for lb, _, _ in body] for la, _, _ in arm],
                        device=env.device)
    return {"arm": pack(arm), "body": pack(body), "mask": mask}


def _world_spheres(env: "ManagerBasedRLEnv", s: dict) -> torch.Tensor:
    robot = env.scene["robot"]
    pos = robot.data.body_pos_w[:, s["ids"]]                       # [N, S, 3]
    quat = robot.data.body_quat_w[:, s["ids"]]                     # [N, S, 4]
    c = s["c"].unsqueeze(0).expand(pos.shape[0], -1, -1)
    return pos + quat_apply(quat.reshape(-1, 4), c.reshape(-1, 3)).reshape(pos.shape)


def self_collision_depth(env: "ManagerBasedRLEnv", margin: float = DEFAULT_MARGIN) -> torch.Tensor:
    """Per right-arm/hand sphere: deepest intrusion (m, >= 0) into any body
    sphere, counting `margin` of required clearance.

    Returns: [num_envs, num_arm_spheres]
    """
    if not hasattr(env, "_self_collision_spheres"):
        env._self_collision_spheres = _build(env)
    sets = env._self_collision_spheres
    a = _world_spheres(env, sets["arm"])                           # [N, A, 3]
    b = _world_spheres(env, sets["body"])                          # [N, B, 3]
    dist = torch.cdist(a, b)                                       # [N, A, B]
    need = sets["arm"]["r"].view(1, -1, 1) + sets["body"]["r"].view(1, 1, -1) + margin
    depth = (need - dist).clamp(min=0.0) * sets["mask"].unsqueeze(0)
    return depth.amax(dim=-1)


def penalty_self_collision(
    env: "ManagerBasedRLEnv",
    margin: float = DEFAULT_MARGIN,
    ref: float = 0.05,
) -> torch.Tensor:
    """Quadratic penalty on the right arm/hand intruding into the rest of the
    body (self_collision_depth). Use a NEGATIVE weight.

        penalty = sum_s (depth_s / ref)^2

    Zero whenever the arm keeps `margin` of clearance, so it never prefers
    one clear posture over another; grows smoothly with depth, so it teaches
    "move the elbow/hand out" rather than forbidding joint values.

    Returns: [num_envs]
    """
    return ((self_collision_depth(env, margin) / ref) ** 2).sum(dim=-1)


def _capsule_depth(env: "ManagerBasedRLEnv", spheres: dict, margin: float) -> torch.Tensor:
    """Per sphere of `spheres`: deepest intrusion (m, >= 0) into the stem,
    branch or trunk capsules (env._tree_segments / _tree_radii)."""
    p = _world_spheres(env, spheres)                               # [N, A, 3]
    a = env._tree_segments[:, :, 0].unsqueeze(1)                   # [N, 1, T, 3]
    b = env._tree_segments[:, :, 1].unsqueeze(1)
    ab = b - a
    ap = p.unsqueeze(2) - a                                        # [N, A, T, 3]
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
    dist = (ap - t.unsqueeze(-1) * ab).norm(dim=-1)                # [N, A, T]
    need = spheres["r"].view(1, -1, 1) + env._tree_radii.view(1, 1, -1) + margin
    return (need - dist).clamp(min=0.0).amax(dim=-1)


def tree_contact_depth(env: "ManagerBasedRLEnv", margin: float = 0.01) -> torch.Tensor:
    """Per right-arm/hand sphere: deepest intrusion (m, >= 0) into the stem,
    branch or trunk, modelled as capsules (env._tree_segments / _tree_radii,
    set every reset by domain_randomization.randomize_tree_and_apple).

    The tree prims are visual only (no colliders), so without this the hand
    can pass straight through the stem or sit between the apple and the
    trunk. Privileged geometry, used in the reward only.

    Returns: [num_envs, num_arm_spheres] (zeros if the tree is not set up)
    """
    if not hasattr(env, "_self_collision_spheres"):
        env._self_collision_spheres = _build(env)
    arm = env._self_collision_spheres["arm"]
    if not hasattr(env, "_tree_segments"):
        return torch.zeros(env.num_envs, len(arm["r"]), device=env.device)
    return _capsule_depth(env, arm, margin)


#: Body links that turn with the waist yaw: the torso (its top sphere covers
#: the head) and the parked left arm. Pelvis and legs never move.
def _turns_with_waist(link: str) -> bool:
    return link == "torso_link" or (link.startswith("left_") and not any(k in link for k in ("hip", "knee", "ankle")))


def body_tree_contact_depth(env: "ManagerBasedRLEnv", margin: float = 0.01) -> torch.Tensor:
    """tree_contact_depth for the body spheres that turn with the waist
    (torso, left arm). Only matters when the waist yaw is driven
    (AppleReachTreeWaist-v1): turning the torso swings the parked left arm
    through the wall.

    Returns: [num_envs, num_turning_spheres] (zeros if the tree is not set up)
    """
    if not hasattr(env, "_self_collision_spheres"):
        env._self_collision_spheres = _build(env)
    sets = env._self_collision_spheres
    if "turning" not in sets:
        names = env.scene["robot"].data.body_names
        body = sets["body"]
        keep = torch.tensor([_turns_with_waist(names[i]) for i in body["ids"].tolist()], device=env.device)
        sets["turning"] = {k: v[keep] for k, v in body.items()}
    turning = sets["turning"]
    if not hasattr(env, "_tree_segments"):
        return torch.zeros(env.num_envs, len(turning["r"]), device=env.device)
    return _capsule_depth(env, turning, margin)


def penalty_body_tree_contact(
    env: "ManagerBasedRLEnv",
    margin: float = 0.01,
    ref: float = 0.02,
) -> torch.Tensor:
    """penalty_tree_contact for the torso / left arm (body_tree_contact_depth).
    Use a NEGATIVE weight.

    Returns: [num_envs]
    """
    return ((body_tree_contact_depth(env, margin) / ref) ** 2).sum(dim=-1)


def penalty_tree_contact(
    env: "ManagerBasedRLEnv",
    margin: float = 0.01,
    ref: float = 0.02,
) -> torch.Tensor:
    """Quadratic penalty on the arm/hand entering the stem, branch or trunk
    (tree_contact_depth). Use a NEGATIVE weight. `ref` small because the
    stem is thin: 2 cm into it already costs 1.0.

    Returns: [num_envs]
    """
    return ((tree_contact_depth(env, margin) / ref) ** 2).sum(dim=-1)
