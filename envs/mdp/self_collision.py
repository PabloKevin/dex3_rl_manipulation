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
