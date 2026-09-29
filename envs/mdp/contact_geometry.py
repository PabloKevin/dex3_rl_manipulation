# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Which FACE of each hand link is touching the apple.

A PhysX ContactSensor reports force per rigid BODY, so on its own it cannot
tell a contact on a finger's pad from one on its side edge or its back --
the same gap reach_rewards._palmar_dorsal_force closes for the palm during
the reach phase. It matters more for the fingers in the grasp phase:
index and middle only flex about their own local z axis (URDF: both
right_hand_index_*_joint and right_hand_middle_*_joint have axis 0 0 1),
each in its own plane, at palm-local z = +28.5 mm and -28.5 mm. An apple
centred between them bulges toward the centreline, so a curling finger
reaches it with the INNER EDGE of its pad first (~20-30 deg off the pad
normal for a 35-45 mm radius fruit), and an apple pushed off to one side
touches the other finger on its SIDE face. The real Dex3-1 taxels sit on
the pad face only, so the two cases have to be told apart twice over:

  - privileged (rewards / safety): every face counts toward "is the fruit
    being hurt", but only pad-and-near-pad contact counts as gripping it.
  - realistic (observations / the grip detector): a pad reads a contact in
    proportion to how much of it lands on the pad -- see pad_face_weight,
    with a per-episode randomized footprint (domain_randomization.
    randomize_tactile_footprint) because the real edge response is unknown.

How the face is found: the per-link sensors track contact POINTS
(track_contact_points, enabled in the grasp env cfg). For a sphere the
surface normal at a contact point is exactly radial, so the direction the
link's surface faces there is (apple_centre - contact_point) -- exact,
independent of friction, which the force direction is not (friction tilts
the net force up to atan(mu) ~ 27 deg off the normal). Compared against
the link's own pad normal, rotated by FK, that gives cos(angle off the
pad): 1 = pad centre, ~0 = side face, < 0 = back.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .observations import _apple_contact_sensors

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Outward normal of each link's pad face, in that link's OWN frame --
# geometric, read off the meshes (envs/assets/cuRobo/meshes, same as the
# URDF's collision geometry), NOT averaged from contacts:
#   - index_* / middle_*: long axis +x, flexion about z, closing rotates +x
#     toward +y, so the pad (the face leading the curl) is +y. The two
#     fingers' links are byte-identical meshes, so their pads must match.
#     The asymmetric thickness (y in [-12.6, +14.7] mm on *_0) is the pad
#     bulge on +y.
#   - thumb_1 / thumb_2: long axis +y, flexion about z toward the LOWER
#     limit, which rotates +y toward +x -- pad +x (thumb_1's x extent is
#     [-12.6, +14.7] mm, the same pad bulge as index_0 turned 90 deg).
#   - palm: +y, apple_grasp_env_cfg.PALM_NORMAL_AXIS.
# right_hand_thumb_0_link (the opposition knuckle) carries no pad and is
# deliberately absent: it only ever counts toward safety (any-face force).
#
# The contact-AVERAGED normals this replaces (grasp_rewards' former
# _PAD_NORMAL_LOCAL) agreed with these to within ~8 deg for index and ~20
# deg for the thumb, but had middle_1 at (0.22, 0.40, 0.89) -- 64 deg off
# its pad, pointing along +z, i.e. the middle finger's side face toward the
# index. That was measured evidence of side contact, not a pad normal, and
# using it as one made the closure/enclosure terms credit the middle finger
# for pushing the apple sideways onto the index.
PAD_NORMAL_LOCAL: dict[str, tuple[float, float, float]] = {
    "right_hand_palm_link":     (0.0, 1.0, 0.0),
    "right_hand_thumb_1_link":  (1.0, 0.0, 0.0),
    "right_hand_thumb_2_link":  (1.0, 0.0, 0.0),
    "right_hand_index_0_link":  (0.0, 1.0, 0.0),
    "right_hand_index_1_link":  (0.0, 1.0, 0.0),
    "right_hand_middle_0_link": (0.0, 1.0, 0.0),
    "right_hand_middle_1_link": (0.0, 1.0, 0.0),
}

# Pad links per finger, for per-finger grip-quality sums.
FINGER_PAD_LINKS: dict[str, list[str]] = {
    "thumb": ["right_hand_thumb_1_link", "right_hand_thumb_2_link"],
    "index": ["right_hand_index_0_link", "right_hand_index_1_link"],
    "middle": ["right_hand_middle_0_link", "right_hand_middle_1_link"],
}


def apple_contact_faces(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    link_names: Sequence[str],
    object_name: str = "apple",
    robot_name: str = "robot",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apple-filtered force on each link, and where on the link it lands.

    Returns (force_w [N, L, 3], cos_pad [N, L]):
      force_w  -- apple-filtered contact force on the link, world frame, N.
      cos_pad  -- cosine of the angle between the link's pad normal and the
                  direction its surface faces at the contact: 1 on the pad,
                  ~0 on the side face, < 0 on the back. 0 where there is no
                  contact (always multiplied by a ~0 force there anyway).

    Contact points come from the per-link sensors' contact_pos_w (NaN when
    not touching). If a sensor has no contact points (track_contact_points
    off, e.g. the reach scene) or a point is missing while force is not,
    falls back to the force direction, -F/|F| -- right for the sign, but
    tilted by friction, see the module docstring.

    Every link in link_names must have an entry in PAD_NORMAL_LOCAL.
    """
    group = _apple_contact_sensors(env, sensor_cfg.name)
    robot: Articulation = env.scene[robot_name]
    apple: RigidObject = env.scene[object_name]

    key = tuple(link_names)
    cache = getattr(env, "_pad_face_cache", None)
    if cache is None:
        cache = env._pad_face_cache = {}
    if key not in cache:
        n = torch.tensor([PAD_NORMAL_LOCAL[l] for l in link_names], device=env.device)
        cache[key] = (
            [robot.data.body_names.index(l) for l in link_names],
            n / n.norm(dim=-1, keepdim=True),
        )
    body_ids, n_local = cache[key]

    force = torch.stack([group[l].data.force_matrix_w[:, 0, 0, :] for l in link_names], dim=1)  # [N, L, 3]
    f_mag = force.norm(dim=-1, keepdim=True)
    from_force = -force / f_mag.clamp(min=1e-6)

    pts = [group[l].data.contact_pos_w for l in link_names]
    if all(p is not None for p in pts):
        p = torch.stack([q[:, 0, 0, :] for q in pts], dim=1)                     # [N, L, 3]
        has_pt = torch.isfinite(p).all(dim=-1, keepdim=True)
        radial = apple.data.root_pos_w.unsqueeze(1) - torch.nan_to_num(p)
        radial = radial / radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        n_out = torch.where(has_pt, radial, from_force)
    else:
        n_out = from_force

    quats = robot.data.body_quat_w[:, body_ids, :]                               # [N, L, 4]
    n_pad_w = quat_apply(
        quats.reshape(-1, 4), n_local.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3)
    ).reshape(env.num_envs, len(link_names), 3)
    cos_pad = (n_out * n_pad_w).sum(dim=-1)
    # No contact at all: no face to speak of.
    cos_pad = torch.where(f_mag.squeeze(-1) > 1e-6, cos_pad, torch.zeros_like(cos_pad))
    return force, cos_pad


def pad_face_weight(
    cos_pad: torch.Tensor,
    full_deg: float | torch.Tensor,
    cut_deg: float | torch.Tensor,
) -> torch.Tensor:
    """1 for contacts within full_deg of the pad normal, falling linearly
    (in cosine) to 0 at cut_deg, 0 beyond. Tensors broadcast against
    cos_pad, so per-env / per-taxel angles work unchanged."""
    cos_full = torch.cos(torch.deg2rad(torch.as_tensor(full_deg, device=cos_pad.device, dtype=cos_pad.dtype)))
    cos_cut = torch.cos(torch.deg2rad(torch.as_tensor(cut_deg, device=cos_pad.device, dtype=cos_pad.dtype)))
    return ((cos_pad - cos_cut) / (cos_full - cos_cut).clamp(min=1e-6)).clamp(0.0, 1.0)


def angle_deg(cos_pad: torch.Tensor) -> torch.Tensor:
    """Diagnostics helper: angle off the pad normal, degrees."""
    return torch.rad2deg(torch.acos(cos_pad.clamp(-1.0, 1.0)))


# Defaults for the realistic pad model when the footprint randomizer hasn't
# run (play configs, probes). Chosen so the tripod's expected inner-edge
# contact (asin(15.5 mm / R) ~ 20-26 deg for R = 35-45 mm: the pad's inner
# edge sits 15.5 mm off the palm centreline) reads fully, while a side-face
# contact (~60-90 deg) reads nothing.
TACTILE_FULL_DEG_DEFAULT = 30.0
TACTILE_CUT_DEG_DEFAULT = 60.0

# Privileged "is this a gripping contact" band per finger, for rewards:
# (full_deg, cut_deg) as in pad_face_weight.
#   - index/middle: generous on purpose. They only flex in their own
#     planes, so on a centred apple their pads' INNER EDGES are the contact
#     (measured 22-37 deg on scripted tripods) and must count fully; a side
#     or back contact must not.
#   - thumb: strict. It is the one digit that can meet the apple squarely
#     (scripted tripods: 4-9 deg off its pad). With the index/middle band a
#     trained policy found it cheaper to swing thumb_0 to its limit and
#     press the thumb's SIDE into the apple (41 deg, 10 mm off the pad
#     centreline) than to flex the thumb from its open pose -- and was paid
#     in full for it.
GRIP_BAND_DEG: dict[str, tuple[float, float]] = {
    "thumb": (25.0, 45.0),
    "index": (45.0, 75.0),
    "middle": (45.0, 75.0),
}

__all__ = [
    "PAD_NORMAL_LOCAL",
    "FINGER_PAD_LINKS",
    "apple_contact_faces",
    "pad_face_weight",
    "angle_deg",
    "TACTILE_FULL_DEG_DEFAULT",
    "TACTILE_CUT_DEG_DEFAULT",
    "GRIP_BAND_DEG",
]
