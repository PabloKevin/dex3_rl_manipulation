# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Harvest basket for the test bench: a slatted cylinder on a stand, on the
robot's right, slightly behind the shoulder line -- outside the zone where
trees and apples are placed (apples are always >= 15 cm in front of the
shoulder, trunks further forward).

Static colliders (never moved), so the dropped apple really lands and stays
in. Slatted so the inside is visible from any camera: vertical slats with
gaps narrower than any apple (randomized radius 32-48 mm), a solid floor and
a top ring.

basket_intrusion() is the matching analytic obstacle for collision checks
(arm / hand / held-apple spheres), treating stand + basket as SOLID up to the
rim: the hand and arm never go below the rim, only the released apple does.
All positions are env-local (robot pelvis at the origin, facing +x).
"""

from __future__ import annotations

import math

import torch

CENTER_XY = (-0.05, -0.50)        # basket axis, env-local
STAND_SIZE = (0.36, 0.36, 0.58)   # box under the basket (x, y, z)
INNER_RADIUS = 0.14
DEPTH = 0.22                      # floor to rim
FLOOR_THICKNESS = 0.01
SLAT_COUNT = 14                   # 2*pi*0.14/14 = 6.3 cm pitch -> ~4.3 cm gaps < any apple
SLAT_WIDTH = 0.02
SLAT_THICKNESS = 0.01
RING_HEIGHT = 0.015

FLOOR_Z = STAND_SIZE[2] + FLOOR_THICKNESS      # top of the floor
RIM_Z = FLOOR_Z + DEPTH                        # top of the slats
OUTER_RADIUS = INNER_RADIUS + SLAT_THICKNESS
STAND_RADIUS = 0.5 * math.hypot(STAND_SIZE[0], STAND_SIZE[1])


def add_to_scene(scene_cfg) -> None:
    """Add the stand, floor, slats and top ring to an InteractiveSceneCfg as
    static colliders. Call before the env is created."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.45, 0.25), roughness=0.8)
    grey = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.27), roughness=0.9)
    coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0)
    cx, cy = CENTER_XY

    def yaw_quat(a):
        return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))

    setattr(scene_cfg, "basket_stand", AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketStand",
        spawn=sim_utils.CuboidCfg(size=STAND_SIZE, collision_props=coll, visual_material=grey),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, STAND_SIZE[2] / 2))))
    setattr(scene_cfg, "basket_floor", AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketFloor",
        spawn=sim_utils.CylinderCfg(radius=OUTER_RADIUS, height=FLOOR_THICKNESS, axis="Z",
                                    collision_props=coll, visual_material=wood),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, STAND_SIZE[2] + FLOOR_THICKNESS / 2))))
    r_mid = INNER_RADIUS + SLAT_THICKNESS / 2
    seg = 2 * r_mid * math.sin(math.pi / SLAT_COUNT) + 0.004       # ring segment length (slight overlap)
    for k in range(SLAT_COUNT):
        a = 2 * math.pi * k / SLAT_COUNT
        setattr(scene_cfg, f"basket_slat_{k:02d}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/BasketSlat{k:02d}",
            spawn=sim_utils.CuboidCfg(size=(SLAT_THICKNESS, SLAT_WIDTH, DEPTH), collision_props=coll, visual_material=wood),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(cx + r_mid * math.cos(a), cy + r_mid * math.sin(a), FLOOR_Z + DEPTH / 2), rot=yaw_quat(a))))
        b = a + math.pi / SLAT_COUNT                                  # ring segment between slats k and k+1
        setattr(scene_cfg, f"basket_ring_{k:02d}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/BasketRing{k:02d}",
            spawn=sim_utils.CuboidCfg(size=(SLAT_THICKNESS, seg, RING_HEIGHT), collision_props=coll, visual_material=wood),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(cx + r_mid * math.cos(b) * math.cos(math.pi / SLAT_COUNT), cy + r_mid * math.sin(b) * math.cos(math.pi / SLAT_COUNT),
                     RIM_Z - RING_HEIGHT / 2), rot=yaw_quat(b))))


def _solid_cylinder_signed_dist(p: torch.Tensor, radius: float, top: float) -> torch.Tensor:
    """Signed distance from points p [..., 3] (env-local) to a solid vertical
    cylinder on the ground (axis CENTER_XY, given radius, from z=0 to top)."""
    rad = torch.hypot(p[..., 0] - CENTER_XY[0], p[..., 1] - CENTER_XY[1])
    dr, dz = rad - radius, p[..., 2] - top
    outside = torch.hypot(dr.clamp(min=0.0), dz.clamp(min=0.0))
    inside = torch.maximum(dr, dz).clamp(max=0.0)
    return outside + inside


def basket_intrusion(centres: torch.Tensor, radii: torch.Tensor, margin: float = 0.015) -> torch.Tensor:
    """Depth (m, >= 0) by which spheres (env-local centres [..., S, 3], radii
    [S] or broadcastable) come within `margin` of the stand + basket (solid up
    to the rim).

    Returns: [..., S]
    """
    d = torch.minimum(_solid_cylinder_signed_dist(centres, OUTER_RADIUS, RIM_Z),
                      _solid_cylinder_signed_dist(centres, STAND_RADIUS, STAND_SIZE[2]))
    return (radii + margin - d).clamp(min=0.0)


def apple_in_basket(apple_local: torch.Tensor, apple_radius: torch.Tensor) -> torch.Tensor:
    """[N] bool: apple centre inside the basket (radially within the inner
    radius, between the floor and the rim)."""
    rad = torch.hypot(apple_local[:, 0] - CENTER_XY[0], apple_local[:, 1] - CENTER_XY[1])
    return (rad < INNER_RADIUS) & (apple_local[:, 2] > FLOOR_Z) & (apple_local[:, 2] < RIM_Z + 0.5 * apple_radius)
