# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Parametric "lathe" (revolved-profile) fruit utilities.

Mirrors the direct-USD-manipulation style already used in tree_utils.py
(create_stem_joints), but builds a fruit mesh instead of a joint.

A fruit's shape is described as a half-profile: radius r as a function of
normalized height h in [0, 1] along the revolution axis. Revolving that
profile 360 degrees gives the solid -- exactly what you'd do in Fusion 360
with a sketch + Revolve feature. Apple and pear use the SAME pipeline; only
the control points describing the silhouette differ.

Two ways to use this in your env:

1. CHEAP / RECOMMENDED for per-episode variability (used as an EventTerm,
   same pattern as reset_apple_pose in terminations_and_events.py):
   Build the mesh ONCE from a profile (e.g. at scene setup, or once via
   Fusion -> imported USD asset). On every reset, only change a small
   anisotropic scale (sx, sy, sz) and a random density, then recompute
   mass = density * volume(sx*sy*sz). No mesh rebuilding -> negligible cost,
   scales fine to many parallel envs.
   -> randomize_fruit_scale_and_mass()

2. OPTIONAL / more expensive: true shape variability by rebuilding the mesh
   itself every reset (e.g. perturbing control points). Fine for num_envs=1
   or a handful of envs; will not scale to hundreds of parallel envs.
   -> build_lathe_mesh() + replace_fruit_mesh()

The volume/mass math (FruitProfile, radius_profile, fruit_volume) has NO
dependency on Isaac Sim / pxr and can be unit-tested with plain python.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ─────────────────────────────────────────────
#  Profile definition
# ─────────────────────────────────────────────

@dataclass
class FruitProfile:
    """Half-profile of a solid of revolution.

    control_points: list of (height_fraction, radius_m), height_fraction in
    [0, 1] from the bottom to the top of the fruit, sorted ascending.
    The first and last radius should be 0 (or very small) so the surface
    closes cleanly at the poles, exactly like the top/bottom of a sphere.
    """
    name: str
    height: float                      # total height, metres
    control_points: Sequence[tuple]     # [(h_frac, r_m), ...]


# Illustrative starting profiles -- tune visually against your renders.
# An apple is flattened top/bottom (calyx + stem dimples) and widest
# slightly BELOW the equator, not a sphere.
APPLE_PROFILE = FruitProfile(
    name="apple",
    height=0.075,
    control_points=[
        (0.00, 0.000),   # bottom calyx dimple, closes to the axis
        (0.04, 0.028),
        (0.18, 0.038),
        (0.42, 0.041),   # widest point, slightly below the equator
        (0.65, 0.039),
        (0.88, 0.024),
        (0.96, 0.010),   # shoulder near the stem
        (1.00, 0.000),   # top, closes to the axis
    ],
)

# A pear is the SAME pipeline: just move the widest point into the lower
# half and add a longer, narrower neck toward the stem.
PEAR_PROFILE = FruitProfile(
    name="pear",
    height=0.095,
    control_points=[
        (0.00, 0.000),
        (0.10, 0.034),
        (0.30, 0.042),   # widest point is low, unlike the apple
        (0.55, 0.038),
        (0.75, 0.026),
        (0.90, 0.016),
        (0.97, 0.007),   # narrow neck
        (1.00, 0.000),
    ],
)

FRUIT_PRESETS = {"apple": APPLE_PROFILE, "pear": PEAR_PROFILE}


# ─────────────────────────────────────────────
#  Profile interpolation (Catmull-Rom-style cubic Hermite, numpy only)
# ─────────────────────────────────────────────

def _tangents(h: np.ndarray, r: np.ndarray) -> np.ndarray:
    n = len(h)
    m = np.zeros(n)
    m[0] = (r[1] - r[0]) / (h[1] - h[0])
    m[-1] = (r[-1] - r[-2]) / (h[-1] - h[-2])
    for i in range(1, n - 1):
        m[i] = (r[i + 1] - r[i - 1]) / (h[i + 1] - h[i - 1])
    return m


def _hermite_eval(h: np.ndarray, r: np.ndarray, m: np.ndarray,
                   h_query: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(h, h_query, side="right") - 1, 0, len(h) - 2)
    h0, h1 = h[idx], h[idx + 1]
    t = (h_query - h0) / (h1 - h0)
    t2, t3 = t * t, t ** 3
    b00 = 2 * t3 - 3 * t2 + 1
    b10 = t3 - 2 * t2 + t
    b01 = -2 * t3 + 3 * t2
    b11 = t3 - t2
    dh = h1 - h0
    return (b00 * r[idx] + b10 * dh * m[idx]
            + b01 * r[idx + 1] + b11 * dh * m[idx + 1])


def radius_profile(profile: FruitProfile, n_samples: int = 129):
    """Returns (z_m, r_m) arrays sampling the smooth silhouette."""
    h_pts = np.array([p[0] for p in profile.control_points], dtype=np.float64)
    r_pts = np.array([p[1] for p in profile.control_points], dtype=np.float64)
    m = _tangents(h_pts, r_pts)
    h_query = np.linspace(0.0, 1.0, n_samples)
    r_query = np.clip(_hermite_eval(h_pts, r_pts, m, h_query), 0.0, None)
    return h_query * profile.height, r_query


def fruit_volume(profile: FruitProfile, n_samples: int = 201) -> float:
    """Disk-method volume: V = pi * integral( r(z)^2 dz ), via Simpson's rule."""
    if n_samples % 2 == 0:
        n_samples += 1  # Simpson needs an even number of intervals
    z, r = radius_profile(profile, n_samples=n_samples)
    f = np.pi * r ** 2
    h = z[1] - z[0]
    s = f[0] + f[-1] + 4.0 * np.sum(f[1:-1:2]) + 2.0 * np.sum(f[2:-2:2])
    return s * h / 3.0


# ─────────────────────────────────────────────
#  Optional full mesh build (path 2 above)
# ─────────────────────────────────────────────

def build_lathe_mesh(profile: FruitProfile, n_radial: int = 28, n_height: int = 40):
    """Revolve the profile into a closed quad mesh.

    Returns (points[N,3], face_vertex_counts, face_vertex_indices) ready to
    hand to UsdGeom.Mesh. Poles (r=0 rings) produce degenerate quads, which
    is harmless for rendering/physics but can be cleaned up with explicit
    triangle fans later if you want a perfectly tidy mesh.
    """
    z, r = radius_profile(profile, n_samples=n_height)
    thetas = np.linspace(0.0, 2 * np.pi, n_radial, endpoint=False)
    points = np.array([
        (ri * np.cos(th), ri * np.sin(th), zi)
        for zi, ri in zip(z, r) for th in thetas
    ])

    face_counts, face_indices = [], []
    for j in range(n_height - 1):
        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            p00 = j * n_radial + i
            p01 = j * n_radial + i_next
            p10 = (j + 1) * n_radial + i
            p11 = (j + 1) * n_radial + i_next
            face_counts.append(4)
            face_indices.extend([p00, p01, p11, p10])
    return points, face_counts, face_indices


def replace_fruit_mesh(prim_path: str, profile: FruitProfile,
                        n_radial: int = 28, n_height: int = 40,
                        collision_approximation: str = "convexHull"):
    """Rebuild the mesh prim at prim_path from `profile` (expensive path).

    NOTE: not runnable/testable outside an Isaac Sim process. The exact
    PhysX/UsdPhysics schema attribute names below match the pattern used in
    tree_utils.py for this project's Isaac Sim version; double check against
    your installed Isaac Sim docs if anything fails to apply.
    """
    try:
        from pxr import UsdGeom, UsdPhysics, Gf
        import omni.usd
    except ImportError:
        print("[fruit_utils] pxr not available - mesh not replaced.")
        return None

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path):
        stage.RemovePrim(prim_path)

    points, face_counts, face_indices = build_lathe_mesh(profile, n_radial, n_height)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)

    prim = mesh.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.GetApproximationAttr().Set(collision_approximation)
    UsdPhysics.MassAPI.Apply(prim)
    return prim


# ─────────────────────────────────────────────
#  Recommended per-episode event (cheap path 1 above)
# ─────────────────────────────────────────────

def randomize_fruit_scale_and_mass(
    env,
    env_ids,
    object_cfg,
    nominal_volume_m3: float,
    scale_jitter: float = 0.08,
    density_range: tuple = (700.0, 850.0),
):
    """Per-episode size + mass variability WITHOUT rebuilding the mesh.

    Hook this into EventsCfg exactly like reset_apple_pose is hooked in
    terminations_and_events.py (mode="reset"). `nominal_volume_m3` should
    be `fruit_volume(APPLE_PROFILE)` (or your imported master mesh's actual
    volume) computed once up front.

    For pear, just swap nominal_volume_m3 / density_range / object_cfg.
    """
    import torch
    asset = env.scene[object_cfg.name]
    n = len(env_ids)
    device = env.device

    sx = 1.0 + (torch.rand(n, device=device) * 2 - 1) * scale_jitter
    sy = 1.0 + (torch.rand(n, device=device) * 2 - 1) * scale_jitter
    sz = 1.0 + (torch.rand(n, device=device) * 2 - 1) * scale_jitter

    volumes = nominal_volume_m3 * (sx * sy * sz)
    densities = torch.empty(n, device=device).uniform_(*density_range)
    masses = densities * volumes

    # Applying the result to USD: writing the scale op + UsdPhysics.MassAPI
    # directly on each env's prim, same low-level pattern as tree_utils.py.
    # This part needs to run inside Isaac Sim and hasn't been exercised
    # against a live stage here -- treat it as a starting point and verify
    # against your installed Isaac Lab version (some versions also expose a
    # tensor API like `asset.root_physx_view.set_masses(...)`, which would
    # be a cleaner alternative if available).
    try:
        from pxr import UsdGeom, UsdPhysics, Gf
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        env_ids_list = env_ids.tolist()
        for k, idx in enumerate(env_ids_list):
            prim_path = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{idx}")
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            xform = UsdGeom.Xformable(prim)
            scale_op = next(
                (op for op in xform.GetOrderedXformOps()
                 if op.GetOpType() == UsdGeom.XformOp.TypeScale), None
            )
            if scale_op is None:
                scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3f(sx[k].item(), sy[k].item(), sz[k].item()))
            UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(masses[k].item())
    except ImportError:
        print("[fruit_utils] pxr not available - scale/mass not applied.")


if __name__ == "__main__":
    # Quick sanity check you can run with plain python (no Isaac Sim needed).
    for name, profile in FRUIT_PRESETS.items():
        v = fruit_volume(profile)
        print(f"{name}: volume = {v*1e6:.2f} cm^3, "
              f"~mass @775 kg/m^3 = {v*775*1000:.1f} g")