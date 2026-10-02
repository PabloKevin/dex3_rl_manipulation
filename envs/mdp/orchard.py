# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""The NOVA commercial orchard (envs/EnvCommercialAppleTrees) as an Isaac Lab
scene, for testing the frozen policies in a real tree instead of the
procedural one.

Source: scripts/export_orchard.py turns the orchard's compiled MuJoCo scene
into envs/EnvCommercialAppleTrees/out/isaac/orchard_export.npz (3 rows x 5
tall-spindle trees, 180 hanging apples, trellis posts/wires, terrain, bins).
build_usd() turns that into orchard.usd: one KINEMATIC rigid body holding
  - every visual mesh (trees and leaves, apples, posts, wires, ground...);
  - primitive colliders fitted to MuJoCo's convex collision pieces: capsules
    for round pieces (trunks, scaffolds, posts, wires), oriented boxes for
    flat ones (bins, crates). Apples have no collider: the task apple is the
    env's own dynamic sphere, the others are visual only (the hand passes
    through them). Branches are rigid here (springs in MuJoCo); leaves have
    no collider in either.
Kinematic so it can be teleported every episode (static colliders can't be
moved through the tensor API): the robot stays at the env origin facing +x,
as every frame convention in this project assumes (the frozen reach
policy's observation is in world axes), and the ORCHARD is turned (0 or 180
deg) and shifted around it.

Per episode (place_orchard_and_apple, mode="reset", in place of
domain_randomization.randomize_tree_and_apple) one hanging apple is picked
and the orchard placed so that apple's grasp target lands where the frozen
reach policy was trained (same shoulder-relative distance / azimuth /
height ranges and safe-workspace map as the grasp task), the robot standing
on the terrain, in the alley, facing that apple's side of the row. A
placement is rejected when the robot's body, the arm's way to the apple,
the apple itself or the basket would overlap wood. It leaves the same
bookkeeping as randomize_tree_and_apple: apple pose, env._stem_anchor_pos,
env._stem_dir (the REAL pedicel axis: straight up, the fruit hangs),
env._tree_segments / _tree_radii (the stem, then every wood capsule within
reach), plus env._orchard_info (which apple, for reports).

Placement screening of the robot against the wood is on only with solid
wood (see place_orchard_and_apple's screen_robot).

ONE env only: the orchard is 12 m x 10 m and _tree_radii is shared across
envs (self_collision.tree_contact_depth).
"""

from __future__ import annotations

import math
import os
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ORCHARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "EnvCommercialAppleTrees", "out", "isaac")
EXPORT_PATH = os.path.join(ORCHARD_DIR, "orchard_export.npz")
USD_PATH = os.path.join(ORCHARD_DIR, "orchard.usd")

KIND_NAMES = ("static", "branch", "apple", "object", "terrain")    # = export_orchard.KIND
APPLE_KIND = 2
#: A collision piece is a capsule when its cross-section is round (second
#: box half-extent within this factor of the smallest), else a box.
ROUND_RATIO = 1.6

_cache: dict = {}


def load_export(path: str = EXPORT_PATH) -> dict:
    if path not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing: run scripts/export_orchard.py (needs MuJoCo >= 3.3.5, see its docstring)")
        z = np.load(path)
        d = {k: z[k] for k in z.files}
        d.update(_colliders(d))
        _cache[path] = d
    return _cache[path]


def _colliders(d: dict) -> dict:
    """Split the non-apple collision pieces into capsules and boxes."""
    keep = d["col_kind"] != APPLE_KIND
    h = d["col_box_h"]
    hs = np.sort(h, axis=1)[:, ::-1]
    round_ = hs[:, 1] <= ROUND_RATIO * np.maximum(hs[:, 2], 1e-4)
    cap, box = keep & round_, keep & ~round_
    R = d["col_box_R"][box].copy()
    flip = np.linalg.det(R) < 0
    R[flip, 2] *= -1.0                                  # right-handed box frames
    return {
        "cap_a": d["col_cap_a"][cap], "cap_b": d["col_cap_b"][cap], "cap_r": d["col_cap_r"][cap],
        "box_c": d["col_box_c"][box], "box_R": R, "box_h": h[box],
    }


# ─────────────────────────────────────────────
#  USD
# ─────────────────────────────────────────────

def _quat_wxyz_from_matrix(M: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix (columns = rotated axes) -> (w, x, y, z)."""
    t = np.trace(M)
    if t > 0:
        s = 2.0 * math.sqrt(1.0 + t)
        return (0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s)
    i = int(np.argmax(np.diag(M)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(max(1.0 + M[i, i] - M[j, j] - M[k, k], 1e-12))
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (M[k, j] - M[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (M[j, i] + M[i, j]) / s
    q[1 + k] = (M[k, i] + M[i, k]) / s
    return tuple(q)


def _z_to(v: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) turning +z onto unit vector v."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, v))
    if c < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    ax = np.cross(z, v)
    q = np.array([1.0 + c, *ax])
    q /= np.linalg.norm(q)
    return tuple(float(x) for x in q)


def apple_prim_name(apple: int, j: int) -> str:
    return f"apple_{apple:03d}_{j}"


def build_usd(export_path: str = EXPORT_PATH, usd_path: str = USD_PATH, force: bool = False) -> str:
    """Write orchard.usd from the export (only when missing or older than the
    export). Needs pxr, i.e. a running Isaac Sim app."""
    if (not force and os.path.exists(usd_path)
            and os.path.getmtime(usd_path) >= os.path.getmtime(export_path)
            and os.path.getmtime(usd_path) >= os.path.getmtime(__file__)):
        return usd_path
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    d = load_export(export_path)
    print(f"[orchard] building {usd_path} from {export_path} ...", flush=True)
    stage = Usd.Stage.CreateNew(usd_path + ".tmp.usd")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Orchard")
    stage.SetDefaultPrim(root.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(1000.0)

    # ── materials, one per colour ──
    UsdGeom.Scope.Define(stage, "/Orchard/Looks")
    mats = {}

    def material(rgba):
        key = tuple(np.round(rgba, 3))
        if key not in mats:
            path = f"/Orchard/Looks/m{len(mats):03d}"
            mat = UsdShade.Material.Define(stage, path)
            sh = UsdShade.Shader.Define(stage, path + "/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, key[:3])))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(key[3]))
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            mats[key] = mat
        return mats[key]

    # ── visual meshes ──
    for kind in KIND_NAMES:
        UsdGeom.Scope.Define(stage, f"/Orchard/Visual/{kind}")
    per_apple: dict[int, int] = {}
    for i in range(len(d["vis_vadr"])):
        va, vn = d["vis_vadr"][i], d["vis_vnum"][i]
        fa, fn = d["vis_fadr"][i], d["vis_fnum"][i]
        kind = KIND_NAMES[int(d["vis_kind"][i])]
        a = int(d["vis_apple"][i])
        if kind == "apple" and a >= 0:
            j = per_apple.get(a, 0); per_apple[a] = j + 1
            nm = apple_prim_name(a, j)
        else:
            nm = f"m{i:04d}"
        mesh = UsdGeom.Mesh.Define(stage, f"/Orchard/Visual/{kind}/{nm}")
        pts = d["vis_verts"][va:va + vn]
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(fn, 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(d["vis_faces"][fa:fa + fn].reshape(-1)))
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        rgba = d["vis_rgba"][i]
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, rgba[:3]))]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material(rgba))

    # ── colliders (invisible primitives) ──
    UsdGeom.Scope.Define(stage, "/Orchard/Collision")

    def collider(prim):
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        prim.CreatePurposeAttr(UsdGeom.Tokens.guide)

    for k, (a, b, r) in enumerate(zip(d["cap_a"], d["cap_b"], d["cap_r"])):
        a, b = a.astype(np.float64), b.astype(np.float64)
        L = float(np.linalg.norm(b - a))
        cap = UsdGeom.Capsule.Define(stage, f"/Orchard/Collision/cap{k:04d}")
        cap.CreateAxisAttr(UsdGeom.Tokens.z)
        cap.CreateRadiusAttr(float(r))
        cap.CreateHeightAttr(L)
        cap.AddTranslateOp().Set(Gf.Vec3d(*((a + b) / 2)))
        q = _z_to((b - a) / L) if L > 1e-6 else (1.0, 0.0, 0.0, 0.0)
        cap.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
        collider(cap)
    for k, (c, R, h) in enumerate(zip(d["box_c"], d["box_R"], d["box_h"])):
        box = UsdGeom.Cube.Define(stage, f"/Orchard/Collision/box{k:04d}")
        box.CreateSizeAttr(1.0)
        box.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        q = _quat_wxyz_from_matrix(R.T.astype(np.float64))
        box.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))
        box.AddScaleOp().Set(Gf.Vec3f(*map(float, 2.0 * np.maximum(h, 0.002))))
        collider(box)

    stage.GetRootLayer().Save()
    del stage
    os.replace(usd_path + ".tmp.usd", usd_path)
    print(f"[orchard] {len(d['vis_vadr'])} visual meshes, {len(d['cap_r'])} capsule + {len(d['box_h'])} box "
          f"colliders, {len(mats)} materials -> {usd_path}", flush=True)
    return usd_path


# ─────────────────────────────────────────────
#  Geometry (numpy, env-local frame)
# ─────────────────────────────────────────────

def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _dist_capsules(p: np.ndarray, a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Surface distance [P, C] from points p [P, 3] to capsules."""
    ab = b - a
    t = np.clip(np.einsum("pcj,cj->pc", p[:, None] - a[None], ab) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    q = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None] - q, axis=-1) - r[None]


def _dist_boxes(p: np.ndarray, c: np.ndarray, R: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Distance [P, B] from points to oriented boxes (0 inside)."""
    loc = np.einsum("bij,pbj->pbi", R, p[:, None] - c[None])
    return np.linalg.norm(np.maximum(np.abs(loc) - h[None], 0.0), axis=-1)


class _Obstacles:
    """The orchard's colliders in env-local coordinates for one placement."""

    def __init__(self, d: dict, rot: np.ndarray, t: np.ndarray, near: np.ndarray, radius: float):
        tf = lambda x: x @ rot.T + t
        a, b = tf(d["cap_a"]), tf(d["cap_b"])
        mid = (a + b) / 2
        half = np.linalg.norm(b - a, axis=1) / 2 + d["cap_r"]
        sel = np.linalg.norm(mid - near, axis=1) < radius + half
        self.a, self.b, self.r = a[sel], b[sel], d["cap_r"][sel]
        c = tf(d["box_c"])
        sel = np.linalg.norm(c - near, axis=1) < radius + np.linalg.norm(d["box_h"], axis=1)
        self.c, self.R, self.h = c[sel], d["box_R"][sel] @ rot.T, d["box_h"][sel]

    def clearance(self, p: np.ndarray, min_radius: float = 0.0) -> np.ndarray:
        """[P] distance from each point to the nearest obstacle surface,
        ignoring capsules thinner than min_radius."""
        out = np.full(len(p), np.inf)
        thick = self.r >= min_radius
        if thick.any():
            out = np.minimum(out, _dist_capsules(p, self.a[thick], self.b[thick], self.r[thick]).min(1))
        if len(self.h):
            out = np.minimum(out, _dist_boxes(p, self.c, self.R, self.h).min(1))
        return out


def _segment_points(a, b, step=0.02):
    n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / step) + 1)
    return np.linspace(a, b, n)


# ─────────────────────────────────────────────
#  Events
# ─────────────────────────────────────────────

def set_apple_radius(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    spawn_radius: float,
    radius_range: tuple[float, float],
) -> None:
    """prestartup: apple radius uniform in radius_range (the orchard's fruit
    are 32-42 mm, inside the 32-48 mm the grasp policy trained on), by USD
    scale like domain_randomization.randomize_apple_scale."""
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    env._apple_radius = torch.full((env.num_envs,), spawn_radius, device=env.device)
    for idx in range(env.num_envs):
        r = random.uniform(*radius_range)
        prim = stage.GetPrimAtPath(f"/World/envs/env_{idx}/Apple")
        if not prim.IsValid():
            continue
        xf = UsdGeom.Xformable(prim)
        op = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeScale), None) \
            or xf.AddScaleOp()
        s = r / spawn_radius
        op.Set(Gf.Vec3f(s, s, s))
        env._apple_radius[idx] = r


def place_orchard_and_apple(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    orchard_cfg: SceneEntityCfg,
    reach_min: float,
    reach_max: float,
    reach_azimuth_center_deg: float,
    reach_azimuth_span_deg: float,
    target_dz_range: tuple[float, float],
    approach_yaw_spread_deg: float,
    standoff: float,
    reach_min_forward: float = 0.0,
    reach_min_horizontal: float = 0.0,
    workspace_map: str | None = None,
    stem_radius: float = 0.003,
    screen_robot: bool = True,
    body_margin: float = 0.03,
    body_ignore_radius: float = 0.0,
    arm_corridor_radius: float = 0.06,
    hand_pocket_radius: float = 0.045,
    apple_margin: float = 0.001,
    basket_radius: float = 0.32,
    basket_top: float = 1.4,
    tree_reach: float = 1.0,
    apples: list[str] | None = None,
    tries_per_apple: int = 40,
) -> None:
    """Per-episode: pick a hanging apple, place the orchard around the robot
    (see the module docstring). `apples` restricts the pick to these names
    (e.g. "gala_r1_t3_07"); approach_yaw_spread_deg is the grasp task's
    branch-yaw spread (the palm comes at the fruit up to that far off +x).
    screen_robot: reject placements where the robot's body or the arm's way
    to the fruit would overlap wood. Only meaningful with solid wood: in a
    spindle canopy the reach envelope of the frozen reach policy (0.25-0.33 m
    from the shoulder) puts the hips inside the low scaffolds for all but
    ~5% of the fruit. With visual-only wood the apple fit and the basket are
    still checked, and the intrusion is measured instead.
    body_ignore_radius: wood thinner than this (flexible shoots, which a
    body standing at the canopy edge brushes aside) is ignored by the body
    check -- never by the arm / hand / apple checks."""
    import omni.usd
    from pxr import UsdGeom

    d = load_export()
    stage = omni.usd.get_context().get_stage()
    apple_obj = env.scene[object_cfg.name]
    orchard = env.scene[orchard_cfg.name]
    robot = env.scene["robot"]
    if env.num_envs != 1:
        raise ValueError("the orchard scene supports exactly one env (num_envs=1)")
    dev = env.device

    # ── one-time: robot geometry, workspace map ──
    if not hasattr(env, "_orchard_robot"):
        from . import self_collision as scm
        origin = env.scene.env_origins[0]
        sets = scm._build(env)
        body = (scm._world_spheres(env, sets["body"])[0] - origin).cpu().numpy()
        arm = (scm._world_spheres(env, sets["arm"])[0] - origin).cpu().numpy()
        shoulder = (robot.data.body_pos_w[0, robot.data.body_names.index("right_shoulder_pitch_link")] - origin)
        env._shoulder_local_xyz = shoulder.clone()
        env._orchard_robot = {
            "p": np.concatenate([body, arm]),
            "r": np.concatenate([sets["body"]["r"].cpu().numpy(), sets["arm"]["r"].cpu().numpy()]),
            "shoulder": shoulder.cpu().numpy().astype(np.float64),
            "has_basket": "basket_stand" in env.scene.keys(),
        }
        if workspace_map is not None:
            m = np.load(workspace_map)
            env._workspace_map = (m["x_edges"], m["y_edges"], m["z_edges"], m["feasible"])
        env._orchard_hidden = []
        env._orchard_info = [None]
    rb = env._orchard_robot
    sh = rb["shoulder"]

    def feasible(x, y, z):
        if workspace_map is None:
            return True
        xe, ye, ze, feas = env._workspace_map
        i = int(math.floor((x - xe[0]) / (xe[1] - xe[0])))
        j = int(math.floor((y - ye[0]) / (ye[1] - ye[0])))
        k = int(math.floor((z - ze[0]) / (ze[1] - ze[0])))
        n = feas.shape
        return 0 <= i < n[0] and 0 <= j < n[1] and 0 <= k < n[2] and bool(feas[i, j, k])

    H, h0, res = d["terrain_h"], d["terrain_origin"], float(d["terrain_res"])

    def terrain(xy):
        f = (np.asarray(xy) - h0) / res
        i = np.clip(f.astype(int), 0, np.array(H.shape) - 2)
        u, v = np.clip(f - i, 0, 1)
        return float((1 - u) * (1 - v) * H[i[0], i[1]] + u * (1 - v) * H[i[0] + 1, i[1]]
                     + (1 - u) * v * H[i[0], i[1] + 1] + u * v * H[i[0] + 1, i[1] + 1])

    r_env = float(env._apple_radius[0]) if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) else 0.04
    centers, ped = d["apple_center"].astype(np.float64), d["apple_pedicel"].astype(np.float64)
    trunk = {(int(r), int(t)): xy for r, t, xy in zip(d["tree_row"], d["tree_idx"], d["tree_trunk_xy"])}
    # the env's sphere must fit where the orchard's fruit hangs
    ok = d["apple_radius"] + d["apple_wood_gap"] >= r_env + apple_margin
    if apples:
        ok &= np.isin(d["apple_name"], apples)
    cand = list(np.nonzero(ok)[0])
    random.shuffle(cand)
    az_lo = reach_azimuth_center_deg - reach_azimuth_span_deg / 2.0
    az_hi = reach_azimuth_center_deg + reach_azimuth_span_deg / 2.0
    rejects = {"height": 0, "body": 0, "arm": 0, "apple": 0, "basket": 0}

    chosen = None
    for k in cand:
        c0 = centers[k]
        dx = c0[0] - trunk[(int(d["apple_row"][k]), int(d["apple_tree"][k]))][0]
        yaw = 0.0 if (dx < -0.05 or (abs(dx) <= 0.05 and random.random() < 0.5)) else math.pi
        rot = _rot_z(yaw)
        c_rot = rot @ c0
        for _ in range(tries_per_apple):
            # where the robot would stand -> terrain -> the apple's height
            # above the robot's feet fixes dz
            guess = c_rot[:2] - np.array([0.55, -0.25])
            z_shift = -terrain(rot.T[:2, :2] @ (guess - c_rot[:2]) + c0[:2])
            dz = c0[2] + z_shift - sh[2]
            if not target_dz_range[0] <= dz <= target_dz_range[1]:
                rejects["height"] += 1
                break
            r_xy_max = math.sqrt(max(reach_max ** 2 - dz ** 2, 1e-4))
            r_xy_min = max(math.sqrt(max(reach_min ** 2 - dz ** 2, 0.0)), reach_min_horizontal)
            if r_xy_min >= r_xy_max:
                rejects["height"] += 1
                break
            r_xy = random.uniform(r_xy_min, r_xy_max)
            az = math.radians(random.uniform(az_lo, az_hi))
            if r_xy * math.cos(az) < reach_min_forward or not feasible(r_xy * math.cos(az), r_xy * math.sin(az), dz):
                continue
            target = np.array([sh[0] + r_xy * math.cos(az), sh[1] + r_xy * math.sin(az), sh[2] + dz])
            th = math.radians(random.uniform(-approach_yaw_spread_deg, approach_yaw_spread_deg))
            b = np.array([-math.cos(th), -math.sin(th), 0.0])            # apple -> palm (toward the robot)
            apple_local = target - b * (r_env + standoff)
            t = apple_local - c_rot
            t[2] = z_shift = -terrain(rot.T[:2, :2] @ -t[:2])     # feet on the terrain, exactly
            apple_local[2] = c0[2] + z_shift
            if not target_dz_range[0] - 0.02 <= apple_local[2] - sh[2] <= target_dz_range[1] + 0.02:
                rejects["height"] += 1
                continue
            obs = _Obstacles(d, rot, t, near=sh, radius=1.6)
            if screen_robot and (obs.clearance(rb["p"], body_ignore_radius) - rb["r"]).min() < body_margin:
                rejects["body"] += 1
                continue
            corridor = _segment_points(sh, target)
            if screen_robot and (obs.clearance(corridor).min() < arm_corridor_radius
                                 or obs.clearance(target[None])[0] < hand_pocket_radius):
                rejects["arm"] += 1
                continue
            if obs.clearance(apple_local[None])[0] < r_env + apple_margin:
                rejects["apple"] += 1
                continue
            if rb["has_basket"]:
                from . import basket as bk
                col = _segment_points((*bk.CENTER_XY, 0.0), (*bk.CENTER_XY, basket_top), 0.05)
                if obs.clearance(col).min() < basket_radius:
                    rejects["basket"] += 1
                    continue
            chosen = (k, yaw, rot, t, apple_local, obs)
            break
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError(f"place_orchard_and_apple: no feasible apple among {len(cand)} candidates "
                           f"(apple radius {r_env * 1000:.0f} mm), rejections {rejects}")
    k, yaw, rot, t, apple_local, obs = chosen

    # ── write the orchard and the apple ──
    ids = env_ids
    origin = env.scene.env_origins[ids]
    pose = torch.zeros(len(ids), 7, device=dev)
    pose[:, :3] = origin + torch.tensor(t, device=dev, dtype=torch.float32)
    pose[:, 3] = math.cos(yaw / 2)
    pose[:, 6] = math.sin(yaw / 2)
    orchard.write_root_pose_to_sim(pose, env_ids=ids)
    orchard.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)

    apple_w = origin + torch.tensor(apple_local, device=dev, dtype=torch.float32)
    root = apple_obj.data.default_root_state[ids].clone()
    root[:, :3] = apple_w
    root[:, 7:] = 0.0
    apple_obj.write_root_state_to_sim(root, env_ids=ids)

    stem = rot @ (ped[k] - centers[k])
    stem /= np.linalg.norm(stem)
    if not hasattr(env, "_stem_anchor_pos"):
        env._stem_anchor_pos = torch.zeros(env.num_envs, 3, device=dev)
        env._stem_dir = torch.zeros(env.num_envs, 3, device=dev)
    env._stem_anchor_pos[ids] = apple_w
    env._stem_dir[ids] = torch.tensor(stem, device=dev, dtype=torch.float32)

    # stem first (as randomize_tree_and_apple), then the wood within reach
    ped_local = rot @ ped[k] + t
    near = np.linalg.norm((obs.a + obs.b) / 2 - sh, axis=1) < tree_reach + np.linalg.norm(obs.b - obs.a, axis=1) / 2
    seg = [np.stack([apple_local + stem * r_env, ped_local])] + [np.stack([a, b]) for a, b in zip(obs.a[near], obs.b[near])]
    rad = [stem_radius] + list(obs.r[near])
    env._tree_segments = torch.tensor(np.stack(seg), device=dev, dtype=torch.float32).unsqueeze(0) + origin.view(-1, 1, 1, 3)
    env._tree_radii = torch.tensor(rad, device=dev, dtype=torch.float32)

    # hide the orchard's own copy of this apple (the env's sphere replaces it)
    for path in env._orchard_hidden:
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            UsdGeom.Imageable(p).MakeVisible()
    env._orchard_hidden = []
    ns = f"/World/envs/env_{int(ids[0])}/Orchard/Visual/apple"
    for j in range(8):
        path = f"{ns}/{apple_prim_name(int(k), j)}"
        p = stage.GetPrimAtPath(path)
        if not p.IsValid():
            break
        UsdGeom.Imageable(p).MakeInvisible()
        env._orchard_hidden.append(path)

    others = np.linalg.norm(centers - centers[k], axis=1)
    rel = apple_local - sh
    env._orchard_info[0] = {
        "apple": str(d["apple_name"][k]), "cultivar": str(d["apple_cultivar"][k]),
        "orchard_radius": float(d["apple_radius"][k]), "wood_gap": float(d["apple_wood_gap"][k]),
        "height": float(apple_local[2]), "facing": "east" if yaw == 0.0 else "west",
        "dist": float(np.linalg.norm(rel)), "azimuth_deg": math.degrees(math.atan2(rel[1], rel[0])),
        "neighbours_10cm": int(((others > 1e-6) & (others < 0.10)).sum()),
        "wood_capsules": len(rad) - 1, "rejects": rejects,
    }
