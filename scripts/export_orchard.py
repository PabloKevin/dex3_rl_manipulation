#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Export the NOVA commercial orchard (envs/EnvCommercialAppleTrees, a MuJoCo
project) into a plain .npz that the Isaac Lab orchard scene
(envs/mdp/orchard.py) turns into a USD.

The orchard's only complete description is the compiled scene
out/g1_scenes/orchard_g1.mjb (its XML and world entrypoint were not copied
over), so this reads that binary and writes, in the MuJoCo world frame (z up,
the G1 in the alleys faces +x toward a row):
  - every visual mesh except the robot's (trees + leaves, apples, posts,
    wires, irrigation, crates, bins, ground cover) and the terrain
    heightfields as meshes, with their colours;
  - every collision piece (MuJoCo convex pieces: trunks, posts, wires,
    scaffold branches, bins...) with a fitted oriented box and capsule, for
    the scene's clearance checks;
  - the 180 hanging apples: centre, radius, pedicel tip (real stem axis),
    cultivar, row/tree, gap to the nearest wood (mj_geomDistance);
  - tree trunks (row/tree -> xy) and a terrain height grid.

Needs MuJoCo >= 3.3.5 (the .mjb was compiled with 3.3.5; 3.3.3 and older
refuse it) -- NOT the Isaac Lab python. E.g.:
  pip install --target /tmp/mj335 mujoco==3.3.5
  PYTHONPATH=/tmp/mj335 python scripts/export_orchard.py
Output: envs/EnvCommercialAppleTrees/out/isaac/orchard_export.npz (~7 MB).
The Isaac side rebuilds its USD whenever this file is newer than it.
"""
import argparse
import os
import re

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCHARD = os.path.join(REPO, "envs", "EnvCommercialAppleTrees")

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mjb", default=os.path.join(ORCHARD, "out", "g1_scenes", "orchard_g1.mjb"))
p.add_argument("--out", default=os.path.join(ORCHARD, "out", "isaac", "orchard_export.npz"))
p.add_argument("--terrain_res", type=float, default=0.10, help="terrain height grid spacing (m)")
args = p.parse_args()

m = mujoco.MjModel.from_binary_path(args.mjb)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
OBJ = mujoco.mjtObj
print(f"[export] {args.mjb}: {m.nbody} bodies, {m.ngeom} geoms (MuJoCo {mujoco.__version__})")

robot = m.body_rootid == m.body("pelvis").id
body_names = [name(OBJ.mjOBJ_BODY, b) for b in range(m.nbody)]
apple_bodies = [b for b in range(m.nbody) if body_names[b].endswith("_body")
                and any(body_names[c] == body_names[b].replace("_body", "_pedicel") for c in range(m.nbody))]
windfalls = set()
KIND = {"static": 0, "branch": 1, "apple": 2, "object": 3, "terrain": 4}


def body_kind(b):
    if b == 0:
        return KIND["static"]
    if b in apple_bodies or body_names[b].endswith("_pedicel"):
        return KIND["apple"]
    jt = [m.jnt_type[j] for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])]
    if any(t == mujoco.mjtJoint.mjJNT_HINGE for t in jt):
        return KIND["branch"]                 # spring-loaded scaffold / shoot
    return KIND["object"]                     # free bodies: crates, windfall fruit


def mesh_world(g):
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    v = m.mesh_vert[a:a + n].astype(np.float64) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return v.astype(np.float32), m.mesh_face[fa:fa + fn].astype(np.int32)


def hfield_world(g):
    h = m.geom_dataid[g]
    nr, nc = m.hfield_nrow[h], m.hfield_ncol[h]
    rx, ry, top, _ = m.hfield_size[h]
    z = m.hfield_data[m.hfield_adr[h]:m.hfield_adr[h] + nr * nc].reshape(nr, nc) * top
    xs, ys = np.meshgrid(np.linspace(-rx, rx, nc), np.linspace(-ry, ry, nr))
    v = np.stack([xs, ys, z], -1).reshape(-1, 3) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    i = np.arange(nr * nc).reshape(nr, nc)
    q = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1).reshape(-1, 4)
    f = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return v.astype(np.float32), f.astype(np.int32)


def fit(v):
    """Oriented box (PCA axes) and capsule around a convex piece."""
    c = v.mean(0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    loc = (v - c) @ vt.T
    lo, hi = loc.min(0), loc.max(0)
    c = c + ((lo + hi) / 2) @ vt
    half = (hi - lo) / 2
    t = (v - c) @ vt[0]
    r = np.linalg.norm((v - c) - np.outer(t, vt[0]), axis=1).max()
    l = max(0.0, (t.max() - t.min()) / 2 - r)
    mid = c + vt[0] * (t.max() + t.min()) / 2
    return c, vt, half, mid - vt[0] * l, mid + vt[0] * l, r


apple_index = {b: k for k, b in enumerate(apple_bodies)}
vis, col = [], []
for g in range(m.ngeom):
    b = m.geom_bodyid[g]
    if robot[b]:
        continue
    kind = body_kind(b)
    apple = apple_index.get(b, apple_index.get(m.body_parentid[b], -1))
    collides = bool(m.geom_contype[g] or m.geom_conaffinity[g])
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
        v, f = hfield_world(g)
        vis.append((v, f, m.geom_rgba[g], KIND["terrain"], -1))
        continue
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    v, f = mesh_world(g)
    if collides:
        col.append((v, f, kind, apple, g))
    elif m.geom_rgba[g][3] > 0 and m.geom_group[g] in (0, 1, 2):
        vis.append((v, f, m.geom_rgba[g], kind, apple))


def pack(items, prefix):
    out = {}
    nv = np.array([len(it[0]) for it in items]); nf = np.array([len(it[1]) for it in items])
    out[prefix + "verts"] = np.concatenate([it[0] for it in items])
    out[prefix + "faces"] = np.concatenate([it[1] for it in items])
    out[prefix + "vadr"] = np.concatenate([[0], np.cumsum(nv)[:-1]]).astype(np.int64)
    out[prefix + "vnum"] = nv.astype(np.int64)
    out[prefix + "fadr"] = np.concatenate([[0], np.cumsum(nf)[:-1]]).astype(np.int64)
    out[prefix + "fnum"] = nf.astype(np.int64)
    return out


data = {}
data.update(pack(vis, "vis_"))
data["vis_rgba"] = np.array([it[2] for it in vis], np.float32)
data["vis_kind"] = np.array([it[3] for it in vis], np.int32)
data["vis_apple"] = np.array([it[4] for it in vis], np.int32)
data.update(pack(col, "col_"))
data["col_kind"] = np.array([it[2] for it in col], np.int32)
data["col_apple"] = np.array([it[3] for it in col], np.int32)
fits = [fit(it[0].astype(np.float64)) for it in col]
data["col_box_c"] = np.array([f[0] for f in fits], np.float32)
data["col_box_R"] = np.array([f[1] for f in fits], np.float32)          # rows = box axes
data["col_box_h"] = np.array([f[2] for f in fits], np.float32)
data["col_cap_a"] = np.array([f[3] for f in fits], np.float32)
data["col_cap_b"] = np.array([f[4] for f in fits], np.float32)
data["col_cap_r"] = np.array([f[5] for f in fits], np.float32)

# ── apples ────────────────────────────────────────────────────────────
eq_of = {}
for e in range(m.neq):
    nm = name(OBJ.mjOBJ_EQUALITY, e)
    eq_of[m.eq_obj1id[e]] = (nm, m.eq_obj2id[e])
wood = [it[4] for it in col if it[2] != KIND["apple"]]
wood_pos = d.geom_xpos[wood]
a_center, a_radius, a_ped, a_name, a_cult, a_row, a_tree, a_gap, a_mass = [], [], [], [], [], [], [], [], []
for b in apple_bodies:
    ped = m.body(body_names[b].replace("_body", "_pedicel")).id
    gs = [g for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
          if m.geom_contype[g] or m.geom_conaffinity[g]]
    g = max(gs, key=lambda g: m.geom_rbound[g])                       # the fruit, not the tiny stem stub
    v, _ = mesh_world(g)
    c = (v.min(0) + v.max(0)) / 2
    eq_name, _ = eq_of.get(ped, ("", -1))
    mt = re.match(r"harvest_apple_(.+)_r(\d+)_t(\d+)_(\d+)", eq_name)
    a_center.append(c); a_radius.append(float(np.ptp(v, 0).mean() / 2)); a_ped.append(d.xpos[ped].copy())
    a_name.append(eq_name.replace("harvest_apple_", "") or body_names[b])
    a_cult.append(mt.group(1) if mt else "?")
    a_row.append(int(mt.group(2)) if mt else -1); a_tree.append(int(mt.group(3)) if mt else -1)
    a_mass.append(float(m.body_mass[b]))
    near = [wood[k] for k in np.nonzero(np.linalg.norm(wood_pos - c, axis=1) < 1.5)[0]]
    gap = 1.0
    for w in near:
        gap = min(gap, mujoco.mj_geomDistance(m, d, g, w, 0.3, None))
    a_gap.append(gap)
data["apple_center"] = np.array(a_center, np.float32)
data["apple_radius"] = np.array(a_radius, np.float32)
data["apple_pedicel"] = np.array(a_ped, np.float32)
data["apple_name"] = np.array(a_name)
data["apple_cultivar"] = np.array(a_cult)
data["apple_row"] = np.array(a_row, np.int32)
data["apple_tree"] = np.array(a_tree, np.int32)
data["apple_wood_gap"] = np.array(a_gap, np.float32)                  # fruit surface -> nearest wood (m)
data["apple_mass"] = np.array(a_mass, np.float32)

# ── trunks: the tall vertical piece nearest each tree's fruit ─────────
caps_dir = data["col_cap_b"] - data["col_cap_a"]
length = np.linalg.norm(caps_dir, axis=1)
vertical = (length > 1.5) & (np.abs(caps_dir[:, 2]) > 0.95 * length) & (data["col_cap_r"] < 0.12)
vmid = (data["col_cap_a"] + data["col_cap_b"]) / 2
trees = sorted(set(zip(a_row, a_tree)))
t_xy = []
for r, t in trees:
    sel = (data["apple_row"] == r) & (data["apple_tree"] == t)
    row_x = np.median(data["apple_center"][data["apple_row"] == r, 0])
    guess = np.array([row_x, np.median(data["apple_center"][sel, 1])])
    k = np.argmin(np.where(vertical, np.linalg.norm(vmid[:, :2] - guess, axis=1), np.inf))
    t_xy.append(vmid[k, :2])
data["tree_row"] = np.array([r for r, _ in trees], np.int32)
data["tree_idx"] = np.array([t for _, t in trees], np.int32)
data["tree_trunk_xy"] = np.array(t_xy, np.float32)

# ── terrain height grid (max over the heightfields) ───────────────────
allv = data["vis_verts"]
lo, hi = allv[:, :2].min(0) - 0.5, allv[:, :2].max(0) + 0.5
xs = np.arange(lo[0], hi[0], args.terrain_res); ys = np.arange(lo[1], hi[1], args.terrain_res)
H = np.zeros((len(xs), len(ys)), np.float32)
hf = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD]
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        best = None
        for g in hf:
            dist = mujoco.mj_rayHfield(m, d, g, np.array([x, y, 10.0]), np.array([0.0, 0.0, -1.0]))
            if dist >= 0:
                z = 10.0 - dist
                best = z if best is None else max(best, z)
        H[i, j] = 0.0 if best is None else best
data["terrain_origin"] = np.array([xs[0], ys[0]], np.float32)
data["terrain_res"] = np.float32(args.terrain_res)
data["terrain_h"] = H
data["source"] = np.array(f"{os.path.abspath(args.mjb)} (MuJoCo {mujoco.__version__})")

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(args.out, **data)
print(f"[export] {len(vis)} visual meshes ({len(data['vis_verts'])} verts), {len(col)} collision pieces, "
      f"{len(a_center)} apples on {len(trees)} trees, terrain {H.shape} "
      f"({H.min():.2f}..{H.max():.2f} m) -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
