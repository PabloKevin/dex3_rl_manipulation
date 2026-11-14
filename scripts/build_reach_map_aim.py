"""Reachable-apple map for the tree sampler from the aim map (envs/assets/aim_map_v5.pt).

User, 2026-11-14: some low / far apples in the harvest video were beyond the arm; they spoil the grasp training
(an episode the fingers cannot win) and the harvest metrics. Measured: the v5 reach never arrives beyond ~0.53 m
from the shoulder (the collision-free pose library with the robot's shoulder-roll limit tops out at 0.536 m), and
the old map (reach_workspace_single_v2_fixed_waist.npz, 5 cm cells, built for an earlier reach with roll up to
+0.35) admits apples to ~0.6 m -- plus the grasp world inherited STRETCH_FRAC: 10% of targets drawn just OUTSIDE it.

feasible(cell) = old map's feasible at the cell centre AND, at EVERY waist yaw of the training range, the best
palm heading / tilt has Q >= --q_min (the aim map's predicted reach success). Same npz keys as the old map
(tree_sampler.Envelope reads x_edges, y_edges, z_edges, feasible: apple centre - right shoulder, torso at yaw 0),
2.5 cm cells. No simulator needed:

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/build_reach_map_aim.py
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("aim", os.path.join(REPO, "envs/mdp/aim.py"))   # torch only, no Isaac
aim_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aim_mod)

p = argparse.ArgumentParser()
p.add_argument("--aim", default=os.path.join(REPO, "envs/assets/aim_map_v5.pt"))
p.add_argument("--old", default=os.path.join(REPO, "envs/assets/reach_workspace_single_v2_fixed_waist.npz"))
p.add_argument("--out", default=os.path.join(REPO, "envs/assets/reach_workspace_aim_v5.npz"))
p.add_argument("--cell", type=float, default=0.025)
p.add_argument("--q_min", type=float, default=0.5)
p.add_argument("--yaws_deg", type=float, nargs="+", default=[-45.0, -30.0, -15.0, 0.0, 15.0])
args = p.parse_args()

A = aim_mod.load(args.aim, "cuda" if torch.cuda.is_available() else "cpu")
old = np.load(args.old)
oxe, oye, oze, ofeas = old["x_edges"], old["y_edges"], old["z_edges"], old["feasible"]


def edges(e):
    n = int(round((e[-1] - e[0]) / args.cell))
    return e[0] + args.cell * np.arange(n + 1)


xe, ye, ze = edges(oxe), edges(oye), edges(oze)
cx, cy, cz = [(e[:-1] + e[1:]) / 2 for e in (xe, ye, ze)]
G = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), -1).reshape(-1, 3)


def old_ok(P):
    idx = [np.floor((P[:, k] - e[0]) / (e[1] - e[0])).astype(int) for k, e in enumerate((oxe, oye, oze))]
    inside = np.all([(i >= 0) & (i < n) for i, n in zip(idx, ofeas.shape)], axis=0)
    ok = np.zeros(len(P), bool)
    ok[inside] = ofeas[tuple(i[inside] for i in idx)]
    return ok


ok_old = old_ok(G)
hs = torch.deg2rad(torch.arange(-45.0, 45.1, 15.0, device=A.device))
ts = torch.deg2rad(torch.arange(-30.0, 30.1, 15.0, device=A.device))
H, T = len(hs), len(ts)
best = np.full((len(G), len(args.yaws_deg)), 0.0, np.float32)
cand = np.nonzero(ok_old)[0]
for k, yd in enumerate(args.yaws_deg):
    for s in range(0, len(cand), 4096):
        ids = cand[s:s + 4096]
        rel = torch.as_tensor(G[ids], dtype=torch.float32, device=A.device)
        n = len(ids)
        q = A.q(rel.view(n, 1, 1, 3).expand(n, H, T, 3), hs.view(1, H, 1).expand(n, H, T),
                ts.view(1, 1, T).expand(n, H, T), torch.full((n, H, T), math.radians(yd), device=A.device))
        best[ids, k] = q.reshape(n, -1).amax(-1).cpu().numpy()
qmin = best.min(1)
feas = ok_old & (qmin >= args.q_min)
shape = (len(cx), len(cy), len(cz))
np.savez(args.out, x_edges=xe, y_edges=ye, z_edges=ze, feasible=feas.reshape(shape),
         q_min_over_yaw=qmin.reshape(shape), q_per_yaw=best.reshape(shape + (len(args.yaws_deg),)),
         yaws_deg=np.array(args.yaws_deg), q_threshold=args.q_min, cell=args.cell,
         source=f"{os.path.basename(args.old)} & aim map {os.path.basename(args.aim)}")
d = np.linalg.norm(G, axis=1)
print(f"[reach map] {args.out}: {feas.sum()} of {ok_old.sum()} old-feasible cells kept "
      f"({100 * feas.sum() / max(ok_old.sum(), 1):.0f}%); old max dist {d[ok_old].max():.3f} m -> new {d[feas].max():.3f} m")
for lo in np.arange(0.40, 0.62, 0.025):
    m = ok_old & (d >= lo) & (d < lo + 0.025)
    if m.any():
        print(f"  dist {lo:.3f}-{lo + 0.025:.3f}: old cells {m.sum():4d}, kept {100 * feas[m].mean():5.1f}%, "
              f"Q spread over yaw (median max-min) {np.median(best[m].max(1) - best[m].min(1)):.2f}")
