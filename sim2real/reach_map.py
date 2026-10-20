#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
reach_map.py -- WHERE the right arm can put the grasp pocket, and with which palm direction: a geometric
volume, looked up at run time (no IK, no optimisation on the robot path).

Built once (offline) by sampling the arm's joint space inside the ROBOT's limits (the toy's / runner's clip,
the right shoulder roll <= +0.0524 rad) with a batched numpy forward kinematics straight from the URDF: every
sample marks the 2 cm cell its pocket falls in (torso frame, relative to the right shoulder: the arm moves
with the torso, so one map serves every waist yaw) with the HEADING of its palm normal relative to the
toy's approach rule (the horizontal line from the torso axis to the point, arm_pose_io.make_plan), when that
normal is near horizontal (|tilt| <= TILT_MAX_DEG: the goals' normals are horizontal). Headings are binned
every 15 deg (-60 .. +60, + = turned LEFT). A cell is then dilated by one cell (the policy's 2 cm reach
tolerance) -- the "interpolated" reachable volume. Samples with the hand inside a simple body envelope are
left out.

    ~/miniconda3/envs/g1_dex3_sim2real/bin/python sim2real/reach_map.py --build      (-> sim2real/reach_map.npz)
    ~/miniconda3/envs/g1_dex3_sim2real/bin/python sim2real/reach_map.py --check RUN_DIR   (the attempts of a run)

At run time (toy_harvest): ReachMap.headings(rel) -> the reachable heading offsets (deg) at the point rel
(torso frame, minus the right shoulder), [] = out of reach.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.path.join(HERE, "reach_map.npz")
#: the robot's right-arm target clip (the toy's / onboard runner's q_lo / q_hi on the robot, 2026-10-19)
ROBOT_LO = np.array([-2.945, -2.156, -2.487, -0.969, -1.874, -1.534, -1.534])
ROBOT_HI = np.array([2.526, 0.0524, 2.487, 2.016, 1.874, 1.534, 1.534])
CELL = 0.02
#: the grid, torso frame relative to the right shoulder (m)
X_RANGE, Y_RANGE, Z_RANGE = (-0.10, 0.66), (-0.70, 0.50), (-0.66, 0.44)
HEADINGS_DEG = np.arange(-60, 61, 15)
TILT_MAX_DEG = 15.0
#: body envelope (torso frame, the torso origin): the hand may not be inside this cylinder above HIP_Z,
#: nor below it closer than LEG_R to the axis
BODY_R, HIP_Z, LEG_R = 0.17, -0.15, 0.24


# ─────────────────────────────────────────────
#  Batched forward kinematics (torso -> palm), from the URDF
# ─────────────────────────────────────────────

def _rot_axis(axis, q):
    """[N,3,3] rotations about a fixed unit axis by angles q [N]."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1.0 - c) * (K @ K)[None]


class ArmFK:
    """The pocket and palm rotation in the TORSO frame for batches of right-arm joint vectors."""

    def __init__(self, urdf=None):
        sys.path.insert(0, HERE)
        sys.path.insert(0, os.path.join(HERE, "onboard"))
        import reach_io as R
        from np_fk import UrdfChains
        self.R = R
        ch = UrdfChains(urdf or R.URDF_DEFAULT, ["right_hand_palm_link", "right_shoulder_pitch_link"])
        def after_torso(chain):
            i = [k for k, j in enumerate(chain) if j["parent"] == "torso_link"]
            return chain[i[0]:]
        self.palm_chain = after_torso(ch.chains["right_hand_palm_link"])
        sh = after_torso(ch.chains["right_shoulder_pitch_link"])
        Rs, ps = np.eye(3), np.zeros(3)
        for j in sh:                                  # the shoulder link's origin (its joint at q = 0 frame)
            ps = ps + Rs @ j["p0"]
            Rs = Rs @ j["R0"]
        self.shoulder = ps
        self.arm_names = list(R.RIGHT_ARM_JOINTS)
        import arm_pose_io as AP
        self.pocket_local = np.asarray(AP.POCKET, float)
        self.normal_local = np.asarray(AP.PALM_NORMAL, float)

    def __call__(self, q):
        """q [N,7] -> (pocket [N,3], palm normal [N,3]) in the torso frame."""
        N = len(q)
        Rm = np.repeat(np.eye(3)[None], N, axis=0)
        p = np.zeros((N, 3))
        for j in self.palm_chain:
            p = p + Rm @ j["p0"]
            Rj = np.repeat(j["R0"][None], N, axis=0)
            if j["type"] in ("revolute", "continuous"):
                k = self.arm_names.index(j["name"]) if j["name"] in self.arm_names else None
                if k is not None:
                    Rj = Rj @ _rot_axis(j["axis"], q[:, k])
            Rm = Rm @ Rj
        return p + Rm @ self.pocket_local, Rm @ self.normal_local


# ─────────────────────────────────────────────
#  The map
# ─────────────────────────────────────────────

def _cells(rel):
    return (np.floor((rel[:, 0] - X_RANGE[0]) / CELL).astype(int), np.floor((rel[:, 1] - Y_RANGE[0]) / CELL).astype(int),
            np.floor((rel[:, 2] - Z_RANGE[0]) / CELL).astype(int))


def _shape():
    return (int(round((X_RANGE[1] - X_RANGE[0]) / CELL)), int(round((Y_RANGE[1] - Y_RANGE[0]) / CELL)),
            int(round((Z_RANGE[1] - Z_RANGE[0]) / CELL)))


def heading_offset_deg(p_torso, normal_torso):
    """The palm normal's heading minus the toy's approach heading (torso axis -> point), deg (+ = left)."""
    ref = np.degrees(np.arctan2(p_torso[..., 1], p_torso[..., 0]))
    h = np.degrees(np.arctan2(normal_torso[..., 1], normal_torso[..., 0]))
    return (h - ref + 180.0) % 360.0 - 180.0


def build(n_samples=6_000_000, chunk=250_000, seed=0, out=MAP_FILE):
    fk = ArmFK()
    rng = np.random.default_rng(seed)
    shape = _shape() + (len(HEADINGS_DEG),)
    hit = np.zeros(shape, bool)
    any_hit = np.zeros(_shape(), bool)
    t0 = time.time()
    kept = 0
    for s0 in range(0, n_samples, chunk):
        q = ROBOT_LO + (ROBOT_HI - ROBOT_LO) * rng.random((chunk, 7))
        pk, n = fk(q)
        # the body: the hand's pocket outside a cylinder around the torso, and the legs below the hips
        r_ax = np.hypot(pk[:, 0], pk[:, 1])
        ok = ~(((pk[:, 2] > HIP_Z) & (r_ax < BODY_R)) | ((pk[:, 2] <= HIP_Z) & (r_ax < LEG_R)))
        rel = pk - fk.shoulder
        ix, iy, iz = _cells(rel)
        ok &= (ix >= 0) & (ix < shape[0]) & (iy >= 0) & (iy < shape[1]) & (iz >= 0) & (iz < shape[2])
        any_hit[ix[ok], iy[ok], iz[ok]] = True
        tilt = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1)))
        ok &= np.abs(tilt) <= TILT_MAX_DEG
        off = heading_offset_deg(pk, n)
        b = np.round((off - HEADINGS_DEG[0]) / 15.0).astype(int)
        ok &= (b >= 0) & (b < len(HEADINGS_DEG)) & (np.abs(off - HEADINGS_DEG[np.clip(b, 0, len(HEADINGS_DEG) - 1)]) <= 7.5)
        hit[ix[ok], iy[ok], iz[ok], b[ok]] = True
        kept += int(ok.sum())
    raw = hit.copy()
    # the policy's 2 cm tolerance: a cell next to a reached one counts (dilation by one cell, 6-neighbourhood)
    d = hit.copy()
    for ax in range(3):
        d |= np.roll(hit, 1, axis=ax) | np.roll(hit, -1, axis=ax)
    np.savez_compressed(out, reach=d, raw=raw, any=any_hit, cell=CELL, x_range=X_RANGE, y_range=Y_RANGE,
                        z_range=Z_RANGE, headings_deg=HEADINGS_DEG, tilt_max_deg=TILT_MAX_DEG, q_lo=ROBOT_LO,
                        q_hi=ROBOT_HI, shoulder_torso=fk.shoulder, n_samples=n_samples)
    print(f"[reach_map] {n_samples} samples ({kept} with a near-horizontal palm in the grid) in "
          f"{time.time() - t0:.0f} s -> {out}: {int(raw.any(-1).sum())} cells reached with some heading "
          f"({int(d.any(-1).sum())} after the 2 cm dilation), any palm direction {int(any_hit.sum())}")


class ReachMap:
    def __init__(self, path=MAP_FILE):
        z = np.load(path)
        self.reach = z["reach"]
        self.cell = float(z["cell"])
        self.lo = np.array([z["x_range"][0], z["y_range"][0], z["z_range"][0]], float)
        self.headings = np.asarray(z["headings_deg"], float)

    def headings_at(self, rel) -> list:
        """rel: the point minus the right shoulder, torso frame -> the reachable palm heading offsets (deg,
        + = turned left of the torso-axis -> point line), smallest |offset| first; [] = out of reach."""
        i = np.floor((np.asarray(rel, float) - self.lo) / self.cell).astype(int)
        if np.any(i < 0) or np.any(i >= np.array(self.reach.shape[:3])):
            return []
        hs = self.headings[self.reach[i[0], i[1], i[2]]]
        return sorted(hs.tolist(), key=lambda h: (abs(h), h))


def check_run(run_dir, path=MAP_FILE):
    """The apples a robot run tried (onboard logs), the map's verdict and the run's outcome."""
    import glob
    import json
    sys.path.insert(0, HERE)
    import reach_io as R
    kin = R.G1Kinematics()
    M = ReachMap(path)
    seen = set()
    for f in sorted(glob.glob(os.path.join(run_dir, "onboard_*.npz"))):
        d = np.load(f, allow_pickle=True)
        m = json.loads(str(d["meta"]))
        st = m["start"]
        gl = st.get("goals") or []
        if not gl or gl[-1]["name"] != "apple":
            continue
        key = tuple(np.round(st["apple"], 2))
        if key in seen:
            continue
        seen.add(key)
        p = np.asarray(gl[-1]["target"], float)
        fr = kin.frames(np.asarray(d["waist"][0], float), R.HOME_Q)
        rel = fr.torso_R.T @ (p - fr.shoulder_pos)
        hs = M.headings_at(rel)
        print(f"{os.path.basename(f)[8:23]} pocket goal from the shoulder {np.round(rel, 3).tolist()} "
              f"({np.linalg.norm(rel):.3f} m): map {'REACHABLE, headings ' + str([int(h) for h in hs]) if hs else 'OUT OF REACH'}"
              f"  | run: {m['result']['outcome'][:48]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--samples", type=int, default=6_000_000)
    ap.add_argument("--check", default=None, help="a robot run folder (logs/sim2real/toy_reach_single/...)")
    a = ap.parse_args()
    if a.build:
        build(a.samples)
    if a.check:
        check_run(a.check)
