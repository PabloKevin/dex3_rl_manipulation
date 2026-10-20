#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
trunk_side.py -- is the apple BESIDE THE TRUNK, and on which side? Depth + colour, no training.

Robot 2026-10-20 (user): an apple in front of the trunk is harvested almost perfectly; an apple to the side of it
gets reached from the front and the hand closes on the apple AND the trunk. Then the hand should come from the
apple's free side: trunk on the left of the apple -> from the right, the palm normal turned LEFT (+); trunk on
the right -> the palm normal turned right (-).

How: the apple projected into the head camera's picture (the calibrated extrinsics the detector uses:
apple_picker/camera_extrinsics.yaml 'unitree'; its camera y points UP), its near-surface depth measured in the
aligned depth image. Two vertical strips beside it (as wide as the fingers sweep, STRIP_W apple radii, from
STRIP_H radii above to below its centre): a pixel is TRUNK when it is solid near the apple's depth (within
DEPTH_TOL, or nearer: in front of it blocks too) and BROWN (HSV, the tree's bark; leaves are green, apples red).
A strip whose trunk share is over SHARE_ON (and over the other strip's) = the trunk on that side.

  python3 sim2real/trunk_side.py --eval        every real attempt with a recorded frame -> a contact sheet
                                               (logs/sim2real/trunk_side/sheet_*.jpg) + verdicts.jsonl
  python3 sim2real/trunk_side.py --eval --labels logs/sim2real/trunk_side/labels.json    + accuracy
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import datetime

import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "logs", "sim2real", "trunk_side")
EXTRINSICS = os.path.expanduser("~/g1_fruits_picker/apple_picker/camera_extrinsics.yaml")
CAM = {"w": 424, "h": 240, "fx": 303.1, "fy": 303.0, "cx": 215.8, "cy": 123.3}
#: strip width / half height, in apple radii (in the picture)
STRIP_W, STRIP_H, STRIP_GAP = 1.3, 1.3, 1.1
#: solid near the apple's depth: within this behind its near surface (m); anything in front counts too
DEPTH_TOL = 0.07
#: bark in OpenCV HSV (H 0..180). The head camera's colour has a MAGENTA cast: the bark reads H 155-180 / 0-20,
#: S 40-200, V 25-110 (2026-10-20 recordings), the red apples the same hues but bright (V > 120), the leaves green
#: (H 35-95). Bark = NOT green and DARK.
GREEN_H = (30, 95)
SAT = (25, 230)
VAL = (15, 115)
#: a strip row "has bark" when this share of its width is bark; a strip whose rows have bark over SHARE_ON = the trunk
ROW_FILL = 0.30
SHARE_ON = 0.45


class Projector:
    def __init__(self, path=EXTRINSICS):
        import yaml
        e = yaml.safe_load(open(path))["unitree"]
        self.A, self.t = np.array(e["A"], float), np.array(e["t"], float)

    def pixel(self, p_torso, r):
        """(u, v, r_px, z of the centre m) of an apple centre (torso frame)."""
        pc = np.linalg.solve(self.A, np.asarray(p_torso, float) - self.t)
        z = float(pc[2])
        return CAM["cx"] + CAM["fx"] * pc[0] / z, CAM["cy"] - CAM["fy"] * pc[1] / z, CAM["fx"] * float(r) / z, z


def trunk_masks(rgb, depth_m, d_apple):
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    brown = ~((h >= GREEN_H[0]) & (h <= GREEN_H[1])) & (s >= SAT[0]) & (s <= SAT[1]) & (v >= VAL[0]) & (v <= VAL[1])
    solid = (depth_m > 0.05) & (depth_m < d_apple + DEPTH_TOL)
    return brown & solid, brown, solid


#: the robot's own hand / forearm masked out of the strips: discs of this radius (m) around their FK points
HAND_MASK_R = 0.07


def hand_points_torso(kin, waist3, arm_q):
    """Torso-frame points along the right forearm and hand (for the mask)."""
    import arm_pose_io as AP
    fr = kin.frames(waist3, arm_q)
    pts = [fr.palm_pos, AP.hand_frame(fr)[0], fr.palm_pos + fr.palm_R @ np.array([0.12, 0.0, 0.0])]
    for link in ("right_wrist_yaw_link", "right_wrist_roll_link", "right_elbow_link"):
        pts.append(np.asarray(kin.data.oMf[kin.model.getFrameId(link)].translation))
    return [fr.torso_R.T @ (p - fr.torso_pos) for p in pts]


def side_scores(rgb, depth_mm, u, v, r_px, z_centre, r_m, mask=None):
    """-> dict(left, right, verdict 'left' | 'right' | 'none', d_apple, boxes). mask: pixels to ignore (the hand)."""
    H, W = depth_mm.shape
    depth = depth_mm.astype(float) * 0.001
    # the apple's near surface: the median of the valid depth inside half its disc, else the model's
    yy, xx = np.ogrid[:H, :W]
    disc = (xx - u) ** 2 + (yy - v) ** 2 <= (0.5 * r_px) ** 2
    vals = depth[disc & (depth > 0.05)]
    d_apple = float(np.median(vals)) if vals.size > 10 else z_centre - r_m
    trunk, _, _ = trunk_masks(rgb, depth, d_apple)
    if mask is not None:
        trunk = trunk & ~mask
    y0, y1 = int(max(0, v - STRIP_H * r_px)), int(min(H, v + STRIP_H * r_px))
    out = {"d_apple": d_apple, "boxes": {}}
    for side, x0, x1 in (("left", u - STRIP_GAP * r_px - STRIP_W * r_px, u - STRIP_GAP * r_px),
                         ("right", u + STRIP_GAP * r_px, u + STRIP_GAP * r_px + STRIP_W * r_px)):
        a, b = int(max(0, x0)), int(min(W, x1))
        if b - a < 3 or y1 - y0 < 3:
            out[side] = float("nan")
            continue
        blk = trunk[y0:y1, a:b]
        # a trunk is a CONTINUOUS vertical band: the share of the strip's rows with bark over ROW_FILL of the width
        # (shadowed leaves are dark and not green too, but patchy)
        out[side] = float((blk.mean(axis=1) >= ROW_FILL).mean())
        out[side + "_px"] = float(blk.mean())
        out["boxes"][side] = (a, y0, b, y1)
    L, Rr = out.get("left", 0.0), out.get("right", 0.0)
    L, Rr = (0.0 if np.isnan(L) else L), (0.0 if np.isnan(Rr) else Rr)
    out["verdict"] = "left" if L >= SHARE_ON and L > Rr else "right" if Rr >= SHARE_ON and Rr > L else "none"
    return out


def hand_mask(proj, pts_torso, shape):
    H, W = shape
    m = np.zeros((H, W), bool)
    yy, xx = np.ogrid[:H, :W]
    for p in pts_torso:
        u, v, rp, z = proj.pixel(p, HAND_MASK_R)
        if z > 0.05:
            m |= (xx - u) ** 2 + (yy - v) ** 2 <= rp ** 2
    return m


# ─────────────────────────────────────────────
#  The trunk as a CYLINDER (3D), and the apple's side of it
# ─────────────────────────────────────────────
#: the picture region searched around the apple (apple radii: half width, above, below)
ROI_HALF_W, ROI_UP, ROI_DOWN = 4.0, 5.0, 6.0
#: bark points kept from this much in front of the apple's near surface to this much behind it (m)
ROI_NEAR, ROI_FAR = 0.12, 0.30
#: RANSAC line: inlier distance (m), iterations, minimum inliers
LINE_TOL, LINE_ITERS, LINE_MIN = 0.03, 150, 60
#: the trunk's radius is kept within (m)
TRUNK_R = (0.015, 0.09)
#: the apple is IN FRONT of the trunk when its centre is off the trunk's axis, across the robot's view, by less than
#: the trunk's radius + this share of the apple's radius
FRONT_FRAC = 0.5
#: a trunk whose surface is further than this from the apple's surface does not matter (m)
NEAR_M = 0.06


def _backproject(proj, u, v, z):
    """Pixels (u, v) at depth z (m, along the optical axis) -> torso-frame points (the calibration's camera y UP)."""
    pc = np.stack([(u - CAM["cx"]) * z / CAM["fx"], -(v - CAM["cy"]) * z / CAM["fy"], z], axis=-1)
    return pc @ proj.A.T + proj.t


def fit_trunk(rgb, depth_mm, proj, apple_torso, r_apple, mask=None, rng=None):
    """The trunk near the apple as a cylinder (torso frame): bark pixels (not green, dark, solid) in a region around
    the apple, back-projected with the depth, a RANSAC 3D line through them; the radius from the band's width, the
    axis pushed that radius away from the camera (the points are its front surface). Then the apple vs the axis:
    'front' (within FRONT_DEG of the trunk -> robot direction), 'right' / 'left' of it as the robot sees it, or
    'far' (surfaces more than NEAR_M apart). -> dict or None (no trunk found)."""
    rng = rng or np.random.default_rng(0)
    u, v, rp, zc = proj.pixel(apple_torso, r_apple)
    H, W = depth_mm.shape
    depth = depth_mm.astype(float) * 0.001
    yy, xx = np.ogrid[:H, :W]
    disc = (xx - u) ** 2 + (yy - v) ** 2 <= (0.5 * rp) ** 2
    vals = depth[disc & (depth > 0.05)]
    d_apple = float(np.median(vals)) if vals.size > 10 else zc - r_apple
    x0, x1 = int(max(0, u - ROI_HALF_W * rp)), int(min(W, u + ROI_HALF_W * rp))
    y0, y1 = int(max(0, v - ROI_UP * rp)), int(min(H, v + ROI_DOWN * rp))
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s_, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bark = ~((h >= GREEN_H[0]) & (h <= GREEN_H[1])) & (s_ >= SAT[0]) & (s_ <= SAT[1]) & (vv >= VAL[0]) & (vv <= VAL[1])
    bark &= (depth > d_apple - ROI_NEAR) & (depth < d_apple + ROI_FAR)
    bark &= ~((xx - u) ** 2 + (yy - v) ** 2 <= (1.05 * rp) ** 2)            # not the apple itself
    if mask is not None:
        bark &= ~mask
    roi = np.zeros_like(bark)
    roi[y0:y1, x0:x1] = True
    bark &= roi
    vs, us = np.nonzero(bark)
    if len(us) < LINE_MIN:
        return None
    P = _backproject(proj, us.astype(float), vs.astype(float), depth[vs, us])
    if len(P) > 4000:
        sel = rng.choice(len(P), 4000, replace=False)
        P, us, vs = P[sel], us[sel], vs[sel]
    best, best_in = None, None
    for _ in range(LINE_ITERS):
        i, j = rng.choice(len(P), 2, replace=False)
        d = P[j] - P[i]
        n = float(np.linalg.norm(d))
        if n < 0.05:
            continue
        d /= n
        rel = P - P[i]
        dist = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
        inl = dist < LINE_TOL
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (P[i], d), inl
    if best_in is None or best_in.sum() < LINE_MIN:
        return None
    Q = P[best_in]
    c = Q.mean(axis=0)
    _, _, Vt = np.linalg.svd(Q - c)
    d = Vt[0] / np.linalg.norm(Vt[0])
    tproj = (Q - c) @ d
    # the radius: the band's half width across the axis (in the picture, at its depth)
    cam_pos = proj.t                                                    # the camera's origin in the torso frame
    view = c - cam_pos
    view /= np.linalg.norm(view)
    side = np.cross(d, view)
    side /= max(np.linalg.norm(side), 1e-9)
    w = (Q - c) @ side
    r = float(np.clip(0.5 * (np.percentile(w, 95) - np.percentile(w, 5)), *TRUNK_R))
    axis_c = c + r * view                                               # behind the surface it was seen on
    p0, p1 = axis_c + float(np.percentile(tproj, 3)) * d, axis_c + float(np.percentile(tproj, 97)) * d
    # the apple vs the axis: the closest axis point, then the direction from the trunk to the apple, horizontal
    A = np.asarray(apple_torso, float)
    s_ax = float(np.clip((A - p0) @ d, 0.0, np.linalg.norm(p1 - p0)))
    foot = p0 + s_ax * d
    to_apple = A - foot
    gap = float(np.linalg.norm(to_apple)) - r - float(r_apple)
    to_robot = cam_pos - foot
    to_robot[2] = 0.0
    to_robot /= max(np.linalg.norm(to_robot), 1e-9)
    ta = to_apple.copy()
    ta[2] = 0.0
    # seen by the robot looking at the trunk: the apple's offset ACROSS the view (+ = toward the robot's left) decides;
    # its depth vs the trunk is noise when the apple touches it (robot recordings 2026-10-20)
    left = np.cross([0.0, 0.0, 1.0], -to_robot)
    lateral = float(to_apple @ left)
    ang = 0.0 if np.linalg.norm(ta) < 1e-6 else float(np.degrees(np.arctan2(ta @ left, ta @ to_robot)))
    if gap > NEAR_M:
        rel = "far"
    elif abs(lateral) <= r + FRONT_FRAC * float(r_apple):
        rel = "front"                                                     # the trunk is behind the apple's centre
    else:
        rel = "left_of_trunk" if lateral > 0 else "right_of_trunk"        # the apple on the trunk's left / right
    # the strip method's naming: where the TRUNK is relative to the apple (for heading_for)
    verdict = {"left_of_trunk": "right", "right_of_trunk": "left"}.get(rel, "none")
    return {"p0": p0, "p1": p1, "r": r, "n": int(best_in.sum()), "apple_rel": rel, "angle_deg": ang, "lateral_m": lateral,
            "gap_m": gap, "verdict": verdict, "d_apple": d_apple, "uv": (u, v, rp)}


def overlay(rgb, proj, apple_torso, r_apple, cyl, title=""):
    """A PIL picture: the apple, the fitted cylinder's outline (orange), the decision."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(rgb, np.uint8))
    d = ImageDraw.Draw(im)
    u, v, rp, _ = proj.pixel(apple_torso, r_apple)
    d.ellipse([u - rp, v - rp, u + rp, v + rp], outline=(0, 255, 255), width=2)
    if cyl is not None:
        pts = [cyl["p0"] + f * (cyl["p1"] - cyl["p0"]) for f in np.linspace(0, 1, 12)]
        pix = [proj.pixel(q, cyl["r"]) for q in pts]
        uv = np.array([[q[0], q[1]] for q in pix])
        dv = uv[-1] - uv[0]
        nrm = np.array([-dv[1], dv[0]]) / max(np.linalg.norm(dv), 1e-6)
        for sg in (-1, 1):
            d.line([tuple(uv[i] + sg * pix[i][2] * nrm) for i in range(len(pix))], fill=(255, 140, 0), width=2)
    d.rectangle([0, 0, im.width, 12], fill=(0, 0, 0))
    d.text((3, 0), title, fill=(255, 255, 0))
    return im


def heading_for(verdict, turn_deg=60.0):
    """The palm heading offset (deg, + = turned left) for the free side: trunk left -> from the right -> +."""
    return {"left": +turn_deg, "right": -turn_deg}.get(verdict, 0.0)


# ─────────────────────────────────────────────
#  Offline evaluation on the recorded runs
# ─────────────────────────────────────────────

def _frames(run):
    out = []
    for seg in sorted(glob.glob(os.path.join(run, "camera", "*", "index.jsonl"))):
        base = os.path.dirname(seg)
        for line in open(seg):
            d = json.loads(line)
            if d["kind"] in ("color", "depth"):
                out.append((float(d["rx"]), d["kind"], os.path.join(base, d["file"])))
    return sorted(out)


def attempts():
    """Every real attempt with a recorded colour + depth frame just before its approach."""
    sys.path.insert(0, HERE)
    import apple_pos_calib as AC
    import reach_io as R
    kin = R.G1Kinematics()
    out = []
    for run in AC._real_runs():
        if not os.path.isdir(os.path.join(run, "camera")):
            continue
        F = _frames(run)
        if not F:
            continue
        for f in sorted(glob.glob(os.path.join(run, "onboard_*.npz"))):
            z = np.load(f, allow_pickle=True)
            m = json.loads(str(z["meta"]))
            st = m["start"]
            gl = st.get("goals") or []
            if not gl or gl[-1]["name"] != "apple":
                continue
            t_ep = time.mktime(datetime.datetime.strptime(st["id"], "%Y%m%d_%H%M%S").timetuple())
            col = [x for x in F if x[1] == "color" and t_ep - 1.5 < x[0] < t_ep - 0.1]
            dep = [x for x in F if x[1] == "depth" and t_ep - 1.5 < x[0] < t_ep - 0.1]
            if not col or not dep:
                continue
            c = col[-1]
            d = min(dep, key=lambda x: abs(x[0] - c[0]))
            if abs(d[0] - c[0]) > 0.25:
                continue
            w = np.asarray(z["waist"][0], float)
            fr = kin.frames(w, R.HOME_Q)
            p = np.asarray(st["apple"], float)
            out.append({"run": os.path.basename(run), "episode": st["id"], "color": c[2], "depth": d[2],
                        "waist": w.tolist(), "arm_q": np.asarray(z["q"][0], float).tolist(),
                        "apple_torso": (fr.torso_R.T @ (p - fr.torso_pos)).tolist(),
                        "radius": float(st.get("apple_radius", 0.04)), "outcome": m["result"].get("outcome", "")})
    return out


def evaluate(labels_path=None, per_sheet=12):
    from PIL import Image, ImageDraw
    sys.path.insert(0, HERE)
    import reach_io as R
    kin = R.G1Kinematics()
    P = Projector()
    rows = attempts()
    labels = json.load(open(labels_path)) if labels_path and os.path.exists(labels_path) else {}
    os.makedirs(OUT_DIR, exist_ok=True)
    tiles, res = [], []
    for k, a in enumerate(rows):
        rgb = np.array(Image.open(a["color"]).convert("RGB"))
        dmm = np.array(Image.open(a["depth"]))
        u, v, rp, zc = P.pixel(a["apple_torso"], a["radius"])
        hm = hand_mask(P, hand_points_torso(kin, np.asarray(a["waist"]), np.asarray(a["arm_q"])), dmm.shape)
        sc = side_scores(rgb, dmm, u, v, rp, zc, a["radius"], hm)
        cy = fit_trunk(rgb, dmm, P, a["apple_torso"], a["radius"], hm)
        r = dict(a, idx=k, u=u, v=v, r_px=rp, left=sc["left"], right=sc["right"], verdict=sc["verdict"],
                 d_apple=sc["d_apple"], cyl=None if cy is None else
                 {"p0": cy["p0"].tolist(), "p1": cy["p1"].tolist(), "r": cy["r"], "n": cy["n"], "rel": cy["apple_rel"],
                  "angle_deg": cy["angle_deg"], "lateral_m": cy["lateral_m"], "gap_m": cy["gap_m"],
                  "verdict": cy["verdict"]})
        res.append(r)
        # a tile: the apple's neighbourhood with the strips, the trunk pixels tinted
        trunk, _, _ = trunk_masks(rgb, dmm.astype(float) * 0.001, sc["d_apple"])
        trunk &= ~hm
        img = rgb.copy()
        img[hm] = (0.6 * img[hm]).astype(np.uint8)
        img[trunk] = (0.5 * img[trunk] + 0.5 * np.array([0, 120, 255])).astype(np.uint8)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.ellipse([u - rp, v - rp, u + rp, v + rp], outline=(0, 255, 255), width=2)
        if cy is not None:                                 # the fitted cylinder: its two outline edges, projected
            pts = [cy["p0"] + f * (cy["p1"] - cy["p0"]) for f in np.linspace(0, 1, 12)]
            pix = [P.pixel(q, cy["r"]) for q in pts]
            uv = np.array([[q[0], q[1]] for q in pix])
            dvec = uv[-1] - uv[0]
            nrm = np.array([-dvec[1], dvec[0]]) / max(np.linalg.norm(dvec), 1e-6)
            for sgn in (-1, 1):
                edge = [tuple(uv[i] + sgn * pix[i][2] * nrm) for i in range(len(pix))]
                dr.line(edge, fill=(255, 140, 0), width=2)
            dr.line([tuple(uv[0]), tuple(uv[-1])], fill=(255, 220, 120), width=1)
        cw = int(max(60, 4.5 * rp))
        crop = im.crop((int(u - cw), int(v - 0.75 * cw), int(u + cw), int(v + 0.75 * cw))).resize((360, 270))
        d2 = ImageDraw.Draw(crop)
        lab = labels.get(str(k), "")
        d2.rectangle([0, 0, 360, 24], fill=(0, 0, 0))
        cyt = "no trunk" if cy is None else (f"{cy['apple_rel']} side {cy['lateral_m'] * 100:+.0f}cm gap "
                                             f"{cy['gap_m'] * 100:.0f}cm r{cy['r'] * 100:.1f}")
        d2.text((3, 1), f"#{k} cyl: {cyt} | strips {sc['verdict']}" + (f" | label {lab}" if lab else ""),
                fill=(255, 255, 0))
        d2.text((3, 12), f"{a['episode'][4:]} {a['outcome'][:22]}", fill=(255, 255, 255))
        tiles.append(crop)
    for s0 in range(0, len(tiles), per_sheet):
        ts = tiles[s0:s0 + per_sheet]
        sheet = Image.new("RGB", (360 * 4, 270 * ((len(ts) + 3) // 4)), "white")
        for i, t in enumerate(ts):
            sheet.paste(t, ((i % 4) * 360, (i // 4) * 270))
        sheet.save(os.path.join(OUT_DIR, f"sheet_{s0 // per_sheet:02d}.jpg"), quality=88)
    with open(os.path.join(OUT_DIR, "verdicts.jsonl"), "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    print(f"[trunk_side] {len(res)} attempts -> {OUT_DIR}/sheet_*.jpg; strips: "
          + ", ".join(f"{k} {sum(r['verdict'] == k for r in res)}" for k in ("left", "right", "none"))
          + "; cylinder: " + ", ".join(f"{k} {sum((r['cyl'] or {}).get('rel', 'no trunk') == k for r in res)}"
                                       for k in ("front", "left_of_trunk", "right_of_trunk", "far", "no trunk")))
    if labels:
        ok = [r for r in res if str(r["idx"]) in labels]
        n_ok = sum(r["verdict"] == labels[str(r["idx"])] for r in ok)
        print(f"  labelled {len(ok)}: {n_ok} correct ({100.0 * n_ok / max(len(ok), 1):.0f}%)")
        for want in ("left", "right", "none"):
            sub = [r for r in ok if labels[str(r["idx"])] == want]
            if sub:
                print(f"   truth {want:5s}: {len(sub)} -> " + ", ".join(
                    f"{k} {sum(r['verdict'] == k for r in sub)}" for k in ("left", "right", "none")))
    return res



# ─────────────────────────────────────────────
#  EVERY TRUNK in the picture: colour first, the depth to filter, 1-2 segment cylinders from the floor
# ─────────────────────────────────────────────
# User 2026-10-20: three trunks now; each a vertical cylinder (maybe one part a bit diagonal), from the ground;
# find them by the bark colour, check by the depth, then fit. Points in the PELVIS frame (fixed to the ground).
#: bark colour (OpenCV HSV): not green, not grey (S >= 25), not bright (the red apples, the light floor)
BARK_V = (15, 130)
#: points kept: depth range from the camera (m), above the floor by (m), up to (m above the floor)
DEPTH_RANGE = (0.20, 1.60)
ABOVE_FLOOR, MAX_HEIGHT = 0.03, 1.60
#: 3D grouping: voxel (m), minimum points / vertical extent (m) of a trunk
VOXEL, MIN_PTS, MIN_EXTENT = 0.03, 120, 0.18
#: a trunk leans at most this much (horizontal m per vertical m: 1 = 45 deg)
MAX_LEAN = 1.0
#: two pieces are the same trunk when their lines pass within this (m) at the heights they share / between them
MERGE_M = 0.06
#: the two-segment fit is used when it lowers the residual by this share and each part has this share of the points
SPLIT_GAIN, SPLIT_MIN_SHARE = 0.35, 0.25


def floor_z_pelvis(kin):
    """The floor's height in the pelvis frame: the feet's sole (the URDF ankle roll links, the sole ~0.03 m below)."""
    import reach_io as R
    kin.frames([0.0, 0.0, 0.0], R.HOME_Q)
    z = [float(kin.data.oMf[kin.model.getFrameId(l)].translation[2]) for l in ("left_ankle_roll_link", "right_ankle_roll_link")]
    return min(z) - 0.03


def bark_points(rgb, depth_mm, proj, fr, floor_z, mask=None):
    """Bark-coloured pixels with a valid depth -> pelvis-frame points (and their pixels)."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    depth = depth_mm.astype(float) * 0.001
    bark = ~((h >= GREEN_H[0]) & (h <= GREEN_H[1])) & (s_ >= SAT[0]) & (s_ <= SAT[1]) & (v >= BARK_V[0]) & (v <= BARK_V[1])
    bark &= (depth > DEPTH_RANGE[0]) & (depth < DEPTH_RANGE[1])
    if mask is not None:
        bark &= ~mask
    vs, us = np.nonzero(bark)
    if len(us) == 0:
        return np.zeros((0, 3)), us, vs
    Pt = _backproject(proj, us.astype(float), vs.astype(float), depth[vs, us])
    Pp = Pt @ fr.torso_R.T + fr.torso_pos
    keep = (Pp[:, 2] > floor_z + ABOVE_FLOOR) & (Pp[:, 2] < floor_z + MAX_HEIGHT)
    return Pp[keep], us[keep], vs[keep]


def _components(P):
    """Connected 3D groups on a voxel grid (26-neighbours) -> list of index arrays."""
    if len(P) == 0:
        return []
    from collections import defaultdict, deque
    vox = np.floor(P / VOXEL).astype(int)
    cells = defaultdict(list)
    for i, k in enumerate(map(tuple, vox)):
        cells[k].append(i)
    seen, out = set(), []
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    for k in cells:
        if k in seen:
            continue
        q, comp = deque([k]), []
        seen.add(k)
        while q:
            c = q.popleft()
            comp.extend(cells[c])
            for d in nb:
                n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if n in cells and n not in seen:
                    seen.add(n)
                    q.append(n)
        out.append(np.array(comp))
    return out


def _line_xy_of_z(P, w=None):
    """Least squares x(z), y(z) = a + b z -> (a [2], b [2], rms m)."""
    z = P[:, 2]
    A = np.stack([np.ones_like(z), z], axis=1)
    coef, *_ = np.linalg.lstsq(A, P[:, :2], rcond=None)
    res = P[:, :2] - A @ coef
    return coef[0], coef[1], float(np.sqrt((res ** 2).sum(1).mean()))


def _robust_line(P, iters=3):
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        a, b, _ = _line_xy_of_z(P[keep])
        res = np.linalg.norm(P[:, :2] - (a + np.outer(P[:, 2], b)), axis=1)
        thr = max(0.025, 2.5 * float(np.median(res[keep])))
        keep = res < thr
        if keep.sum() < 20:
            break
    a, b, rms = _line_xy_of_z(P[keep])
    return a, b, rms, keep


def fit_trunk_polyline(P, cam_pelvis, floor_z):
    """One trunk's points -> {"pts": [floor, (knee), top] axis points, "r", "n"} or None. x, y as linear functions
    of the height (a vertical-ish trunk), one line or two joined at a knee, the lean capped, the axis pushed the
    trunk's radius away from the camera (the points are its front), extended down to the floor."""
    a, b, rms1, keep = _robust_line(P)
    Q = P[keep]
    if len(Q) < MIN_PTS or float(np.ptp(Q[:, 2])) < MIN_EXTENT:
        return None
    z0, z1 = float(Q[:, 2].min()), float(Q[:, 2].max())
    best = None
    if len(Q) > 2 * MIN_PTS:
        for zk in np.quantile(Q[:, 2], np.linspace(SPLIT_MIN_SHARE, 1 - SPLIT_MIN_SHARE, 9)):
            lo, hi = Q[Q[:, 2] <= zk], Q[Q[:, 2] > zk]
            if len(lo) < 40 or len(hi) < 40:
                continue
            al, bl, rl = _line_xy_of_z(lo)
            ah, bh, rh = _line_xy_of_z(hi)
            r2 = float(np.sqrt((rl ** 2 * len(lo) + rh ** 2 * len(hi)) / len(Q)))
            if best is None or r2 < best[0]:
                best = (r2, zk, (al, bl), (ah, bh))
    def xy(ab, z):
        return ab[0] + ab[1] * z
    if best is not None and best[0] < (1 - SPLIT_GAIN) * rms1:
        _, zk, lo_ab, hi_ab = best
        knee = (xy(lo_ab, zk) + xy(hi_ab, zk)) / 2.0
        segs = [(lo_ab, floor_z, zk), (hi_ab, zk, z1)]
        pts = [np.r_[xy(lo_ab, floor_z), floor_z], np.r_[knee, zk], np.r_[xy(hi_ab, z1), z1]]
    else:
        pts = [np.r_[xy((a, b), floor_z), floor_z], np.r_[xy((a, b), z1), z1]]
        segs = [((a, b), floor_z, z1)]
    # the lean: a part more leaning than MAX_LEAN is not a trunk (a branch / a leaf edge)
    for ab, _, _ in segs:
        if float(np.linalg.norm(ab[1])) > MAX_LEAN:
            return None
    # the radius: the band's half width across the view, at each height
    mid = np.mean(pts, axis=0)
    view = mid - cam_pelvis
    view[2] = 0.0
    view /= max(np.linalg.norm(view), 1e-9)
    across = np.array([-view[1], view[0], 0.0])
    w = []
    for zc in np.linspace(z0, z1, 8):
        sl = Q[np.abs(Q[:, 2] - zc) < 0.03]
        if len(sl) > 15:
            w.append(0.5 * float(np.ptp(sl @ across)))
    r = float(np.clip(np.median(w) if w else 0.03, *TRUNK_R))
    pts = [p + r * view for p in pts]
    return {"pts": [np.asarray(p, float) for p in pts], "r": r, "n": int(len(Q))}


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab))), a + t * ab


def trunk_axis_dist(tr, p):
    """(distance of p to the trunk's axis polyline, the closest axis point)."""
    best = (1e9, None)
    for a, b in zip(tr["pts"][:-1], tr["pts"][1:]):
        d = _seg_dist(np.asarray(p, float), a, b)
        if d[0] < best[0]:
            best = d
    return best


#: the colour mask cleaned with an opening of this size (px): leaf stems / apple rims (thin) drop out, bark bands stay
OPEN_PX = 7
#: a band: at least this long (px) and this elongated (length / width), within this angle (deg) of the picture's vertical
BAND_MIN_LEN, BAND_MIN_ASPECT, BAND_MAX_TILT = 50, 2.5, 50
#: pieces of one band (leaves in front) are joined when their centre lines pass within this (px)
BAND_JOIN_PX = 10
#: the trunk's depth: the median over its band; the band is dropped when the depth is beyond (m)
TRUNK_MAX_DEPTH = 1.25


def _band_line(us, vs):
    """Robust centre line of a band: u = a + b v (the picture's x as a function of its row) -> (a, b, rms px)."""
    keep = np.ones(len(us), bool)
    for _ in range(3):
        A = np.stack([np.ones(keep.sum()), vs[keep]], axis=1)
        coef, *_ = np.linalg.lstsq(A, us[keep], rcond=None)
        res = us - (coef[0] + coef[1] * vs)
        width = float(np.percentile(np.abs(res[keep]), 90))
        keep = np.abs(res) < max(3.0, 1.5 * width)
    A = np.stack([np.ones(keep.sum()), vs[keep]], axis=1)
    coef, *_ = np.linalg.lstsq(A, us[keep], rcond=None)
    res = us[keep] - (coef[0] + coef[1] * vs[keep])
    return float(coef[0]), float(coef[1]), float(np.sqrt((res ** 2).mean())), keep


#: a row's bark run counts as a trunk cross-section when this wide (px)
RUN_W = (5, 80)
#: a run continues a track when its centre is within this (px, or half the track's width) of the track's, rows apart
#: at most TRACK_GAP, and its width within TRACK_WRATIO of the track's
TRACK_GAP, TRACK_WRATIO = 8, 3.0


def _runs(row):
    """(start, end) of the True runs of a 1-D bool array."""
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def detect_trunk_bands(rgb, depth_mm, mask=None):
    """The trunks as BANDS in the picture (colour first): the bark mask cleaned, then TRACED row by row from the
    bottom up -- a run of bark of a trunk's width that continues a track's centre and width extends it (a trunk is a
    long near-vertical band; leaves and stems give short / wide / wandering runs) -- tracks broken by leaves joined
    when their centre lines agree. -> [{"a", "b" (centre u = a + b v), "v0", "v1", "half_w" px, "depth" m, "n",
    "two"}], the longest first."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bark = (~((h >= GREEN_H[0]) & (h <= GREEN_H[1])) & (s_ >= SAT[0]) & (s_ <= SAT[1])
            & (v >= BARK_V[0]) & (v <= BARK_V[1])).astype(np.uint8)
    if mask is not None:
        bark[mask] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX))
    bark = cv2.morphologyEx(bark, cv2.MORPH_OPEN, k)
    bark = cv2.morphologyEx(bark, cv2.MORPH_CLOSE, k).astype(bool)
    H, W = bark.shape
    tracks, done = [], []
    for vrow in range(H - 1, -1, -1):
        runs = [(float(a0), float(a1)) for a0, a1 in _runs(bark[vrow]) if RUN_W[0] <= a1 - a0 + 1 <= RUN_W[1]]
        used = set()
        for tr in tracks:
            c_pred = tr["c"][-1] + tr["slope"] * (vrow - tr["v"][-1])        # slope = du / dv
            best = None
            for j, (a0, a1) in enumerate(runs):
                if j in used:
                    continue
                c, w = 0.5 * (a0 + a1), a1 - a0 + 1
                if abs(c - c_pred) <= max(6.0, 0.5 * tr["w"][-1]) and 1 / TRACK_WRATIO <= w / tr["w"][-1] <= TRACK_WRATIO:
                    if best is None or abs(c - c_pred) < best[0]:
                        best = (abs(c - c_pred), j, c, w)
            if best is not None:
                _, j, c, w = best
                used.add(j)
                tr["v"].append(vrow); tr["c"].append(c); tr["w"].append(w)
                if len(tr["v"]) >= 6:                       # the track's own lean (px per row, going up)
                    tr["slope"] = float(np.polyfit(tr["v"][-12:], tr["c"][-12:], 1)[0])
        for tr in list(tracks):
            if tr["v"][-1] - vrow > TRACK_GAP:
                tracks.remove(tr)
                done.append(tr)
        for j, (a0, a1) in enumerate(runs):
            if j not in used:
                tracks.append({"v": [vrow], "c": [0.5 * (a0 + a1)], "w": [a1 - a0 + 1], "slope": 0.0})
    done += tracks
    pieces = []
    for tr in done:
        if len(tr["v"]) < 12:
            continue
        vs, cs, ws = np.array(tr["v"], float), np.array(tr["c"], float), np.array(tr["w"], float)
        bb, aa = np.polyfit(vs, cs, 1)
        if np.degrees(np.arctan(abs(bb))) > BAND_MAX_TILT:
            continue
        pieces.append({"vs": vs, "cs": cs, "ws": ws})
    # pieces of one trunk (leaves in front): joined when their centre lines agree
    def line(pc):
        bb, aa = np.polyfit(pc["vs"], pc["cs"], 1)
        return aa, bb
    depth = depth_mm.astype(float) * 0.001

    def _piece_depth(pc):
        ui = np.clip(np.round(pc["cs"]).astype(int), 0, W - 1)
        dd = depth[pc["vs"].astype(int), ui]
        dd = dd[(dd > DEPTH_RANGE[0]) & (dd < DEPTH_RANGE[1])]
        return float(np.median(dd)) if dd.size else 9.0
    joined = True
    while joined:
        joined = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                ai, bi = line(pieces[i])
                aj, bj = line(pieces[j])
                pi, pj = pieces[i], pieces[j]
                di = abs((ai + bi * np.median(pj["vs"])) - np.median(pj["cs"]))
                dj = abs((aj + bj * np.median(pi["vs"])) - np.median(pi["cs"]))
                wi, wj = np.median(pi["ws"]), np.median(pj["ws"])
                if max(di, dj) < max(BAND_JOIN_PX, 0.7 * max(wi, wj)) and abs(np.degrees(np.arctan(bi) - np.arctan(bj))) < 20 \
                        and 0.4 <= wi / wj <= 2.5 and abs(_piece_depth(pi) - _piece_depth(pj)) < 0.15:
                    pieces[i] = {k2: np.r_[pi[k2], pj[k2]] for k2 in ("vs", "cs", "ws")}
                    pieces.pop(j)
                    joined = True
                    break
            if joined:
                break
    depth = depth_mm.astype(float) * 0.001
    bands = []
    for pc in pieces:
        vs, cs, ws = pc["vs"], pc["cs"], pc["ws"]
        bb, aa = np.polyfit(vs, cs, 1)
        v0, v1 = float(vs.min()), float(vs.max())
        half_w = 0.5 * float(np.median(ws)) / np.sqrt(1 + bb * bb)
        length = (v1 - v0) * np.sqrt(1 + bb * bb)
        if length < BAND_MIN_LEN or length / (2 * half_w) < BAND_MIN_ASPECT:
            continue
        ui = np.clip(np.round(cs).astype(int), 0, W - 1)
        dd = depth[vs.astype(int), ui]
        dd = dd[(dd > DEPTH_RANGE[0]) & (dd < DEPTH_RANGE[1])]
        if dd.size < 8:
            continue
        dz = float(np.median(dd))
        if dz > TRUNK_MAX_DEPTH:
            continue
        band = {"a": float(aa), "b": float(bb), "v0": v0, "v1": v1, "half_w": half_w, "depth": dz, "n": int(len(vs)),
                "two": None}
        if len(vs) > 60:                                    # a knee: the lower and upper halves leaning differently
            vm = float(np.median(vs))
            lo, hi = vs >= vm, vs < vm
            bl, al = np.polyfit(vs[lo], cs[lo], 1)
            bh, ah = np.polyfit(vs[hi], cs[hi], 1)
            rms1 = float(np.sqrt(np.mean((cs - (aa + bb * vs)) ** 2)))
            rms2 = float(np.sqrt(np.mean(np.r_[cs[lo] - (al + bl * vs[lo]), cs[hi] - (ah + bh * vs[hi])] ** 2)))
            if abs(np.degrees(np.arctan(bl) - np.arctan(bh))) > 10 and rms2 < 0.6 * rms1:
                band["two"] = ((float(al), float(bl)), (float(ah), float(bh)), vm)
        bands.append(band)
    bands.sort(key=lambda bd: -bd["n"])
    return bands


#: the band is sampled in bins of this many rows for its depth (the camera looks down: the depth changes along it)
BIN_ROWS = 8


#: at most this many trunks (trees) in the scene: per picture and in the map (user 2026-10-20: three trees now;
#: toy_harvest --max_trunks)
MAX_TRUNKS = 3


def detect_trunks(rgb, depth_mm, proj, fr, floor_z, mask=None, max_trunks=None):
    """Every trunk the picture shows -> [{"pts" (axis points, pelvis frame, from the floor up), "r", "n"}]: the
    bands found by colour (detect_trunk_bands); along each, per BIN_ROWS rows, the median depth of the band's own
    pixels places its centre in 3D (the depth only places / checks: a bin without depth is skipped), x and y fitted
    as lines of the height (one, or two at the band's knee), the lean capped, the axis pushed the trunk's radius
    behind its visible front, extended down to the floor along its lowest part."""
    max_trunks = MAX_TRUNKS if max_trunks is None else int(max_trunks)
    depth = depth_mm.astype(float) * 0.001
    H, W = depth.shape
    cam = fr.torso_R @ proj.t + fr.torso_pos
    out = []
    for bd in detect_trunk_bands(rgb, depth_mm, mask)[:max_trunks + 4]:
        two = bd["two"]
        P3, vk = [], (None if two is None else two[2])
        for v0 in np.arange(bd["v0"], bd["v1"] + 1, BIN_ROWS):
            vc = v0 + 0.5 * BIN_ROWS
            ab = (bd["a"], bd["b"]) if two is None else (two[0] if vc >= two[2] else two[1])
            uc = ab[0] + ab[1] * vc
            hw = max(2, int(round(bd["half_w"])))
            ys = np.arange(int(v0), int(min(H, v0 + BIN_ROWS)))
            xs = np.arange(int(max(0, uc - hw)), int(min(W, uc + hw + 1)))
            if len(ys) == 0 or len(xs) == 0:
                continue
            dd = depth[np.ix_(ys, xs)].ravel()
            dd = dd[(dd > DEPTH_RANGE[0]) & (dd < DEPTH_RANGE[1])]
            if dd.size < 6:
                continue
            z = float(np.median(dd))
            p = fr.torso_R @ _backproject(proj, np.array([uc]), np.array([vc]), np.array([z]))[0] + fr.torso_pos
            P3.append((vc, p))
        if len(P3) < 4:
            continue
        V = np.array([q[0] for q in P3])
        Q = np.array([q[1] for q in P3])
        if float(np.median(np.linalg.norm(Q - cam, axis=1))) > TRUNK_MAX_DEPTH + 0.15:
            continue
        zq = Q[:, 2]
        z_lo, z_hi = float(zq.min()), float(zq.max())
        if z_hi - z_lo < MIN_EXTENT:
            continue
        # the TRAINING trunk: vertical from the floor to a kink, then one part leaning <= 15 deg from it
        base, kink, lean, rms, keep = fit_training_trunk(Q, floor_z)
        if rms > 0.05 or keep.sum() < 4:                        # does not follow a trunk's axis
            continue
        z_hi = float(zq[keep].max())
        pts = training_trunk_pts(base, kink, lean, floor_z, z_hi)
        zc_mean = float(np.median(np.linalg.norm(Q - cam, axis=1)))
        r = float(np.clip(bd["half_w"] * zc_mean / CAM["fx"], *TRUNK_R))
        view = np.mean(pts, axis=0) - cam
        view[2] = 0.0
        view /= max(np.linalg.norm(view), 1e-9)
        pts = [np.asarray(p, float) + r * view for p in pts]
        out.append({"pts": pts, "r": r, "n": bd["n"], "depth": zc_mean, "kink_m": float(kink - floor_z),
                    "lean_deg": float(np.degrees(np.arctan(np.linalg.norm(lean)))), "rms": rms})
    # one trunk seen as two bands (leaves across it): the same 3D axis -> keep the longer one
    keep = []
    for t in out:
        mid = 0.5 * (t["pts"][1] + t["pts"][-1])
        if all(trunk_axis_dist(k, mid)[0] > 0.06 for k in keep):
            keep.append(t)
    return keep[:max_trunks]                                   # the bands come longest first


#: THE TRAINING TRUNK (envs/apple_reach_spindle_env_cfg.SPINDLE, tree_sampler.trunk_axis): vertical from the ground
#: up to a kink (m above the floor), then ONE straight part from the kink, leaning 3-15 deg in any direction:
#: axis xy(z) = base + lean * max(z - kink, 0). The real trunks are fitted with exactly this model (user 2026-10-20).
KINK_RANGE = (0.5, 0.9)
LEAN_DEG_MAX = 15.0
#: the kink is searched on this grid (m); an upper part shorter than this (m) over the kink cannot show its lean
KINK_STEP, LEAN_MIN_SPAN = 0.02, 0.10


def fit_training_trunk(Q, floor_z, iters=3):
    """Trunk centre points Q (pelvis frame, N x 3) -> (base xy [2], kink z (pelvis), lean [2] (dx/dz, dy/dz), rms m,
    keep mask) with the training model: x, y = base + lean * max(z - kink, 0), the lower part VERTICAL, the kink in
    KINK_RANGE above the floor, |lean| <= tan(LEAN_DEG_MAX). Points not above the kink by LEAN_MIN_SPAN: a vertical
    trunk (no lean seen), the kink at the visible top / the range's middle."""
    Q = np.asarray(Q, float)
    z = Q[:, 2]
    lean_max = math.tan(math.radians(LEAN_DEG_MAX))
    keep = np.ones(len(Q), bool)

    def solve(k, sel):
        h = np.maximum(z[sel] - k, 0.0)
        if float(h.max(initial=0.0)) < LEAN_MIN_SPAN:              # nothing seen above the kink: vertical
            base = np.median(Q[sel, :2], axis=0)
            lean = np.zeros(2)
        else:
            A = np.stack([np.ones_like(h), h], axis=1)
            coef, *_ = np.linalg.lstsq(A, Q[sel, :2], rcond=None)
            base, lean = coef[0], coef[1]
            n = float(np.linalg.norm(lean))
            if n > lean_max:                                       # capped: the base refitted with the capped lean
                lean = lean * lean_max / n
                base = np.mean(Q[sel, :2] - np.outer(h, lean), axis=0)
        res = np.linalg.norm(Q[:, :2] - (base + np.outer(np.maximum(z - k, 0.0), lean)), axis=1)
        return base, lean, res

    best = None
    for _ in range(iters):
        best = None
        for kf in np.arange(KINK_RANGE[0], KINK_RANGE[1] + 1e-9, KINK_STEP):
            k = floor_z + float(kf)
            base, lean, res = solve(k, keep)
            rms = float(np.sqrt(np.mean(res[keep] ** 2)))
            if best is None or rms < best[3] - 1e-4:
                best = (base, k, lean, rms, res)
        base, k, lean, rms, res = best
        thr = max(0.025, 2.5 * float(np.median(res[keep])))
        new = res < thr
        if new.sum() < 4 or np.array_equal(new, keep):
            break
        keep = new
    base, k, lean, rms, res = best
    if not np.any(lean):                                           # no lean seen: the kink where the view ends
        k = float(np.clip(z[keep].max(), floor_z + KINK_RANGE[0], floor_z + KINK_RANGE[1]))
    return np.asarray(base, float), float(k), np.asarray(lean, float), rms, keep


def training_trunk_pts(base, kink, lean, floor_z, z_top):
    """The axis polyline of a training-model trunk: the floor, the kink, the top (the kink only when above it)."""
    pts = [np.r_[base, floor_z]]
    if z_top > kink + 1e-3:
        pts += [np.r_[base, kink], np.r_[base + lean * (z_top - kink), z_top]]
    else:
        pts += [np.r_[base, max(z_top, floor_z + 0.05)]]
    return [np.asarray(p, float) for p in pts]


def apple_vs_trunks(trunks, apple_pelvis, r_apple, cam_pelvis):
    """The nearest trunk to the apple and the apple's side of it (seen from the robot) -> (index or None,
    'front' | 'left_of_trunk' | 'right_of_trunk' | 'far' | 'none', lateral m, gap m)."""
    if not trunks:
        return None, "none", 0.0, 0.0
    A = np.asarray(apple_pelvis, float)
    ds = [trunk_axis_dist(t, A) for t in trunks]
    k = int(np.argmin([d[0] for d in ds]))
    dist, foot = ds[k]
    r = trunks[k]["r"]
    gap = dist - r - float(r_apple)
    to_robot = cam_pelvis - foot
    to_robot[2] = 0.0
    to_robot /= max(np.linalg.norm(to_robot), 1e-9)
    left = np.cross([0.0, 0.0, 1.0], -to_robot)
    lateral = float((A - foot) @ left)
    if gap > NEAR_M:
        rel = "far"
    elif abs(lateral) <= r + FRONT_FRAC * float(r_apple):
        rel = "front"
    else:
        rel = "left_of_trunk" if lateral > 0 else "right_of_trunk"
    return k, rel, lateral, gap


def merge_trunk_map(old, new, tol=0.08, max_trunks=None):
    """The trunks seen so far (pelvis frame) updated with this picture's: a new one replaces an old one whose axis
    passes within tol (the newer view wins), the others stay; at most max_trunks (the newest views first)."""
    out = list(new)
    for t in old:
        mid = 0.5 * (t["pts"][0] + t["pts"][-1])
        if all(trunk_axis_dist(n, mid)[0] > tol for n in new):
            out.append(t)
    return out[:MAX_TRUNKS if max_trunks is None else int(max_trunks)]


def draw_trunks(rgb, proj, fr, trunks, apple_torso=None, r_apple=0.04, sel=None, title=""):
    """The picture with every trunk's outline (orange: the one beside the apple, yellow: the others)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(rgb, np.uint8))
    d = ImageDraw.Draw(im)
    for k, t in enumerate(trunks):
        col = (255, 120, 0) if k == sel else (255, 230, 80)
        for a, b in zip(t["pts"][:-1], t["pts"][1:]):
            pts = [a + f * (b - a) for f in np.linspace(0, 1, 10)]
            pix = []
            for q in pts:
                qt = fr.torso_R.T @ (q - fr.torso_pos)
                pix.append(proj.pixel(qt, t["r"]))
            pix = [x for x in pix if x[3] > 0.1]
            if len(pix) < 2:
                continue
            uv = np.array([[x[0], x[1]] for x in pix])
            dv = uv[-1] - uv[0]
            nrm = np.array([-dv[1], dv[0]]) / max(np.linalg.norm(dv), 1e-6)
            for sg in (-1, 1):
                d.line([tuple(uv[i] + sg * pix[i][2] * nrm) for i in range(len(pix))], fill=col, width=2)
    if apple_torso is not None:
        u, v, rp, _ = proj.pixel(apple_torso, r_apple)
        d.ellipse([u - rp, v - rp, u + rp, v + rp], outline=(0, 255, 255), width=2)
    d.rectangle([0, 0, im.width, 12], fill=(0, 0, 0))
    d.text((3, 0), title, fill=(255, 255, 0))
    return im


def evaluate_trunks(runs=None, out_name="trunks"):
    """Every recorded attempt (or the given runs): all trunks drawn + the apple's side -> sheets."""
    from PIL import Image
    sys.path.insert(0, HERE)
    import reach_io as R
    kin = R.G1Kinematics()
    P = Projector()
    floor = floor_z_pelvis(kin)
    rows = attempts()
    if runs:
        rows = [r for r in rows if r["run"] in runs]
    tiles, res = [], []
    for k, a in enumerate(rows):
        rgb = np.array(Image.open(a["color"]).convert("RGB"))
        dmm = np.array(Image.open(a["depth"]))
        w3 = np.asarray(a["waist"], float)
        fr = kin.frames(w3, R.HOME_Q)
        hm = hand_mask(P, hand_points_torso(kin, w3, np.asarray(a["arm_q"])), dmm.shape)
        trunks = detect_trunks(rgb, dmm, P, fr, floor, hm)
        apple_p = fr.torso_R @ np.asarray(a["apple_torso"]) + fr.torso_pos
        cam = fr.torso_R @ P.t + fr.torso_pos
        sel, rel, lat, gap = apple_vs_trunks(trunks, apple_p, a["radius"], cam)
        res.append(dict(idx=k, run=a["run"], episode=a["episode"], n_trunks=len(trunks), rel=rel, lateral=lat, gap=gap,
                        trunks=[{"pts": [p.tolist() for p in t["pts"]], "r": t["r"], "n": t["n"]} for t in trunks]))
        tiles.append(draw_trunks(rgb, P, fr, trunks, a["apple_torso"], a["radius"], sel,
                                 f"#{k} {len(trunks)} trunks; apple {rel} ({lat * 100:+.0f} cm, gap {gap * 100:.0f})"))
    os.makedirs(OUT_DIR, exist_ok=True)
    for s0 in range(0, len(tiles), 12):
        ts = tiles[s0:s0 + 12]
        W, H = ts[0].size
        sheet = Image.new("RGB", (W * 3, H * ((len(ts) + 2) // 3)), "white")
        for i, t in enumerate(ts):
            sheet.paste(t, ((i % 3) * W, (i // 3) * H))
        sheet.save(os.path.join(OUT_DIR, f"{out_name}_{s0 // 12:02d}.jpg"), quality=88)
    with open(os.path.join(OUT_DIR, f"{out_name}.jsonl"), "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    print(f"[trunks] {len(res)} attempts: trunks found per picture " +
          str(np.bincount([r['n_trunks'] for r in res]).tolist()) + "; apple: " +
          ", ".join(f"{k} {sum(r['rel'] == k for r in res)}" for k in ("front", "left_of_trunk", "right_of_trunk", "far", "none")))
    return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--trunks", nargs="*", default=None, help="every trunk per picture (the given runs, else all)")
    ap.add_argument("--max_trunks", type=int, default=MAX_TRUNKS, help="at most this many trunks (trees) per picture")
    a = ap.parse_args()
    MAX_TRUNKS = a.max_trunks
    if a.eval:
        evaluate(a.labels)
    if a.trunks is not None:
        evaluate_trunks(a.trunks or None)
