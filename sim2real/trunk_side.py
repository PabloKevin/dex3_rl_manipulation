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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    if a.eval:
        evaluate(a.labels)
