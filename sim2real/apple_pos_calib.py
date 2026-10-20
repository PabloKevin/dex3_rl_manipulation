#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
apple_pos_calib.py -- calibrate the PERCEIVED apple position from REAL grasps: perceived (the detector, pelvis frame)
-> where the apple really was. The truth signal: an apple that was HELD after the grasp sat in the grasp pocket,
so its centre was (to ~1-2 cm) where the pocket ended. Each sample = (perceived apple, pocket at the grasp) of
one held grasp; the correction is fitted on the residual (truth - perceived) in the TORSO frame (x forward, y
left, z up) as a function of the apple's place (its height relative to the right shoulder, its distance, the
picture row / column where the camera saw it).

  python3 sim2real/apple_pos_calib.py --collect           robot runs -> logs/sim2real/apple_calib/samples.jsonl
  python3 sim2real/apple_pos_calib.py --fit               the models + leave-one-out errors -> apple_calib.json
Models compared (leave-one-out over the samples): none, a constant offset, ridge-linear in the features, and a
small MLP (only when there are >= MLP_MIN samples: with fewer it cannot be validated). The fitted file is used
by toy_harvest (--apple_calib, default on when the file exists) ON TOP of the dashboard's offset.
New samples are added automatically by toy_harvest at every held grasp (the same record).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUNS = os.path.join(REPO, "logs", "sim2real", "toy_reach_single")
SESSIONS = os.path.expanduser("~/g1_fruits_picker/logs")
OUT_DIR = os.path.join(REPO, "logs", "sim2real", "apple_calib")
SAMPLES = os.path.join(OUT_DIR, "samples.jsonl")
CALIB = os.path.join(OUT_DIR, "apple_calib.json")
#: the head camera at 424 px (camera_info on the robot)
CAM = {"w": 424.0, "h": 240.0, "fx": 303.1, "fy": 303.0, "cx": 215.8, "cy": 123.3}
#: travel below this = held (the picker's GRASP_TRAVEL_FRAC_TH), for runs without the toy's held check
HELD_TRAVEL = 0.65
#: the correction is never larger than this (m)
MAX_SHIFT = 0.04
MLP_MIN = 40
RIDGE = 0.5


#: the fingertip pad spheres (envs/mdp/self_collision.HAND_SPHERES: the tip ones), link frame, and their radius
TIPS = [("right_hand_thumb_2_link", (0.0, 0.038, 0.0)), ("right_hand_index_1_link", (0.036, 0.0, 0.0)),
        ("right_hand_middle_1_link", (0.036, 0.0, 0.0))]
TIP_R = 0.015


def fingertip_centre(kin, waist3, arm_q, hand_q, r_apple, pocket):
    """The apple's centre from the closed hand (pelvis frame): the three fingertip pads (FK of the MEASURED
    finger angles) lie on its surface -> the point at r_apple + TIP_R from each, the one on the palm's side
    (started at the pocket, a weak pull toward it breaks the mirror ambiguity). -> (centre, rms of the fit m)."""
    import reach_io as R
    named = {f"right_hand_{n}_joint": float(v) for n, v in zip(R.DEX3_ORDER, hand_q)}
    kin.frames(waist3, arm_q, named)                 # sets data.oMf
    pts = []
    for link, off in TIPS:
        X = kin.data.oMf[kin.model.getFrameId(link)]
        pts.append(np.asarray(X.translation) + np.asarray(X.rotation) @ np.asarray(off))
    P = np.array(pts)
    d0 = float(r_apple) + TIP_R
    c = np.asarray(pocket, float).copy()
    for _ in range(50):
        v = c - P
        dist = np.linalg.norm(v, axis=1)
        r = dist - d0
        J = v / dist[:, None]
        Jr = np.vstack([J, 0.1 * np.eye(3)])
        rr = np.concatenate([r, 0.1 * (c - pocket)])
        step = np.linalg.lstsq(Jr, -rr, rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-5:
            break
    return c, float(np.sqrt(np.mean((np.linalg.norm(c - P, axis=1) - d0) ** 2)))


def features(rel_sh, uv):
    """[height vs the shoulder (m), horizontal distance from the shoulder (m), picture row (0..1), column (0..1)]."""
    return np.array([rel_sh[2], float(np.hypot(rel_sh[0], rel_sh[1])), uv[1] / CAM["h"], uv[0] / CAM["w"]])


def _real_sessions():
    out = []
    for s in sorted(glob.glob(os.path.join(SESSIONS, "session_*_toy"))):
        try:
            if "[robot]" in open(os.path.join(s, "robot_launch.log")).read():
                out.append(os.path.basename(s)[8:27])          # 2026-10-19_13-19-32
        except OSError:
            pass
    return out


def _real_runs():
    sess = _real_sessions()
    all_s = sorted(os.path.basename(s)[8:27] for s in glob.glob(os.path.join(SESSIONS, "session_*_toy")))
    runs = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*_robot"))):
        t = os.path.basename(d)[:19]
        prev = [s for s in all_s if s <= t]
        if prev and prev[-1] in sess:
            runs.append(d)
    return runs


def _outcomes(brief):
    """[(hh:mm:ss, held?)] in the order of the run's grasps: toy_harvest's 'apple N: picked / ...' lines, else the
    pose toy's 'grasp closed (fingers N% of the way)'."""
    out = []
    for line in brief:
        m = re.match(r"(\d\d:\d\d:\d\d) \[\w+\] apple \d+: (\w+)", line)
        if m:
            out.append((m.group(1), m.group(2) == "picked"))
            continue
        m = re.search(r"(\d\d:\d\d:\d\d).*grasp closed \(fingers (\d+)% of the way\)", line)
        if m:
            out.append((m.group(1), int(m.group(2)) / 100.0 < HELD_TRAVEL))
    return out


def collect():
    sys.path.insert(0, HERE)
    import arm_pose_io as AP
    import reach_io as R
    kin = R.G1Kinematics()
    rows = []
    for d in _real_runs():
        try:
            brief = open(os.path.join(d, "brief.log")).read().splitlines()
        except OSError:
            continue
        full = []
        try:
            full = open(os.path.join(d, "toy_full.log")).read().splitlines()
        except OSError:
            pass
        outs = _outcomes(brief) or _outcomes(full)
        for f in sorted(glob.glob(os.path.join(d, "onboard_*.npz"))):
            z = np.load(f, allow_pickle=True)
            m = json.loads(str(z["meta"]))
            st = m["start"]
            gl = st.get("goals") or []
            if not gl or gl[-1]["name"] != "apple" or len(z["q"]) < 5:
                continue
            ep_t = st["id"][9:]                                   # HHMMSS
            ep_hms = f"{ep_t[:2]}:{ep_t[2:4]}:{ep_t[4:6]}"
            nxt = [o for o in outs if o[0] >= ep_hms]
            if not nxt:
                continue
            held = nxt[0][1]
            w0 = np.asarray(z["waist"][0], float)
            w1 = np.asarray(z["waist"][-1], float)
            fr0 = kin.frames(w0, R.HOME_Q)
            perceived = np.asarray(st["apple"], float)
            fr1 = kin.frames(w1, np.asarray(z["q"][-1], float))
            pocket = AP.hand_frame(fr1)[0]
            rel_sh = fr0.torso_R.T @ (perceived - fr0.shoulder_pos)
            c = fr0.camera_R.T @ (perceived - fr0.camera_pos)
            uv = [CAM["cx"] - CAM["fx"] * c[1] / c[0], CAM["cy"] - CAM["fy"] * c[2] / c[0]]
            rows.append({"run": os.path.basename(d), "episode": st["id"], "held": bool(held),
                         "perceived_pelvis": perceived.tolist(), "pocket_pelvis": pocket.tolist(),
                         "goal_pelvis": gl[-1]["target"], "waist": w0.tolist(),
                         "residual_torso": (fr0.torso_R.T @ (pocket - perceived)).tolist(),
                         "rel_shoulder_torso": rel_sh.tolist(), "uv": uv,
                         "outcome": m["result"].get("outcome", ""), "src": "logs"})
    os.makedirs(OUT_DIR, exist_ok=True)
    keep = []
    if os.path.exists(SAMPLES):                                   # the toy's own records stay
        keep = [json.loads(l) for l in open(SAMPLES) if l.strip() and json.loads(l).get("src") == "toy"]
    with open(SAMPLES, "w") as fh:
        for r in keep + rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[collect] {len(rows)} grasps from {len(_real_runs())} real runs ({sum(r['held'] for r in rows)} held) "
          f"+ {len(keep)} toy records -> {SAMPLES}")


# ─────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────

def _ridge_fit(X, Y, lam):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    Rg = lam * np.eye(Xa.shape[1])
    Rg[-1, -1] = 0.0
    return np.linalg.solve(Xa.T @ Xa + Rg, Xa.T @ Y)


def _ridge_pred(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def _mlp_fit(X, Y, hidden=8, iters=3000, lr=0.01, wd=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.5, (X.shape[1], hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.1, (hidden, Y.shape[1])); b2 = Y.mean(0)
    for _ in range(iters):
        H = np.tanh(X @ W1 + b1)
        P = H @ W2 + b2
        G = 2 * (P - Y) / len(X)
        gW2 = H.T @ G + wd * W2; gb2 = G.sum(0)
        GH = (G @ W2.T) * (1 - H ** 2)
        gW1 = X.T @ GH + wd * W1; gb1 = GH.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return (W1, b1, W2, b2)


def _mlp_pred(M, X):
    W1, b1, W2, b2 = M
    return np.tanh(X @ W1 + b1) @ W2 + b2


def fit(out=CALIB):
    rows = [json.loads(l) for l in open(SAMPLES) if l.strip()]
    tips = [r for r in rows if r.get("held") and r.get("truth_pelvis") is not None and r.get("tip_fit_rms_m", 1) < 0.01]
    if len(tips) >= 6:
        # the fingertip truth (toy_harvest records): residual = the apple's centre from the closed hand - perceived
        for r in tips:
            r["residual_torso"] = r["residual_truth_torso"]
        S = tips
        print(f"truth: the closed hand's fingertips ({len(S)} grasps)")
    else:
        S = [r for r in rows if r["held"]]
        print(f"truth: the pocket at the grasp (WEAK: a held apple may sit 2-4 cm off the pocket; only {len(tips)} "
              f"fingertip records, >= 6 needed)")
    if len(S) < 3:
        sys.exit(f"{len(S)} held grasps: too few")
    X = np.array([features(np.asarray(r["rel_shoulder_torso"]), r["uv"]) for r in S])
    Y = np.array([r["residual_torso"] for r in S])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    print(f"{len(S)} held grasps (of {len(rows)}): residual = pocket at the grasp - perceived, torso frame [cm]")
    print("   height vs shoulder  dist  row  col  |  residual x / y / z  |  run / episode / outcome")
    for x, y, r in zip(X, Y, S):
        print(f"   {x[0] * 100:+6.1f} cm  {x[1] * 100:5.1f}  {x[2]:.2f} {x[3]:.2f}  |  {y[0] * 100:+5.1f} {y[1] * 100:+5.1f} "
              f"{y[2] * 100:+5.1f}  |  {r['run'][:19]} {r['episode'][9:]} {r['outcome'][:28]}")
    n = len(S)
    loo = {"none": [], "constant": [], "linear": [], "mlp": []}
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        loo["none"].append(Y[i])
        loo["constant"].append(Y[i] - Y[tr].mean(0))
        if n >= 6:
            W = _ridge_fit(Xn[tr], Y[tr], RIDGE)
            loo["linear"].append(Y[i] - _ridge_pred(W, Xn[i:i + 1])[0])
        if n >= MLP_MIN:
            M = _mlp_fit(Xn[tr], Y[tr])
            loo["mlp"].append(Y[i] - _mlp_pred(M, Xn[i:i + 1])[0])
    print("\nleave-one-out error of the corrected apple (cm):    3D RMS   z RMS   z mean")
    best, best_e = "none", 1e9
    for k, v in loo.items():
        if not v:
            print(f"   {k:9s} (needs {'6' if k == 'linear' else MLP_MIN} samples)")
            continue
        v = np.array(v)
        e3, ez = float(np.sqrt((v ** 2).sum(1).mean())), float(np.sqrt((v[:, 2] ** 2).mean()))
        print(f"   {k:9s}                                    {e3 * 100:5.2f}   {ez * 100:5.2f}   {v[:, 2].mean() * 100:+5.2f}")
        if e3 < best_e - 1e-4:
            best, best_e = k, e3
    model = {"model": best, "n": n, "mu": mu.tolist(), "sd": sd.tolist(), "loo_rms_cm": best_e * 100,
             "features": ["height_vs_shoulder_m", "dist_from_shoulder_m", "row_0_1", "col_0_1"]}
    if best == "constant":
        model["b"] = Y.mean(0).tolist()
    elif best == "linear":
        model["W"] = _ridge_fit(Xn, Y, RIDGE).tolist()
    elif best == "mlp":
        model["mlp"] = [a.tolist() for a in _mlp_fit(Xn, Y)]
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(model, open(out, "w"), indent=1)
    print(f"-> {out}: '{best}'")


class AppleCalib:
    """The fitted correction: perceived apple (pelvis) -> corrected (pelvis). Clipped to MAX_SHIFT."""

    def __init__(self, path=CALIB):
        self.m = json.load(open(path))

    def shift_torso(self, rel_sh, uv):
        m = self.m
        if m["model"] == "none":
            return np.zeros(3)
        if m["model"] == "constant":
            d = np.asarray(m["b"], float)
        else:
            x = (features(np.asarray(rel_sh, float), uv) - np.asarray(m["mu"])) / np.asarray(m["sd"])
            if m["model"] == "linear":
                d = _ridge_pred(np.asarray(m["W"]), x[None])[0]
            else:
                d = _mlp_pred(tuple(np.asarray(a) for a in m["mlp"]), x[None])[0]
        nrm = float(np.linalg.norm(d))
        return d * (MAX_SHIFT / nrm) if nrm > MAX_SHIFT else d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    if a.fit:
        fit()
