#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
harvest_moves.py -- the geometry of the autonomous harvest (toy_harvest.py) that is not the policy:
  HOME / READY  the arm tucked by the hip (where the waist turns) and the arm-pose policy's start; the
                scripted joint move between them goes through home_ready_via
  BASKET        fixed to the ground; the drop point at the deposit waist (reachable, pulled toward the torso
                axis) -- the arm-pose policy goes there, as to an apple
  (DETACH       toy_harvest replays the approach backwards -- no IK anywhere in the harvest)
The settings are in harvest_config.json. Check a change without the robot:
    ~/miniconda3/envs/g1_dex3_sim2real/bin/python sim2real/harvest_moves.py
numpy + reach_io's pinocchio model.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arm_pose_io as AP  # noqa: E402
import reach_io as R  # noqa: E402

CONFIG = os.path.join(HERE, "harvest_config.json")
#: reach_io's HOME_Q as shipped (the policy's start on the robot): READY when the config has none
READY_DEFAULT = np.array(R.HOME_Q, float).copy()
DEFAULTS = {
    "home_q": [0.3, -0.6, 0.8, 1.0, 0.0, 0.0, 0.0],
    "ready_q": None,
    "home_ready_via": [0.07, -0.78, 0.53, 0.95, 0.0, 0.0, 0.0],
    "deposit_start_q": [0.02, -1.11, 0.41, 0.49, -0.24, -0.26, 0.47],
    "left_scan_q": [0.19, 0.52, 0.29, 0.68, 0.14, 0.01, -0.01],
    "grasp_check": {"baseline_s": 0.3, "read_s": 0.3, "held_n": 0.3, "strong_n": 1.0, "empty_n": 0.1,
                    "travel_min": 0.15, "travel_empty": 0.85},
    "basket": {"from_shoulder_m": [0.0, -0.522, -0.049], "waist_range_deg": [-15.0, 0.0], "max_reach_m": 0.50,
               "accept_m": 0.06, "exclude_radius_m": 0.30},
    "moves": {"speed_rad_s": 0.6},
    "detach": {"pull_m": 0.08, "pull_speed_m_s": 0.03, "kp": 250.0, "kd": 14.0, "tau_cap_nm": 18.0,
               "stall_s": 4.0, "snap_dq": 0.6, "speed_stop": 3.0,
               "return_speed_stop": 2.5},
    "view": {"edge_frac": 0.12, "margin_px": 3, "image_w": 424, "fx": 303.1, "cx": 215.8, "labels": ["red_apple"]},
    "waist": {"lock_window_s": 0.3, "lock_range_deg": 0.3, "lock_tol_deg": 2.0, "lock_timeout_s": 1.5},
    "reach": {"map": "reach_map.npz", "palm_turn_cost_deg": 15.0, "waist_turn_cost_deg": 10.0},
}


def load_config(path: str | None = None) -> dict:
    """DEFAULTS, overridden by the file's values (keys starting with '_' are notes)."""
    cfg = json.loads(json.dumps(DEFAULTS))
    path = path or CONFIG
    if os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update({kk: vv for kk, vv in v.items() if not kk.startswith("_")})
            else:
                cfg[k] = v
    cfg["home_q"] = np.asarray(cfg["home_q"], float)
    cfg["ready_q"] = READY_DEFAULT.copy() if cfg.get("ready_q") is None else np.asarray(cfg["ready_q"], float)
    cfg["home_ready_via"] = [] if not cfg.get("home_ready_via") else [np.asarray(cfg["home_ready_via"], float)]
    cfg["deposit_start_q"] = None if cfg.get("deposit_start_q") is None else np.asarray(cfg["deposit_start_q"], float)
    cfg["left_scan_q"] = None if cfg.get("left_scan_q") is None else np.asarray(cfg["left_scan_q"], float)
    return cfg


# ─────────────────────────────────────────────
#  The basket
# ─────────────────────────────────────────────

def basket_point(kin: "R.G1Kinematics", cfg: dict, waist_roll_pitch=(0.0, 0.0)) -> np.ndarray:
    """The basket point, pelvis frame (fixed to the ground): the right shoulder + from_shoulder_m, the waist at 0."""
    fr = kin.frames([0.0, float(waist_roll_pitch[0]), float(waist_roll_pitch[1])], cfg["home_q"])
    return fr.shoulder_pos + fr.torso_R @ np.asarray(cfg["basket"]["from_shoulder_m"], float)


def basket_drop(kin: "R.G1Kinematics", cfg: dict, waist3):
    """With the waist at waist3: (drop point, u = the horizontal torso-axis -> basket direction, the basket
    point, how far the drop is short of it [m]). The drop is the basket point pulled along -u until it is
    max_reach_m from the right shoulder."""
    B = basket_point(kin, cfg, (waist3[1], waist3[2]))
    fr = kin.frames(waist3, cfg["home_q"])
    u = B - fr.torso_pos
    u[2] = 0.0
    dB = float(np.linalg.norm(u))
    u /= max(dB, 1e-9)
    reach = float(cfg["basket"]["max_reach_m"])
    if np.linalg.norm(B - fr.shoulder_pos) <= reach:
        return B.copy(), u, B, 0.0
    lo, hi = 0.0, dB
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if np.linalg.norm(B - m * u - fr.shoulder_pos) <= reach:
            hi = m
        else:
            lo = m
    return B - hi * u, u, B, hi


def gravity_target(kin, waist3, q, kp, named=None, clip=0.12):
    """The target that makes the arm sit AT q under kp: q + g / kp (its gravity sag added)."""
    g = kin.gravity_torque(waist3, q, named=named)
    return np.asarray(q, float) + np.clip(g / max(float(kp), 1.0), -clip, clip)


# ─────────────────────────────────────────────
#  Check the settings without the robot
# ─────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--picker_dir", default=os.path.expanduser("~/g1_fruits_picker/apple_picker"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    kin = R.G1Kinematics()
    try:
        sys.path.insert(0, a.picker_dir)
        import _paths  # noqa: F401
        import body_collision as BC
        line = BC.DEFAULT_MARGIN_M
    except Exception as e:
        BC, line = None, -0.011
        print(f"(no body cover check: {e})")
    yaws = np.radians([-45, -30, -15, 0, 15])

    def clear(q):
        return min(float(BC.self_clearance("right", np.asarray(q, float), waist=float(w))) for w in yaws) \
            if BC else float("nan")

    print(f"body cover: the contact line is {line * 100:+.1f} cm; worst over the scan's waist yaws -45..+15")
    for name in ("home_q", "ready_q"):
        q = cfg[name]
        fr = kin.frames([0, 0, 0], q)
        pk = fr.palm_pos + fr.palm_R @ AP.POCKET
        rel = fr.torso_R.T @ (pk - fr.torso_pos)
        print(f"  {name:8s} {np.round(q, 3).tolist()}  pocket (torso) {np.round(rel, 3).tolist()}, "
              f"{np.hypot(rel[0], rel[1]):.2f} m from the waist axis, cover {clear(q) * 100:+.1f} cm")
    pts = [cfg["home_q"], *cfg["home_ready_via"], cfg["ready_q"]]
    path = [p0 + f * (p1 - p0) for p0, p1 in zip(pts[:-1], pts[1:]) for f in np.linspace(0, 1, 12)]
    print(f"  HOME <-> READY{' via ' + str(np.round(pts[1], 2).tolist()) if len(pts) > 2 else ''}: cover worst "
          f"{min(clear(q) for q in path) * 100:+.1f} cm, "
          f"~{1.875 * sum(np.max(np.abs(b - c)) for b, c in zip(pts[:-1], pts[1:])) / cfg['moves']['speed_rad_s']:.1f} s "
          f"at {cfg['moves']['speed_rad_s']:g} rad/s")
    if cfg["deposit_start_q"] is not None:
        q = cfg["deposit_start_q"]
        path = [cfg["home_q"] + f * (q - cfg["home_q"]) for f in np.linspace(0, 1, 15)]
        fr = kin.frames([0, 0, 0], q)
        rel = fr.torso_R.T @ (fr.palm_pos + fr.palm_R @ AP.POCKET - fr.torso_pos)
        print(f"  deposit start {np.round(q, 2).tolist()}: pocket (torso) {np.round(rel, 3).tolist()}, cover "
              f"{clear(q) * 100:+.1f} cm; HOME -> it (joint line) worst {min(clear(x) for x in path) * 100:+.1f} cm")
    if cfg["left_scan_q"] is not None and BC is not None:
        ql = cfg["left_scan_q"]
        cl = min(float(BC.self_clearance("left", ql, waist=float(w))) for w in np.radians(np.arange(-46, 17, 2)))
        print(f"  left arm during the harvest {np.round(ql, 2).tolist()}: cover vs the body / legs worst {cl * 100:+.1f} cm "
              f"over waist -46..+16")
    b = cfg["basket"]
    B = basket_point(kin, cfg)
    fr0 = kin.frames([0, 0, 0], cfg["home_q"])
    print(f"basket point (pelvis) {np.round(B, 3).tolist()}: {np.round(fr0.torso_R.T @ (B - fr0.torso_pos), 3).tolist()} "
          f"from the torso at waist 0; deposit at a waist in {b['waist_range_deg']} deg (the nearest)")
    for wd in b["waist_range_deg"]:
        w3 = np.array([math.radians(float(wd)), 0.0, 0.0])
        drop, u, _, short = basket_drop(kin, cfg, w3)
        fr = kin.frames(w3, cfg["home_q"])
        rel = fr.torso_R.T @ (drop - fr.shoulder_pos)
        print(f"  waist {float(wd):+.0f} deg: the drop point {short * 100:.1f} cm short of the basket point (max reach "
              f"{float(b['max_reach_m']):.2f} m), from the shoulder (torso frame) {np.round(rel, 3).tolist()}")


if __name__ == "__main__":
    main()
