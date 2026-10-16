# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach calibration on the real robot: the math (pure numpy + reach_io's
pinocchio model, no robot, no Isaac Lab). Used by sim2real/dual/calib_reach.py
(record), sim2real/fit_reach_calib.py (fit) and run_toy.py single --calib
(apply).

What is calibrated. The single reach policy ends with its palm somewhere for
an apple it was told is at `apple_fed`. On the robot an operator then moves the
palm to where it SHOULD have ended (jog or hand-guided) and records it. The
error has many sources (camera calibration, radius / centre estimate, joint
offsets, sag, sim2real), but to first order the policy tracks the apple it is
fed one to one, so the correction is applied where it is cheapest: the apple
fed to the policy is shifted by Delta(apple), the palm's recorded error
(ideal - end) plus any shift already fed during that trial. Delta is fitted
as a constant, or constant + linear in the apple position (ridge) with enough
trials, in the PELVIS frame (the policy's own frame), and clipped.

Orientation errors are recorded and reported, not corrected: shifting the
apple cannot turn the palm.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

#: never shift the fed apple by more than this (m), whatever the fit says
MAX_SHIFT_M = 0.05
#: linear model only from this many trials on; below: the mean shift
MIN_TRIALS_LINEAR = 8
#: ridge on the linear part (per metre of apple offset), in m of shift
RIDGE = 0.02
#: outside the fitted apples' box grown by this, only the constant is applied
EXTRAPOLATE_M = 0.05


# ─────────────────────────────────────────────
#  Jog: damped least-squares IK on the palm point + orientation
# ─────────────────────────────────────────────

def rotvec_to_R(v) -> np.ndarray:
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def R_to_rotvec(R) -> np.ndarray:
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(a))
    return w * a


def palm_pose(kin, waist3, arm7, named=None):
    """(palm point [3], palm rotation [3, 3]) in the pelvis frame."""
    fr = kin.frames(waist3, arm7, named)
    return fr.palm_ref, fr.palm_R


def solve_palm_ik(kin, waist3, arm7, p_goal, R_goal, lo, hi, named=None, use_waist=False,
                  waist_lim=math.radians(45.0), iters=60, damp=0.02, rot_w=0.3, null_w=0.02):
    """Arm (and optionally waist yaw) joints putting the palm point at p_goal and the palm
    frame at R_goal, starting from (waist3, arm7) and pulled toward it in the null space:
    a small nudge stays a small joint change. Numerical Jacobian on reach_io's model.
    Returns (arm7, waist_yaw, pos_err_m, rot_err_rad)."""
    q0 = np.concatenate([np.asarray(arm7, float), [float(waist3[0])]])
    q = q0.copy()
    n = 8 if use_waist else 7
    lo8 = np.concatenate([np.asarray(lo, float), [-waist_lim]])
    hi8 = np.concatenate([np.asarray(hi, float), [waist_lim]])

    def pose(qq):
        w = np.array([qq[7], waist3[1], waist3[2]])
        return palm_pose(kin, w, qq[:7], named)

    for _ in range(iters):
        p, R = pose(q)
        e = np.concatenate([p_goal - p, rot_w * R_to_rotvec(R_goal @ R.T)])
        if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) / rot_w < 2e-3:
            break
        J = np.zeros((6, n))
        h = 1e-5
        for j in range(n):
            dq = np.zeros(8)
            dq[j] = h
            p2, R2 = pose(q + dq)
            J[:3, j] = (p2 - p) / h
            J[3:, j] = rot_w * R_to_rotvec(R2 @ R.T) / h
        JJt = J @ J.T + damp ** 2 * np.eye(6)
        Jp = J.T @ np.linalg.solve(JJt, np.eye(6))
        step = Jp @ e + (np.eye(n) - Jp @ J) @ (null_w * (q0[:n] - q[:n]))
        q[:n] = np.clip(q[:n] + np.clip(step, -0.05, 0.05), lo8[:n], hi8[:n])
    p, R = pose(q)
    return q[:7], float(q[7]), float(np.linalg.norm(p_goal - p)), float(np.linalg.norm(R_to_rotvec(R_goal @ R.T)))


# ─────────────────────────────────────────────
#  Float: a slip-spring the operator can push around
# ─────────────────────────────────────────────

class SlipSpring:
    """Hand-guiding through a position-only PD link (g1_dual's RL stream ignores
    feed-forward torque). The target is q_ref + g(q) / kp: the PD's own offset
    carries the arm's weight, so it stays where it is with a soft kp. q_ref slips
    toward the measured q only when the operator pushes it more than `deadband`
    away, at most `vmax`: the arm follows a firm push and stops where it is left,
    while a gravity-model error below kp * deadband only shows as a small static
    offset, never a drift."""

    def __init__(self, q_start, kp=30.0, kd=3.0, deadband=0.04, vmax=0.3, dt=0.02):
        self.q_ref = np.asarray(q_start, float).copy()
        self.kp, self.kd, self.db, self.vmax, self.dt = float(kp), float(kd), float(deadband), float(vmax), float(dt)

    def step(self, q_meas, g_torque):
        e = np.asarray(q_meas, float) - self.q_ref
        slip = np.sign(e) * np.maximum(np.abs(e) - self.db, 0.0)
        self.q_ref = self.q_ref + np.clip(slip, -self.vmax * self.dt, self.vmax * self.dt)
        return self.q_ref + np.asarray(g_torque, float) / self.kp


# ─────────────────────────────────────────────
#  Trials, fit, apply
# ─────────────────────────────────────────────

def file_sha1(path: str) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def append_trial(path: str, trial: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(trial) + "\n")


def load_trials(paths) -> list[dict]:
    rows = []
    for p in ([paths] if isinstance(paths, str) else paths):
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def trial_delta(t: dict) -> np.ndarray:
    """The shift (pelvis frame) that would have made this trial end at the ideal: the
    palm's error (ideal - end, measured) + what was already fed on top of the true apple."""
    fed = np.asarray(t["apple_fed"], float) - np.asarray(t["apple_true"], float)
    return np.asarray(t["palm_ideal"], float) - np.asarray(t["palm_end"], float) + fed


def fit(trials: list[dict], ridge: float = RIDGE, min_linear: int = MIN_TRIALS_LINEAR) -> dict:
    """Delta(apple) = b [+ A (apple - p0)], least squares with a ridge on A; the
    leave-one-out RMS of both models, the linear one only when it beats the constant."""
    P = np.array([t["apple_true"] for t in trials], float)
    D = np.array([trial_delta(t) for t in trials], float)
    n = len(trials)
    if n == 0:
        raise ValueError("no trials")
    p0 = P.mean(0)

    def fit_lin(Pi, Di):
        X = np.hstack([np.ones((len(Pi), 1)), Pi - p0])
        reg = np.diag([0.0, ridge, ridge, ridge]) * len(Pi)
        W = np.linalg.solve(X.T @ X + reg, X.T @ Di)
        return W[0], W[1:].T

    def loo(lin):
        if n < 2:
            return float("nan")
        err = []
        for i in range(n):
            m = np.arange(n) != i
            if lin:
                b, A = fit_lin(P[m], D[m])
                pred = b + A @ (P[i] - p0)
            else:
                pred = D[m].mean(0)
            err.append(np.sum((pred - D[i]) ** 2))
        return float(math.sqrt(np.mean(err)))

    loo_c = loo(False)
    out = {"frame": "pelvis", "n": n, "p0": p0.tolist(), "b": D.mean(0).tolist(), "A": np.zeros((3, 3)).tolist(),
           "model": "const", "loo_rms_m_const": loo_c, "max_shift_m": MAX_SHIFT_M,
           "box_lo": (P.min(0) - EXTRAPOLATE_M).tolist(), "box_hi": (P.max(0) + EXTRAPOLATE_M).tolist(),
           "raw_rms_m": float(math.sqrt(np.mean(np.sum(D ** 2, 1)))), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if n >= min_linear:
        loo_l = loo(True)
        out["loo_rms_m_linear"] = loo_l
        if loo_l < loo_c:
            b, A = fit_lin(P, D)
            out.update(model="linear", b=b.tolist(), A=A.tolist())
    return out


class ReachCalib:
    """Delta(apple) from a fit_reach_calib.py JSON: added to the apple fed to the policy."""

    def __init__(self, path: str):
        with open(path) as f:
            self.c = json.load(f)
        self.path = path
        self.b = np.asarray(self.c["b"], float)
        self.A = np.asarray(self.c["A"], float)
        self.p0 = np.asarray(self.c["p0"], float)
        self.lo, self.hi = np.asarray(self.c["box_lo"], float), np.asarray(self.c["box_hi"], float)
        self.max_shift = float(self.c.get("max_shift_m", MAX_SHIFT_M))

    def delta(self, apple_pelvis) -> np.ndarray:
        p = np.asarray(apple_pelvis, float)
        d = self.b.copy()
        if self.c["model"] == "linear" and np.all(p >= self.lo) and np.all(p <= self.hi):
            d = d + self.A @ (p - self.p0)
        n = float(np.linalg.norm(d))
        return d if n <= self.max_shift else d * (self.max_shift / n)

    def describe(self) -> str:
        return (f"{os.path.basename(self.path)}: {self.c['model']} from {self.c['n']} trials, constant "
                f"{np.round(self.b * 100, 1).tolist()} cm (pelvis), LOO RMS "
                f"{self.c.get('loo_rms_m_' + self.c['model'], float('nan')) * 100:.1f} cm")
