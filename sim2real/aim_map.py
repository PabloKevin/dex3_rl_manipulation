# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""The aim map on the robot (envs/mdp/aim.py WITHOUT torch): the waist yaw and the palm's approach direction for
an apple, chosen before the reach. numpy only, Python 3.8.

    Q(apple relative to the right shoulder in the torso frame, palm heading, palm tilt, waist yaw)
        -> P(the AppleArmPose-v5 reach ends on the fruit, settled, clear of the body, without sweeping it)

fitted on the frozen arm policy's own rollouts (scripts/build_aim_map.py; --fit_only <rollouts.npz> rewrites the
map and its .npz copy for this module). choose() tries the waist yaws of the map's range at which the head camera
still sees the apple whole, with 7 palm headings x 5 tilts each, and returns the best -- a rotation about the
waist axis and ~700 evaluations of a small MLP: no kinematics solved at run time.
Q near 0 for every option = the apple is out of this arm's reach from here (too far / too low): skip it.

Check against the torch version: python sim2real/aim_map.py --check envs/assets/aim_map_v5.pt
"""
from __future__ import annotations

import math

import numpy as np


class AimMap:
    def __init__(self, path: str):
        d = np.load(path, allow_pickle=True)
        self.W = [d[f"W{i}"] for i in range(int(d["n_layers"]))]
        self.b = [d[f"b{i}"] for i in range(int(d["n_layers"]))]
        self.mean, self.std = d["mean"], d["std"]
        self.shoulder = d["shoulder_torso"]
        self.cam_A, self.cam_t = d["cam_A"], d["cam_t"]
        self.cam = {k: float(d[f"cam_{k}"]) for k in ("fx", "fy", "cx", "cy", "width", "height", "radius", "margin_px")}
        self.z_range = tuple(float(v) for v in d["cam_z_range"])
        self.yaw_range_deg = tuple(float(v) for v in d["yaw_range_deg"])

    def q(self, rel, heading, tilt, yaw) -> np.ndarray:
        x = np.concatenate([rel, heading[..., None], tilt[..., None], yaw[..., None]], axis=-1)
        h = (x - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if i < len(self.W) - 1:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0.0)))          # ELU
        return 1.0 / (1.0 + np.exp(-h[..., 0]))

    def sees(self, p_torso) -> np.ndarray:
        c = self.cam
        pc = (p_torso - self.cam_t) @ self.cam_A
        x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
        zz = np.maximum(z, 1e-6)
        u = c["cx"] + c["fx"] * x / zz
        v = c["cy"] - c["fy"] * y / zz
        rp = c["fx"] * c["radius"] / zz + c["margin_px"]
        return (z >= self.z_range[0]) & (z <= self.z_range[1]) & (u - rp >= 0) & (u + rp <= c["width"]) \
            & (v - rp >= 0) & (v + rp <= c["height"])

    def choose(self, p_axis, yaw_now: float, turn_waist: bool = True, yaw_step_deg: float = 5.0,
               headings_deg=(-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0), tilts_deg=(-30.0, -15.0, 0.0, 15.0, 30.0),
               yaw_cost: float = 0.06, plain_cost: float = 0.04, need_view: bool = True) -> dict:
        """p_axis [3]: apple - torso origin in the PELVIS axes. -> {"yaw", "heading", "tilt" (rad), "q", "q_now",
        "normal_torso" [3] (palm normal in the torso frame once the waist is at yaw), "seen"}."""
        p_axis = np.asarray(p_axis, float)
        lo, hi = self.yaw_range_deg
        ys = np.radians(np.arange(lo, hi + 1e-6, yaw_step_deg))
        yaw = np.concatenate([ys, [float(yaw_now)]])
        hs, ts = np.radians(headings_deg), np.radians(tilts_deg)
        c, s = np.cos(-yaw), np.sin(-yaw)
        p_t = np.stack([c * p_axis[0] - s * p_axis[1], s * p_axis[0] + c * p_axis[1], np.full_like(yaw, p_axis[2])], -1)
        rel = p_t - self.shoulder
        seen = self.sees(p_t) if need_view else np.ones(len(yaw), bool)
        Y, H, T = len(yaw), len(hs), len(ts)
        relg = np.broadcast_to(rel[:, None, None, :], (Y, H, T, 3))
        hg = np.broadcast_to(hs[None, :, None], (Y, H, T))
        tg = np.broadcast_to(ts[None, None, :], (Y, H, T))
        yg = np.broadcast_to(yaw[:, None, None], (Y, H, T))
        q = self.q(relg, hg, tg, yg)
        score = q - yaw_cost * np.abs(yg - yaw_now) / math.radians(45.0) \
            - plain_cost * (np.abs(hg) / math.radians(45.0) + np.abs(tg) / math.radians(30.0))
        ok = np.broadcast_to(seen[:, None, None], (Y, H, T)).copy()
        ok[-1] = True
        if not turn_waist:
            ok[:-1] = False
        score = np.where(ok, score, -1e9)
        iy, ih, it = np.unravel_index(int(np.argmax(score)), score.shape)
        u = p_t[iy].copy()
        u[2] = 0.0
        u /= max(float(np.linalg.norm(u)), 1e-6)
        ch, sh = math.cos(hs[ih]), math.sin(hs[ih])
        n = np.array([ch * u[0] - sh * u[1], sh * u[0] + ch * u[1], 0.0]) * math.cos(ts[it])
        n[2] = math.sin(ts[it])
        return {"yaw": float(yaw[iy]), "heading": float(hs[ih]), "tilt": float(ts[it]), "q": float(q[iy, ih, it]),
                "q_now": float(q[-1].max()), "normal_torso": n, "seen": bool(seen[iy])}


def export_npz(pt_path: str, npz_path: str | None = None) -> str:
    """envs/assets/aim_map_v5.pt (torch) -> the .npz this module loads."""
    import torch

    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = d["state_dict"]
    keys = sorted({int(k.split(".")[0]) for k in sd})
    out = {"n_layers": len(keys), "mean": d["mean"].numpy(), "std": d["std"].numpy(),
           "shoulder_torso": d["shoulder_torso"].numpy(), "cam_A": np.asarray(d["camera"]["A"], float),
           "cam_t": np.asarray(d["camera"]["t"], float), "cam_z_range": np.asarray(d["camera"]["z_range"], float),
           "yaw_range_deg": np.asarray(d["yaw_range_deg"], float)}
    for i, k in enumerate(keys):
        out[f"W{i}"], out[f"b{i}"] = sd[f"{k}.weight"].numpy(), sd[f"{k}.bias"].numpy()
    for k in ("fx", "fy", "cx", "cy", "width", "height", "radius", "margin_px"):
        out[f"cam_{k}"] = float(d["camera"][k])
    npz_path = npz_path or pt_path[:-3] + ".npz"
    np.savez(npz_path, **out)
    return npz_path


if __name__ == "__main__":
    import argparse
    import importlib.util
    import os

    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--check", required=True, help="the torch map (.pt): export the .npz and compare both")
    a = ap_.parse_args()
    import torch

    npz = export_npz(a.check)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("aim_t", os.path.join(here, "envs", "mdp", "aim.py"))
    aim_t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aim_t)
    T, Nmap = aim_t.load(a.check), AimMap(npz)
    rng = np.random.default_rng(0)
    worst, same = 0.0, 0
    for _ in range(300):
        p = rng.uniform([0.2, -0.45, -0.3], [0.65, 0.3, 0.6])
        y0 = float(rng.uniform(-0.7, 0.25))
        r1 = Nmap.choose(p, y0)
        r2 = T.choose(torch.tensor(p[None], dtype=torch.float32), torch.tensor([y0]))
        worst = max(worst, abs(r1["q"] - float(r2["q"][0])))
        same += int(abs(r1["yaw"] - float(r2["yaw"][0])) < 1e-4 and abs(r1["heading"] - float(r2["heading"][0])) < 1e-4
                    and abs(r1["tilt"] - float(r2["tilt"][0])) < 1e-4)
    print(f"[aim_map] {npz}: 300 random apples, same choice as torch in {same}, largest |dQ| {worst:.2e}")
