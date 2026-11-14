# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Aim map: the WAIST yaw and the palm's approach direction for an apple, chosen before the reach.

User, 2026-11-13: "the waist is really important ... maybe another policy in the loop just controlling the
waist, so as to give a proper position previous to reaching" (the reaches that drove the waist themselves did
not converge). Choosing the waist yaw is ONE decision per apple, taken before the arm moves; what makes a yaw
good is whether the arm policy then reaches the fruit. So the waist "policy" is a value function learned from
the arm policy's own rollouts, and an arg-max over a handful of candidates:

    Q(apple relative to the shoulder in the torso frame, palm heading, palm tilt, waist yaw) -> P(the reach ends
    on the goal, settled, clear of the body, without sweeping the fruit)

scripts/build_aim_map.py rolls the frozen arm policy out on random goals (robot only, thousands of envs) and
fits Q (a small MLP). choose() turns the torso (candidate yaws) -- the apple's position in the torso frame is a
rotation about the waist axis, no kinematics solved -- keeps the yaws at which the head camera still sees the
apple whole, and returns the (yaw, heading, tilt) with the best Q, preferring the smallest waist move and the
plainest approach among near-equals. Nothing here needs the simulator: torch only (the robot stack can load it).
"""

from __future__ import annotations

import math

import torch


class AimMap:
    def __init__(self, data: dict, device="cpu"):
        self.device = torch.device(device)
        dims = data["dims"]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers).to(self.device)
        self.net.load_state_dict(data["state_dict"])
        self.net.eval()
        self.mean = torch.as_tensor(data["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(data["std"], dtype=torch.float32, device=self.device)
        self.shoulder = torch.as_tensor(data["shoulder_torso"], dtype=torch.float32, device=self.device)
        cam = data["camera"]
        self.cam = cam
        self.cam_A = torch.as_tensor(cam["A"], dtype=torch.float32, device=self.device)
        self.cam_t = torch.as_tensor(cam["t"], dtype=torch.float32, device=self.device)
        self.yaw_range = data.get("yaw_range_deg", (-45.0, 15.0))
        self.meta = {k: v for k, v in data.items() if k not in ("state_dict",)}

    @torch.no_grad()
    def q(self, rel: torch.Tensor, heading: torch.Tensor, tilt: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """[...] success probability. rel [..., 3] goal - right shoulder (torso frame), angles in rad."""
        x = torch.cat([rel, heading.unsqueeze(-1), tilt.unsqueeze(-1), yaw.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.net((x - self.mean) / self.std)).squeeze(-1)

    def sees(self, p_torso: torch.Tensor) -> torch.Tensor:
        """[...] bool: the head camera sees an apple at p_torso [..., 3] (torso frame) whole."""
        c = self.cam
        pc = (p_torso - self.cam_t) @ self.cam_A
        x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
        zz = z.clamp(min=1e-6)
        u = c["cx"] + c["fx"] * x / zz
        v = c["cy"] - c["fy"] * y / zz
        rp = c["fx"] * c["radius"] / zz + c["margin_px"]
        lo, hi = c["z_range"]
        return (z >= lo) & (z <= hi) & (u - rp >= 0) & (u + rp <= c["width"]) & (v - rp >= 0) & (v + rp <= c["height"])

    @torch.no_grad()
    def choose(self, p_axis: torch.Tensor, yaw_now: torch.Tensor, turn_waist: bool = True,
               yaw_step_deg: float = 5.0, headings_deg=(-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0),
               tilts_deg=(-30.0, -15.0, 0.0, 15.0, 30.0), yaw_cost: float = 0.06, plain_cost: float = 0.04,
               need_view: bool = True) -> dict:
        """p_axis [n, 3]: apple - torso origin in the PELVIS axes (x forward, z up; the waist axis is the
        vertical through the torso origin). yaw_now [n] rad. Returns dict of [n] tensors: yaw (rad), heading,
        tilt (rad), q (the chosen option's success probability), q_now (the best at the current yaw), and
        normal_torso [n, 3] (the palm normal in the torso frame once the waist is at yaw).
        Score = Q - yaw_cost x |turn| / 45 deg - plain_cost x (|heading| / 45 deg + |tilt| / 30 deg)."""
        dev = self.device
        p_axis = p_axis.to(dev).float()
        yaw_now = yaw_now.to(dev).float()
        n = p_axis.shape[0]
        lo, hi = self.yaw_range
        ys = torch.deg2rad(torch.arange(lo, hi + 1e-6, yaw_step_deg, device=dev))
        hs = torch.deg2rad(torch.tensor(headings_deg, device=dev))
        ts = torch.deg2rad(torch.tensor(tilts_deg, device=dev))
        Y, H, T = len(ys), len(hs), len(ts)
        yaw = torch.cat([ys.view(1, Y).expand(n, Y), yaw_now.view(n, 1)], dim=1)          # [n, Y+1] (last: now)
        c, s = torch.cos(-yaw), torch.sin(-yaw)
        px = c * p_axis[:, :1] - s * p_axis[:, 1:2]
        py = s * p_axis[:, :1] + c * p_axis[:, 1:2]
        p_t = torch.stack([px, py, p_axis[:, 2:3].expand(n, Y + 1)], dim=-1)               # [n, Y+1, 3] torso frame
        rel = p_t - self.shoulder
        seen = self.sees(p_t) if need_view else torch.ones(n, Y + 1, dtype=torch.bool, device=dev)
        relg = rel.view(n, Y + 1, 1, 1, 3).expand(n, Y + 1, H, T, 3)
        hg = hs.view(1, 1, H, 1).expand(n, Y + 1, H, T)
        tg = ts.view(1, 1, 1, T).expand(n, Y + 1, H, T)
        yg = yaw.view(n, Y + 1, 1, 1).expand(n, Y + 1, H, T)
        q = self.q(relg, hg, tg, yg)                                                        # [n, Y+1, H, T]
        turn = (yg - yaw_now.view(n, 1, 1, 1)).abs()
        score = q - yaw_cost * turn / math.radians(45.0) \
            - plain_cost * (hg.abs() / math.radians(45.0) + tg.abs() / math.radians(30.0))
        ok = seen.view(n, Y + 1, 1, 1).expand_as(score).clone()
        ok[:, -1] = True                                         # staying is always allowed
        if not turn_waist:
            ok[:, :-1] = False
        score = torch.where(ok, score, torch.full_like(score, -1e9))
        flat = score.view(n, -1).argmax(dim=-1)
        iy = flat // (H * T)
        ih = (flat // T) % H
        it = flat % T
        ar = torch.arange(n, device=dev)
        yaw_c, h_c, t_c = yaw[ar, iy], hs[ih], ts[it]
        q_c = q[ar, iy, ih, it]
        q_now = q[:, -1].reshape(n, -1).amax(dim=-1)
        u = p_t[ar, iy].clone()
        u[:, 2] = 0.0
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ch, sh_ = torch.cos(h_c), torch.sin(h_c)
        hdir = torch.stack([ch * u[:, 0] - sh_ * u[:, 1], sh_ * u[:, 0] + ch * u[:, 1], torch.zeros_like(ch)], dim=-1)
        nrm = hdir * torch.cos(t_c).unsqueeze(-1)
        nrm[:, 2] = torch.sin(t_c)
        return {"yaw": yaw_c, "heading": h_c, "tilt": t_c, "q": q_c, "q_now": q_now, "normal_torso": nrm,
                "seen": seen[ar, iy]}


def load(path: str, device="cpu") -> AimMap:
    return AimMap(torch.load(path, map_location="cpu", weights_only=False), device)
