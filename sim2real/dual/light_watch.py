"""light_watch.py -- toy_reach.Watch without the per-step heavy checks, for RL mode.

Removed from the 50 Hz loop (the robot's daemon guards contact now, from motor torque):
  - body-collision clearance (two sphere-model checks per step)
  - the torch safety monitor and the RNEA external-torque model
Kept, all cheap: lowstate / daemon plumbing (STOP), joint speed (STOP: an arm falling or
pushed), motor temperature (STOP), and as WARN only: palm speed / acceleration, fence,
regress, inside-the-apple, clipped targets, late steps. The policy may miss the apple;
it is not stopped for it.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import toy_reach as T  # noqa: E402

R = T.R


class LightWatch(T.Watch):
    def __init__(self, args, kin):
        super().__init__(args, kin)            # loads body_collision (used once, for the path home)
        self.brief = None
        self.hz = 0.0
        self._t_brief = 0.0

    def step(self, s, fr, q_cmd, v_cmd, speed, ctx):
        a, sm = self.args, self.sm
        out = []
        if s.age > 0.1:
            out.append((sm.STOP, f"lowstate {s.age * 1000:.0f} ms old"))
        if ctx.get("commanding"):
            if s.frozen:
                out.append((sm.STOP, "the daemon FROZE the arm (|q_ref - q| > 1.2 rad)"))
            if s.mode is not None and s.mode != "qref":
                out.append((sm.STOP, f"the robot left RL mode (now '{s.mode}')"))
            if s.daemon_age is not None and s.daemon_age > 1.0:
                out.append((sm.WARN, f"daemon state {s.daemon_age:.1f} s old"))
        dt = R.DT if self.t_prev is None else max(0.005, s.t - self.t_prev)
        self.t_prev = s.t
        ctx["dt"] = dt
        self.hz = 0.9 * self.hz + 0.1 * (1.0 / dt) if self.hz else 1.0 / dt
        if dt > 2.0 * R.DT:
            out.append((sm.WARN, f"control step {dt * 1000:.0f} ms late (50 Hz = 20 ms)"))
        jv = float(np.max(np.abs(s.arm_dq)))
        if jv > a.joint_speed_stop:
            out.append((sm.STOP, f"joint speed {jv:.2f} rad/s > {a.joint_speed_stop:.2f} (falling or pushed)"))
        if self.palm_prev is not None:
            v = float(np.linalg.norm((fr.palm_ref - self.palm_prev) / dt))
            self.palm_v = 0.5 * self.palm_v + 0.5 * v
        self.palm_prev = fr.palm_ref.copy()
        if self.palm_v > a.palm_speed_stop:
            out.append((sm.WARN, f"palm speed {self.palm_v:.2f} m/s"))
        if ctx.get("segment") is not None:
            p0, p1 = ctx["segment"]
            mid, rad = 0.5 * (p0 + p1), 0.5 * float(np.linalg.norm(p1 - p0))
            off = float(np.linalg.norm(fr.palm_ref - mid)) - rad
            ctx["off_path"] = off
            if off > a.fence:
                out.append((sm.WARN, f"palm {off * 100:.0f} cm outside the reach region"))
            grow = float(np.linalg.norm(fr.palm_ref - p1)) - float(np.linalg.norm(p0 - p1))
            if grow > a.regress:
                out.append((sm.WARN, f"palm {grow * 100:.0f} cm further from the apple than at the start"))
        gap = ctx.get("gap")
        if gap is not None and gap < -0.02:
            out.append((sm.WARN, f"palm {-gap * 100:.1f} cm inside the (virtual) apple"))
        if ctx.get("clipped"):
            out.append((sm.WARN, "target at the joint limit: " + ", ".join(ctx["clipped"])))
        if s.arm_temp is not None and np.isfinite(s.arm_temp).any():
            tmax = float(np.nanmax(s.arm_temp))
            if tmax > a.temp_stop:
                out.append((sm.STOP, f"motor temperature {tmax:.0f} C"))
            elif tmax > a.temp_warn:
                out.append((sm.WARN, f"motor temperature {tmax:.0f} C"))
        self.dq_prev = s.arm_dq.copy()
        if self.brief is not None and ctx.get("commanding") and s.t - self._t_brief > 2.0:
            self._t_brief = s.t
            self.brief("MOVE", f"policy {self.hz:.0f} Hz, palm {self.palm_v:.2f} m/s"
                               + (f", {ctx['gap'] * 100:+.1f} cm to the apple" if ctx.get("gap") is not None else ""),
                       fold_s=0.0)
        level = max([lv for lv, _ in out], default=sm.OK)
        return level, out

    def print_new(self, out, t):
        for lvl, why in out:
            key = " ".join(why.split(" ")[:2])
            if t - self.last_print.get(key, -9.0) >= 1.0:
                self.last_print[key] = t
                col = T.C_RED if lvl >= self.sm.STOP else T.C_YEL
                print(f"{col}  t={t:6.2f}s {self.sm.LEVEL_NAMES[lvl].upper():5s} {why}{T.C_OFF}", flush=True)
                if self.brief is not None:
                    self.brief("STOP" if lvl >= self.sm.STOP else "WARN", why)
