#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
toy_arm_pose.py -- the point-to-point arm policy (AppleArmPose-v1) on the real robot, steered by the
waypoints of scripts/play_arm_pose_reach.py: a STAGING point at the apple's height, --stage_radius from
the torso axis (the dashboard's "stage" field overrides it, 0.15 m by default), then the APPLE (the
grasp pocket on the apple centre), the palm normal along the horizontal torso -> apple line.

Everything else is the single toy's flow (toy_reach_single.ToySingle: home, detection / dashboard click,
GO, hold, return home); only the policy step is this file's. The hand is held in the reach's cup during
the policy (the policy trained with it). arm_pose_io.py builds the inputs (validate_arm_pose_io.py).

Through run_toy.py (kind "pose", dual daemon, --onboard to run the policy step on the Jetson):
  apple_picker/g1_launch.sh toy -- pose --policy logs/rsl_rl/apple_arm_pose/<run>/exported/policy.pt \\
      --speed 0.3 --target detection [--onboard]
Own options: --stage_radius M (0.15)  --min_stage_back M (0.08)  --apple_offset M (0: the pocket on the
apple centre; negative = stop short)  --stage_via (fly through the staging point)  --fingers {free,home}
--waypoint_timeout S (4 / --speed)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arm_pose_io as AP  # noqa: E402
import toy_reach_single as S  # noqa: E402

T, R = S.T, S.R
C_CYN, C_GRN, C_RED, C_YEL = T.C_CYN, T.C_GRN, T.C_RED, T.C_YEL


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stage_radius", type=float, default=AP.STAGE_RADIUS)
    p.add_argument("--min_stage_back", type=float, default=AP.MIN_STAGE_BACK)
    p.add_argument("--apple_offset", type=float, default=0.0)
    p.add_argument("--stage_via", action="store_true")
    p.add_argument("--fingers", choices=["free", "home"], default="free")
    p.add_argument("--waypoint_timeout", type=float, default=None,
                   help=f"s per waypoint (default {AP.WAYPOINT_TIMEOUT_S:g} / --speed: the trained budget, played "
                        f"1/speed slower)")
    mine, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest + ["--layout_ok"]
    a = S.parse_args()
    AP.use_constants()                       # after the single toy's own switch: the arm-pose policy's limiter
    a.layout_name = "arm_pose"
    a.visible = False
    if a.waist != "hold":
        print("  arm-pose policy: no waist output -> --waist hold (the torso stays where it is)")
        a.waist = "hold"
    user_limit = "--time_limit" in sys.argv or any(x.startswith("--time_limit=") for x in sys.argv)
    for k, v in vars(mine).items():
        setattr(a, k, v)
    if a.waypoint_timeout is None:
        a.waypoint_timeout = AP.WAYPOINT_TIMEOUT_S / a.speed
    if not user_limit:                       # both waypoints get their whole budget
        a.time_limit = 2.0 * a.waypoint_timeout + 2.0
    print(f"  AppleArmPose policy (35 in / 7 out): staging {a.stage_radius:.2f} m from the torso axis (the "
          f"dashboard's 'stage' field overrides it), then the apple; the hand in the reach's cup")
    return a


class ToyArmPose(S.ToySingle):
    def stage_radius_now(self) -> float:
        v = getattr(self.io, "stage_radius", None)
        return float(v) if v is not None else float(self.args.stage_radius)

    def hand_cup(self):
        if getattr(self.io, "commands", False) and self.args.hand != "keep":
            self.io.hand_cmd(np.array([AP.HAND_CUP[f"right_hand_{n}_joint"] for n in R.DEX3_ORDER]))

    def make_plan(self, fr):
        a = self.args
        plan, desc = AP.make_plan(fr, self.apple, self.stage_radius_now(), a.min_stage_back, a.apple_offset,
                                  a.stage_via, a.fingers)
        self.say(f"  plan: {desc}", C_CYN)
        return plan

    def _brief(self, kind, msg):
        """run_toy's short log, when there is one (run_toy.DualToyMixin)."""
        b = getattr(self, "brief", None)
        if b is not None:
            b(kind, msg, fold_s=0.0)

    def policy_loop(self):
        """The arm-pose policy through the plan's goals (laptop loop; run_toy --onboard runs it on the Jetson)."""
        a, io, W = self.args, self.io, self.watch
        sm = W.sm
        L = R.make_limiters("arm_pose")
        prev_raw = np.zeros(AP.N_OUT)
        s = io.read()
        fr = self.frames(s)
        plan = self.make_plan(fr)
        self.ctx.update(segment=(fr.palm_ref.copy(), self.apple.copy()), lo=self.lo, hi=self.hi, commanding=True)
        W.reset()
        self.set_phase("POLICY")
        self.ctx["doing"] = "arm-pose policy: staging, then the apple"
        self.hand_cup()
        wi, trk, t_wp = 0, AP.Tracker(), 0.0
        outcome, stopped = None, False
        t_start = io.now()
        t_next, t_print = t_start, -1.0
        level = sm.OK
        try:
            while True:
                t_tick = io.now()
                t = t_tick - t_start
                if io.estop_why is not None:
                    stopped, outcome = True, f"E-STOP: {io.estop_why}"
                    self.halted = outcome
                    break
                s = io.read()
                fr = self.frames(s)
                g = plan[wi]
                d, ang, _ = AP.errors(fr, g)
                obs = AP.build_obs(s.arm_q, s.arm_dq, fr, g, prev_raw)
                raw = self.pol(obs)
                tgt = L["arm"].step(raw[:7], s.arm_q, a.speed)
                q_cmd = np.clip(tgt, self.lo, self.hi)
                hit = np.abs(q_cmd - tgt) > 1e-6
                if hit.any():
                    L["arm"].prev = q_cmd.copy()
                    L["arm"].vel[hit] = 0.0
                gap = float(np.linalg.norm(self.apple - fr.palm_ref)) - a.apple_radius
                self.ctx.update(gap=gap, clipped=[R.RIGHT_ARM_JOINTS[j][6:-6] for j in np.flatnonzero(hit)],
                                speed_now=a.speed)
                level, out = W.step(s, fr, q_cmd, L["arm"].vel, a.speed, self.ctx)
                W.print_new(out, t)
                if level >= sm.STOP and a.safety == "on":
                    stopped = True
                    outcome = "SAFETY STOP: " + "; ".join(w for l, w in out if l >= sm.STOP)
                    self.halted = outcome
                    break
                io.send_arm(q_cmd)
                self.q_sent = q_cmd.copy()
                prev_raw = raw
                self.rows.append(dict(t=t, obs=obs, raw=raw, q_cmd=q_cmd, q=s.arm_q, dq=s.arm_dq, waist=s.waist_q,
                                      palm=fr.palm_ref, goal=g.target, wp=wi, d=d, ang=ang, gap=gap, age=s.age,
                                      compute=io.now() - t_tick))
                t_wp += R.DT
                reached = trk.step(fr, g)
                if reached or t_wp >= a.waypoint_timeout:
                    msg = (f"  {g.name} {'reached' if reached else 'TIMEOUT'} after {t_wp:.2f} s: pocket "
                           f"{d * 100:.1f} cm, normal {math.degrees(ang):.0f} deg")
                    self.say(msg, C_GRN if reached else C_YEL)
                    if g.name == "apple":
                        if reached:
                            self.reach_info = dict(t=t, pocket_err_mm=d * 1000, normal_err_deg=math.degrees(ang),
                                                   palm=fr.palm_ref.tolist(), q=s.arm_q.tolist())
                            outcome = f"REACHED in {t:.2f} s (pocket {d * 100:.1f} cm, normal {math.degrees(ang):.0f} deg)"
                        else:
                            outcome = f"TIME LIMIT at the apple (pocket {d * 100:.1f} cm, normal {math.degrees(ang):.0f} deg)"
                        break
                    wi, t_wp = wi + 1, 0.0
                    trk.reset()
                if t - t_print >= 0.5:
                    t_print = t
                    print(f"  t={t:5.2f}s {g.name:8s} pocket {d * 100:5.1f} cm  normal {math.degrees(ang):4.0f} deg  "
                          f"palm v {W.palm_v:.2f} m/s  [{sm.LEVEL_NAMES[level]}]", flush=True)
                self.status(s)
                if t >= a.time_limit:
                    outcome = f"TIME LIMIT {a.time_limit:.0f} s at the {g.name} (pocket {d * 100:.1f} cm)"
                    break
                t_next += R.DT
                if io.now() - t_next > R.DT:
                    t_next = io.now()
                io.sleep_until(t_next)
        except KeyboardInterrupt:
            stopped, outcome = True, "OPERATOR STOP (Ctrl-C)"
            self.halted = outcome
        self.ctx["commanding"] = False
        self.say(f"  policy: {outcome}", C_GRN if outcome.startswith("REACHED") else C_RED if stopped else C_YEL)
        return outcome, stopped


def main():
    args = parse_args()
    toy = ToyArmPose(args)
    out = toy.run()
    sys.exit(0 if out and str(out).startswith("REACHED") else 1)


if __name__ == "__main__":
    main()
