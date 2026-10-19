#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
toy_arm_pose.py -- the point-to-point arm policy (AppleArmPose-v1) on the real robot, steered by the
waypoints of scripts/play_arm_pose_reach.py: a STAGING point at the apple's height, --stage_radius from
the torso axis (the dashboard's "stage" field overrides it, 0.24 m by default), then the APPLE (the
grasp pocket on the apple centre), the palm normal along the horizontal torso -> apple line.

Everything else is the single toy's flow (toy_reach_single.ToySingle: home, detection / dashboard click,
GO, hold, return home); only the policy step is this file's. The hand is held in the reach's cup during
the policy (the policy trained with it). arm_pose_io.py builds the inputs (validate_arm_pose_io.py).

Through run_toy.py (kind "pose", dual daemon, --onboard to run the policy step on the Jetson):
  apple_picker/g1_launch.sh toy -- pose --policy logs/rsl_rl/apple_arm_pose/<run>/exported/policy.pt \\
      --speed 0.3 --target detection [--onboard]
Own options: --stage_radius M (0.24)  --apple_offset_xyz X Y Z (cm, torso frame; the dashboard's "offset" fields)  --min_stage_back M (0.08)  --apple_offset M (0: the pocket on the
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
    p.add_argument("--apple_offset_xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="cm, torso frame (x forward, y left, z up): the apple goal moved by this; the dashboard's "
                        "'offset' fields override it")
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
    """The single toy's flow with the arm-pose policy step: APPROACH (staging -> apple), GRASP (the picker's
    own close: HandController.hand_close_start + settle_and_hold, the arm held where it reached), RETURN
    (the same staging point -> home, the hand closed on the fruit), RELEASE at home (the picker's open for the
    next grasp). The base flow then holds and moves home; ENTER / RUN TOY in the launcher = the next apple."""

    def stage_radius_now(self) -> float:
        v = getattr(self.io, "stage_radius", None)
        return float(v) if v is not None else float(self.args.stage_radius)

    def apple_offset_now(self) -> np.ndarray:
        """cm, torso frame: the dashboard's offset fields, else --apple_offset_xyz."""
        v = getattr(self.io, "apple_offset_cm", None)
        return np.asarray(v if v is not None else self.args.apple_offset_xyz, float)

    def hand_cup(self):
        if getattr(self.io, "commands", False) and self.args.hand != "keep":
            self.io.hand_cmd(np.array([AP.HAND_CUP[f"right_hand_{n}_joint"] for n in R.DEX3_ORDER]))

    def _brief(self, kind, msg):
        """run_toy's short log, when there is one (run_toy.DualToyMixin)."""
        b = getattr(self, "brief", None)
        if b is not None:
            b(kind, msg, fold_s=0.0)

    # ---------- the thumb: re-seated at home ----------
    def reseat_thumb(self):
        """thumb_0 (the thumb's rotation) is held by the picker at its 30 deg rest with kp 0.5, too weak for
        its friction: robot logs 2026-10-16 read it anywhere from 19 to 49 deg, never moving in a run --
        wherever the power-on left it. With the hand open at home, its gain is raised THUMB_RESEAT_KP_X times
        for at most THUMB_RESEAT_S (until it is within THUMB_TOL_DEG), then restored."""
        io = self.io
        app = getattr(io, "app", None)
        if not getattr(io, "commands", False) or app is None or self.args.hand == "keep":
            return
        target = math.radians(THUMB_REST_DEG)
        q = io.hand.hand_finger_q("right") if hasattr(io, "hand") else None
        if q is None:
            return
        err0 = math.degrees(float(q[0]) - target)
        if abs(err0) <= THUMB_SKIP_DEG:
            self._brief("INFO", f"thumb_0 at {math.degrees(float(q[0])):.0f} deg (rest {THUMB_REST_DEG:.0f}): fine")
            return
        kp0 = list(getattr(app, "hand_kp_vec", None) or [float(getattr(app, "hand_kp", 0.5))] * 7)
        kp = list(kp0)
        kp[0] = kp0[0] * THUMB_RESEAT_KP_X
        app.hand_kp_vec = kp
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < THUMB_RESEAT_S:
                self.hold(self.q_sent if self.q_sent is not None else R.HOME_Q, 0.1)
                q = io.hand.hand_finger_q("right")
                if q is not None and abs(math.degrees(float(q[0]) - target)) <= THUMB_TOL_DEG:
                    break
        finally:
            app.hand_kp_vec = kp0
        q = io.hand.hand_finger_q("right")
        now = float("nan") if q is None else math.degrees(float(q[0]))
        self._brief("INFO", f"thumb_0 re-seated: {THUMB_REST_DEG + err0:.0f} -> {now:.0f} deg (rest "
                            f"{THUMB_REST_DEG:.0f}; its kp x{THUMB_RESEAT_KP_X:g} for {time.monotonic() - t0:.1f} s)")

    def at_home(self):
        """run_toy calls it at home, before looking for the fruit."""
        if not getattr(self, "_reseated", False):
            self._reseated = True
            self.reseat_thumb()

    def plan(self, s, apple_pelvis=None):
        if self.args.target != ["detection"]:     # a given target: no detection step, the thumb here
            self.at_home()
        return super().plan(s, apple_pelvis)

    # ---------- the plan ----------
    def make_plans(self, fr):
        a = self.args
        # the calibration offset (torso frame) moves the apple goal -- and with it the staging height
        off = self.apple_offset_now()
        apple = np.asarray(self.apple, float) + fr.torso_R @ (off / 100.0)
        if np.any(np.abs(off) > 1e-9):
            self.say(f"  apple goal offset (torso x forward, y left, z up): {np.round(off, 2).tolist()} cm", C_CYN)
        plan, desc = AP.make_plan(fr, apple, self.stage_radius_now(), a.min_stage_back, a.apple_offset,
                                  a.stage_via, a.fingers)
        AP.set_stage_advance(plan[0], a.speed)
        back = AP.make_return_plan(plan, AP.home_goal(fr), a.speed)
        self.say(f"  plan: {desc}; staging passed at {AP.STAGE_ADVANCE_D * 100:.0f} cm / {AP.STAGE_ADVANCE_DEG:.0f} deg "
                 f"or after {plan[0].advance_s:.1f} s", C_CYN)
        return plan, back

    # ---------- the grasp: the picker's traditional close ----------
    def grasp_close(self):
        """HandController.hand_close_start (the configured close, thumb_0 at its rest), the arm held where it
        is, settle_and_hold polled while the fingers travel (the picker's CLOSE + LIFT), CLOSE_HOLD_S."""
        io = self.io
        hand = getattr(io, "hand", None)
        if not getattr(io, "commands", False) or hand is None or self.args.hand == "keep":
            return "grasp skipped (no hand output)"
        node = getattr(io, "node", None)
        try:
            if node is not None and hasattr(node, "_capture_grasp_tactile_baseline"):
                node._capture_grasp_tactile_baseline("right")
        except Exception:
            pass
        self.set_phase("GRASP")
        self.ctx["doing"] = "closing the hand (the picker's close)"
        if not hand.hand_close_start("right"):
            return "grasp: the hand cannot be commanded"
        hold_s = float(getattr(node, "close_hold_s_right", 0.0) or 0.0)
        hold_s = max(hold_s, CLOSE_HOLD_S)
        t0 = time.monotonic()
        q_hold = self.q_sent if self.q_sent is not None else self.gravity_hold(io.read())
        while time.monotonic() - t0 < hold_s:
            if io.estop_why is not None:
                return f"E-STOP: {io.estop_why}"
            self.hold(q_hold, 0.1)
            try:
                hand.settle_and_hold("right")
            except Exception:
                pass
        tv = None
        try:
            tv = hand.grasp_travel_frac("right")
        except Exception:
            pass
        return "grasp closed" + (f" (fingers {tv * 100:.0f}% of the way)" if tv is not None else "")

    def release_at_home(self):
        """At home with the fruit: the picker's open for the next grasp (HandController.open_hands_for_new_grasp:
        the fitted hold dropped, the configured close targets back, the hand opened), the arm held while the
        fingers open (RELEASE_S); the next run puts the reach's cup back at GO."""
        io = self.io
        hand = getattr(io, "hand", None)
        self._carrying = False
        if not getattr(io, "commands", False) or hand is None or self.args.hand == "keep":
            return ""
        self.set_phase("RELEASE")
        self.ctx["doing"] = "opening the hand at home (the fruit is released)"
        try:
            hand.open_hands_for_new_grasp()
        except Exception as e:
            return f"release failed ({e})"
        q_hold = self.q_sent if self.q_sent is not None else self.gravity_hold(io.read())
        t0 = time.monotonic()
        while time.monotonic() - t0 < RELEASE_S:
            if io.estop_why is not None:
                return f"E-STOP: {io.estop_why}"
            self.hold(q_hold, 0.1)
        return "released at home"

    def settle_tick(self):
        """While the arm carries the fruit: the picker keeps re-fitting the hold (its LIFT)."""
        hand = getattr(self.io, "hand", None)
        if hand is not None and getattr(self, "_carrying", False):
            try:
                hand.settle_and_hold("right")
            except Exception:
                pass

    # ---------- the policy over a list of goals (laptop) ----------
    def run_goals(self, goals, leg):
        a, io, W = self.args, self.io, self.watch
        sm = W.sm
        L = R.make_limiters("arm_pose")
        prev_raw = np.zeros(AP.N_OUT)
        W.reset()
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
                g = goals[wi]
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
                self.settle_tick()
                self.rows.append(dict(t=t, obs=obs, raw=raw, q_cmd=q_cmd, q=s.arm_q, dq=s.arm_dq, waist=s.waist_q,
                                      palm=fr.palm_ref, goal=g.target, wp=wi, leg=0 if leg == "approach" else 1,
                                      d=d, ang=ang, gap=gap, age=s.age, compute=io.now() - t_tick))
                t_wp += R.DT
                ev = AP.goal_event(g, trk, fr, t_wp, a.waypoint_timeout)
                if ev:
                    msg = f"{g.name} {ev} after {t_wp:.2f} s: pocket {d * 100:.1f} cm, normal {math.degrees(ang):.0f} deg"
                    self.say("  " + msg, C_YEL if ev == "timeout" else C_GRN)
                    if wi == len(goals) - 1:
                        if g.name == "apple" and ev == "reached":
                            self.reach_info = dict(t=t, pocket_err_mm=d * 1000, normal_err_deg=math.degrees(ang),
                                                   palm=fr.palm_ref.tolist(), q=s.arm_q.tolist())
                        outcome = f"{g.name.upper()} {'REACHED' if ev == 'reached' else 'TIMEOUT'} in {t:.2f} s " \
                                  f"(pocket {d * 100:.1f} cm, normal {math.degrees(ang):.0f} deg)"
                        break
                    wi, t_wp = wi + 1, 0.0
                    trk.reset()
                if t - t_print >= 0.5:
                    t_print = t
                    print(f"  t={t:5.2f}s {g.name:12s} pocket {d * 100:5.1f} cm  normal {math.degrees(ang):4.0f} deg  "
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
        return outcome, stopped

    def _start_policy(self):
        s = self.io.read()
        fr = self.frames(s)
        plan, back = self.make_plans(fr)
        self.ctx.update(segment=(fr.palm_ref.copy(), self.apple.copy()), lo=self.lo, hi=self.hi, commanding=True)
        self.set_phase("POLICY")
        self.ctx["doing"] = "arm-pose policy: staging, then the apple"
        self.hand_cup()
        return plan, back

    def _release_if_home(self, out_g, out_b, stopped):
        """The fruit is let go only at home: after a return that reached home or ran out its time there."""
        if out_g.startswith("grasp closed") and not stopped and str(out_b).startswith("HOME") \
                and self.io.estop_why is None:
            return self.release_at_home()
        return ""

    def _finish_policy(self, out_a, out_g, out_b, stopped, out_r=""):
        self.ctx["commanding"] = False
        self._carrying = False
        parts = [out_a] + ([out_g] if out_g else []) + ([f"return: {out_b}"] if out_b else []) + \
            ([out_r] if out_r else [])
        outcome = "; ".join(parts)
        # REACHED (the base flow then holds and moves home) when the apple was reached or held to its timeout
        if not stopped and out_a.startswith("APPLE"):
            outcome = "REACHED " + outcome
        self.say(f"  policy: {outcome}", C_GRN if outcome.startswith("REACHED") else C_RED if stopped else C_YEL)
        return outcome, stopped

    def policy_loop(self):
        """APPROACH -> GRASP -> RETURN with the policy on the laptop (run_toy --onboard: policy_loop_onboard)."""
        plan, back = self._start_policy()
        out_a, stopped = self.run_goals(plan, "approach")
        self.fr_at_apple = self.frames(self.io.read())
        out_g = out_b = ""
        if not stopped and out_a.startswith("APPLE") and self.io.estop_why is None:
            out_g = self.grasp_close()
            if not out_g.startswith("E-STOP"):
                self.set_phase("RETURN")
                self.ctx["doing"] = "arm-pose policy: back through the staging point to home"
                self._carrying = True
                out_b, stopped = self.run_goals(back, "return")
        out_r = self._release_if_home(out_g, out_b, stopped)
        return self._finish_policy(out_a, out_g, out_b, stopped, out_r)

    def policy_loop_onboard(self):
        """The same three steps with the two policy legs on the Jetson (dual/onboard_link.py)."""
        from onboard_link import run_policy_onboard
        plan, back = self._start_policy()
        out_a, stopped = run_policy_onboard(self, extra={"goals": [g.to_dict() for g in plan]})
        self.fr_at_apple = self.frames(self.io.read())
        out_g = out_b = ""
        if not stopped and str(out_a).startswith("APPLE") and self.io.estop_why is None:
            out_g = self.grasp_close()
            if not out_g.startswith("E-STOP"):
                self.set_phase("RETURN")
                self._carrying = True
                out_b, stopped = run_policy_onboard(self, extra={"goals": [g.to_dict() for g in back]},
                                                    tick=self.settle_tick)
        out_r = self._release_if_home(out_g, str(out_b), stopped)
        return self._finish_policy(str(out_a), out_g, str(out_b), stopped, out_r)


#: the picker's thumb_0 rest (full_move2.HAND_REST_THUMB_DEG) and its re-seat at home
THUMB_REST_DEG = 30.0
THUMB_SKIP_DEG = 5.0
THUMB_TOL_DEG = 3.0
THUMB_RESEAT_KP_X = 3.0
THUMB_RESEAT_S = 1.5
#: s the closed hand is held (with the picker's settle) before the arm goes back
CLOSE_HOLD_S = 1.5
#: s the arm stays still at home while the hand opens and the fruit falls out
RELEASE_S = 1.5


def main():
    args = parse_args()
    toy = ToyArmPose(args)
    out = toy.run()
    sys.exit(0 if out and str(out).startswith("REACHED") else 1)


if __name__ == "__main__":
    main()
