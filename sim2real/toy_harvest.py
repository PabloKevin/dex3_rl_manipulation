#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
toy_harvest.py -- AUTONOMOUS harvest with the arm-pose toy (toy_arm_pose.py) and a waist SCAN.

The arm rests at HOME, tucked down by the hip (harvest_config.json home_q: nothing in front of the chest, so
the waist turns without pushing the apples); the arm-pose policy starts and ends at READY (the old home, in
front of the belly); HOME <-> READY is a scripted joint move (harvest_moves.py). One GO, then (no clicks):
  DETECT    the arm at HOME, the waist still: the apples the head camera sees NOW (seen within FRESH_S),
            reachable (in front of the right shoulder: the arm-pose policy's task region), not harvested or
            given up yet (a memory of positions fixed to the ground) -> the LOWEST one
  CENTER    that apple not whole in the picture left-right (its box at the left / right edge, or its centre
            too near it -- above / below does not count): the waist turns to centre it first, then DETECT
  APPROACH  HOME -> READY, then the policy: staging -> apple            GRASP  the picker's close
  DETACH    the hand pulled slowly straight back (stiffer, the pull force capped, the stem's snap caught;
            no safety stop for the snap -- it is expected here)
  RETURN    the policy: the same staging point -> READY; then HOME
  DEPOSIT   the waist to the basket's yaw (-15 deg), HOME -> READY, the policy: staging -> the basket
            (fixed to the ground where the right arm points stretched out to the right at waist 0)
  RELEASE   the hand opens; the policy back to READY, HOME, the waist back to where it was; DETECT again
  SCAN      nothing left in view: the waist turns RIGHT (to --right_deg) on a slow quintic; the moment a new
            reachable apple shows, it brakes -- a quintic from the current yaw / speed / acceleration -- to
            centre the new apples in the view, waits until the waist is still ("locked"), then DETECT. At the
            right limit with nothing: the same to the LEFT (--left_deg). At the left limit: the waist home
            (0 deg), the run ends.
An apple that fails --max_tries times is skipped. The dashboard's END button ends the run after the current
step (the waist goes home); STOP / the remote (E-STOP) work as always. Whatever ends the run, the hand opens
(run_toy.open_hand_at_end). The dashboard's state ring shows these states.

    apple_picker/g1_launch.sh toy -- harvest --policy logs/rsl_rl/apple_arm_pose/<run>/exported/policy.pt \\
        --speed 0.3 --target detection [--onboard]
Own options: --right_deg (-45)  --left_deg (15)  --scan_speed_deg (5, deg/s peak)  --scan_acc_deg (8, deg/s^2)
--max_tries (2)  --no_deposit  --deposit_test (one deposit, no apple: to place the basket)  --harvest_config
FILE (harvest_config.json)  and everything of toy_arm_pose.py
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
import harvest_moves as HM  # noqa: E402
import toy_arm_pose as TA  # noqa: E402

S, T, R = TA.S, TA.T, TA.R
C_CYN, C_GRN, C_RED, C_YEL = T.C_CYN, T.C_GRN, T.C_RED, T.C_YEL

#: the dashboard's state ring (status "states"), in the order of one harvest
HARVEST_STATES = ["DETECT", "CENTER", "APPROACH", "GRASP", "DETACH", "RETURN", "DEPOSIT", "RELEASE", "SCAN"]
#: an apple is "in view now" when the tracker saw it within this (s; ~10 Hz camera + latency)
FRESH_S = 0.8
#: the same apple (ground-fixed memory): within this (m)
SAME_APPLE_M = 0.06
#: the waist is "locked" when within LOCK_DEG of its target and slower than LOCK_SPEED_DEG for LOCK_HOLD_S
LOCK_DEG, LOCK_SPEED_DEG, LOCK_HOLD_S, LOCK_TIMEOUT_S = 0.6, 1.0, 0.5, 5.0
#: grasp: fingers closed further than this share of their travel = nothing in the hand
EMPTY_TRAVEL = 0.85
#: the arm-pose policy's task region (envs/apple_arm_pose_env_cfg.TASK_FILTER), relative to the right shoulder,
#: torso frame: where its goals were trained
REACH_FORWARD_MIN, REACH_MAX, REACH_DZ = 0.10, 0.60, (-0.45, 0.30)
#: and not across the body: at most this far to the LEFT of the right shoulder (loopback 2026-10-19: an apple
#: 30 cm left of it, with the waist at its +15 deg limit, ended 35 cm away)
REACH_LEFT_MAX = 0.18
#: the hand closes only when the apple waypoint ended this close (real timeouts ended 2.3-4.5 cm off);
#: further = a missed try, the arm goes home
GRASP_MAX_ERR = 0.05
#: a detector box belongs to an apple of the tracker within this (m, torso frame)
BOX_MATCH_M = 0.06
#: an apple at the picture's edge: the waist turns to centre it at most this many times, then it is given up
MAX_CENTER = 2
#: the waist turn that centres an apple must be at least this (deg); less = it cannot be centred (a limit)
MIN_CENTER_DEG = 1.5
#: s over which the stiffness goes back from the detach's to the policy's (the arm kept where it is)
GAIN_BACK_S = 0.3


# ─────────────────────────────────────────────
#  Quintic waist trajectories
# ─────────────────────────────────────────────

class Quintic:
    """p(t) on [0, T]: p, v, a given at t = 0, v = a = 0 at p1 (rest at the end)."""

    def __init__(self, p0, v0, a0, p1, T_):
        self.p0, self.p1, self.T = float(p0), float(p1), max(float(T_), 1e-3)
        Tt = self.T
        c0, c1, c2 = float(p0), float(v0), 0.5 * float(a0)
        A = np.array([[Tt ** 3, Tt ** 4, Tt ** 5], [3 * Tt ** 2, 4 * Tt ** 3, 5 * Tt ** 4], [6 * Tt, 12 * Tt ** 2, 20 * Tt ** 3]])
        b = np.array([p1 - (c0 + c1 * Tt + c2 * Tt ** 2), -(c1 + 2 * c2 * Tt), -2 * c2])
        self.c = np.concatenate([[c0, c1, c2], np.linalg.solve(A, b)])

    def at(self, t):
        t = min(max(float(t), 0.0), self.T)
        c = self.c
        p = c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3 + c[4] * t ** 4 + c[5] * t ** 5
        v = c[1] + 2 * c[2] * t + 3 * c[3] * t ** 2 + 4 * c[4] * t ** 3 + 5 * c[5] * t ** 4
        a = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t ** 2 + 20 * c[5] * t ** 3
        return p, v, a

    def peaks(self, n=200):
        ts = np.linspace(0.0, self.T, n)
        pva = np.array([self.at(t) for t in ts])
        return float(np.max(np.abs(pva[:, 1]))), float(np.max(np.abs(pva[:, 2])))


def plan_quintic(p0, v0, a0, p1, vmax, amax, t_min=0.5) -> Quintic:
    """The shortest quintic (to 2 %) whose speed and acceleration stay within vmax / amax."""
    d = abs(p1 - p0)
    T_ = max(t_min, 1.875 * d / vmax, math.sqrt(5.7735 * d / amax) if d > 0 else 0.0, 2.0 * abs(v0) / amax)
    for _ in range(80):
        q = Quintic(p0, v0, a0, p1, T_)
        pv, pa = q.peaks()
        if pv <= vmax * 1.001 and pa <= amax * 1.001:
            return q
        T_ *= 1.06
    return q


# ─────────────────────────────────────────────
#  The harvest
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--right_deg", type=float, default=-45.0)
    p.add_argument("--left_deg", type=float, default=15.0)
    p.add_argument("--scan_speed_deg", type=float, default=5.0)
    p.add_argument("--scan_acc_deg", type=float, default=8.0)
    p.add_argument("--max_tries", type=int, default=2)
    p.add_argument("--no_deposit", action="store_true", help="release at home instead of the basket")
    p.add_argument("--deposit_test", action="store_true",
                   help="one deposit with no apple (the hand opens over the basket), then the run ends: to place "
                        "the basket / check harvest_config.json's basket")
    p.add_argument("--harvest_config", default=HM.CONFIG)
    p.add_argument("--empty_travel", type=float, default=EMPTY_TRAVEL,
                   help="fingers closed further than this share of their travel = nothing in the hand")
    mine, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    a = TA.parse_args()
    for k, v in vars(mine).items():
        setattr(a, k, v)
    if not -46.0 <= a.right_deg < 0.0 <= a.left_deg <= 46.0:
        sys.exit("--right_deg must be in [-46, 0), --left_deg in [0, 46]")
    a.hcfg = HM.load_config(a.harvest_config)
    # HOME is the tucked pose: every "home" of the toy flow (the first move, holds, the move at the end)
    R.HOME_Q = np.asarray(a.hcfg["home_q"], float).copy()
    b = a.hcfg["basket"]
    print(f"  HARVEST: autonomous (lowest apple first), waist scan {a.right_deg:+.0f} .. {a.left_deg:+.0f} deg at "
          f"{a.scan_speed_deg:g} deg/s; home = the arm tucked {np.round(R.HOME_Q, 2).tolist()}, the policy from READY "
          f"{np.round(a.hcfg['ready_q'], 2).tolist()}; deposit "
          + ("at home" if a.no_deposit else f"in the basket at waist {float(b['waist_deg']):+.0f} deg "
                                            f"({os.path.basename(a.harvest_config)})"))
    return a


class ToyHarvest(TA.ToyArmPose):
    def __init__(self, args):
        super().__init__(args)
        self.hcfg = args.hcfg
        self.ik = HM.ArmIK(self.kin, self.lo, self.hi)
        self.hstate, self.hstate_t = "DETECT", time.monotonic()
        self.memory = []                         # [{"p": pelvis xyz, "tries": n, "done": bool, "centered": n}]
        self.n_picked = self.n_tries = 0
        self.t_cycle0, self.last_cycle_s = None, None
        self._carrying = False
        self.scan_trigger = None
        pub = getattr(self.io, "publish", None)
        if pub is not None:
            def publish(doc, target):        # the dashboard: the harvest's own state ring and counters
                doc = dict(doc, state=self.hstate, states=HARVEST_STATES,
                           state_since_s=round(time.monotonic() - self.hstate_t, 1),
                           metrics={"successful_drops": self.n_picked, "total_picks_attempted": self.n_tries,
                                    "last_cycle_time_s": self.last_cycle_s})
                return pub(doc, target)
            self.io.publish = publish

    # ---- state / log ----
    def hset(self, st, doing=""):
        if st != self.hstate:
            self.hstate, self.hstate_t = st, time.monotonic()
        if st not in ("GRASP", "RELEASE"):        # grasp_close / release_at_home log those themselves
            self.set_phase(st)
        if doing:
            self.ctx["doing"] = doing

    def end_requested(self) -> bool:
        t = getattr(self.io, "harvest_stop_t", None)
        return t is not None and t > self._t_go

    # ---- detection ----
    def _camera(self):
        ci = getattr(self.io, "cam_info", None) or {}
        v = self.hcfg["view"]
        return float(ci.get("w", v["image_w"])), float(ci.get("fx", v["fx"])), float(ci.get("cx", v["cx"]))

    def in_picture(self, p, r, fr, boxes):
        """Is the apple at p (pelvis) whole in the head camera's picture left-right? -> (whole, its bearing in
        the camera [rad, + left], why). The detector's box when one matches, else its sphere projected."""
        v = self.hcfg["view"]
        W, fx, cx = self._camera()
        pt = fr.torso_R.T @ (p - fr.torso_pos)
        box = None
        for q_t, bb in boxes:
            if np.linalg.norm(q_t - pt) < BOX_MATCH_M:
                box = bb
                break
        if box is not None:
            x1, x2 = box[0], box[2]
            src = "box"
        else:
            c = fr.camera_R.T @ (p - fr.camera_pos)
            if c[0] <= 0.05:
                return False, 0.0, "behind the camera"
            u = cx - fx * c[1] / c[0]
            rp = fx * float(r or self.args.apple_radius) / c[0]
            x1, x2 = u - rp, u + rp
            src = "projected"
        uc = 0.5 * (x1 + x2)
        bearing = math.atan((cx - uc) / fx)
        m = float(v["margin_px"])
        if x1 < m or x2 > W - m:
            return False, bearing, f"cut by the {'left' if x1 < m else 'right'} edge ({src} x {x1:.0f}..{x2:.0f} of {W:.0f} px)"
        if abs(uc - W / 2) > (W / 2) * (1.0 - float(v["edge_frac"])):
            return False, bearing, f"its centre near the {'left' if uc < W / 2 else 'right'} edge ({src} x {uc:.0f} of {W:.0f} px)"
        return True, bearing, src

    def visible_apples(self):
        """[{"p": pelvis xyz, "r", "whole", "bearing", "why"}] the camera sees NOW, reachable, not harvested /
        given up."""
        io = self.io
        try:
            snap = io.node.detection.get_snapshot()
        except Exception:
            return []
        now = time.time()
        s = io.read()
        fr = self.frames(s)
        t_box, boxes = getattr(io, "det_boxes", (0.0, []))
        boxes = boxes if time.monotonic() - t_box < FRESH_S else []
        out = []
        seen = list(snap.seen_stamps) + [0.0] * len(snap.apples_all)
        radii = list(snap.radii) + [None] * len(snap.apples_all)
        for i, pt in enumerate(snap.apples_all):
            if now - float(seen[i]) > FRESH_S:
                continue
            pt = np.asarray(pt, float)
            p = fr.torso_R @ pt + fr.torso_pos
            rel = fr.torso_R.T @ (p - fr.shoulder_pos)
            if rel[0] < REACH_FORWARD_MIN or np.linalg.norm(rel) > REACH_MAX or not REACH_DZ[0] <= rel[2] <= REACH_DZ[1] \
                    or rel[1] > REACH_LEFT_MAX:
                continue
            m = self._memo(p)
            if m is not None and (m["done"] or m["tries"] >= self.args.max_tries):
                continue
            whole, bearing, why = self.in_picture(p, radii[i], fr, boxes)
            out.append({"p": p, "r": radii[i], "whole": whole, "bearing": bearing, "why": why})
        return out

    def _memo(self, p, create=False):
        for m in self.memory:
            if np.linalg.norm(m["p"] - p) < SAME_APPLE_M:
                return m
        if create:
            m = {"p": np.asarray(p, float).copy(), "tries": 0, "done": False, "centered": 0}
            self.memory.append(m)
            return m
        return None

    def give_up(self, p):
        m = self._memo(p, create=True)
        m["tries"] = max(m["tries"], self.args.max_tries)

    def look(self, seconds):
        """Hold the arm and the waist still while the tracker updates; then what it sees."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            self.hold(self.q_sent if self.q_sent is not None else R.HOME_Q, 0.1)
        return self.visible_apples()

    @staticmethod
    def lowest(apples):
        return min(apples, key=lambda x: float(x["p"][2]))

    def auto_choose(self):
        """run_toy's fruit choice in autonomous mode: no click. What the camera sees from home is only reported
        (and planned, for the base flow's checks): the run itself starts with a DETECT."""
        a = self.args
        self.hset("DETECT", "looking for apples (autonomous: the lowest first)")
        apples = self.look(a.detect_s)
        s = self.io.read()
        fr = self.frames(s)
        if not apples:
            self.brief("INFO", "no reachable apple in view: the run starts with a waist scan")
            self.plan(s, apple_pelvis=fr.palm_ref + fr.torso_R @ np.array([0.25, 0.0, 0.0]))   # a placeholder
            return None
        x = self.lowest(apples)
        self.brief("INFO", f"{len(apples)} reachable apple(s) in view, the lowest at {np.round(x['p'], 3).tolist()} "
                           f"(pelvis){'' if x['whole'] else ': ' + x['why']}")
        self.plan(s, apple_pelvis=x["p"])
        return None

    # ---- the waist ----
    def waist_now(self):
        s = self.io.read()
        return float(s.waist_q[0]), float(s.waist_dq[0])

    def waist_run(self, traj, watch=None):
        """Stream the arm's hold and the waist trajectory at 50 Hz. watch(): checked every 0.1 s; a non-None
        value stops the run early -> (value, p, v, a at that moment)."""
        q_hold = self.q_sent if self.q_sent is not None else R.HOME_Q
        t0 = self.io.now()
        t_next, t_watch = t0, 0.0
        while True:
            t = self.io.now() - t0
            if self.io.estop_why is not None:
                return "estop", None
            p, v, acc = traj.at(t)
            self.io.send_arm(q_hold, waist=p, waist_dq=v)
            self._waist_cmd = (p, v, acc)
            self.status(self.io.read())
            if watch is not None and t - t_watch >= 0.1:
                t_watch = t
                w = watch()
                if w is not None:
                    return w, (p, v, acc)
            if t >= traj.T:
                return None, (traj.p1, 0.0, 0.0)
            t_next += R.DT
            self.io.sleep_until(t_next)

    def waist_lock(self, target):
        """Hold the waist command at target until the measured waist is there and still."""
        q_hold = self.q_sent if self.q_sent is not None else R.HOME_Q
        t0 = time.monotonic()
        still_since = None
        while time.monotonic() - t0 < LOCK_TIMEOUT_S:
            if self.io.estop_why is not None:
                return False
            self.io.send_arm(q_hold, waist=target, waist_dq=0.0)
            w, dw = self.waist_now()
            ok = abs(math.degrees(w - target)) < LOCK_DEG and abs(math.degrees(dw)) < LOCK_SPEED_DEG
            still_since = (still_since or time.monotonic()) if ok else None
            if still_since is not None and time.monotonic() - still_since >= LOCK_HOLD_S:
                return True
            time.sleep(R.DT)
        self.brief("WARN", f"waist not locked in {LOCK_TIMEOUT_S:.0f} s (at {math.degrees(self.waist_now()[0]):+.1f}, "
                           f"target {math.degrees(target):+.1f} deg): going on")
        return True

    def waist_to(self, target, why):
        """A quintic waist move (the arm held as it is) and the lock. -> False on an E-STOP."""
        a = self.args
        w, dw = self.waist_now()
        p0, v0, a0 = getattr(self, "_waist_cmd", (w, 0.0, 0.0))
        if abs(p0 - w) > math.radians(2.0):
            p0, v0, a0 = w, 0.0, 0.0
        if abs(target - p0) < math.radians(0.3) and abs(v0) < 1e-3:
            return self.waist_lock(target)
        traj = plan_quintic(p0, v0, a0, target, math.radians(a.scan_speed_deg), math.radians(a.scan_acc_deg))
        self.brief("STEP", f"waist {math.degrees(p0):+.0f} -> {math.degrees(target):+.0f} deg ({why}, {traj.T:.1f} s)")
        ev, _ = self.waist_run(traj)
        if ev == "estop":
            return False
        return self.waist_lock(target)

    def center_on(self, x):
        """Turn the waist so that apple x sits in the middle of the picture. -> True (turned), False (it
        cannot be centred: a waist limit, or tried MAX_CENTER times), None (E-STOP)."""
        a = self.args
        m = self._memo(x["p"], create=True)
        w, _ = self.waist_now()
        target = float(np.clip(w + x["bearing"], math.radians(a.right_deg), math.radians(a.left_deg)))
        if m["centered"] >= MAX_CENTER or abs(math.degrees(target - w)) < MIN_CENTER_DEG:
            self.brief("WARN", f"apple {np.round(x['p'], 3).tolist()} not whole in the picture ({x['why']}) and "
                               f"it cannot be centred ({'tried ' + str(m['centered']) + 'x' if m['centered'] >= MAX_CENTER else 'the waist limit'}): skipped")
            self.give_up(x["p"])
            return False
        m["centered"] += 1
        self.hset("CENTER", "turning the waist to see the whole apple")
        self.brief("INFO", f"the lowest apple is not whole in the picture ({x['why']}): the waist turns "
                           f"{math.degrees(target - w):+.0f} deg to centre it first")
        return True if self.waist_to(target, "centre the apple") else None

    def scan(self):
        """Turn the waist toward the scan side until new apples show, centre them, lock. -> True (apples to
        harvest), False (the scan is over: both limits done), None (E-STOP / END)."""
        a = self.args
        while True:
            if self.end_requested():
                return None
            limit = math.radians(a.right_deg if self.scan_dir < 0 else a.left_deg)
            w, _ = self.waist_now()
            if (limit - w) * self.scan_dir <= math.radians(1.0):
                if self.scan_dir < 0:
                    self.brief("INFO", f"the right limit ({a.right_deg:+.0f} deg) reached with nothing new: now to the left")
                    self.scan_dir = +1
                    continue
                return False
            self.hset("SCAN", f"waist scanning {'right' if self.scan_dir < 0 else 'left'} for apples")
            p0, v0, a0 = getattr(self, "_waist_cmd", (w, 0.0, 0.0))
            if abs(p0 - w) > math.radians(2.0):
                p0, v0, a0 = w, 0.0, 0.0
            traj = plan_quintic(p0, v0, a0, limit, math.radians(a.scan_speed_deg), math.radians(a.scan_acc_deg))
            self.brief("STEP", f"scan {'right' if self.scan_dir < 0 else 'left'}: waist {math.degrees(p0):+.0f} -> "
                               f"{math.degrees(limit):+.0f} deg ({traj.T:.1f} s, {a.scan_speed_deg:g} deg/s)")

            def watch():
                if self.end_requested():
                    return "end"
                return self.visible_apples() or None
            ev, st = self.waist_run(traj, watch)
            if ev in ("estop", "end"):
                return None
            if ev is None:
                continue                       # at the limit: the loop flips the side or ends
            # new apples in view: brake, centred on them (their mean bearing in the camera), within the limits
            p, v, acc = st
            target = float(np.clip(p + float(np.mean([x["bearing"] for x in ev])), math.radians(a.right_deg),
                                   math.radians(a.left_deg)))
            brake = plan_quintic(p, v, acc, target, math.radians(a.scan_speed_deg), math.radians(a.scan_acc_deg))
            self.brief("STEP", f"{len(ev)} apple(s) in view at waist {math.degrees(p):+.0f} deg: braking to "
                               f"{math.degrees(target):+.0f} deg to centre them ({brake.T:.1f} s)")
            ev2, _ = self.waist_run(brake)
            if ev2 == "estop":
                return None
            self.waist_lock(target)
            self.scan_trigger = [x["p"] for x in ev]
            return True

    # ---- HOME <-> READY (scripted) ----
    def joint_move(self, points, label):
        """A joint path from where the arm is commanded through `points` at the config's speed, checked
        against the body cover as the toy's own moves (and the stream's speed / acceleration watch). -> error
        or None."""
        io, a = self.io, self.args
        s = io.read()
        q0 = self.q_sent if self.q_sent is not None else self.gravity_hold(s)
        pts = [np.asarray(q0, float)] + [np.asarray(p, float) for p in points]
        speed = float(self.hcfg["moves"]["speed_rad_s"])
        traj = T.joint_path_traj(pts, speed)
        ok, c0, worst, c_end, bad, note = self.path_check(traj, s)
        if not ok:
            return f"{label}: the path collides with the body (body cover worst {worst * 100:+.1f} cm)"
        self.brief("INFO", f"{label}: {len(traj) * R.DT:.1f} s, body cover worst {worst * 100:+.1f} cm", fold_s=0.0)
        keep = a.move_speed
        a.move_speed = speed                       # the stream's speed watch scales with it
        try:
            return self.stream(traj, label=label)
        finally:
            a.move_speed = keep

    def to_ready(self):
        return self.joint_move([*self.hcfg["home_ready_via"], self.hcfg["ready_q"]], "home -> ready")

    def to_home(self):
        err = self.joint_move([*self.hcfg["home_ready_via"], R.HOME_Q], "-> home")
        if err and "collides" in err:
            # not from READY (a leg that ended elsewhere): the toy's own search of a clear path home
            s = self.io.read()
            q0 = self.q_sent if self.q_sent is not None else self.gravity_hold(s)
            via, why = self.plan_home(q0, s)
            if via is None:
                return f"no clear path home from here ({why})"
            keep = self.args.move_speed
            self.args.move_speed = float(self.hcfg["moves"]["speed_rad_s"])
            try:
                return self.move_to(q0, R.HOME_Q, "move home", via=via)
            finally:
                self.args.move_speed = keep
        return err

    # ---- DETACH: the apple off its stem without the snap ----
    def detach(self, u_apple):
        """The hand pulled straight back (along -u_apple, the approach) by pull_m at pull_speed_m_s, at the
        detach's stiffer kp / kd, the arm's sag compensated. The pull waits while the commanded lead on a
        shoulder / elbow joint would mean more than tau_cap_nm (the stem holds: little stored spring); a joint
        faster than snap_dq = the stem broke -> the target is put where the arm IS for 0.4 s (the spring let
        go). -> "snapped" | "pulled" | "stalled" (the tension let off) | "SAFETY STOP: ..." | "E-STOP: ..."."""
        D = self.hcfg["detach"]
        io, a = self.io, self.args
        s = io.read()
        w3 = np.asarray(s.waist_q, float)
        named = T.hand_named(s)
        path, e_ik, _ = HM.detach_path(self.ik, np.asarray(s.arm_q, float), w3, -np.asarray(u_apple, float),
                                       float(D["pull_m"]))
        path = np.asarray(path)
        kp, kd, cap = float(D["kp"]), float(D["kd"]), float(D["tau_cap_nm"])
        T_pull = max(1.0, 1.875 * float(D["pull_m"]) / float(D["pull_speed_m_s"]))
        tp = waited = 0.0
        q_catch, t_catch, peak, lead_max = None, None, 0.0, 0.0
        result = None
        q_des = path[0]
        self.brief("STEP", f"detach: pull {float(D['pull_m']) * 100:.0f} cm back in {T_pull:.1f} s at kp {kp:g} "
                           f"(pull force capped at {cap:g} N m per joint, the snap caught)")
        t_next = io.now()
        while True:
            if io.estop_why is not None:
                return f"E-STOP: {io.estop_why}"
            s = io.read()
            q, jv = np.asarray(s.arm_q, float), float(np.max(np.abs(s.arm_dq)))
            peak = max(peak, jv)
            if jv > float(D["speed_stop"]):
                q_h = HM.gravity_target(self.kin, w3, q, kp, named)
                for _ in range(int(0.5 / R.DT)):
                    io.send_arm(q_h, kp=kp, kd=kd)
                    time.sleep(R.DT)
                self.q_sent = q_h
                return f"SAFETY STOP: joint speed {jv:.2f} rad/s > {float(D['speed_stop']):.2f} while detaching"
            if q_catch is None:
                f = T.quintic(tp / T_pull) * (len(path) - 1)
                i = int(min(f, len(path) - 2))
                q_des = path[i] + (f - i) * (path[i + 1] - path[i])
                lead = float(np.max(np.abs(q_des[:4] - q[:4])))
                lead_max = max(lead_max, lead)
                if jv > float(D["snap_dq"]) and tp > 0.1:
                    q_catch, t_catch = q.copy(), io.now()          # the stem broke: the spring let go here
                    q_des = q_catch
                elif lead * kp > cap:
                    waited += R.DT                                  # the stem holds: the pull waits
                    if waited > float(D["stall_s"]):
                        result = "stalled"
                        break
                else:
                    tp = min(T_pull, tp + R.DT)
                    if tp >= T_pull and lead * kp < 0.5 * cap:
                        result = "pulled"
                        break
            else:
                q_des = q_catch
                if io.now() - t_catch >= 0.4:
                    result = "snapped"
                    break
            q_cmd = np.clip(HM.gravity_target(self.kin, w3, q_des, kp, named), self.lo, self.hi)
            io.send_arm(q_cmd, kp=kp, kd=kd)
            self.q_sent = q_cmd
            self.status(s)
            t_next += R.DT
            io.sleep_until(t_next)
        if result == "stalled":
            # the tension off before anything else: the target eased onto where the arm is
            q_from = q_des.copy()
            n = int(0.5 / R.DT)
            for k in range(1, n + 1):
                s = io.read()
                q_d = q_from + T.quintic(k / n) * (np.asarray(s.arm_q, float) - q_from)
                io.send_arm(np.clip(HM.gravity_target(self.kin, w3, q_d, kp, named), self.lo, self.hi), kp=kp, kd=kd)
                time.sleep(R.DT)
        # back to the policy's stiffness, the arm kept where it is
        s = io.read()
        q_here = np.asarray(s.arm_q, float)
        n = max(1, int(GAIN_BACK_S / R.DT))
        for k in range(1, n + 1):
            f = T.quintic(k / n)
            kk, dd = kp + f * (a.kp - kp), kd + f * (a.kd - kd)
            q_cmd = np.clip(HM.gravity_target(self.kin, w3, q_here, kk, named), self.lo, self.hi)
            io.send_arm(q_cmd, kp=kk, kd=dd)
            self.q_sent = q_cmd
            time.sleep(R.DT)
        self.brief("INFO" if result != "stalled" else "WARN",
                   f"detach: {result} after {tp:.1f} s of pull ({waited:.1f} s waiting on the stem), peak joint speed "
                   f"{peak:.2f} rad/s, largest lead {lead_max:.3f} rad (= {lead_max * kp:.0f} N m), IK line "
                   f"{e_ik * 1000:.1f} mm", fold_s=0.0)
        return result

    # ---- the policy legs ----
    def _legs_laptop(self, goals, leg):
        return self.run_goals(goals, leg)

    def _legs_onboard(self, goals, leg):
        from onboard_link import run_policy_onboard
        out, stopped = run_policy_onboard(self, extra={"goals": [g.to_dict() for g in goals]},
                                          tick=self.settle_tick if self._carrying else None)
        return str(out), stopped

    def legs(self, goals, leg):
        f = self._legs_onboard if getattr(self, "_onboard_mode", False) else self._legs_laptop
        return f(goals, leg)

    def _ready_plan(self, target, name):
        """At READY: (staging -> target, staging -> READY) for the arm-pose policy (the apple's plan)."""
        a = self.args
        fr = self.frames(self.io.read())
        plan, _ = AP.make_plan(fr, target, self.stage_radius_now(), a.min_stage_back, 0.0, a.stage_via, a.fingers)
        plan[1].name = name
        AP.set_stage_advance(plan[0], a.speed)
        back = AP.make_return_plan(plan, AP.home_goal(fr), a.speed)
        back[-1].name = "ready"
        return plan, back

    # ---- the basket ----
    def deposit(self, carrying=True):
        """From HOME with the apple: the waist to the basket's yaw, READY, the policy to the basket, the hand
        opens, the policy back to READY, HOME, the waist back. -> (outcome, stopped)"""
        io, b = self.io, self.hcfg["basket"]
        w_back = self.waist_now()[0]
        self.hset("DEPOSIT", f"the waist to {float(b['waist_deg']):+.0f} deg for the basket")
        if not self.waist_to(math.radians(float(b["waist_deg"])), "the basket"):
            return f"E-STOP: {io.estop_why}", True
        err = self.to_ready()
        if err:
            return err, True
        s = io.read()
        drop, u, B, short = HM.basket_drop(self.kin, self.hcfg, np.asarray(s.waist_q, float))
        plan, back = self._ready_plan(drop, "basket")
        self.hset("DEPOSIT", "arm-pose policy: staging, then over the basket")
        self.brief("INFO", f"basket: the drop point {short * 100:.1f} cm short of the basket point (the arm's reach)",
                   fold_s=0.0)
        out, stopped = self.legs(plan, "deposit")
        if stopped or io.estop_why is not None:
            return out, True
        d_end = float(np.linalg.norm(AP.hand_frame(self.frames(io.read()))[0] - drop))
        if d_end > float(b["accept_m"]):
            self.brief("WARN", f"the hand ended {d_end * 100:.1f} cm from the drop point (> {float(b['accept_m']) * 100:.0f}):"
                               " released there anyway (better than carrying it on)")
        self.hset("RELEASE", "opening the hand over the basket")
        self.release_at_home()
        self._carrying = False
        self.hset("DEPOSIT", "arm-pose policy: back to ready")
        out_b, stopped = self.legs(back, "return")
        if stopped or io.estop_why is not None:
            return out_b, True
        err = self.to_home()
        if err:
            return err, True
        if not self.waist_to(w_back, "back where the scan was"):
            return f"E-STOP: {io.estop_why}", True
        return "deposited", False

    # ---- one apple ----
    def harvest_one(self, p, r):
        """APPROACH, GRASP, DETACH, RETURN, DEPOSIT (RELEASE) for the apple at p (pelvis). -> (outcome, stop)"""
        a, io = self.args, self.io
        self.apple = np.asarray(p, float)
        self.apple_true = self.apple.copy()
        if r:
            a.apple_radius = float(r)
        m = self._memo(self.apple, create=True)
        m["tries"] += 1
        self.n_tries += 1
        t0 = time.monotonic()
        self.hset("APPROACH", "the arm to ready")
        err = self.to_ready()
        if err:
            return err, True
        s = io.read()
        fr = self.frames(s)
        plan, back = self.make_plans(fr)
        back[-1].name = "ready"
        self.hset("APPROACH", "arm-pose policy: staging, then the apple")
        self.hand_cup()
        out_a, stopped = self.legs(plan, "approach")
        self.fr_at_apple = self.frames(io.read())
        if stopped or io.estop_why is not None:
            return out_a, True
        held = False
        d_end = float(np.linalg.norm(AP.hand_frame(self.fr_at_apple)[0] - plan[-1].target))
        if str(out_a).startswith("APPLE") and d_end > GRASP_MAX_ERR:
            self.brief("WARN", f"the hand ended {d_end * 100:.1f} cm from the apple goal (> {GRASP_MAX_ERR * 100:.0f}): "
                               f"no grasp, a missed try ({m['tries']} of {a.max_tries})")
        elif str(out_a).startswith("APPLE"):
            self.hset("GRASP", "closing the hand (the picker's close)")
            out_g = self.grasp_close()
            if out_g.startswith("E-STOP"):
                return out_g, True
            self.hset("DETACH", "pulling the apple off its stem (slow, stiff, the snap caught)")
            out_d = self.detach(plan[-1].normal)
            if out_d.startswith(("E-STOP", "SAFETY")):
                return out_d, True
            if out_d == "stalled":
                self.brief("WARN", f"the stem held: the hand lets the apple go (try {m['tries']} of {a.max_tries})")
                self.hset("RELEASE", "the stem held: letting the apple go")
                self.release_at_home()
            else:
                tv = None
                try:
                    tv = io.hand.grasp_travel_frac("right")
                except Exception:
                    pass
                held = tv is None or tv < a.empty_travel
                if not held:
                    self.brief("WARN", f"the hand closed {tv * 100:.0f}% of the way: nothing in it (try {m['tries']} of "
                                       f"{a.max_tries})")
        self._carrying = held
        self.hset("RETURN", "arm-pose policy: back through the staging point to ready")
        out_b, stopped = self.legs(back, "return")
        if stopped or io.estop_why is not None:
            return out_b, True
        self.hset("RETURN", "the arm home")
        err = self.to_home()
        if err:
            return err, True
        if held:
            if self.hcfg.get("basket") and not a.no_deposit:
                out_dp, stop = self.deposit()
                if stop:
                    return out_dp, True
            else:
                self.hset("RELEASE", "opening the hand at home")
                self.release_at_home()
            m["done"] = True
            self.n_picked += 1
            self.last_cycle_s = time.monotonic() - t0
        self._carrying = False
        return ("picked" if held else f"missed ({out_a})"), False

    # ---- the campaign (run_toy calls it once, after GO) ----
    def policy_loop(self):
        return self.campaign(onboard=False)

    def policy_loop_onboard(self):
        return self.campaign(onboard=True)

    def campaign(self, onboard):
        a, io = self.args, self.io
        self._onboard_mode = onboard
        self._t_go = time.monotonic()
        self.scan_dir = -1                     # right first
        self.ctx.update(lo=self.lo, hi=self.hi, commanding=True)
        stopped, why = False, ""
        if a.deposit_test:
            self._carrying = False
            out, stopped = self.deposit(carrying=False)
            why = f"deposit test: {out}"
            if not stopped:
                self.waist_to(0.0, "home")
        while not a.deposit_test:
            if io.estop_why is not None:
                stopped, why = True, f"E-STOP: {io.estop_why}"
                break
            if self.end_requested():
                why = "END pressed on the dashboard"
                break
            self.hset("DETECT", "looking for apples (the lowest first)")
            apples = self.look(a.detect_s)
            trig, self.scan_trigger = self.scan_trigger, None
            if not apples and trig:
                # seen while turning but not once the waist is still (a flicker, or out of reach there):
                # never again a reason to stop -- else the scan would brake on it forever
                for q in trig:
                    self.give_up(q)
                self.brief("WARN", f"{len(trig)} apple(s) seen while turning were not confirmed with the waist "
                                   "still: ignored from now on")
            if not apples:
                more = self.scan()
                if more is None:
                    stopped = io.estop_why is not None
                    why = f"E-STOP: {io.estop_why}" if stopped else "END pressed on the dashboard"
                    break
                if not more:
                    why = "no more apples (both scan limits reached)"
                    break
                continue
            x = self.lowest(apples)
            if not x["whole"]:
                c = self.center_on(x)
                if c is None:
                    stopped, why = True, f"E-STOP: {io.estop_why}"
                    break
                continue                        # look again (centred, or that apple given up)
            self.brief("INFO", f"next: apple {np.round(x['p'], 3).tolist()} (pelvis), the lowest of {len(apples)}")
            out, stop = self.harvest_one(x["p"], x["r"])
            self.brief("DONE" if out == "picked" else "WARN", f"apple {self.n_tries}: {out} ({self.n_picked} picked)")
            if stop:
                stopped, why = True, out
                break
        if not stopped and io.estop_why is None and not a.deposit_test:
            self.hset("SCAN", "the waist home")
            self.waist_to(0.0, "home")
        self.ctx["commanding"] = False
        outcome = (f"HARVEST: {self.n_picked} picked in {self.n_tries} attempt(s); {why}")
        if not stopped:
            outcome = "REACHED " + outcome      # the base flow holds and moves home
        self.say(f"  {outcome}", C_GRN if not stopped else C_RED)
        return outcome, stopped


def main():
    args = parse_args()
    toy = ToyHarvest(args)
    out = toy.run()
    sys.exit(0 if out and str(out).startswith("REACHED") else 1)


if __name__ == "__main__":
    main()
