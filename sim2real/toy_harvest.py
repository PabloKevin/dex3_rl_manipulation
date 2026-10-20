#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
toy_harvest.py -- AUTONOMOUS harvest with the arm-pose toy (toy_arm_pose.py) and a waist SCAN.

The arm rests at HOME, tucked down by the hip (harvest_config.json home_q: nothing in front of the chest, so
the waist turns without pushing the apples); the arm-pose policy starts and ends at READY (the old home, in
front of the belly); HOME <-> READY is a scripted joint move (harvest_moves.py). One GO, then (no clicks):
  DETECT    the arm at HOME, the waist still: the apples the head camera sees NOW (seen within FRESH_S), RED
            (the detector's class), not by the basket, reachable from some waist yaw of the scan (reach_map.py: a
            geometric volume of where the pocket can go with which palm heading, sampled once offline -- a
            table look-up, no IK), not harvested or given up yet -> the LOWEST one
  AIM       the waist yaw and palm heading to take it from (look-ups: the smallest palm turn + waist turn, the
            apple whole in the picture left-right there); a turn of the waist first when needed, then DETECT
  APPROACH  HOME -> READY, then the policy: staging -> apple (the palm turned as AIM chose)
  GRASP     the picker's close
  DETACH    the hand pulled slowly back along its own approach path, replayed backwards (stiffer, the pull
            easing off as the stem's tension builds, the snap caught; no safety stop for the snap -- it is
            expected here; the policy leg after it stops only above detach.return_speed_stop)
  RETURN    the policy: the same staging point -> READY; then HOME
  DEPOSIT   the waist into the basket's yaw range (-30..0 deg, only as far as needed), HOME -> READY, the
            policy straight to the drop, the palm facing the ground, the fingers along the arm (the basket is
            fixed to the ground where the right arm points stretched out to the right at waist 0)
  RELEASE   the hand opens; the policy back to READY, HOME, the waist back to where it was; DETECT again
  SCAN      nothing left in view: the waist turns RIGHT (to --right_deg) on a slow quintic; the moment a new
            reachable apple shows, it brakes -- a quintic from the current yaw / speed / acceleration -- to
            centre the new apples in the view, waits until the waist is steady, then DETECT. At the right
            limit with nothing: the same to the LEFT (--left_deg). At the left limit: the waist home (0 deg),
            the run ends. (The robot smooths the streamed waist between packets: g1_dual dual-2.)
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
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arm_pose_io as AP  # noqa: E402
import harvest_moves as HM  # noqa: E402
import reach_map as RMAP  # noqa: E402
import toy_arm_pose as TA  # noqa: E402

S, T, R = TA.S, TA.T, TA.R
C_CYN, C_GRN, C_RED, C_YEL = T.C_CYN, T.C_GRN, T.C_RED, T.C_YEL

#: the dashboard's state ring (status "states"), in the order of one harvest
HARVEST_STATES = ["DETECT", "AIM", "APPROACH", "GRASP", "DETACH", "RETURN", "DEPOSIT", "RELEASE", "SCAN"]
#: the picker's generic lines in a harvest: none (the harvest says its own, more specific ones) or its own key
GENERIC_VOICE = {"reaching": None, "got_it": None, "missed_it": None, "found_one": None, "looking_for_fruit": None,
                 "ask_fruit": None, "need_help": "rl_safety", "finish_my_work": "rl_done"}
#: an apple is "in view now" when the tracker saw it within this (s; ~10 Hz camera + latency)
FRESH_S = 0.8
#: the same apple (ground-fixed memory): within this (m)
SAME_APPLE_M = 0.06
#: grasp: fingers closed further than this share of their travel = nothing in the hand
EMPTY_TRAVEL = 0.85
#: the waist yaws AIM considers (deg step over the scan range)
AIM_STEP_DEG = 2.0
#: the hand closes only when the apple waypoint ended this close (real timeouts ended 2.3-4.5 cm off);
#: further = a missed try, the arm goes home
GRASP_MAX_ERR = 0.05
#: a detector box belongs to an apple of the tracker within this (m, torso frame)
BOX_MATCH_M = 0.06
#: AIM turns the waist for one apple at most this many times (the camera then looks again each time)
MAX_CENTER = 2
#: a waist turn smaller than this (deg) is not made: the apple is harvested from here
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
    p.add_argument("--waist_kp", type=float, default=None,
                   help="a softer waist (60..380) while the harvest moves it (the robot's dual-2 daemon; default: "
                        "its own 380); the arm's policy legs always run at 380")
    p.add_argument("--no_deposit", action="store_true", help="release at home instead of the basket")
    p.add_argument("--deposit_test", action="store_true",
                   help="one deposit with no apple (the hand opens over the basket), then the run ends: to place "
                        "the basket / check harvest_config.json's basket")
    p.add_argument("--harvest_config", default=HM.CONFIG)
    p.add_argument("--grip_pct", type=float, default=100.0,
                   help="grip strength, %% of the picker's finger kp, when the dashboard has none (its 'grip' field wins)")
    p.add_argument("--trunk_side", choices=["off", "shadow", "act"], default="act",
                   help="apple beside the trunk (sim2real/trunk_side.py: depth + colour): act = reach from its free side "
                        "(the palm turned up to 60 deg), shadow = only log / save the picture, off")
    p.add_argument("--reactive", choices=["off", "on"], default="on",
                   help="stop an approach that pushes on something before the apple (pads / arm lag), back off, retry "
                        "from the side; a stalled detach (a trunk does not come off) also makes the next try a side one")
    p.add_argument("--apple_calib", nargs="?", const="default", default=None,
                   help="correct the perceived apple with sim2real/apple_pos_calib.py's fit (default file "
                        "logs/sim2real/apple_calib/apple_calib.json), before the dashboard's offset; off by default")
    p.add_argument("--empty_travel", type=float, default=None,
                   help="fingers closed further than this share of their travel = nothing in the hand (when the pads "
                        "do not decide; default harvest_config.json grasp_check.travel_empty)")
    mine, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    a = TA.parse_args()
    for k, v in vars(mine).items():
        setattr(a, k, v)
    if not -46.0 <= a.right_deg < 0.0 <= a.left_deg <= 46.0:
        sys.exit("--right_deg must be in [-46, 0), --left_deg in [0, 46]")
    a.hcfg = HM.load_config(a.harvest_config)
    a.want_pads = True                       # the held check reads the hand's pads (toy_reach reads them for --grasp)
    # HOME is the tucked pose: every "home" of the toy flow (the first move, holds, the move at the end)
    R.HOME_Q = np.asarray(a.hcfg["home_q"], float).copy()
    b = a.hcfg["basket"]
    print(f"  HARVEST: autonomous (lowest apple first), waist scan {a.right_deg:+.0f} .. {a.left_deg:+.0f} deg at "
          f"{a.scan_speed_deg:g} deg/s; home = the arm tucked {np.round(R.HOME_Q, 2).tolist()}, the policy from READY "
          f"{np.round(a.hcfg['ready_q'], 2).tolist()}; deposit "
          + ("at home" if a.no_deposit else f"in the basket, waist in {b['waist_range_deg']} deg, palm down "
                                            f"({os.path.basename(a.harvest_config)})"))
    return a


class ToyHarvest(TA.ToyArmPose):
    def __init__(self, args):
        super().__init__(args)
        self.hcfg = args.hcfg
        mv = self.hcfg["moves"]
        # scripted moves: a measured acceleration over move_acc_stop must last acc_persist ticks (robot 2026-10-20:
        # a single 12.4 rad/s^2 sample stopped a 0.7 rad/s^2 move, nothing seen; readings arriving bunched make the
        # velocity difference over a few ms look huge) and is taken over at least acc_dt_min_s
        args.move_acc_stop = max(float(args.move_acc_stop), float(mv.get("acc_stop_rad_s2", 20.0)))
        self.acc_persist = int(mv.get("acc_persist", 3))
        self.acc_dt_min = float(mv.get("acc_dt_min_s", 0.015))
        if args.empty_travel is not None:
            self.hcfg["grasp_check"]["travel_empty"] = float(args.empty_travel)
        self.rmap = RMAP.ReachMap(os.path.join(HERE, self.hcfg["reach"]["map"]))
        self.basket_B = HM.basket_point(self.kin, self.hcfg)
        self._no_box_warned = False
        self.acal = None
        if args.apple_calib:
            import apple_pos_calib as AC
            path = AC.CALIB if args.apple_calib == "default" else args.apple_calib
            try:
                self.acal = AC.AppleCalib(path)
                print(f"  apple calibration {path}: model '{self.acal.m['model']}' from {self.acal.m['n']} grasps "
                      f"(leave-one-out {self.acal.m['loo_rms_cm']:.1f} cm)")
            except Exception as e:
                print(f"  apple calibration {path} unusable ({e}): none")
        self.hstate, self.hstate_t = "DETECT", time.monotonic()
        self.memory = []                         # [{"p": pelvis xyz, "tries": n, "done": bool, "centered": n}]
        self.n_picked = self.n_tries = 0
        self.t_cycle0, self.last_cycle_s = None, None
        self._carrying = False
        self.scan_trigger = None
        speak = getattr(self.io, "speak", None)
        if speak is not None:
            def harvest_speak(key, **kw):     # the picker's generic lines -> the harvest's own (sim2real/voice rl_*)
                key = GENERIC_VOICE.get(key, key)
                if key:
                    return speak(key, **kw)
            self.io.speak = harvest_speak
        self._too_far_said, self._too_far_new, self._scan_said = [], False, None
        self._trunk_pel = None
        pub = getattr(self.io, "publish", None)
        if pub is not None:
            def publish(doc, target):        # the dashboard: the harvest's own state ring and counters
                tr = getattr(self, "_trunk_pel", None)
                if tr is not None:              # the dashboard's MuJoCo view draws it (torso frame, as the plan)
                    try:
                        fr = self.frames(self.io.read())
                        to_t = lambda q: (fr.torso_R.T @ (np.asarray(q, float) - fr.torso_pos)).round(4).tolist()
                        doc = dict(doc, trunk={"p0": to_t(tr["p0"]), "p1": to_t(tr["p1"]), "r": round(tr["r"], 4),
                                               "rel": tr["rel"]})
                    except Exception:
                        pass
                doc = dict(doc, state=self.hstate, states=HARVEST_STATES,
                           state_since_s=round(time.monotonic() - self.hstate_t, 1),
                           metrics={"successful_drops": self.n_picked, "total_picks_attempted": self.n_tries,
                                    "last_cycle_time_s": self.last_cycle_s})
                return pub(doc, target)
            self.io.publish = publish

    # ---- the voice ----
    def say_rl(self, key, max_age_s=3.0, **kw):
        """One of the harvest's clips (sim2real/voice/rl_*.wav), dropped when it could only play late."""
        v = getattr(self, "voice", None)                 # run_toy's DualToyMixin
        if v is not None:
            v(key, max_age_s=max_age_s, **kw)

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

    def _recent_boxes(self):
        """The detector's boxes of the last FRESH_S: [(torso xyz, [x1, y1, x2, y2], label)], newest first; None
        when the detector sends no boxes at all (then nothing is filtered by picture or class)."""
        hist = list(getattr(self.io, "det_hist", []) or [])
        now = time.monotonic()
        out = [b for t, bs in reversed(hist) if now - t < FRESH_S for b in bs]
        return out if (out or any(bs for _, bs in hist)) else None

    def in_picture(self, p, r, fr, boxes):
        """Is the apple at p (pelvis) whole in the head camera's picture left-right, seen from frames fr? ->
        (whole, its bearing in the camera [rad, + left], why, label). The detector's newest box when one
        matches (boxes: _recent_boxes), else its sphere projected (the URDF camera)."""
        v = self.hcfg["view"]
        W, fx, cx = self._camera()
        pt = fr.torso_R.T @ (p - fr.torso_pos)
        box, label = None, None
        for q_t, bb, lab in (boxes or []):
            if np.linalg.norm(q_t - pt) < BOX_MATCH_M:
                box, label = bb, lab
                break
        if box is not None:
            x1, x2 = box[0], box[2]
            src = "box"
        else:
            c = fr.camera_R.T @ (p - fr.camera_pos)
            if c[0] <= 0.05:
                return False, 0.0, "behind the camera", label
            u = cx - fx * c[1] / c[0]
            rp = fx * float(r or self.args.apple_radius) / c[0]
            x1, x2 = u - rp, u + rp
            src = "projected"
        uc = 0.5 * (x1 + x2)
        bearing = math.atan((cx - uc) / fx)
        m = float(v["margin_px"])
        if x1 < m or x2 > W - m:
            return False, bearing, f"cut by the {'left' if x1 < m else 'right'} edge ({src} x {x1:.0f}..{x2:.0f} of {W:.0f} px)", label
        if abs(uc - W / 2) > (W / 2) * (1.0 - float(v["edge_frac"])):
            return False, bearing, f"its centre near the {'left' if uc < W / 2 else 'right'} edge ({src} x {uc:.0f} of {W:.0f} px)", label
        return True, bearing, src, label

    def headings_here(self, p, fr):
        """The palm heading offsets (deg) that reach the apple goal p (pelvis) with the waist of frames fr."""
        return self.rmap.headings_at(fr.torso_R.T @ (np.asarray(p, float) - fr.shoulder_pos))

    def aim(self, p, r, s, whole_now):
        """Where to harvest the apple at p (pelvis) from: the waist yaw within the scan range and the palm
        heading turn, both looked up (reach_map, the URDF camera) -- no IK. Cost: palm turn /
        palm_turn_cost_deg + waist turn / waist_turn_cost_deg; the apple must be whole in the picture there
        (at the current yaw: the detector's box says so). -> {"w", "h", "turn_deg"} or None (reachable at no yaw)."""
        a, c = self.args, self.hcfg["reach"]
        w_now = float(s.waist_q[0])
        best = None
        ws = np.radians(np.arange(a.right_deg, a.left_deg + 1e-6, AIM_STEP_DEG))
        for w in np.concatenate([[w_now], ws]):
            w3 = np.array([w, s.waist_q[1], s.waist_q[2]])
            fr = self.kin.frames(w3, R.HOME_Q)
            hs = self.headings_here(p, fr)
            if not hs:
                continue
            here = abs(w - w_now) < 1e-6
            whole = whole_now if here else self.in_picture(p, r, fr, None)[0]
            if not whole:
                continue
            cost = abs(hs[0]) / float(c["palm_turn_cost_deg"]) + abs(math.degrees(w - w_now)) / float(c["waist_turn_cost_deg"])
            if best is None or cost < best[0] - 1e-9:
                best = (cost, w, hs[0])
        if best is None:
            return None
        return {"w": best[1], "h": best[2], "turn_deg": math.degrees(best[1] - w_now)}

    def visible_apples(self, with_aim=True):
        """[{"p": pelvis xyz, "r", "whole", "bearing", "why", "aim"}] the camera sees NOW: red (the detector's
        class), not by the basket, reachable from some waist yaw of the scan (reach_map), not harvested /
        given up."""
        io = self.io
        try:
            snap = io.node.detection.get_snapshot()
        except Exception:
            return []
        now = time.time()
        s = io.read()
        fr = self.frames(s)
        boxes = self._recent_boxes()
        if boxes is None and not self._no_box_warned:
            self._no_box_warned = True
            self.brief("WARN", "the detector sends no picture boxes: no class (red) and no whole-in-picture check")
        labels = set(self.hcfg["view"].get("labels") or [])
        excl = float(self.hcfg["basket"].get("exclude_radius_m", 0.0))
        out = []
        seen = list(snap.seen_stamps) + [0.0] * len(snap.apples_all)
        radii = list(snap.radii) + [None] * len(snap.apples_all)
        for i, pt in enumerate(snap.apples_all):
            if now - float(seen[i]) > FRESH_S:
                continue
            p = fr.torso_R @ np.asarray(pt, float) + fr.torso_pos
            if excl > 0.0 and np.hypot(*(p - self.basket_B)[:2]) < excl:
                continue                                    # in / by the basket
            m = self._memo(p)
            if m is not None and (m["done"] or m["tries"] >= self.args.max_tries):
                continue
            whole, bearing, why, label = self.in_picture(p, radii[i], fr, boxes)
            if boxes is not None and labels and label not in labels:
                continue                                    # not a red apple (or no box of it now)
            x = {"p": p, "r": radii[i], "whole": whole, "bearing": bearing, "why": why, "label": label}
            if with_aim:
                x["aim"] = self.aim(p, radii[i], s, whole)
                if x["aim"] is None:
                    if not any(np.linalg.norm(q - p) < SAME_APPLE_M for q in self._too_far_said):
                        self._too_far_said.append(p.copy())
                        self._too_far_new = True
                    continue                                # out of the arm's reach at every yaw of the scan
            out.append(x)
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
        """Hold the waist command at target until the measured yaw is steady (within lock_range_deg over
        lock_window_s) and within lock_tol_deg of it, or lock_timeout_s."""
        L = self.hcfg["waist"]
        q_hold = self.q_sent if self.q_sent is not None else R.HOME_Q
        t0 = time.monotonic()
        hist = []
        while time.monotonic() - t0 < float(L["lock_timeout_s"]):
            if self.io.estop_why is not None:
                return False
            self.io.send_arm(q_hold, waist=target, waist_dq=0.0)
            now = time.monotonic()
            hist = [(t, w) for t, w in hist if now - t <= float(L["lock_window_s"])] + [(now, self.waist_now()[0])]
            ws = np.degrees([w for _, w in hist])
            if now - hist[0][0] >= 0.9 * float(L["lock_window_s"]) and np.ptp(ws) <= float(L["lock_range_deg"]) \
                    and abs(float(ws[-1]) - math.degrees(target)) <= float(L["lock_tol_deg"]):
                return True
            time.sleep(R.DT)
        self.brief("WARN", f"waist not steady in {float(L['lock_timeout_s']):g} s (at {math.degrees(self.waist_now()[0]):+.1f}, "
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

    def aim_turn(self, x):
        """Turn the waist to the yaw AIM chose for apple x. -> True (turned), None (E-STOP)."""
        m = self._memo(x["p"], create=True)
        m["centered"] += 1
        ai = x["aim"]
        why = "not whole in the picture: " + x["why"] if not x["whole"] else \
            ("out of reach from here" if not self.headings_here(x["p"], self.frames(self.io.read())) else "easier from there")
        self.hset("AIM", "turning the waist to reach / see the apple")
        self.say_rl("rl_aim")
        self.brief("INFO", f"apple {np.round(x['p'], 3).tolist()} ({why}): the waist turns {ai['turn_deg']:+.0f} deg, "
                           f"palm {ai['h']:+.0f} deg from the approach line there")
        return True if self.waist_to(ai["w"], "aim at the apple") else None

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
            if self._scan_said != self.scan_dir:
                self._scan_said = self.scan_dir
                self.say_rl("rl_scan_right" if self.scan_dir < 0 else "rl_scan_left")
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
            self.say_rl("rl_apple_found")
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
    def replay_path(self, pull_m):
        """The approach leg's own measured arm path (the onboard runner's 25 Hz state, or the laptop leg's rows),
        played BACKWARDS from the apple until the pocket is pull_m from where it ended -- no IK. Only the samples
        that move steadily away from the apple are kept (the hovering at its end drops out), lightly smoothed,
        moved to start exactly where the arm IS (the offset fading out along the path) and resampled evenly in
        pocket distance. -> (joint path [N, 7], its pocket length m) or (None, why)."""
        rec = getattr(self, "_leg_approach", None) or []
        if len(rec) < 3:
            return None, "no recorded approach"
        s = self.io.read()
        w3 = np.asarray(s.waist_q, float)
        qs = [np.asarray(q, float) for _, _, q in rec]
        pk = [AP.hand_frame(self.kin.frames(w3, q))[0] for q in qs]
        end = pk[-1]
        # backwards from the apple: the samples moving steadily away from it, until pull_m ALONG the path
        keep, d_last, arc = [len(qs) - 1], 0.0, 0.0
        for i in range(len(qs) - 2, -1, -1):
            d = float(np.linalg.norm(pk[i] - end))
            if d > d_last + 0.002:
                arc += float(np.linalg.norm(pk[i] - pk[keep[-1]]))
                keep.append(i)
                d_last = d
                if arc >= pull_m:
                    break
        if arc < 0.03:
            return None, f"the recorded approach moved only {arc * 100:.1f} cm"
        Q = np.array([qs[i] for i in keep])
        if len(Q) >= 5:
            Q[1:-1] = (Q[:-2] + Q[1:-1] + Q[2:]) / 3.0
        Q = Q + np.linspace(1.0, 0.0, len(Q))[:, None] * (np.asarray(s.arm_q, float) - Q[0])
        P = np.array([AP.hand_frame(self.kin.frames(w3, q))[0] for q in Q])
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        L = min(float(cum[-1]), float(pull_m))          # the last 25 Hz step may overshoot pull_m: cut there
        u = np.linspace(0.0, L, max(10, int(L / 0.002)))
        Qu = np.stack([np.interp(u, cum, Q[:, j]) for j in range(7)], axis=1)
        return Qu, L

    def detach(self):
        """The hand pulled back along its own approach (replay_path) by pull_m at pull_speed_m_s on a quintic
        time scale, at the detach's stiffer kp / kd (ramped in over GAIN_BACK_S, the arm kept where it is), the
        sag compensated. The pull slows smoothly to a stop as the commanded lead on a shoulder / elbow joint
        approaches tau_cap_nm (the stem holds: little stored spring); a joint faster than snap_dq = the stem
        broke -> the target is put where the arm IS for 0.4 s. Up to speed_stop rad/s is no safety stop here.
        -> "snapped" | "pulled" | "stalled" (the tension let off) | "skipped: ..." | "SAFETY STOP: ..." | "E-STOP: ..."."""
        D = self.hcfg["detach"]
        io, a = self.io, self.args
        path, L = self.replay_path(float(D["pull_m"]))
        if path is None:
            self.brief("WARN", f"detach: {L} -- no pull, the policy takes the apple back", fold_s=0.0)
            return f"skipped: {L}"
        s = io.read()
        w3 = np.asarray(s.waist_q, float)
        named = T.hand_named(s)
        kp, kd, cap = float(D["kp"]), float(D["kd"]), float(D["tau_cap_nm"])
        T_pull = max(1.0, 1.875 * L / float(D["pull_speed_m_s"]))
        tp = waited = 0.0
        q_catch, t_catch, peak, lead_max = None, None, 0.0, 0.0
        result = None
        q_des, q_des_prev = path[0], None
        self.brief("STEP", f"detach: the approach replayed backwards, {L * 100:.0f} cm in {T_pull:.1f} s at kp {kp:g} "
                           f"(the pull eases off near {cap:g} N m per joint, the snap caught)")
        t0 = io.now()
        t_next = t0
        while True:
            if io.estop_why is not None:
                return f"E-STOP: {io.estop_why}"
            s = io.read()
            q, jv = np.asarray(s.arm_q, float), float(np.max(np.abs(s.arm_dq)))
            peak = max(peak, jv)
            g = T.quintic((io.now() - t0) / GAIN_BACK_S)
            kk, dd = a.kp + g * (kp - a.kp), a.kd + g * (kd - a.kd)
            if jv > float(D["speed_stop"]):
                q_h = HM.gravity_target(self.kin, w3, q, kk, named)
                for _ in range(int(0.5 / R.DT)):
                    io.send_arm(q_h, kp=kk, kd=dd)
                    time.sleep(R.DT)
                self.q_sent = q_h
                return f"SAFETY STOP: joint speed {jv:.2f} rad/s > {float(D['speed_stop']):.2f} while detaching"
            if q_catch is None:
                f = T.quintic(tp / T_pull) * (len(path) - 1)
                i = int(min(f, len(path) - 2))
                q_des = path[i] + (f - i) * (path[i + 1] - path[i])
                tau = float(np.max(np.abs(q_des[:4] - q[:4]))) * kk
                lead_max = max(lead_max, tau / max(kk, 1.0))
                # the snap: the arm FASTER than the pull asks (the stem let go), not the pull's own speed
                v_des = (q_des - q_des_prev) / R.DT if q_des_prev is not None else np.zeros(7)
                q_des_prev = q_des.copy()
                if float(np.max(np.abs(np.asarray(s.arm_dq, float) - v_des))) > float(D["snap_dq"]) and tp > 0.1:
                    q_catch, t_catch = q.copy(), io.now()          # the stem broke: the spring let go here
                    q_des = q_catch
                else:
                    rate = float(np.clip((cap - tau) / (0.25 * cap), 0.0, 1.0))   # eases off near the cap
                    if rate <= 0.0:
                        waited += R.DT
                        if waited > float(D["stall_s"]):
                            result = "stalled"
                            break
                    tp = min(T_pull, tp + rate * R.DT)
                    if tp >= T_pull and tau < 0.5 * cap:
                        result = "pulled"
                        break
            else:
                q_des = q_catch
                if io.now() - t_catch >= 0.4:
                    result = "snapped"
                    break
            q_cmd = np.clip(HM.gravity_target(self.kin, w3, q_des, kk, named), self.lo, self.hi)
            io.send_arm(q_cmd, kp=kk, kd=dd)
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
                   f"detach: {result} after {tp:.1f} of {T_pull:.1f} s of pull ({waited:.1f} s waiting on the stem), "
                   f"peak joint speed {peak:.2f} rad/s, largest lead {lead_max:.3f} rad", fold_s=0.0)
        return result

    # ---- the left arm: opened during the harvest (off the left leg when the waist turns) ----
    def left_arm_to(self, q, why, timeout_s=8.0):
        """The LEFT arm through v20's own daemon-side joint trajectory (grasp_q; in RL mode the dual daemon
        passes the other arm's commands to v20), the right arm held meanwhile. -> True when done."""
        arm = getattr(self.io, "arm", None)
        if q is None or arm is None or not getattr(self.io, "commands", False):
            return False
        try:
            arm.reset_grasp_q("left")
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout_s:
                if self.io.estop_why is not None:
                    return False
                if arm.ensure_grasp_q("left", [np.asarray(q, float)], ramp_s=2.5, seg_s=0.8, hold_s=0.3):
                    self.brief("INFO", f"left arm {why} ({time.monotonic() - t0:.1f} s)", fold_s=0.0)
                    return True
                self.hold(self.q_sent if self.q_sent is not None else R.HOME_Q, 0.1)
        except Exception as e:
            self.brief("WARN", f"left arm not moved ({why}): {e}")
            return False
        self.brief("WARN", f"left arm: no confirmation in {timeout_s:.0f} s ({why})")
        return False

    def left_arm_open(self):
        q = self.hcfg.get("left_scan_q")
        if q is None:
            return
        try:
            self._left_q0 = self.io.arm._theta("left")
        except Exception:
            self._left_q0 = None
        self.left_arm_to(q, "opened for the harvest (off the left leg when the waist turns)")

    def left_arm_back(self):
        q0 = getattr(self, "_left_q0", None)
        if q0 is not None and self.hcfg.get("left_scan_q") is not None:
            self.left_arm_to(q0, "back where it was")

    # ---- is the apple in the hand? (the pads, else the fingers' travel) ----
    def _pad_reader(self):
        if getattr(self, "tcal", None) is None:
            path = getattr(self.args, "tactile_calib", None) or (T.TIO.DEFAULT_CALIB if os.path.exists(T.TIO.DEFAULT_CALIB) else None)
            if path:
                try:
                    self.tcal = T.TIO.TactileCalib.load(path)
                except Exception as e:
                    self.brief("WARN", f"tactile calibration unreadable ({e}): the fingers' travel decides")
        a = self.args
        return T.PadTactile(a.tactile_gain, a.tactile_top, a.tactile_map, getattr(self, "tcal", None))

    def grasp_baseline(self):
        """Right before the close: the pads' baseline (the arm held)."""
        G = self.hcfg["grasp_check"]
        self._pads = self._pad_reader()
        t0 = time.monotonic()
        while time.monotonic() - t0 < float(G["baseline_s"]):
            s = self.io.read()
            if s.pads is not None:
                self._pads.add_baseline(s.pads)
            self.hold(self.q_sent if self.q_sent is not None else R.HOME_Q, 0.05)
        self._pads_ok = self._pads.finish_baseline()

    def tactile_verdict(self, when):
        """The calibrated pads alone -> (True held / False empty / None not clear, text). Clear = at least 2 fingers
        >= held_n or one >= strong_n (held), every finger < empty_n (empty)."""
        G = self.hcfg["grasp_check"]
        io = self.io
        pads = getattr(self, "_pads", None)
        if pads is None or not getattr(self, "_pads_ok", False) or pads.noise is None \
                or not np.any(np.nan_to_num(pads.noise) > 0):
            return None, "pads unusable (no data / no noise)"
        F = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < float(G["read_s"]):
            s = io.read()
            if s.pads is not None:
                t9, _, _ = pads.taxels(s.pads)
                F.append([t9[0] + t9[1], t9[4] + t9[5], t9[2] + t9[3]])      # thumb, index, middle [N]
            self.hold(self.q_sent if self.q_sent is not None else R.HOME_Q, 0.05)
        if not F:
            return None, "no pad readings"
        f = np.mean(F, axis=0)
        txt = f"pads thumb {f[0]:.2f} / index {f[1]:.2f} / middle {f[2]:.2f} N"
        if int(np.sum(f >= float(G["held_n"]))) >= 2 or float(f.max()) >= float(G["strong_n"]):
            return True, txt
        if float(f.max()) < float(G["empty_n"]):
            return False, txt
        return None, txt + " (not clear)"

    def travel_verdict(self):
        G = self.hcfg["grasp_check"]
        tv = None
        try:
            tv = self.io.hand.grasp_travel_frac("right")
        except Exception:
            pass
        if tv is None:
            return None, "travel n/a"
        return bool(float(G["travel_min"]) <= tv < float(G["travel_empty"])), f"travel {tv * 100:.0f}%"

    def vision_verdict(self):
        """At READY the hand is in the head camera's picture (bottom right): the detector's red apple at the grasp
        pocket = held. Robot 2026-10-20 (camera recordings): held -> a red_apple box at the hand, empty -> none.
        -> (True / False / None no detections at all, text)."""
        G = self.hcfg["grasp_check"]
        io = self.io
        t0 = time.monotonic()
        while time.monotonic() - t0 < float(G["vision_s"]):
            self.hold(self.q_sent if self.q_sent is not None else R.HOME_Q, 0.1)
        s = io.read()
        fr = self.frames(s)
        pk = fr.torso_R.T @ (AP.hand_frame(fr)[0] - fr.torso_pos)
        hist = list(getattr(io, "det_hist", []) or [])
        recent = [bs for t, bs in hist if time.monotonic() - t <= float(G["vision_s"])]
        if not recent:
            return None, "no detector messages"
        labels = set(self.hcfg["view"].get("labels") or ["red_apple"])
        # where the pocket is in the PICTURE (FK + the detector's own camera calibration): a red apple's box over it,
        # and its centre within vision_m (an apple of the tree behind the hand overlaps in the picture, not in depth)
        import trunk_side as TS
        if getattr(self, "_tproj", None) is None:
            self._tproj = TS.Projector()
        u, v, _, _ = self._tproj.pixel(pk, 0.04)
        mpx = float(G.get("vision_px", 10))
        best, hit = None, False
        for bs in recent:
            for x, bb, lab in bs:
                if lab not in labels:
                    continue
                d = float(np.linalg.norm(x - pk))
                best = d if best is None else min(best, d)
                if bb[0] - mpx <= u <= bb[2] + mpx and bb[1] - mpx <= v <= bb[3] + mpx and d <= float(G["vision_m"]):
                    hit = True
        if hit:
            return True, f"a red apple's box over the hand (pixel {u:.0f},{v:.0f}), {best * 100:.0f} cm from the pocket"
        return False, (f"no red apple over the hand (pixel {u:.0f},{v:.0f})"
                       + (f"; nearest red apple {best * 100:.0f} cm" if best is not None else ""))

    def say_check(self, when, held, how):
        self.brief("INFO" if held else "WARN", f"{when}: {'HELD' if held else 'EMPTY'} by {how}", fold_s=0.0)
        self.say_rl("rl_got_it" if held else "rl_empty")

    # ---- the grip strength (the dashboard's 'grip' %: the picker's finger kp scaled; thumb_0 = its rotation, kept) ----
    def grip_pct_now(self):
        v = getattr(self.io, "grip_pct", None)
        return float(v) if v is not None else float(self.args.grip_pct)

    def grip_apply(self):
        app = getattr(self.io, "app", None)
        if app is None or not getattr(self.io, "commands", False):
            return
        if getattr(self, "_grip_kp0", None) is None:
            self._grip_kp0 = list(getattr(app, "hand_kp_vec", None) or [float(getattr(app, "hand_kp", 0.5))] * 7)
        pct = float(np.clip(self.grip_pct_now(), 30.0, 120.0))
        kp = [self._grip_kp0[0]] + [k * pct / 100.0 for k in self._grip_kp0[1:]]
        app.hand_kp_vec = kp
        self.brief("INFO", f"grip {pct:g}% (finger kp {np.round(kp, 3).tolist()}; 100% = {np.round(self._grip_kp0, 3).tolist()})",
                   fold_s=0.0)

    def grip_restore(self):
        app = getattr(self.io, "app", None)
        if app is not None and getattr(self, "_grip_kp0", None) is not None:
            app.hand_kp_vec = list(self._grip_kp0)

    def release_at_home(self):
        out = super().release_at_home()
        self.grip_restore()
        return out

    # ---- is the apple beside the trunk? (depth + colour, trunk_side.py) ----
    def trunk_check(self, p, r):
        """The trunk near apple p (pelvis) as a CYLINDER from the camera's picture + depth (trunk_side.fit_trunk), and
        the apple's side of it -> (verdict 'left' | 'right' (where the TRUNK is, seen from the robot) | 'none' |
        'n/a', text). The cylinder is kept (pelvis frame: it stays put when the waist turns) for the dashboard's
        MuJoCo view; the picture with its outline is saved in the run folder."""
        import trunk_side as TS
        rgbd = self.io.latest_rgbd() if hasattr(self.io, "latest_rgbd") else None
        if rgbd is None:
            return "n/a", "no camera picture"
        rgb, depth, age = rgbd
        r = float(r or self.args.apple_radius)
        try:
            if getattr(self, "_tproj", None) is None:
                self._tproj = TS.Projector()
            s = self.io.read()
            fr = self.frames(s)
            pt = fr.torso_R.T @ (np.asarray(p, float) - fr.torso_pos)
            hm = TS.hand_mask(self._tproj, TS.hand_points_torso(self.kin, np.asarray(s.waist_q, float),
                                                                 np.asarray(s.arm_q, float)), depth.shape)
            cyl = TS.fit_trunk(rgb, depth, self._tproj, pt, r, hm)
        except Exception as e:
            return "n/a", f"trunk check failed: {e}"
        if cyl is None:
            self._trunk_pel = None
            txt = f"no trunk found near it (picture {age:.2f} s old)"
            verdict = "none"
        else:
            to_pel = lambda q: fr.torso_R @ np.asarray(q, float) + fr.torso_pos
            self._trunk_pel = {"p0": to_pel(cyl["p0"]), "p1": to_pel(cyl["p1"]), "r": cyl["r"], "rel": cyl["apple_rel"]}
            txt = (f"the apple {cyl['apple_rel'].replace('_', ' ')} (off its axis {cyl['lateral_m'] * 100:+.0f} cm across "
                   f"the view, surfaces {cyl['gap_m'] * 100:.0f} cm apart; trunk r {cyl['r'] * 100:.1f} cm, "
                   f"{cyl['n']} points; picture {age:.2f} s old)")
            verdict = cyl["verdict"]
        try:
            TS.overlay(rgb, self._tproj, pt, r, cyl, f"{verdict}: {txt}").save(
                os.path.join(self.log_dir, f"trunk_{time.strftime('%H%M%S')}_{verdict}.jpg"), quality=90)
        except Exception:
            pass
        return verdict, txt

    def side_heading(self, p, fr, sign):
        """The largest palm turn (60 / 45 / 30 deg) toward `sign` that the reach map says reaches p from here."""
        hs = set(self.headings_here(p, fr))
        for h in (60.0, 45.0, 30.0):
            if sign * h in hs:
                return sign * h
        return None

    def choose_heading(self, x, h_aim):
        """The palm heading for apple x: AIM's, unless the trunk is beside it (or an earlier try was blocked): then
        from its free side."""
        a = self.args
        m = self._memo(x["p"], create=True)
        verdict, txt = ("n/a", "off") if a.trunk_side == "off" else self.trunk_check(x["p"], x["r"])
        if verdict != "n/a":
            self.brief("INFO", f"trunk check: {verdict} ({txt})", fold_s=0.0)
        sign = {"left": +1.0, "right": -1.0}.get(verdict, 0.0)     # trunk left -> from the right -> palm turned left
        why = f"the trunk on its {verdict}"
        if m.get("blocked", 0) > 0 and (sign == 0.0 or a.trunk_side != "act"):
            sign = +1.0 if m["blocked"] % 2 == 1 else -1.0          # blocked before, no trunk seen: one side, then the other
            why = f"blocked {m['blocked']}x before"
        elif a.trunk_side != "act":
            return h_aim
        if sign == 0.0:
            return h_aim
        st = self.io.read()
        fr = self.frames(st)
        h = self.side_heading(x["p"], fr, sign)
        if h is None:
            # another waist yaw of the scan where the side approach reaches it (and it stays whole in the picture)
            w_now = float(st.waist_q[0])
            best = None
            for w in np.radians(np.arange(a.right_deg, a.left_deg + 1e-6, AIM_STEP_DEG)):
                frw = self.kin.frames(np.array([w, st.waist_q[1], st.waist_q[2]]), R.HOME_Q)
                hw = self.side_heading(x["p"], frw, sign)
                if hw is not None and self.in_picture(x["p"], x["r"], frw, None)[0] \
                        and (best is None or abs(w - w_now) < abs(best[0] - w_now)):
                    best = (w, hw)
            if best is None:
                self.brief("WARN", f"{why}, but no side approach reaches it from any waist yaw (reach map): the usual one")
                return h_aim
            self.hset("AIM", "turning the waist so that the side approach reaches it")
            self.say_rl("rl_aim")
            self.brief("INFO", f"{why}: the side approach reaches it with the waist at {math.degrees(best[0]):+.0f} deg",
                       fold_s=0.0)
            if not self.waist_to(best[0], "the side approach"):
                return h_aim
            h = best[1]
        self.brief("INFO", f"{why}: reaching from its {'right' if sign > 0 else 'left'} side, the palm {h:+.0f} deg",
                   fold_s=0.0)
        return h

    # ---- the reactive fallback: an approach pushing on something before the apple ----
    def approach_watch_start(self):
        """The pads' baseline with the hand open in the cup (nothing touched yet), the watch armed."""
        self._blocked = None
        self._watch_lead_since = None
        self._watch_t = 0.0
        if self.args.reactive != "on":
            self._watch_pads = None
            return
        self._watch_pads = self._pad_reader()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.3:
            s = self.io.read()
            if s.pads is not None:
                self._watch_pads.add_baseline(s.pads)
            self.hold(self.q_sent if self.q_sent is not None else R.HOME_Q, 0.05)
        if not self._watch_pads.finish_baseline():
            self._watch_pads = None

    def approach_watch(self):
        """run_policy_onboard's tick during the APPROACH: blocked = (a pad >= block_n, or the arm's lag beyond its
        gravity sag >= block_lead_rad for block_s) while the palm is still > block_far_m from the apple's surface."""
        G = self.hcfg["reactive"]
        if self.args.reactive != "on" or self._blocked is not None:
            return
        now = time.monotonic()
        if now - self._watch_t < 0.05:
            return
        self._watch_t = now
        link = getattr(self, "_onboard", None)
        st = getattr(link, "state", None) or {}
        if st.get("phase") != "POLICY" or st.get("palm") is None or st.get("q") is None:
            return
        far = float(np.linalg.norm(np.asarray(st["palm"], float) - self.apple)) - float(self.args.apple_radius) \
            > float(G["block_far_m"])
        if not far:
            self._watch_lead_since = None
            return
        why = None
        s = self.io.read()
        pads = getattr(self, "_watch_pads", None)
        if pads is not None and s.pads is not None and pads.noise is not None and np.any(np.nan_to_num(pads.noise) > 0):
            t9, _, _ = pads.taxels(s.pads)
            if float(np.max(t9)) >= float(G["block_n"]):
                why = f"a pad pressed {float(np.max(t9)):.1f} N before the apple"
        q, qc = np.asarray(st["q"], float), np.asarray(st.get("q_cmd", st["q"]), float)
        g = self.kin.gravity_torque(s.waist_q, q)
        lead = float(np.max(np.abs((qc - q) - g / float(self.args.kp))[:4]))
        if lead >= float(G["block_lead_rad"]):
            self._watch_lead_since = self._watch_lead_since or now
            if now - self._watch_lead_since >= float(G["block_s"]):
                why = why or f"the arm held back {lead:.2f} rad beyond its sag for {now - self._watch_lead_since:.1f} s"
        else:
            self._watch_lead_since = None
        if why:
            self._blocked = why
            try:
                link.stop(f"blocked: {why}")
            except Exception:
                pass

    # ---- calibration data: where the apple REALLY was, from the closed hand ----
    def grasp_snapshot(self, goal):
        s = self.io.read()
        if s.hand_q is None:
            return None
        return {"t": time.time(), "arm_q": np.asarray(s.arm_q, float).tolist(), "waist": np.asarray(s.waist_q, float).tolist(),
                "hand_q": np.asarray(s.hand_q, float).tolist(), "goal_pelvis": np.asarray(goal, float).tolist()}

    def record_grasp(self, snap, held, detail):
        """One line of logs/sim2real/apple_calib/samples.jsonl (sim2real/apple_pos_calib.py fits on them): the perceived
        apple, the pocket, and the apple's centre from the closed hand's fingertips (the truth)."""
        if snap is None:
            return
        try:
            import apple_pos_calib as AC
            w3, q, hq = (np.asarray(snap[k], float) for k in ("waist", "arm_q", "hand_q"))
            fr = self.kin.frames(w3, q, T.hand_named(type("S", (), {"hand_q": hq})()))
            pocket = AP.hand_frame(fr)[0]
            D = np.asarray(self.apple_true, float)
            truth, rms = AC.fingertip_centre(self.kin, w3, q, hq, float(self.args.apple_radius), pocket)
            fr0 = self.kin.frames(w3, R.HOME_Q)
            Rt = fr0.torso_R
            c = fr0.camera_R.T @ (D - fr0.camera_pos)
            uv = [AC.CAM["cx"] - AC.CAM["fx"] * c[1] / c[0], AC.CAM["cy"] - AC.CAM["fy"] * c[2] / c[0]]
            rec = dict(snap, run=os.path.basename(self.log_dir), episode=time.strftime("%Y%m%d_%H%M%S"), held=bool(held),
                       detail=detail, perceived_pelvis=D.tolist(), pocket_pelvis=pocket.tolist(),
                       truth_pelvis=truth.tolist(), tip_fit_rms_m=rms, apple_radius=float(self.args.apple_radius),
                       residual_torso=(Rt.T @ (pocket - D)).tolist(), residual_truth_torso=(Rt.T @ (truth - D)).tolist(),
                       rel_shoulder_torso=(Rt.T @ (D - fr0.shoulder_pos)).tolist(), uv=uv,
                       offset_cm=np.asarray(self.apple_offset_now(), float).tolist(), src="toy")
            os.makedirs(AC.OUT_DIR, exist_ok=True)
            with open(AC.SAMPLES, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            self.brief("INFO", f"calibration record: the apple's centre from the fingertips is "
                               f"{np.round(Rt.T @ (truth - D) * 100, 1).tolist()} cm from the perceived one (torso frame; "
                               f"fit {rms * 1000:.0f} mm)", fold_s=0.0)
        except Exception as e:
            self.brief("WARN", f"calibration record not written: {e}")

    # ---- the policy legs ----
    def _legs_laptop(self, goals, leg, speed_stop=None):
        n0 = len(self.rows)
        keep = self.args.joint_speed_stop
        if speed_stop:
            self.args.joint_speed_stop = max(keep, speed_stop)
        try:
            out = self.run_goals(goals, leg)
        finally:
            self.args.joint_speed_stop = keep
        self._leg_rec = [(r["t"], goals[min(int(r["wp"]), len(goals) - 1)].name, np.asarray(r["q"], float))
                         for r in self.rows[n0:]]
        return out

    def _legs_onboard(self, goals, leg, speed_stop=None):
        from onboard_link import run_policy_onboard
        extra = {"goals": [g.to_dict() for g in goals]}
        if speed_stop:
            extra["joint_speed_stop"] = max(float(self.args.joint_speed_stop), float(speed_stop))
        tick = self.settle_tick if self._carrying else (self.approach_watch if leg == "approach" else None)
        out, stopped = run_policy_onboard(self, extra=extra, tick=tick)
        if leg == "approach" and getattr(self, "_blocked", None):
            return f"BLOCKED: {self._blocked}", False             # our own stop: the toy backs off and goes on
        return str(out), stopped

    def legs(self, goals, leg, speed_stop=None):
        """A policy leg (onboard or laptop); speed_stop: a joint-speed stop above the usual one for this leg."""
        f = self._legs_onboard if getattr(self, "_onboard_mode", False) else self._legs_laptop
        return f(goals, leg, speed_stop)

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
    def basket_goal(self, s):
        """The drop: the reachable point over the basket, the palm facing the ground, the fingers along the arm
        (the horizontal shoulder -> drop line)."""
        drop, u, B, short = HM.basket_drop(self.kin, self.hcfg, np.asarray(s.waist_q, float))
        fr = self.frames(s)
        up = fr.torso_R[:, 2]
        f = drop - fr.shoulder_pos
        f = f - np.dot(f, up) * up
        f /= max(np.linalg.norm(f), 1e-9)
        return AP.Goal("basket", drop, -up, f, True, False), short

    def deposit(self, carrying=True):
        """From HOME with the apple: the waist into the basket's yaw range (only as far as needed), READY, the
        policy straight to the drop (palm down), the hand opens, the policy back to READY, HOME, the waist
        back. -> (outcome, stopped)"""
        io, b = self.io, self.hcfg["basket"]
        w_back = self.waist_now()[0]
        lo, hi = (math.radians(float(v)) for v in b["waist_range_deg"])
        w_dep = float(np.clip(w_back, lo, hi))
        self.hset("DEPOSIT", f"the waist to {math.degrees(w_dep):+.0f} deg for the basket")
        self.say_rl("rl_basket")
        if abs(w_dep - w_back) > math.radians(0.5) and not self.waist_to(w_dep, "the basket"):
            return f"E-STOP: {io.estop_why}", True
        start = self.hcfg.get("deposit_start_q")
        err = self.joint_move([start], "home -> deposit start") if start is not None else self.to_ready()
        if err:
            return err, True
        s = io.read()
        ready = AP.home_goal(self.frames(s))
        ready.name = "deposit start" if start is not None else "ready"
        goal, short = self.basket_goal(s)
        self.hset("DEPOSIT", "arm-pose policy: over the basket, the palm down")
        self.brief("INFO", f"basket: the drop point {short * 100:.1f} cm short of the basket point (the arm's reach), "
                           f"palm down, fingers along the arm", fold_s=0.0)
        out, stopped = self.legs([goal], "deposit")
        if stopped or io.estop_why is not None:
            return out, True
        d_end = float(np.linalg.norm(AP.hand_frame(self.frames(io.read()))[0] - goal.target))
        if d_end > float(b["accept_m"]):
            self.brief("WARN", f"the hand ended {d_end * 100:.1f} cm from the drop point (> {float(b['accept_m']) * 100:.0f}):"
                               " released there anyway (better than carrying it on)")
        self.hset("RELEASE", "opening the hand over the basket")
        self.release_at_home()
        self.say_rl("rl_deposited")
        self._carrying = False
        self.hset("DEPOSIT", f"arm-pose policy: back to the {ready.name}")
        out_b, stopped = self.legs([ready], "return")
        if stopped or io.estop_why is not None:
            return out_b, True
        err = self.joint_move([R.HOME_Q], "deposit start -> home") if start is not None else self.to_home()
        if err:
            return err, True
        if abs(w_dep - w_back) > math.radians(0.5) and not self.waist_to(w_back, "back where the scan was"):
            return f"E-STOP: {io.estop_why}", True
        return "deposited", False

    # ---- one apple ----
    @staticmethod
    def turn_plan(plan, back, heading_deg, fr):
        """The palm normal of the staging and apple goals turned by heading_deg about the torso's vertical (+ =
        left), the staging point moved onto the turned approach line (as far from the apple as before)."""
        if abs(heading_deg) < 1e-6:
            return
        up = fr.torso_R[:, 2]
        h = math.radians(heading_deg)
        n0 = plan[-1].normal
        n = math.cos(h) * n0 + math.sin(h) * np.cross(up, n0) + (1 - math.cos(h)) * np.dot(up, n0) * up
        n /= np.linalg.norm(n)
        goal = plan[-1].target
        d = plan[0].target - goal
        d_h = d - np.dot(d, up) * up
        stage = goal - float(np.linalg.norm(d_h)) * (n - np.dot(n, up) * up) / max(np.linalg.norm(n - np.dot(n, up) * up), 1e-9) \
            + np.dot(d, up) * up
        for g in (plan[0], plan[1], back[0]):
            g.normal = n.copy()
        plan[0].target = stage.copy()
        back[0].target = stage.copy()

    def harvest_one(self, p, r, heading=0.0):
        """APPROACH, GRASP, DETACH, RETURN, DEPOSIT (RELEASE) for the apple at p (pelvis), the palm turned
        heading deg from the approach rule (reach_map). -> (outcome, stop)"""
        a, io = self.args, self.io
        self.apple = np.asarray(p, float)
        self.apple_true = self.apple.copy()
        if self.acal is not None:
            import apple_pos_calib as AC
            fr0 = self.frames(io.read())
            c = fr0.camera_R.T @ (self.apple - fr0.camera_pos)
            uv = [AC.CAM["cx"] - AC.CAM["fx"] * c[1] / c[0], AC.CAM["cy"] - AC.CAM["fy"] * c[2] / c[0]]
            d = self.acal.shift_torso(fr0.torso_R.T @ (self.apple - fr0.shoulder_pos), uv)
            self.apple = self.apple + fr0.torso_R @ d
            self.brief("INFO", f"apple calibration: perceived apple moved {np.round(d * 100, 1).tolist()} cm (torso)",
                       fold_s=0.0)
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
        if heading:
            self.turn_plan(plan, back, heading, fr)
            self.brief("INFO", f"palm turned {heading:+.0f} deg from the approach line (+ = left; the staging point on "
                               f"the turned line)", fold_s=0.0)
        self.hset("APPROACH", "arm-pose policy: staging, then the apple")
        self.hand_cup()
        self.approach_watch_start()
        out_a, stopped = self.legs(plan, "approach")
        self._leg_approach = list(getattr(self, "_leg_rec", []) or [])
        self.fr_at_apple = self.frames(io.read())
        if stopped or io.estop_why is not None:
            return out_a, True
        held = grasped = pending = False
        snap, detail = None, ""
        if str(out_a).startswith("BLOCKED"):
            m["blocked"] = m.get("blocked", 0) + 1
            self.brief("WARN", f"approach {out_a}: backing off; the next try comes from the side (try {m['tries']} of "
                               f"{a.max_tries})")
        d_end = float(np.linalg.norm(AP.hand_frame(self.fr_at_apple)[0] - plan[-1].target))
        if str(out_a).startswith("APPLE") and d_end > GRASP_MAX_ERR:
            self.brief("WARN", f"the hand ended {d_end * 100:.1f} cm from the apple goal (> {GRASP_MAX_ERR * 100:.0f}): "
                               f"no grasp, a missed try ({m['tries']} of {a.max_tries})")
        elif str(out_a).startswith("APPLE"):
            self.hset("GRASP", "closing the hand (the picker's close)")
            self.say_rl("rl_grasping")
            self.grasp_baseline()
            self.grip_apply()
            out_g = self.grasp_close()
            if out_g.startswith("E-STOP"):
                return out_g, True
            snap = self.grasp_snapshot(plan[-1].target)
            self.hset("DETACH", "pulling the apple off its stem (slow, stiff, the snap caught)")
            self.say_rl("rl_pulling")
            out_d = self.detach()
            grasped = True
            if out_d.startswith(("E-STOP", "SAFETY")):
                return out_d, True
            if out_d == "stalled":
                m["blocked"] = m.get("blocked", 0) + 1                # a trunk in the hand does not come off either
                self.say_rl("rl_stem_strong")
                self.brief("WARN", f"the stem held (or the hand has the trunk too): the hand lets go; the next try comes "
                                   f"from the side (try {m['tries']} of {a.max_tries})")
                self.hset("RELEASE", "the stem held: letting the apple go")
                self.release_at_home()
            else:
                v, tact = self.tactile_verdict("after the detach")
                tv_held, trav = self.travel_verdict()
                pending = v is None                      # the pads not clear: the camera decides at READY
                held = True if pending else v
                detail = f"after the detach: pads {tact}; {trav}"
                if not pending:
                    self.say_check("after the detach", held, f"the pads ({tact}; {trav})")
                    self.record_grasp(snap, held, detail)
                else:
                    self.brief("INFO", f"after the detach: the pads do not decide ({tact}; {trav}): the camera "
                                       f"checks at ready", fold_s=0.0)
                if not held:
                    self.brief("WARN", f"nothing in the hand after the detach (try {m['tries']} of {a.max_tries}): "
                                       f"no deposit")
                    self.release_at_home()
        self._carrying = held
        self.hset("RETURN", "arm-pose policy: back through the staging point to ready")
        # a stem that lets go late (the detach skipped, or pulled without a snap) jerks the arm here: not a stop
        out_b, stopped = self.legs(back, "return",
                                   speed_stop=self.hcfg["detach"].get("return_speed_stop") if grasped else None)
        if stopped or io.estop_why is not None:
            return out_b, True
        if held and grasped and pending:
            # the hand at READY is in the camera's picture: the detector's apple in it, or not
            self.hset("RETURN", "showing the hand to the camera: is the apple in it?")
            v, how = self.vision_verdict()
            if v is None:
                v, trav = self.travel_verdict()
                how = f"the fingers' travel ({trav}; {how})"
                v = True if v is None else v
            else:
                how = f"the camera ({how})"
            held = v
            self.say_check("at ready", held, how)
            self.record_grasp(snap, held, detail + f"; at ready: {how}")
            if not held:
                self.brief("WARN", f"nothing in the hand (try {m['tries']} of {a.max_tries}): no deposit, the next apple")
                self.hset("RELEASE", "nothing in the hand: opening it")
                self.release_at_home()
                self._carrying = False
        self.hset("RETURN", "the arm home")
        err = self.to_home()
        if err:
            return err, True
        if held and grasped and not pending:
            v, tact = self.tactile_verdict("back at home")
            if v is False:
                self.brief("WARN", f"the apple was lost on the way back ({tact}): no deposit, the next apple")
                self.say_rl("rl_dropped")
                self.release_at_home()
                self._carrying = False
                return f"lost on the way back ({out_a})", False
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
        self.say_rl("rl_harvest_start", max_age_s=5.0)
        self.left_arm_open()
        if a.waist_kp:
            self.io.waist_kp = float(a.waist_kp)
            self.brief("INFO", f"waist kp {a.waist_kp:g} while the harvest moves it (the daemon's own 380 otherwise; "
                               f"needs the robot's dual-2 daemon)", fold_s=0.0)
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
            if self._too_far_new:
                self._too_far_new = False
                self.say_rl("rl_too_far")
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
                    self.say_rl("rl_no_more", max_age_s=5.0)
                    break
                continue
            x = self.lowest(apples)
            ai = x["aim"]
            m = self._memo(x["p"], create=True)
            if abs(ai["turn_deg"]) >= MIN_CENTER_DEG:
                if m["centered"] < MAX_CENTER:
                    c = self.aim_turn(x)
                    if c is None:
                        stopped, why = True, f"E-STOP: {io.estop_why}"
                        break
                    continue                    # look again from there
                fr = self.frames(io.read())
                hs = self.headings_here(x["p"], fr)
                if not (x["whole"] and hs):
                    self.brief("WARN", f"apple {np.round(x['p'], 3).tolist()}: still not reachable / whole after "
                                       f"{m['centered']} waist turns: skipped")
                    self.give_up(x["p"])
                    continue
                ai = {"w": self.waist_now()[0], "h": hs[0], "turn_deg": 0.0}
            self.brief("INFO", f"next: apple {np.round(x['p'], 3).tolist()} (pelvis), the lowest of {len(apples)}"
                               + (f", palm {ai['h']:+.0f} deg" if ai["h"] else ""))
            h = self.choose_heading(x, ai["h"])
            out, stop = self.harvest_one(x["p"], x["r"], heading=h)
            self.brief("DONE" if out == "picked" else "WARN", f"apple {self.n_tries}: {out} ({self.n_picked} picked)")
            if stop:
                stopped, why = True, out
                break
        if not stopped and io.estop_why is None and not a.deposit_test:
            self.hset("SCAN", "the waist home")
            self.waist_to(0.0, "home")
        if io.estop_why is None:
            self.left_arm_back()
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
