#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
toy_follow.py -- FOLLOW an apple held in a person's hand, GRASP it on command, HOLD it, let it go on command.
The arm-pose policy (AppleArmPose, toy_arm_pose.py) runs on the LAPTOP with its goal updated every tick from
the head camera; the waist turns in parallel (its own thread) to keep the apple in view and in reach. Nothing
on the Jetson changes: the same dual-daemon RL stream as every toy (run_toy.py, kind "follow").

States (the dashboard's state ring):
  REACH    (the start) the palm follows the apple, --standoff_cm in front of it, the palm facing it; the policy
           pauses once there (the trained stop test) and starts again when the apple moves --resume_cm away.
           Apple not seen for --lost_s: the arm holds where it is. Out of the arm's reach (reach_map.py, a
           table look-up -- no IK): the arm holds, the waist turns if that brings it into reach.
  GRASP    the pocket onto the apple itself (from wherever the arm is: reached first when it is far); within
           --grasp_lock_cm the goal is fixed (the hand hides the apple from the camera) and the waist stops;
           reached (or close enough at the waypoint timeout) -> the picker's close -> HOLD. Too far at the
           timeout -> back to REACH.
  HOLD     the arm and the waist hold their position, the hand keeps the grip (the picker's settle).
  RELEASE  (on "reach" from HOLD) the hand opens, then REACH again.
WAIST (thread "waist_follow", beside the policy): from the tracked apple it decides the waist yaw -- nothing while
the apple sits comfortably in the picture (--view_keep of the half-width) and in reach; else the nearest yaw
(within --right_deg .. --left_deg) that centres it (--view_aim) AND reaches it, else one that only centres it.
Apple lost after leaving the picture: the waist turns toward the side it left by (--search_s). The thread plans
quintic yaw trajectories (speed / acceleration limited); the 50 Hz loop streams them with the arm.

COMMANDS -- the laptop microphone (follow_listener.py: vosk, offline; a spoken command counts only when the whole
utterance is the command, e.g. "grasp" / "robot, grasp it" -- not a sentence that contains the word; --listen off to
disable) or typed in the launcher terminal + ENTER:
    reach  (r, follow, release, open; es: alcanza, sigue, suelta, abre)
    grasp  (g, grab; es: agarra, coge, toma)
    end    (e, q, finish; es: termina, fin -- the arm home, the run ends)
English and Spanish are heard side by side (two vosk models, vosk-model-small-es-0.42 added 2026-10-21).
The dashboard's END button ends it too; STOP / the remote (E-STOP) as always; Ctrl-C lowers the arm.

    apple_picker/g1_launch.sh toy -- follow --speed 0.3
(default --policy: the robot's AppleArmPose v1 export, logs/rsl_rl/apple_arm_pose/2026-10-19_13-31-55)
Offline: run_toy.py follow --mock --realtime --speed 0.3 --mock_apples 0.4,-0.1,0 --fake_apple 0.40 -0.18 0.05
         --fake_motion_cm 8 --script 15:grasp,30:reach,45:end --listen off --yes
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import queue
import select
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arm_pose_io as AP  # noqa: E402
import toy_harvest as H  # noqa: E402

TA, S, T, R = H.TA, H.S, H.T, H.R
C_CYN, C_GRN, C_RED, C_YEL = T.C_CYN, T.C_GRN, T.C_RED, T.C_YEL

#: the dashboard's state ring (toy_harvest's publish reads its module's HARVEST_STATES: this process only)
FOLLOW_STATES = ["REACH", "GRASP", "HOLD", "RELEASE"]
H.HARVEST_STATES = FOLLOW_STATES
#: an apple sighting moves the tracked apple this share of the way (per camera frame, ~10 Hz)
TRACK_ALPHA = 0.6
#: deg between the waist yaws of the follower's look-up table
WAIST_TABLE_STEP_DEG = 0.5
#: s between two waist decisions (the thread's look-ups)
WAIST_DECIDE_S = 0.1
#: rad: palm normal error that restarts a paused policy (the apple turned around the torso)
RESUME_ANG = math.radians(25.0)
#: GRASP: s the pocket must stay within --grasp_close_cm (palm slower than GRASP_STILL_V m/s) before the close
GRASP_SETTLE_S = 0.3
GRASP_STILL_V = 0.05
#: s the arm is held with the hand open after a release, before following again
RELEASE_HOLD_S = 1.0
#: typed / spoken words -> commands
TERM_WORDS = {"alcanza": "reach", "agarra": "grasp", "termina": "end",
              "reach": "reach", "r": "reach", "follow": "reach", "f": "reach", "release": "reach", "open": "reach",
              "grasp": "grasp", "g": "grasp", "grab": "grasp",
              "end": "end", "e": "end", "q": "end", "quit": "end", "finish": "end", "home": "end"}
LISTENER = os.path.join(HERE, "follow_listener.py")
#: the detector's fruit, torso frame (fruits_to_torso.py), one message per picture
DETECTED_TOPIC = "/fruits_detected_torso"
HELP = ("commands: reach (r) | grasp (g) | end (q) -- type + ENTER here, or say reach / grasp / finish "
        "(alcanza / agarra / termina)")


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--standoff_cm", type=float, default=6.0,
                   help="REACH: the palm this far in front of the apple (pocket goal pulled back along the palm normal)")
    p.add_argument("--grasp_depth_cm", type=float, default=0.0,
                   help="GRASP: the pocket goal this far past the apple centre (negative = short of it)")
    p.add_argument("--grasp_lock_cm", type=float, default=4.0,
                   help="GRASP: within this the goal stops following the camera (the hand hides the apple) and the waist stops")
    p.add_argument("--grasp_close_cm", type=float, default=3.0,
                   help="GRASP: close once the pocket stayed within this of the apple, the palm still, for "
                        "GRASP_SETTLE_S (the trained stop test wants 2 cm; mock / robot policies hover at 2.5-3)")
    p.add_argument("--labels", default="apple",
                   help="detector classes followed: comma-separated substrings of the class label (default: any *apple*)")
    p.add_argument("--max_apple_m", type=float, default=0.80, help="ignore fruit further than this from the right shoulder")
    p.add_argument("--det_latency", type=float, default=0.20,
                   help="s from the picture to its detection here: the sighting is put in the pelvis frame with the waist "
                        "yaw of that moment")
    p.add_argument("--lost_s", type=float, default=0.8, help="no sighting for this long = the apple is lost")
    p.add_argument("--track_jump_m", type=float, default=0.20,
                   help="a sighting further than this from the tracked apple is another fruit (until the track is lost)")
    p.add_argument("--goal_deadband_cm", type=float, default=1.5, help="the policy goal moves only when the apple moved this much")
    p.add_argument("--resume_cm", type=float, default=4.0, help="REACH: a paused policy starts again beyond this")
    p.add_argument("--waist_follow", choices=["on", "off"], default="on")
    p.add_argument("--waist_speed_deg", type=float, default=15.0, help="waist peak speed, deg/s")
    p.add_argument("--waist_acc_deg", type=float, default=30.0, help="waist peak acceleration, deg/s^2")
    p.add_argument("--view_keep", type=float, default=0.55,
                   help="the waist stays while the apple's centre is within this share of the picture's half-width")
    p.add_argument("--view_aim", type=float, default=0.25, help="... and when it turns, it brings it within this share")
    p.add_argument("--search_s", type=float, default=6.0,
                   help="s the waist searches toward the side a lost apple left the picture by (0 = no search)")
    p.add_argument("--listen", choices=["mic", "off"], default="mic", help="the laptop microphone (follow_listener.py)")
    p.add_argument("--mic_device", default="default", help="ALSA capture device (arecord -L)")
    p.add_argument("--mic_conf", type=float, default=0.5, help="vosk word confidence needed for a spoken command")
    p.add_argument("--mic_mode", choices=["both", "free", "grammar"], default="both",
                   help="follow_listener.py: both = a grammar recogniser picks the command, a free one checks it was "
                        "not a sentence (default); free = transcription only; grammar = any command word")
    p.add_argument("--mic_lang", default="en,es", help="languages listened to: en, es or en,es")
    p.add_argument("--session_s", type=float, default=1800.0, help="the run ends after this long")
    p.add_argument("--script", default=None, help="tests: timed commands, s after GO, e.g. 15:grasp,30:reach,45:end")
    p.add_argument("--fake_apple", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="tests: a synthetic apple (pelvis frame) instead of the detector, seen through the same path")
    p.add_argument("--fake_motion_cm", type=float, default=0.0, help="tests: the fake apple sways left-right this much")
    p.add_argument("--fake_period_s", type=float, default=10.0)
    mine, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    a = H.parse_args()
    for k, v in vars(mine).items():
        setattr(a, k, v)
    a.time_limit = a.session_s
    a.labels = [x.strip() for x in a.labels.split(",") if x.strip()]
    print(f"  FOLLOW: the palm {a.standoff_cm:g} cm in front of the apple in the hand; grasp / hold / release on "
          f"command ({'microphone + ' if a.listen == 'mic' else ''}terminal); waist "
          + (f"follows in {a.right_deg:+.0f} .. {a.left_deg:+.0f} deg at {a.waist_speed_deg:g} deg/s"
             if a.waist_follow == "on" else "held"))
    return a


# ─────────────────────────────────────────────
#  The waist: its own thread
# ─────────────────────────────────────────────

class WaistFollower(threading.Thread):
    """Decides the waist yaw from the tracked apple (look-up tables, no FK here: the torso / camera / shoulder
    frames depend on the waist only and are tabulated at start) and plans quintic trajectories to it. The main
    loop streams command() (yaw, yaw rate at io.now()) in every packet with the arm."""

    def __init__(self, toy, w0, roll_pitch):
        super().__init__(daemon=True, name="waist_follow")
        a = self.a = toy.args
        self.io, self.rmap = toy.io, toy.rmap
        self.lo, self.hi = math.radians(a.right_deg), math.radians(a.left_deg)
        self.vmax, self.amax = math.radians(a.waist_speed_deg), math.radians(a.waist_acc_deg)
        self.cam = toy._camera()                                  # (W, fx, cx)
        self.ys = np.radians(np.arange(a.right_deg, a.left_deg + 1e-6, WAIST_TABLE_STEP_DEG))
        self.tab = []
        for w in self.ys:
            fr = toy.kin.frames([float(w), float(roll_pitch[0]), float(roll_pitch[1])], R.HOME_Q)
            self.tab.append((fr.torso_R.copy(), fr.shoulder_pos.copy(), fr.camera_R.copy(), fr.camera_pos.copy()))
        w0 = float(np.clip(w0, self.lo, self.hi))
        self.lock = threading.Lock()          # traj / apple
        self.dlock = threading.Lock()         # a decision vs freeze()
        self.traj, self.t0 = H.Quintic(w0, 0.0, 0.0, w0, 0.5), self.io.now()
        self.apple, self.fresh = None, False
        self.frozen = False
        self.enabled = a.waist_follow == "on"
        self.last_u, self.lost_since, self.searching = None, None, False
        self._seen_t = None
        self.notes = collections.deque(maxlen=50)
        self._note_t, self._note_why, self._quiet = -1e9, "", 0
        self.n_turns = 0
        self._stop = False

    # ---- read by the main loop ----
    def command(self):
        with self.lock:
            tr, t0 = self.traj, self.t0
        p, v, _ = tr.at(self.io.now() - t0)
        if not self.lo <= p <= self.hi:
            return float(np.clip(p, self.lo, self.hi)), 0.0
        return float(p), float(v)

    def state(self):
        with self.lock:
            tr, t0 = self.traj, self.t0
        return tr.at(self.io.now() - t0)

    def set_apple(self, p, fresh, w_meas=None, t_seen=None):
        """The tracked apple (pelvis), fresh or not, the measured yaw now and the time of its last sighting."""
        with self.lock:
            self.apple = None if p is None else np.asarray(p, float).copy()
            self.fresh = bool(fresh)
            if p is not None and t_seen is not None and w_meas is not None and t_seen != self._seen_t:
                # where it really was in the picture at its newest sighting (the measured yaw, not the planned one):
                # what tells a lost apple that left by an edge from one hidden in mid-picture
                self._seen_t = t_seen
                self.last_u = self._u(self._idx(float(w_meas)), self.apple)

    def freeze(self):
        """Stop the waist (a quintic brake from its current speed) and make no more decisions."""
        with self.dlock:
            self.frozen = True
            p, v, acc = self.state()
            self._retarget(p + v * abs(v) / (2.0 * self.amax))

    def unfreeze(self):
        self.frozen = False
        self.last_u, self.lost_since, self.searching = None, None, False

    def settled(self):
        with self.lock:
            tr, t0 = self.traj, self.t0
        return self.io.now() - t0 >= tr.T

    def stop(self):
        self._stop = True

    # ---- the decisions ----
    def _retarget(self, goal, vmax=None):
        """A quintic from the current yaw / rate / acceleration to goal. The start is clamped inside the limits
        (a start at the acceleration limit leaves plan_quintic no feasible time: it returned a ~100x stretched
        polynomial that drifted on at the old rate -- mock 2026-10-21, 28 deg/s into the yaw clip); a plan that
        still breaks them is replaced by one from zero acceleration."""
        goal = float(np.clip(goal, self.lo, self.hi))
        vmax = vmax or self.vmax
        now = self.io.now()
        with self.lock:
            p, v, acc = self.traj.at(now - self.t0)
        p = float(np.clip(p, self.lo, self.hi))
        v = float(np.clip(v, -self.vmax, self.vmax))
        acc = float(np.clip(acc, -0.9 * self.amax, 0.9 * self.amax))
        tr = H.plan_quintic(p, v, acc, goal, vmax, self.amax, t_min=0.3)
        pv, pa = tr.peaks()
        if pv > 1.05 * max(vmax, abs(v)) or pa > 1.05 * self.amax or tr.T > 30.0:
            tr = H.plan_quintic(p, v, 0.0, goal, vmax, self.amax, t_min=0.3)
        with self.lock:
            self.traj, self.t0 = tr, now

    def _idx(self, w):
        return int(np.clip(round((w - self.ys[0]) / math.radians(WAIST_TABLE_STEP_DEG)), 0, len(self.ys) - 1))

    def _u(self, i, p):
        """The apple's column in the picture with the waist at self.ys[i] (None: behind the camera)."""
        _, _, cR, cp = self.tab[i]
        c = cR.T @ (p - cp)
        if c[0] <= 0.05:
            return None
        W, fx, cx = self.cam
        return cx - fx * c[1] / c[0]

    def _reach(self, i, p):
        tR, sh, _, _ = self.tab[i]
        return bool(self.rmap.headings_at(tR.T @ (p - sh)))

    def run(self):
        while not self._stop:
            time.sleep(WAIST_DECIDE_S)
            if not self.enabled or self.frozen:
                continue
            try:
                with self.dlock:
                    if not self.frozen:
                        self._decide()
            except Exception as e:                    # never kill the thread: the main loop streams command()
                self.notes.append(("WARN", f"waist decision failed: {e}"))

    def _decide(self):
        with self.lock:
            p_app, fresh, w_plan = self.apple, self.fresh, self.traj.p1
        W, fx, cx = self.cam
        half = 0.5 * W
        i = self._idx(w_plan)
        if p_app is not None and fresh:
            self.lost_since = None
            self.searching = False
            u = self._u(i, p_app)
            ok_view = u is not None and abs(u - cx) <= self.a.view_keep * half
            ok_reach = self._reach(i, p_app)
            if ok_view and ok_reach:
                return
            us = [self._u(j, p_app) for j in range(len(self.ys))]
            order = np.argsort(np.abs(self.ys - w_plan))
            centred = [j for j in order if us[j] is not None and abs(us[j] - cx) <= self.a.view_aim * half]
            both = [j for j in centred if self._reach(j, p_app)]
            if both:
                j, how = both[0], "centred and in reach"
            elif centred:
                if ok_view:
                    return                            # in view but out of reach everywhere it is centred: stay
                j, how = centred[0], "centred (out of the arm's reach there too)"
            else:
                seen = [j for j in range(len(self.ys)) if us[j] is not None]
                if not seen:
                    return
                j = min(seen, key=lambda k: abs(us[k] - cx))
                how = "as near the centre as the waist limits allow"
            goal = float(self.ys[j])
            if abs(goal - w_plan) > math.radians(0.75):
                why = "near the picture's edge" if not ok_view else "out of the arm's reach"
                self._retarget(goal)
                self.n_turns += 1
                now = self.io.now()
                if why != self._note_why or now - self._note_t >= 2.0:     # small re-aims while tracking: one line / 2 s
                    more = f" (+{self._quiet} smaller re-aims since the last line)" if self._quiet else ""
                    self.notes.append(("INFO", f"waist {math.degrees(w_plan):+.0f} -> {math.degrees(goal):+.0f} deg: "
                                               f"the apple was {why}; there it is {how}{more}"))
                    self._note_t, self._note_why, self._quiet = now, why, 0
                else:
                    self._quiet += 1
            return
        # not seen (now): search toward the side it left the picture by, once
        now = self.io.now()
        if self.lost_since is None:
            self.lost_since = now
        if self.a.search_s <= 0:
            return
        if self.searching:
            if now - self.lost_since > self.a.search_s:
                self.searching = False
                self.last_u = None
                p, v, _ = self.state()
                self._retarget(p + v * abs(v) / (2.0 * self.amax))
                self.notes.append(("WARN", f"apple not found in {self.a.search_s:g} s of search: the waist stops "
                                           f"at {math.degrees(p):+.0f} deg"))
            return
        if self.last_u is not None and abs(self.last_u - cx) > self.a.view_keep * half:
            side = 1.0 if self.last_u < cx else -1.0
            goal = self.hi if side > 0 else self.lo
            self.searching = True
            self._retarget(goal, vmax=0.5 * self.vmax)
            self.notes.append(("INFO", f"apple lost at the picture's {'left' if side > 0 else 'right'} edge: the "
                                       f"waist searches that way (to {math.degrees(goal):+.0f} deg, {self.a.search_s:g} s)"))


# ─────────────────────────────────────────────
#  The apple in the hand
# ─────────────────────────────────────────────

class AppleTrack:
    """One apple followed through the detector's sightings (pelvis frame), the newest weighted TRACK_ALPHA."""

    def __init__(self, a):
        self.a = a
        self.p, self.r, self.label, self.t_seen, self.n = None, None, "", -1e9, 0

    def fresh(self, t):
        return self.p is not None and t - self.t_seen <= self.a.lost_s

    def update(self, cands, t, ref):
        """cands: [(pelvis xyz, radius, label)]; ref: where to pick a new apple near (the right shoulder)."""
        if not cands:
            return None
        if self.p is not None:
            d = [float(np.linalg.norm(c[0] - self.p)) for c in cands]
            k = int(np.argmin(d))
            if d[k] <= self.a.track_jump_m:
                p, r, lab = cands[k]
                self.p = self.p + TRACK_ALPHA * (p - self.p)
                self.r, self.label, self.t_seen = r or self.r, lab, t
                self.n += 1
                return "same"
            if self.fresh(t):
                return None                           # another fruit: keep the one followed
        p, r, lab = min(cands, key=lambda c: float(np.linalg.norm(c[0] - ref)))
        self.p, self.r, self.label, self.t_seen, self.n = np.asarray(p, float).copy(), r, lab, t, 1
        return "new"


# ─────────────────────────────────────────────
#  Commands: terminal + microphone
# ─────────────────────────────────────────────

class Commands:
    def __init__(self, toy):
        self.toy, self.q = toy, queue.Queue()
        self.on = False
        self.proc = None
        self.mic_ok = False

    def start(self):
        a = self.toy.args
        self.on = True
        threading.Thread(target=self._terminal, daemon=True, name="follow_terminal").start()
        if a.listen == "mic":
            self._start_listener()
        if a.script:
            for item in a.script.split(","):
                ts, c = item.split(":")
                threading.Timer(float(ts), lambda c=c: self.q.put(("script", c.strip(), c.strip()))).start()

    def _start_listener(self):
        a, toy = self.toy.args, self.toy
        try:
            err = open(os.path.join(toy.log_dir, "listener.log"), "a", buffering=1)
            self.proc = subprocess.Popen([sys.executable, "-u", LISTENER, "--device", a.mic_device, "--min_conf",
                                          str(a.mic_conf), "--mode", a.mic_mode, "--lang", a.mic_lang],
                                         stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL, text=True)
        except Exception as e:
            toy.brief("WARN", f"microphone listener not started ({e}): terminal commands only", fold_s=0.0)
            return
        threading.Thread(target=self._mic, daemon=True, name="follow_mic").start()

    def _mic(self):
        for line in self.proc.stdout:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("ready"):
                self.mic_ok = True
                self.toy.brief("INFO", "microphone listening (laptop): say reach / grasp / finish (alcanza / agarra / termina)", fold_s=0.0)
            elif d.get("cmd") and self.on:
                self.q.put(("voice", d["cmd"], f"'{d.get('text')}' ({d.get('conf')})"))
        if self.on:
            self.toy.brief("WARN", f"the microphone listener ended (code {self.proc.poll()}; see listener.log): "
                                   "terminal commands only", fold_s=0.0)

    def _terminal(self):
        while self.on:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
            except Exception:
                return
            if not r or not self.on:
                continue
            line = sys.stdin.readline()
            if not line:
                return                                # stdin closed
            w = line.strip().lower()
            c = TERM_WORDS.get(w)
            if c:
                self.q.put(("terminal", c, w))
            else:
                sys.__stdout__.write(f"{C_CYN}  {HELP}{T.C_OFF}\n")
                sys.__stdout__.flush()

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        self.on = False
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(2.0)
            except Exception:
                self.proc.kill()


# ─────────────────────────────────────────────
#  The toy
# ─────────────────────────────────────────────

class ToyFollow(H.ToyHarvest):
    def __init__(self, args):
        super().__init__(args)
        self.hstate = "REACH"
        self.track = AppleTrack(args)
        self._det_q = collections.deque(maxlen=200)       # (io time of arrival, JSON text)
        self._whist = collections.deque(maxlen=200)       # (io time, waist q3): the yaw at a picture's moment
        self._fake_due = []                               # --fake_apple: (delivery time, JSON text)
        self._fake_next = 0.0
        self.waist = None
        self.cmds = Commands(self)
        self.n_grasps = 0
        self.follow_rows = []                             # -> follow.npz (run_goals' rows stay in self.rows)
        node = getattr(self.io, "node", None)
        if node is not None and args.fake_apple is None:
            from std_msgs.msg import String
            node.create_subscription(String, DETECTED_TOPIC,
                                     lambda m: self._det_q.append((self.io.now(), m.data)), 10)

    # ---- the base flow's choice at home: nothing to choose (the apple is in a hand) ----
    def auto_choose(self):
        self.hset("REACH", "waiting for GO")
        s = self.io.read()
        fr = self.frames(s)
        self.plan(s, apple_pelvis=fr.palm_ref + fr.torso_R @ np.array([0.25, 0.0, 0.0]))   # a placeholder
        return None

    # ---- the waist stays where the follower holds it in every scripted hold (grasp close, release) ----
    def stream(self, traj, *args, **kw):
        wf = self.waist
        if wf is not None and wf.enabled and wf.frozen and not args and kw.get("waist_cmd") is None \
                and kw.get("waist_to") is None:
            kw["waist_cmd"] = wf.command()[0]
        return super().stream(traj, *args, **kw)

    # ---- the apple: detections -> pelvis frame with the waist of the picture's moment ----
    def _waist_at(self, t):
        best = None
        for tt, w in reversed(self._whist):
            best = w
            if tt <= t:
                break
        return best

    def _fake_tick(self, now, fr):
        a = self.args
        if a.fake_apple is None:
            return
        while self._fake_due and self._fake_due[0][0] <= now:
            self._det_q.append(self._fake_due.pop(0))
        if now >= self._fake_next:
            self._fake_next = now + 0.1
            ph = 2.0 * math.pi * (now - self._t_go) / max(a.fake_period_s, 1e-3)
            p = np.asarray(a.fake_apple, float) + np.array([0.0, a.fake_motion_cm / 100.0 * math.sin(ph), 0.0])
            c = fr.camera_R.T @ (p - fr.camera_pos)
            W_, fx, cx = self._camera()
            if c[0] <= 0.05 or not 0.0 <= cx - fx * c[1] / c[0] <= W_:
                return                                       # outside the picture: not detected
            pt = fr.torso_R.T @ (p - fr.torso_pos)          # the camera's view now (torso frame at the picture)
            obj = {"cartesian": {"x": pt[0], "y": pt[1], "z": pt[2]}, "bbox": [0, 0, 1, 1], "class_label": "red_apple",
                   "radius_m": 0.04}
            self._fake_due.append((now + a.det_latency, json.dumps({"objects": [obj]})))
        while self._fake_due and self._fake_due[0][0] <= now:
            self._det_q.append(self._fake_due.pop(0))

    def ingest(self, s, fr):
        a = self.args
        while self._det_q:
            t_arr, data = self._det_q.popleft()
            try:
                objs = json.loads(data).get("objects", [])
            except Exception:
                continue
            w3 = self._waist_at(t_arr - a.det_latency)
            frc = fr if w3 is None else self.kin.frames(w3, s.arm_q)
            cands = []
            for o in objs:
                lab, c = str(o.get("class_label", "")), o.get("cartesian")
                if c is None or (a.labels and not any(x in lab for x in a.labels)):
                    continue
                p = frc.torso_R @ np.array([float(c["x"]), float(c["y"]), float(c["z"])]) + frc.torso_pos
                if np.linalg.norm(p - fr.shoulder_pos) > a.max_apple_m:
                    continue
                cands.append((p, o.get("radius_m"), lab))
            ev = self.track.update(cands, t_arr, fr.shoulder_pos)
            if ev == "new":
                self.brief("INFO", f"following a {self.track.label or 'fruit'} at {np.round(self.track.p, 3).tolist()} "
                                   f"(pelvis)", fold_s=0.0)

    # ---- the policy goal ----
    def make_goal(self, apple, fr, back_m, h_prev=0.0):
        """Goal (pocket back_m short of the apple along the palm normal), reachable?, palm heading (deg).
        The palm faces along the horizontal torso-axis -> apple line, turned to a heading of the reach map
        when the straight one does not reach (the last one kept while it still does)."""
        off = self.apple_offset_now()
        p = np.asarray(apple, float) + fr.torso_R @ (off / 100.0)
        u = p - fr.torso_pos
        u[2] = 0.0
        u /= max(np.linalg.norm(u), 1e-6)
        hs = self.rmap.headings_at(fr.torso_R.T @ (p - back_m * u - fr.shoulder_pos))
        h = h_prev if h_prev in hs else (hs[0] if hs else 0.0)
        c, sn = math.cos(math.radians(h)), math.sin(math.radians(h))
        n = np.array([c * u[0] - sn * u[1], sn * u[0] + c * u[1], 0.0])
        return AP.Goal("apple", p - back_m * n, n, self.finger0.copy(), False, False), bool(hs), h

    # ---- the transitions ----
    def settle_waist(self, max_s=1.5):
        """Hold the arm while the frozen waist's brake runs out."""
        t0 = time.monotonic()
        while self.waist is not None and not self.waist.settled() and time.monotonic() - t0 < max_s:
            if self.io.estop_why is not None:
                return
            p, v = self.waist.command()
            self.io.send_arm(self.q_sent, waist=p, waist_dq=v)
            self.io.sleep_until(self.io.now() + R.DT)

    def hand_ctl(self):
        """The picker's hand controller (None in --mock, whose io.hand is the fingers' state)."""
        h = getattr(self.io, "hand", None)
        return h if hasattr(h, "hand_close_start") else None

    def close_hand(self, d):
        self.waist.freeze()
        self.settle_waist()
        self.grip_apply()
        if self.hand_ctl() is None:
            out = "grasp skipped (no hand controller: mock)"
            self.hold(self.q_sent, TA.CLOSE_HOLD_S)
        else:
            out = self.grasp_close()
        self._carrying = True
        self.n_grasps += 1
        self.n_picked = self.n_grasps                     # the dashboard's counter
        self.brief("DONE" if out.startswith("grasp closed") else "WARN",
                   f"GRASP: {out}; pocket {d * 100:.1f} cm from the goal when it closed", fold_s=0.0)
        self.say_rl("rl_got_it")
        return out

    def release(self):
        self.waist.freeze()
        self.settle_waist()
        self.hset("RELEASE", "opening the hand")
        out = self.release_at_home() if self.hand_ctl() is not None else "no hand controller (mock)"
        self._carrying = False
        self.hand_cup()
        self.hold(self.q_sent, RELEASE_HOLD_S)
        self.brief("INFO", f"RELEASE: {out or 'hand open'}", fold_s=0.0)

    # ---- the run ----
    def policy_loop(self):
        return self.follow()

    def policy_loop_onboard(self):
        return "follow runs its policy on the laptop (live goals): no --onboard", True

    def follow(self):
        a, io, W = self.args, self.io, self.watch
        sm = W.sm
        self._t_go = time.monotonic()
        self.ctx.update(lo=self.lo, hi=self.hi, commanding=True, segment=None)
        self.left_arm_open()
        self.hset("REACH", "home -> ready")
        err = self.to_ready()
        if err:
            return f"home -> ready: {err}", True
        s = io.read()
        self.finger0 = AP.hand_frame(self.frames(s))[2]
        self.hand_cup()
        self.waist = WaistFollower(self, float(s.waist_q[0]), (float(s.waist_q[1]), float(s.waist_q[2])))
        self.waist.start()
        self.cmds.start()
        sys.__stdout__.write(f"{C_CYN}  FOLLOWING. {HELP}{T.C_OFF}\n")
        sys.__stdout__.flush()
        L = R.make_limiters("arm_pose")
        prev_raw = np.zeros(AP.N_OUT)
        W.reset()
        trk = AP.Tracker()
        st, running, goal, goal_apple, h = "REACH", False, None, None, 0.0
        reach_ok, locked, t_g0, sub, t_close = True, False, 0.0, "", 0.0
        outcome, stopped = None, False
        t_start = io.now()
        t_next, t_print = t_start, -1.0
        self.hset("REACH", "following the apple")
        try:
            while True:
                t_tick = io.now()
                t = t_tick - t_start
                if io.estop_why is not None:
                    stopped, outcome = True, f"E-STOP: {io.estop_why}"
                    self.halted = outcome
                    break
                if self.end_requested():
                    outcome = "END pressed on the dashboard"
                    break
                if t >= a.session_s:
                    outcome = f"session time {a.session_s:.0f} s over"
                    break
                end = False
                for src, c, txt in self.cmds.poll():
                    self.brief("STEP", f"command '{c}' ({src} {txt}) in {st}", fold_s=0.0)
                    if c == "end":
                        end = True
                    elif c == "grasp" and st == "REACH":
                        if not self.track.fresh(t_tick):
                            self.brief("WARN", "grasp: no apple in view -- still following", fold_s=0.0)
                            continue
                        st, locked, t_g0, goal, running, t_close = "GRASP", False, t, None, False, 0.0
                        self.hset("GRASP", "to the apple, then close")
                        self.say_rl("rl_grasping")
                    elif c == "reach" and st == "GRASP":
                        st, goal, running = "REACH", None, False
                        self.waist.unfreeze()
                        self.hset("REACH", "following the apple (grasp cancelled)")
                    elif c == "reach" and st == "HOLD":
                        self.release()
                        t_next, W.t_prev = io.now(), None          # the release held the arm itself
                        st, goal, running = "REACH", None, False
                        self.waist.unfreeze()
                        self.hset("REACH", "following the apple")
                    else:
                        self.brief("INFO", f"'{c}' ignored in {st}", fold_s=0.0)
                if end:
                    outcome = "end commanded"
                    break
                while self.waist.notes:
                    kind, msg = self.waist.notes.popleft()
                    self.brief(kind, msg, fold_s=0.0)
                s = io.read()
                self._whist.append((t_tick, np.asarray(s.waist_q, float).copy()))
                fr = self.frames(s)
                self._fake_tick(t_tick, fr)
                self.ingest(s, fr)
                fresh = self.track.fresh(t_tick)
                apple = self.track.p
                self.waist.set_apple(apple if st != "HOLD" else None, fresh and st != "HOLD", float(s.waist_q[0]),
                                     self.track.t_seen)
                active = False
                if st == "REACH":
                    if fresh and (goal is None or np.linalg.norm(apple - goal_apple) > a.goal_deadband_cm / 100.0):
                        goal, reach_ok, h = self.make_goal(apple, fr, a.standoff_cm / 100.0, h)
                        goal_apple = apple.copy()
                    active = fresh and goal is not None and reach_ok
                    sub = "following" if active else ("apple out of reach" if fresh else "apple not seen")
                elif st == "GRASP":
                    if not locked and fresh and (goal is None or
                                                 np.linalg.norm(apple - goal_apple) > a.goal_deadband_cm / 100.0):
                        goal, reach_ok, h = self.make_goal(apple, fr, -a.grasp_depth_cm / 100.0, h)
                        goal_apple = apple.copy()
                    active = goal is not None
                    sub = "goal fixed" if locked else "to the apple"
                d = ang = float("nan")
                q_cmd = None
                if active:
                    d, ang, _ = AP.errors(fr, goal)
                    if st == "GRASP" and not locked and d < a.grasp_lock_cm / 100.0:
                        locked = True
                        self.waist.freeze()
                        self.brief("INFO", f"GRASP: pocket {d * 100:.1f} cm from the apple: goal fixed, waist stops",
                                   fold_s=0.0)
                    if not running and (st == "GRASP" or d > a.resume_cm / 100.0 or ang > RESUME_ANG):
                        running = True
                        trk.reset()
                        L["arm"].vel[:] = 0.0
                    if running:
                        obs = AP.build_obs(s.arm_q, s.arm_dq, fr, goal, prev_raw)
                        raw = self.pol(obs)
                        tgt = L["arm"].step(raw[:7], s.arm_q, a.speed)
                        q_cmd = np.clip(tgt, self.lo, self.hi)
                        hit = np.abs(q_cmd - tgt) > 1e-6
                        if hit.any():
                            L["arm"].prev = q_cmd.copy()
                            L["arm"].vel[hit] = 0.0
                        self.ctx["clipped"] = [R.RIGHT_ARM_JOINTS[j][6:-6] for j in np.flatnonzero(hit)]
                        prev_raw = raw
                        if trk.step(fr, goal):
                            if st == "REACH":
                                running = False
                                self.brief("INFO", f"at the apple ({d * 100:.1f} cm, {math.degrees(ang):.0f} deg): "
                                                   f"holding until it moves {a.resume_cm:g} cm", fold_s=0.0)
                elif running:
                    running = False
                if q_cmd is None:
                    q_cmd = self.q_sent if self.q_sent is not None else self.gravity_hold(s)
                    self.ctx["clipped"] = []
                self.ctx.update(gap=None, speed_now=a.speed,
                                doing=f"{st.lower()}: {sub}" if st != "HOLD" else "hold: the apple in the hand")
                level, out = W.step(s, fr, q_cmd, L["arm"].vel if running else np.zeros(7), a.speed, self.ctx)
                W.print_new(out, t)
                if level >= sm.STOP and a.safety == "on":
                    stopped = True
                    outcome = "SAFETY STOP: " + "; ".join(w for l, w in out if l >= sm.STOP)
                    self.halted = outcome
                    break
                wp, wv = self.waist.command()
                io.send_arm(q_cmd, waist=wp if self.waist.enabled else None, waist_dq=wv)
                self.q_sent = np.asarray(q_cmd, float).copy()
                self._waist_cmd = (wp, wv, 0.0)
                self.settle_tick()
                nan3 = np.full(3, np.nan)
                self.follow_rows.append(dict(t=t, st=FOLLOW_STATES.index(st), run=running, q_cmd=q_cmd, q=s.arm_q,
                                      dq=s.arm_dq, waist=s.waist_q, wcmd=np.array([wp, wv]), palm=fr.palm_ref,
                                      goal=goal.target if goal is not None else nan3,
                                      apple=apple if apple is not None else nan3, fresh=fresh, d=d, ang=ang,
                                      age=s.age, compute=io.now() - t_tick))
                # GRASP: close when reached, or close enough when its time is up
                if st == "GRASP" and goal is None and t - t_g0 >= 2.0 * a.lost_s:
                    self.brief("WARN", "GRASP: the apple was not seen again: following", fold_s=0.0)
                    st, running = "REACH", False
                    self.waist.unfreeze()
                    self.hset("REACH", "following the apple")
                elif st == "GRASP" and active:
                    near = d <= a.grasp_close_cm / 100.0 and W.palm_v < GRASP_STILL_V
                    t_close = t_close + R.DT if near else 0.0
                    reached = trk.arrived or t_close >= GRASP_SETTLE_S
                    if reached or t - t_g0 >= a.waypoint_timeout:
                        if reached or d <= H.GRASP_MAX_ERR:
                            running = False
                            self.hset("GRASP", "closing the hand")
                            g_out = self.close_hand(d)
                            t_next, W.t_prev = io.now(), None      # the close held the arm itself
                            if g_out.startswith("E-STOP"):
                                stopped, outcome = True, g_out
                                break
                            st = "HOLD"
                            self.hset("HOLD", "holding the apple")
                        else:
                            self.brief("WARN", f"GRASP: the pocket still {d * 100:.1f} cm from the apple after "
                                               f"{a.waypoint_timeout:.0f} s: not closing -- following again", fold_s=0.0)
                            self.say_rl("rl_too_far")
                            st, goal, running = "REACH", None, False
                            self.waist.unfreeze()
                            self.hset("REACH", "following the apple")
                if t - t_print >= 1.0:
                    t_print = t
                    print(f"  t={t:6.1f}s {st:5s} {sub:20s} pocket {d * 100:5.1f} cm  apple "
                          f"{np.round(apple, 3).tolist() if apple is not None else None} "
                          f"{'fresh' if fresh else 'old'}  waist {math.degrees(s.waist_q[0]):+5.1f} "
                          f"(cmd {math.degrees(wp):+5.1f}) deg  [{sm.LEVEL_NAMES[level]}]", flush=True)
                self.status(s)
                t_next += R.DT
                if io.now() - t_next > R.DT:
                    t_next = io.now()
                io.sleep_until(t_next)
        except KeyboardInterrupt:
            stopped, outcome = True, "OPERATOR STOP (Ctrl-C)"
            self.halted = outcome
        finally:
            self.cmds.stop()
            self.save_follow_rows()
        if self.waist is not None:
            self.waist.freeze()
        summary = (f"FOLLOW: {self.n_grasps} grasp(s), {self.waist.n_turns} waist turn(s) in "
                   f"{io.now() - t_start:.0f} s; {outcome}")
        if not stopped and io.estop_why is None:
            err = self.wrap_up()
            if err:
                self.waist.stop()
                return summary + f"; back home: {err}", True
        self.waist.stop()
        self.ctx["commanding"] = False
        self.apple_true = None                        # no "palm from the REAL apple" line: the apple was a hand's
        if not stopped:
            summary = "REACHED " + summary            # the base flow then holds and ends at home
        self.say(f"  {summary}", C_GRN if not stopped else C_RED)
        return summary, stopped

    def save_follow_rows(self):
        rows = self.follow_rows
        if not rows:
            return
        try:
            np.savez_compressed(os.path.join(self.log_dir, "follow.npz"),
                                **{k: np.array([r[k] for r in rows]) for k in rows[0]})
            per = np.diff([r["t"] for r in rows])
            late = int(np.sum(per > 0.03)) if len(per) else 0
            self.brief("INFO", f"{len(rows)} follow steps ({late} late > 30 ms, compute median "
                               f"{np.median([r['compute'] for r in rows]) * 1000:.1f} ms) -> follow.npz", fold_s=0.0)
        except Exception as e:
            self.brief("WARN", f"follow.npz not written: {e}")

    def wrap_up(self):
        """The end: the hand opens, the policy to READY, the scripted move HOME, the waist back to 0."""
        self.settle_waist()
        if self._carrying:
            self.release()
        self.hset("REACH", "back to ready")
        s = self.io.read()
        g = AP.home_goal(self.kin.frames(s.waist_q, self.hcfg["ready_q"]))
        g.name = "ready"
        out, stopped = self.run_goals([g], "return")
        self.brief("INFO", f"policy to ready: {out}", fold_s=0.0)
        if stopped:
            return out
        err = self.to_home()
        if err:
            return err
        p, v = self.waist.command()
        self._waist_cmd = (p, 0.0 if self.waist.settled() else v, 0.0)
        if not self.waist_to(0.0, "home"):
            return "E-STOP while the waist went home"
        if self.io.estop_why is None:
            self.left_arm_back()
        return None


def main():
    args = parse_args()
    toy = ToyFollow(args)
    out = toy.run()
    sys.exit(0 if out and str(out).startswith("REACHED") else 1)


if __name__ == "__main__":
    main()
