#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
SIM2REAL TOY TEST -- the SINGLE reach policy (AppleReachSingle-v1) drives the
real G1's right arm and waist to one point, through the same robot interface,
safety watch and sequence as toy_reach.py (imported, not changed: toy_reach.py
keeps running the waist / tree policies exactly as before).

What differs from toy_reach.py, all from envs/apple_reach_single_env_cfg.py
(sim2real/validate_single_io.py checks each against Isaac Lab):
  - inputs: reach_io's "single" layout (40): no palm touch input, the hand phase
    last. It has the "waist" layout's size, so the policy is loaded by NAME
    (layout="single"), never by its size -- and --policy is required.
  - hand: OPEN at home (thumb_1 0.6, the rest 0), closed to the partial cup
    (index / middle 1.05 rad) once the palm point is level with the apple
    centre (3 cm), ramped over 0.3 s / --speed; the phase is the policy input.
    thumb_0 stays at the picker's runtime lock (30 deg; the sim has 0), as in
    toy_reach.py.
  - goal: the apple centre (radius + 2 cm) out from the palm point along the
    palm normal (no contact); REACHED within --reach_tol (default 2 cm, the
    training tolerance). The approach is the horizontal line pelvis -> apple
    (nothing from the stem), so a virtual apple sits beyond the target point
    along it.
  - no --grasp: the grasp policy was trained after the touch reach with the
    other cup; the single reach's hand-off is not built yet.

Robot side as toy_reach.py: only pane 1 of launch_apple_picker.sh
(ssh_robot_launch.sh: cameras + arm_mpc_server), NOT apple_picker_sm.py, and
live_locator / fruits_to_torso for --target detection. Daemon v20 (no qref
waist) is fine: the policy's waist goes out as set_waist ramps (approximate,
as toy_reach.py says at the start of a run); v22 streams it.

  cd ~/dex3_rl_manipulation
  P=logs/rsl_rl/apple_reach_single_v1/<run>/exported/policy_<it>.pt
  python3 sim2real/toy_reach_single.py --policy $P --mock   --speed 0.3 --target 0.30 -0.20 0.10
  python3 sim2real/toy_reach_single.py --policy $P --shadow --speed 0.3 --target 0.30 -0.20 0.10
  python3 sim2real/toy_reach_single.py --policy $P          --speed 0.3 --target 0.30 -0.20 0.10
  python3 sim2real/toy_reach_single.py --policy $P          --speed 0.3 --target detection

Every other option is toy_reach.py's (python3 sim2real/toy_reach.py --help).
--target X Y Z (default --target_is palm): where the palm's grasp point should
end; the virtual apple centre is radius + 2 cm beyond it along pelvis -> target.
Logs: logs/sim2real/toy_reach_single/<time>_<mode>/.
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
import toy_reach as T  # noqa: E402

R = T.R
C_RED, C_YEL, C_GRN, C_CYN, C_OFF = T.C_RED, T.C_YEL, T.C_GRN, T.C_CYN, T.C_OFF
#: palm local +X: wrist -> fingertips (reach_rewards.PALM_FINGER_AXIS)
PALM_FINGER_AXIS = np.array([1.0, 0.0, 0.0])

#: THIS process's DDS config (unitree_sdk2py's own template + discovery settings). The SDK
#: default announces the laptop by multicast every 30 s (/tmp/cdds.LOG: SPDPInterval 30 s,
#: no peers) -- the ~20-35 s a fresh process waits before the robot's daemon sees it
#: (estop_link.py, toy_reach.enter_qref). Here the robot is also a unicast peer, announced
#: every second. Only this script's process: the picker and toy_reach.py keep the SDK default.
FAST_DDS_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
                </Interfaces>
            </General>
            <Discovery>
                <SPDPInterval>$__SPDP__$s</SPDPInterval>
                <Peers>
                    <Peer Address="$__PEER__$"/>
                </Peers>
            </Discovery>
            <Tracing>
                <Verbosity>config</Verbosity>
                <OutputFile>/tmp/cdds.LOG</OutputFile>
            </Tracing>
        </Domain>
    </CycloneDDS>'''


def use_fast_dds(robot_ip: str, spdp_s: float) -> None:
    """Before the picker's robot app calls ChannelFactoryInitialize (the SDK fills in the
    interface name from the template it imported)."""
    import unitree_sdk2py.core.channel as ch
    ch.ChannelConfigHasInterface = (FAST_DDS_TEMPLATE.replace("$__PEER__$", robot_ip)
                                    .replace("$__SPDP__$", f"{spdp_s:g}"))


def parse_args():
    """toy_reach.py's options, with the single policy's own --hand and defaults."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--hand", choices=["schedule", "keep"], default="schedule",
                   help="schedule: open at home, the trained closing at the apple's height (default)")
    p.add_argument("--hand_ramp_s", type=float, default=None,
                   help=f"s of the closing ramp (default {R.SINGLE_HAND_RAMP_S} / --speed)")
    p.add_argument("--layout_ok", action="store_true",
                   help="run a --policy whose path does not say 'single' (it must still be AppleReachSingle-v1)")
    p.add_argument("--dds_default", action="store_true",
                   help="keep the SDK's DDS config (multicast every 30 s: ~20-35 s until the daemon sees us)")
    p.add_argument("--robot_ip", default="192.168.123.164", help="unicast DDS peer (the robot's Jetson)")
    p.add_argument("--spdp_s", type=float, default=1.0, help="s between this process's DDS announcements")
    p.add_argument("--palm_touch", choices=["virtual", "zero"], default="virtual",
                   help="AppleReachSingleTactile-v1 only (41 inputs): its palm-touch input from the perceived "
                        "apple's contact with the palm (virtual, as toy_reach.py feeds the spindle) or always 0 "
                        "(the real palm reading is not used)")
    p.add_argument("--single_version", choices=["auto", "1", "2"], default="auto",
                   help="1: AppleReachSingle-v1 (2 cm short, hand closes level with the apple); 2: "
                        "AppleReachSingle-v2 (apple enclosed, hand closes within 10 cm, apple-centred envelope); "
                        "auto (default): 2 when the policy path (or its .txt) says single_v2")
    p.add_argument("--visible", action="store_true",
                   help="an AppleReachVisible policy (37 in / 7 out, waist held): auto when the path says "
                        "apple_reach_visible")
    p.add_argument("-h", "--help", action="store_true")
    mine, rest = p.parse_known_args()
    if mine.help:
        print(__doc__)
        print("toy_reach_single.py options:\n  --hand {schedule,keep}  --hand_ramp_s S  --layout_ok  "
              "--single_version {auto,1,2}  --palm_touch {virtual,zero}\n"
              "  --dds_default  --robot_ip IP  --spdp_s S\n"
              "toy_reach.py options (all apply, except --hand / --touch / --grasp):")
        sys.argv = [sys.argv[0], "--help"]
        T.parse_args()
    if "--policy" not in rest and not any(x.startswith("--policy=") for x in rest):
        sys.exit("--policy is required: the exported AppleReachSingle-v1 policy (40 in / 8 out; the default of "
                 "toy_reach.py is a WAIST policy of the same size, which this script would feed the wrong inputs)")
    if "--grasp" in rest:
        sys.exit("--grasp is not available with the single reach (no hand-off to the grasp policy is trained "
                 "for it yet)")
    user_tol = "--reach_tol" in rest or any(x.startswith("--reach_tol=") for x in rest)
    sys.argv = [sys.argv[0]] + rest + ["--hand", "keep", "--touch", "zero"]
    a = T.parse_args()
    a.visible = mine.visible or "apple_reach_visible" in os.path.abspath(a.policy)
    a.visible_version = R.visible_version_of(a.policy) if a.visible else 0
    a.single_version = R.single_version_of(a.policy) if mine.single_version == "auto" else int(mine.single_version)
    if a.visible:
        # single v2's goal / hand / limiter, waist pitch 0, v2's clip (reach_io.use_visible)
        a.single_version = 2
        R.use_layout_constants("visible", a.visible_version)
        R.ENVELOPE_WAIST = R.ENVELOPE_SINGLE_V2
        if a.waist != "hold":
            print("  visible reach: the policy has no waist output -> --waist hold (the torso stays where it is)")
            a.waist = "hold"
        print(f"  AppleReachVisible-v{a.visible_version} policy: 37 inputs / 7 outputs, the apple in the torso's yaw "
              f"frame (the waist may be turned), the apple measured once at home (no live tracking yet)")
    elif a.single_version == 2:
        R.use_single_v2()
        R.ENVELOPE_WAIST = R.ENVELOPE_SINGLE_V2     # apple centres, not grasp targets (ToySingle.check_apple)
    print(f"  single reach v{a.single_version}: goal {R.SINGLE_STANDOFF * 100:.1f} cm off the surface "
          f"(tol {R.SINGLE_POS_TOL * 100:.1f} cm), hand closes "
          + ("once the palm point is level with the apple" if R.SINGLE_HAND_TRIGGER == "level"
             else f"within {R.SINGLE_HAND_NEAR * 100:.0f} cm of the apple"))
    a.hand = mine.hand
    a.palm_touch = mine.palm_touch
    a.layout_ok = mine.layout_ok
    a.dds_default, a.robot_ip, a.spdp_s = mine.dds_default, mine.robot_ip, mine.spdp_s
    a.hand_ramp_s = mine.hand_ramp_s or R.SINGLE_HAND_RAMP_S / a.speed
    if not user_tol:
        a.reach_tol = R.SINGLE_POS_TOL
    if "single" not in os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a.policy)))) \
            and "single" not in a.policy and not a.layout_ok and not a.visible:
        sys.exit(f"{a.policy}: the path does not say 'single'. Only an AppleReachSingle-v1 policy may run here "
                 "(--layout_ok if it is one)")
    if a.log_dir is None:
        mode = "mock" if a.mock else "shadow" if a.shadow else "robot"
        a.log_dir = os.path.join(T.REPO, "logs", "sim2real", "toy_reach_single",
                                 f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{mode}")
    return a


class ToySingle(T.Toy):
    def __init__(self, args):
        self.t_launch = time.monotonic()
        if not args.mock and not args.dds_default:
            use_fast_dds(args.robot_ip, args.spdp_s)
            print(f"  DDS (this process only): robot {args.robot_ip} as a unicast peer, announced every "
                  f"{args.spdp_s:g} s (--dds_default: the SDK's 30 s multicast)")
        super().__init__(args)
        print(f"  [time] {time.monotonic() - self.t_launch:5.1f} s  robot app + policy loaded")
        connect = self.io.connect

        def connect_timed():
            t = time.monotonic()
            connect()
            print(f"  [time] {time.monotonic() - self.t_launch:5.1f} s  lowstate + arm daemon seen "
                  f"({time.monotonic() - t:.1f} s)", flush=True)
        self.io.connect = connect_timed
        # toy_reach picks the layout by size ("waist" for 40 inputs): load it by name. 41 inputs =
        # AppleReachSingleTactile-v1 (the palm touch input added back)
        if getattr(args, "visible", False):
            self.pol = R.ReachPolicy(args.policy, layout="visible")
        else:
            try:
                self.pol = R.ReachPolicy(args.policy, layout="single")
            except ValueError:
                self.pol = R.ReachPolicy(args.policy, layout="single_tactile")
        self.layout = self.pol.layout
        if self.layout == "single_tactile":
            print(f"  TACTILE single reach (41 inputs): palm touch input = "
                  + ("the perceived apple's contact with the palm (virtual)" if args.palm_touch == "virtual"
                     else "always 0"))
        self.sched = R.SingleHandSchedule(ramp_s=args.hand_ramp_s)
        self.hand_sent = None
        hand_set = self.io.hand_set

        def hand_set_single(kind):
            # "schedule": the start of the trained schedule (open); the rest as toy_reach
            if kind == "schedule":
                self.sched.reset()
                self.send_hand(force=True)
                print("  hand: OPEN (the single reach's start; it closes to the partial cup at the apple's "
                      "height); thumb_0 stays at the picker's runtime lock (30 deg; the sim has 0)")
            else:
                hand_set(kind)
        self.io.hand_set = hand_set_single

    def palm_touch(self, fr) -> float:
        """The tactile policy's palm-touch input (N); 0 for the 40-input single layout."""
        if self.layout != "single_tactile" or self.args.palm_touch != "virtual":
            return 0.0
        return float(R.virtual_touch_force(fr, self.apple, self.args.apple_radius))

    def enter_qref(self, q_hold, timeout_s=45.0):
        t = time.monotonic()
        err = super().enter_qref(q_hold, timeout_s)
        print(f"  [time] {time.monotonic() - self.t_launch:5.1f} s  daemon in qref, receiving this stream "
              f"({time.monotonic() - t:.1f} s)", flush=True)
        return err

    def send_hand(self, force=False):
        q = self.sched.targets()
        if force or self.hand_sent is None or float(np.max(np.abs(q - self.hand_sent))) > 1e-4:
            self.io.hand_cmd(q)
            self.hand_sent = q.copy()

    # ---------- plan: the approach is pelvis -> apple, the goal 2 cm off the surface ----------
    def _yaw(self, p):
        a = self.args
        return math.radians(a.approach_deg) if a.approach_deg is not None else R.single_approach_yaw(p)

    @staticmethod
    def _u(yaw):
        return np.array([math.cos(yaw), math.sin(yaw), 0.0])

    def _target_from_apple(self, apple, r, yaw):
        return np.asarray(apple, float) - (r + R.SINGLE_STANDOFF) * self._u(yaw)

    def check_apple(self, apple, target, sh, quiet=False):
        """toy_reach's envelope check on what the policy's map holds: the grasp target (v1)
        or the apple centre (v2's map, R.ENVELOPE_SINGLE_V2)."""
        return self.check(apple if R.SINGLE_VERSION == 2 else target, sh, quiet=quiet)

    def plan(self, s, apple_pelvis=None):
        a = self.args
        fr = self.frames(s)
        sh = self.shoulder0(s)
        if apple_pelvis is not None:
            p, a.target_is = np.asarray(apple_pelvis, float), "apple"
        elif a.target == ["detection"]:
            cands = self.io.detection_targets()
            if not cands:
                sys.exit("no fruit detected (/fruits_detected_torso: are live_locator + fruits_to_torso running?)")
            p_t, rad = cands[0]
            a.apple_radius = rad or a.apple_radius
            p = fr.torso_R @ np.asarray(p_t, float) + fr.torso_pos
            print(f"  (shadow) first detected fruit: {np.round(p_t, 3).tolist()} (torso)")
        else:
            p_in = np.array([float(v) for v in a.target])
            p = fr.torso_R @ p_in + fr.torso_pos if a.frame == "torso" else p_in
        r = a.apple_radius
        if a.target_is == "apple":
            apple = p
            yaw = self._yaw(apple)
            target = self._target_from_apple(apple, r, yaw)
        else:
            target = p
            yaw = self._yaw(target)
            apple = target + (r + R.SINGLE_STANDOFF) * self._u(yaw)
        self.apple, self.target, self.approach_yaw, self.shoulder = apple, target, yaw, sh
        up = R.single_fingers_up(apple, sh)
        print(f"  palm grasp point goal (pelvis): {np.round(target, 3).tolist()}   "
              f"{'apple centre (detected)' if apple_pelvis is not None else 'virtual apple centre'}: "
              f"{np.round(apple, 3).tolist()} (r {r * 1000:.0f} mm, goal {R.SINGLE_STANDOFF * 100:.0f} cm off its "
              f"surface), approach {math.degrees(yaw):+.0f} deg (pelvis -> apple); trained: fingers "
              f"{'UP, thumb below' if up else 'DOWN, thumb above'} (apple {(apple[2] - sh[2]) * 100:+.0f} cm from "
              f"the shoulder, switch at +{R.SINGLE_FINGER_SWITCH_DZ * 100:.0f})")
        chk = self.check_apple(apple, target, sh)
        if not chk["ok"]:
            if a.force:
                self.say("target outside the envelope -- running anyway (--force)", C_YEL)
            else:
                sys.exit("target outside this policy's training envelope (--force to run it anyway)")

    def plan_from_detections(self):
        """toy_reach.Toy.plan_from_detections with the single reach's approach and goal."""
        a, io = self.args, self.io
        print(f"  sampling the camera's fruit for {a.detect_s:.1f} s (arm held at home) ...", flush=True)
        tracks, n = [], 0
        t_end = io.now() + a.detect_s
        t_info = -1.0
        while io.now() < t_end:
            err = self.hold(R.HOME_Q, 0.2, label="hold home" if io.commands else "")
            if err:
                return err
            s = io.read()
            fr = self.frames(s)
            n += 1
            if io.now() - t_info >= 1.0:
                t_info = io.now()
                print(f"    tracker: {io.detection_info()}", flush=True)
            for p_t, rad in io.detection_targets():
                p = fr.torso_R @ np.asarray(p_t, float) + fr.torso_pos
                tr = next((t for t in tracks if np.linalg.norm(np.mean(t["p"], 0) - p) < 0.03), None)
                if tr is None:
                    tracks.append({"p": [p], "r": [rad]})
                else:
                    tr["p"].append(p)
                    tr["r"].append(rad)
        if n == 0 or not tracks:
            return ("no fruit tracked. The tracker's last word: " + io.detection_info()
                    + "  (no_msg = nothing on /fruits_detected_torso: live_locator + fruits_to_torso running? "
                    "no_pts = messages with no accepted fruit: see the reasons)")
        s = io.read()
        sh = self.shoulder0(s)
        rows = []
        print(f"  {len(tracks)} fruit seen in {n} samples (pelvis frame; torso frame at waist 0):")
        for i, t in enumerate(tracks):
            P = np.array(t["p"])
            mean = P.mean(0)
            spread = float(np.max(np.linalg.norm(P - mean, axis=1)))
            radii = [r for r in t["r"] if r]
            r = float(np.median(radii)) if radii else a.apple_radius
            chk = self.check_apple(mean, self._target_from_apple(mean, r, self._yaw(mean)), sh, quiet=True)
            seen = len(P) / n
            ok = chk["ok"] and seen >= 0.6 and spread <= a.detect_spread
            why = ("REACHABLE" if ok else "no: " + "; ".join(
                (chk["why"] if not chk["ok"] else []) + ([f"seen in {seen * 100:.0f}% of samples"] if seen < 0.6 else [])
                + ([f"position spread {spread * 100:.1f} cm"] if spread > a.detect_spread else [])))
            print(f"    #{i}: {np.round(mean, 3).tolist()}  r {r * 1000:.0f} mm  seen {seen * 100:3.0f}%  spread "
                  f"{spread * 1000:.0f} mm  {chk['r']:.2f} m / {chk['az_deg']:+.0f} deg from the shoulder  -> {why}")
            rows.append((i, mean, r, ok, chk["r"]))
        if a.pick is not None:
            pick = next((row for row in rows if row[0] == a.pick), None)
            if pick is None or not pick[3]:
                return f"fruit #{a.pick} is not reachable / not stable (see the list)"
        else:
            okr = [row for row in rows if row[3]]
            if not okr:
                return "no detected fruit is reachable and stable for this policy (see the list)"
            pick = min(okr, key=lambda row: row[4])
        a.apple_radius = pick[2]
        print(f"  -> fruit #{pick[0]}: centre {np.round(pick[1], 3).tolist()} (pelvis), r {pick[2] * 1000:.0f} mm")
        self.plan(s, apple_pelvis=pick[1])
        return None

    # ---------- status: the seat error instead of the touch gap ----------
    def status(self, s, extra=""):
        now = self.io.now()
        if now - self.status_t < 0.2:
            return
        self.status_t = now
        c = self.ctx
        target = None if self.apple is None else self.to_torso(self.apple, s)
        doc = {
            "t": round(time.time(), 3), "state": f"RL_TOY_REACH_SINGLE:{self.phase}",
            "state_since_s": round(now - self.phase_t0, 1), "step": self.phase,
            "doing": c.get("doing", ""), "side": "right", "extra": extra,
            "target": None if target is None else {"x": float(target[0]), "y": float(target[1]),
                                                     "z": float(target[2]), "side": "right"},
            "waist_deg": round(math.degrees(float(s.waist_q[0])), 1),
            "arms": {"right": {"controller": "rl", "mode": s.mode, "has_path": False, "enabled": None},
                     "left": {"controller": None, "mode": None, "has_path": False, "enabled": None}},
            "events": [dict(e) for e in self.events], "halted": self.halted is not None,
            "halt_reason": self.halted,
            "rl": {"policy": os.path.basename(self.args.policy), "layout": self.layout, "speed": self.args.speed,
                   "speed_now": c.get("speed_now"), "waist_mode": self.args.waist,
                   "gap_mm": None if c.get("seat") is None else round(1000 * c["seat"], 1),
                   "hand_phase": round(self.sched.phase, 2),
                   "touch_N": None, "clearance_cm": None if c.get("clear") is None
                   else round(100 * c["clear"], 1), "palm_speed": round(self.watch.palm_v, 3),
                   "off_path_cm": None if c.get("off_path") is None else round(100 * c["off_path"], 1),
                   "external_torque_frac": c.get("resid"), "safety": c.get("level_name", "ok"),
                   "warnings": c.get("warnings", [])},
        }
        self.io.publish(doc, target)

    def _fingers(self, fr):
        """(angle of the fingers from vertical [deg], 'up'/'down')."""
        f = fr.palm_R @ PALM_FINGER_AXIS
        fz = float(np.clip(f[2], -1.0, 1.0))
        return math.degrees(math.acos(abs(fz))), ("up" if fz > 0 else "down")

    # ---------- the policy ----------
    def policy_loop(self):
        a, io, W = self.args, self.io, self.watch
        sm = W.sm
        L = R.make_limiters(self.layout)
        prev_raw = np.zeros(self.pol.n_out)
        s = io.read()
        fr = self.frames(s)
        self.ctx.update(segment=(fr.palm_ref.copy(), self.apple.copy()), lo=self.lo, hi=self.hi, commanding=True)
        W.reset()
        self.sched.reset()
        self.path_cmd = []
        self.set_phase("POLICY")
        self.ctx["doing"] = "single reach policy driving the arm"
        settled_since = reached_t = None
        stopped, outcome = False, None
        waist_sent, waist_sent_t = 0.0, -9.0
        level = sm.OK
        latched_t = None
        print(f"  [time] {time.monotonic() - self.t_launch:5.1f} s  policy starts (includes your ENTER answers)",
              flush=True)
        want_up = R.single_fingers_up(self.apple, self.shoulder)
        if a.waist == "follow" and not io.qref_waist:
            self.say("waist: FOLLOW through set_waist ramps (this daemon has no qref waist, e.g. v20) -- "
                     "approximate: the torso lags the policy's waist target", C_YEL)
        t_start = io.now()
        t_next = t_start
        t_print = -1.0
        prof = {k: 0.0 for k in ("read", "obs+policy", "watch", "send", "status+log")}
        self.prof = prof
        tw = time.perf_counter
        try:
            while True:
                t_tick = io.now()
                p0 = tw()
                if io.estop_why is not None:
                    stopped, outcome = True, f"E-STOP: {io.estop_why}"
                    self.halted = outcome
                    break
                s = io.read()
                p1 = tw()
                prof["read"] += p1 - p0
                t = t_tick - t_start
                fr = self.frames(s)
                gap = float(np.linalg.norm(self.apple - fr.palm_ref)) - a.apple_radius
                seat = R.single_seat_error(fr, self.apple, a.apple_radius)
                # the hand schedule BEFORE the observation (the sim's interval event)
                was = self.sched.latched
                phase = self.sched.step(fr.palm_ref, self.apple)
                if self.sched.latched and not was:
                    latched_t = t
                    self.say(f"  t={t:.2f} s: palm level with the apple -> the hand closes to the partial cup "
                             f"over {a.hand_ramp_s:.1f} s", C_CYN)
                if a.hand == "schedule":
                    self.send_hand()
                touch = self.palm_touch(fr)
                st = R.ReachState(arm_q=s.arm_q, arm_dq=s.arm_dq, waist_q=s.waist_q, waist_dq=s.waist_dq,
                                  apple=self.apple, touch_force=touch, hand_phase=phase)
                obs, _ = R.build_obs(self.layout, self.kin, st, prev_raw, fr)
                raw = self.pol(obs)
                slow = a.safety == "on" and level >= sm.SLOW
                speed = a.speed * (a.safety_slow if slow else 1.0)
                tgt = L["arm"].step(raw[:7], s.arm_q, speed)
                q_cmd = np.clip(tgt, self.lo, self.hi)
                hit = np.abs(q_cmd - tgt) > 1e-6
                if hit.any():
                    L["arm"].prev = q_cmd.copy()
                    L["arm"].vel[hit] = 0.0
                clipped = [R.RIGHT_ARM_JOINTS[j][6:-6] for j in np.flatnonzero(hit)]
                w_cmd = w_des = float("nan")
                if "waist" in L:                       # the visible layout has no waist output
                    w_cmd = float(np.clip(L["waist"].step(raw[7:8], s.waist_q[:1], speed)[0],
                                          -self.waist_lim, self.waist_lim))
                    w_des = float(np.clip(L["waist"].desired[0], -self.waist_lim, self.waist_lim))
                now = io.now()
                if ("waist" in L and a.waist == "follow" and not io.qref_waist and abs(w_des - waist_sent) > math.radians(2.0)
                        and now - waist_sent_t > 0.3):
                    dur = max(0.5, 1.875 * abs(w_des - float(s.waist_q[0])) / (R.WAIST_MAX_VELOCITY * speed))
                    io.waist_to(w_des, dur)
                    waist_sent, waist_sent_t = w_des, now
                self.ctx.update(gap=gap, seat=seat, touch=round(touch, 2), clipped=clipped, speed_now=speed)
                p2 = tw()
                prof["obs+policy"] += p2 - p1
                level, out = W.step(s, fr, q_cmd, L["arm"].vel, a.speed, self.ctx)
                p3 = tw()
                prof["watch"] += p3 - p2
                self.ctx["level_name"] = sm.LEVEL_NAMES[level]
                self.ctx["warnings"] = [w for _, w in out][:4]
                W.print_new(out, t)
                if level >= sm.STOP and a.safety == "on":
                    stopped = True
                    outcome = "SAFETY STOP: " + "; ".join(w for l, w in out if l >= sm.STOP)
                    self.halted = outcome
                    break
                if out and any(l >= sm.STOP for l, _ in out) and a.safety != "on":
                    hard = [w for l, w in out if l >= sm.STOP and (w.startswith("lowstate") or "daemon" in w)]
                    if hard:
                        stopped, outcome = True, "STOP: " + "; ".join(hard)
                        self.halted = outcome
                        break
                stream_waist = "waist" in L and a.waist == "follow" and io.qref_waist
                p4 = tw()
                io.send_arm(q_cmd, waist=w_cmd if stream_waist else None,
                            waist_dq=float(L["waist"].vel[0]) if stream_waist else 0.0)
                prof["send"] += tw() - p4
                p5 = tw()
                self.q_sent = q_cmd.copy()
                prev_raw = raw
                self.path_cmd.append(q_cmd.copy())
                # reached: the apple centre within --reach_tol of the seat (radius + 2 cm along the
                # palm normal) -- at once (freeze), or held there with the arm still (--after_reach policy)
                still = float(np.max(np.abs(s.arm_dq))) < 0.08 * max(a.speed, 0.3) + 0.04
                if a.after_reach == "freeze":
                    arrived = seat <= a.reach_tol
                else:
                    if seat < a.reach_tol and still:
                        settled_since = settled_since if settled_since is not None else t
                    else:
                        settled_since = None
                    arrived = settled_since is not None and t - settled_since >= a.settle_s
                if arrived and reached_t is None:
                    reached_t = t
                    tilt = math.degrees(math.asin(np.clip(fr.palm_normal[2], -1, 1)))
                    head = math.degrees(math.atan2(fr.palm_normal[1], fr.palm_normal[0]) - self.approach_yaw)
                    fdev, fdir = self._fingers(fr)
                    self.reach_info = dict(t=t, seat_err_mm=seat * 1000, surface_gap_mm=gap * 1000, tilt_deg=tilt,
                                           heading_deg=head, fingers=fdir, fingers_from_vertical_deg=fdev,
                                           fingers_trained="up" if want_up else "down", hand_phase=phase,
                                           palm=fr.palm_ref.tolist(), q=s.arm_q.tolist())
                    self.say(f"  REACHED at t={t:.2f} s: seat error {seat * 1000:.1f} mm (palm {gap * 100:.1f} cm "
                             f"off the surface, goal {R.SINGLE_STANDOFF * 100:.0f}), palm normal {tilt:+.0f} deg from "
                             f"horizontal, {head:+.0f} deg from the approach; fingers {fdir} {fdev:.0f} deg from "
                             f"vertical (trained: {'up' if want_up else 'down'}); hand phase {phase:.2f}", C_GRN)
                    if a.after_reach == "freeze":
                        outcome = f"REACHED in {t:.2f} s (seat error {seat * 1000:.1f} mm)"
                        break
                    self.set_phase("HOLD")
                    self.ctx["doing"] = "policy holding at the goal"
                self.rows.append(dict(
                    t=t, obs=obs, raw=raw, q_cmd=q_cmd, q=s.arm_q, dq=s.arm_dq, tau=s.arm_tau,
                    waist=s.waist_q, waist_dq=s.waist_dq, waist_des=w_des, waist_cmd=w_cmd, palm=fr.palm_ref,
                    normal=fr.palm_normal, gap=gap, seat=seat, hand_phase=phase,
                    hand_cmd=self.sched.targets(), hand_q=np.full(7, np.nan) if s.hand_q is None else s.hand_q,
                    speed=speed, level=level, clear=self.ctx.get("clear", np.nan),
                    resid=self.ctx.get("resid", np.nan), off_path=self.ctx.get("off_path", np.nan),
                    age=s.age, compute=io.now() - t_tick))
                if t - t_print >= 0.5:
                    t_print = t
                    fdev, fdir = self._fingers(fr)
                    print(f"  t={t:5.2f}s seat {seat * 1000:5.1f} mm  surface {gap * 100:+5.1f} cm  hand {phase:.2f}  "
                          f"palm v {W.palm_v:.2f} m/s  clear {self.ctx.get('clear', np.nan) * 100:+5.1f} cm  "
                          f"fingers {fdir} {fdev:3.0f}  waist {math.degrees(s.waist_q[0]):+5.1f} "
                          f"(cmd {math.degrees(w_cmd):+5.1f})  [{sm.LEVEL_NAMES[level]}{' SLOW' if slow else ''}]",
                          flush=True)
                self.status(s)
                prof["status+log"] += tw() - p5
                if reached_t is not None and t - reached_t >= a.hold_s:
                    outcome = f"REACHED in {reached_t:.2f} s, held by the policy {a.hold_s:.1f} s " \
                              f"(seat error {seat * 1000:.1f} mm)"
                    break
                if t >= a.time_limit:
                    outcome = (f"TIME LIMIT {a.time_limit:.0f} s (seat error {seat * 1000:.1f} mm, hand "
                               f"{'closed at t=%.1f s' % latched_t if latched_t is not None else 'never closed'})")
                    break
                t_next += R.DT
                lag = io.now() - t_next
                if lag > R.DT:
                    W.print_new([(sm.WARN, f"control loop {lag * 1000:.0f} ms late")], t)
                    t_next = io.now()
                io.sleep_until(t_next)
        except KeyboardInterrupt:
            stopped, outcome = True, "OPERATOR STOP (Ctrl-C)"
            self.halted = outcome
        self.ctx["commanding"] = False
        col = C_GRN if outcome.startswith("REACHED") else C_RED if stopped else C_YEL
        self.say(f"  policy: {outcome}", col)
        return outcome, stopped

    def shadow_run(self):
        """Read-only: the single policy on the real robot's state, nothing sent (the hand
        schedule runs on the measured palm, its targets are not sent either)."""
        a, io, W = self.args, self.io, self.watch
        L = R.make_limiters(self.layout)
        prev_raw = np.zeros(self.pol.n_out)
        s = io.read()
        fr = self.frames(s)
        self.ctx.update(segment=(fr.palm_ref.copy(), self.apple.copy()), lo=self.lo, hi=self.hi, commanding=False)
        W.reset()
        self.sched.reset()
        self.set_phase("SHADOW")
        n = int(a.shadow_s / R.DT)
        t_next = io.now()
        first = None
        res = []
        for k in range(n):
            s = io.read()
            fr = self.frames(s)
            phase = self.sched.step(fr.palm_ref, self.apple)
            obs, _ = R.build_obs(self.layout, self.kin, R.ReachState(s.arm_q, s.arm_dq, s.waist_q, s.waist_dq,
                                                                     self.apple, touch_force=self.palm_touch(fr),
                                                                     hand_phase=phase), prev_raw, fr)
            raw = self.pol(obs)
            q_cmd = np.clip(L["arm"].step(raw[:7], s.arm_q, a.speed), self.lo, self.hi)
            self.ctx["gap"] = float(np.linalg.norm(self.apple - fr.palm_ref)) - a.apple_radius
            self.ctx["seat"] = R.single_seat_error(fr, self.apple, a.apple_radius)
            level, out = W.step(s, fr, q_cmd, L["arm"].vel, a.speed, self.ctx)
            W.print_new(out, k * R.DT)
            prev_raw = raw
            first = first if first is not None else (obs.copy(), raw.copy())
            g = self.kin.gravity_torque(s.waist_q, s.arm_q, named=T.hand_named(s))
            res.append((s.arm_tau, g))
            self.status(s, extra="SHADOW: nothing is commanded")
            t_next += R.DT
            io.sleep_until(t_next)
        obs0, raw0 = first
        np.set_printoptions(precision=3, suppress=True, linewidth=140)
        print(f"\n  first observation ({self.layout} layout):")
        rows = [("arm q - default", 0, 7), ("arm dq", 7, 14), ("apple - palm", 14, 17),
                ("palm quat", 17, 21), ("prev raw", 21, 28), ("ring", 28, 36),
                ("waist q, dq, prev raw", 36, 39), ("hand phase", 39, 40)]
        if self.layout == "single_tactile":
            rows = rows[:5] + [("palm touch", 28, 29), ("ring", 29, 37), ("waist q, dq, prev raw", 37, 40),
                               ("hand phase", 40, 41)]
        if self.layout == "visible":
            rows = rows[:6] + [("hand phase", 36, 37)]
        for nm, i, j in rows:
            print(f"    {nm:22s} {obs0[i:j]}")
        print(f"  first raw action {raw0}\n  -> first joint targets would move the arm toward "
              f"{np.round(np.clip(R.ARM_DEFAULT + 0.5 * raw0[:7], self.lo, self.hi), 3).tolist()} (before the limiter)")
        print(f"  seat error now {self.ctx['seat'] * 1000:.0f} mm; hand schedule "
              f"{'LATCHED (palm level with the apple)' if self.sched.latched else 'open (palm not level with the apple)'}")
        tau = np.array([r[0] for r in res])
        g = np.array([r[1] for r in res])
        print("  external-torque model at rest (tau_est - RNEA gravity), per joint [N*m]:")
        for j, nme in enumerate(R.RIGHT_ARM_JOINTS):
            d = tau[:, j] - g[:, j]
            print(f"    {nme[6:-6]:16s} tau_est {tau[:, j].mean():+6.2f}  model {g[:, j].mean():+6.2f}  "
                  f"residual {d.mean():+6.2f} +- {d.std():.2f}  ({abs(d.mean()) / T.ARM_RATED[j] * 100:4.1f}% of rated; "
                  f"WARN at {12 * a.resid_scale:.0f}%)")
        return self.finish("shadow run done (nothing was commanded)")


def main():
    args = parse_args()
    toy = ToySingle(args)
    if args.real and not getattr(toy.io, "loopback", False):
        print(f"{C_RED}REAL ROBOT. Someone at the robot with the remote / emergency stop, the area "
              f"in front of the right arm clear.{C_OFF}")
    out = toy.run()
    sys.exit(0 if out and (out.startswith("REACHED") or out.startswith("shadow")) else 1)


if __name__ == "__main__":
    main()
