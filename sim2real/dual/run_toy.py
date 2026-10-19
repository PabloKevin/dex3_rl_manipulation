#!/usr/bin/env python3
"""run_toy.py -- the sim2real toys through the dual daemon's RL mode. The toy files are
imported unchanged; three of their pieces are swapped here:

  robot IO    PickerIO -> DualIO     (rt/arm_rl/*: RL mode, waist in the stream, heartbeat)
  safety      Watch -> LightWatch    (no per-step collision model: the daemon guards contact)
  prompts     one GO before the policy (ENTER here, or GO on the dashboard); the rest auto

  python3 sim2real/dual/run_toy.py single --policy P --speed 0.3 --target detection
  python3 sim2real/dual/run_toy.py reach  --policy P --speed 0.3 --target X Y Z
  python3 sim2real/dual/run_toy.py visible --policy P --speed 0.3 --target detection [--onboard]
                      an AppleReachVisible policy (37 in / 7 out, waist held) through the single toy
  python3 sim2real/dual/run_toy.py apple|grasp|chain ...     (as toy_apple / toy_grasp / toy_chain)
  python3 sim2real/dual/run_toy.py calib --policy P --speed 0.3 --target detection
                      the single toy + a calibration session after the policy stops: jog / float the
                      palm to where it should have ended, SPACE records it (sim2real/dual/calib_reach.py)
  python3 sim2real/dual/run_toy.py v1 --speed 0.3            the harvest test bench's reach (reach_v1.py):
                      grasp_handoff_reach_apple.pt (default --policy), arm only (--waist hold), hand open,
                      the dashboard-clicked apple (--target detection); toy_reach.py's sequence
  python3 sim2real/dual/run_toy.py tree --speed 0.3          AppleReachTree-v1 (apple_reach_tree 2026-10-05_01-33-22
                      model_2000, default --policy): arm only (--waist hold), hand open, the palm-touch input
                      always 0 N (--touch zero), the clearance ring all free; the policy is stopped (arm frozen)
                      when the palm's surface is --stop_cm (default 1) from the apple's surface

Own options (everything else goes to the toy):
  --verbose           the toy's full output in this terminal (default: only the short log here,
                      the full output in <run>/toy_full.log)
  --closer_cm X       tell the policy the apple is X cm further along the approach, so it ends
                      X cm closer to the real one (the dashboard's 'closer' field overrides it)
  --strict_envelope   refuse a target outside the policy's training envelope (default: run it,
                      with a warning -- an imprecise reach beats no motion)
  --lag_gate RAD      hold the arm command (policy limiter and scripted moves) while it is more than RAD
                      ahead of the measured arm (default 0.12, under the daemon's 0.15 'stuck' threshold;
                      0 = off): a delayed arm is waited for instead of run away from; after 2 s of
                      holding it lets go (a blocked arm is the daemon's 'stuck' stop, as before)
  --calib JSON        single / calib: shift the apple fed to the policy by the fitted correction
                      (sim2real/fit_reach_calib.py; on top of --closer_cm)
  --stop_cm X         tree: freeze the arm when the palm surface is X cm from the apple surface (forward
                      kinematics: |apple - palm point| - radius - the palm's 16 mm contact height)
  --onboard           single: the POLICY phase runs on the Jetson (sim2real/onboard/rl_runner.py, its own
                      process beside the daemon; dual/onboard_link.py); the rest of the toy stays here.
                      Needs the runner deployed + started and the policy exported to .npz (onboard_link.py)
The old toys (sim2real/toy_*.py) still talk to v20 directly and keep working.
"""
from __future__ import annotations

import atexit
import json
import os
import select
import signal
import sys
import threading
import time

import numpy as np

HERE =os.path.dirname(os.path.abspath(__file__))
SIM2REAL = os.path.dirname(HERE)
sys.path[:0] = [HERE, SIM2REAL]
import toy_reach as T  # noqa: E402
from brief_log import BriefLog  # noqa: E402
from dual_io import V20_HOLD_KD, V20_HOLD_KP, DualIO  # noqa: E402
from home_paths import HomePaths  # noqa: E402
from light_watch import LightWatch  # noqa: E402

R = T.R
C_CYN, C_OFF, C_YEL, C_RED = T.C_CYN, T.C_OFF, T.C_YEL, T.C_RED
STRICT = False
VERBOSE = False
#: --dds_default for the toys whose parser has no such option (all but single)
DDS_DEFAULT = False
CLOSER_CM = 0.0
#: --calib: a sim2real/fit_reach_calib.py correction (reach_calib.ReachCalib), single / calib only
CALIB = None
#: s of the stiffness ramp kp 100 -> v20's 400 before "bye" (see DualToyMixin.handover_to_hold)
HANDOVER_S = 1.0
#: rad: corner rounding of the scripted joint paths (smooth_joint_path_traj)
CORNER_BLEND_RAD = 0.20
#: the paths home found by earlier runs (home_paths.json)
HOME_PATHS = HomePaths()
#: --lag_gate: rad the arm command may lead the measured arm (0 = off)
LAG_GATE = 0.12
#: s of continuous holding after which the gate lets the command go on
LAG_GATE_MAX_S = 2.0
#: --onboard: the policy phase runs on the Jetson (dual/onboard_link.py)
ONBOARD = False


class LagGate:
    """Counts the gate's holds (one per run, read by DualToyMixin)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_steps, self.n_spells, self.longest_s, self.gave_up = 0, 0, 0.0, 0
        self._since = None

    def check(self, lead: float, now: float) -> bool:
        """-> True = hold this step."""
        if LAG_GATE <= 0 or lead <= LAG_GATE:
            if self._since is not None:
                self.longest_s = max(self.longest_s, now - self._since)
            self._since = None
            return False
        if self._since is None:
            self._since = now
            self.n_spells += 1
        if now - self._since > LAG_GATE_MAX_S:
            if now - self._since - R.DT <= LAG_GATE_MAX_S:
                self.gave_up += 1
            return False
        self.n_steps += 1
        return True

    def summary(self):
        if not self.n_spells:
            return None
        return (f"lag gate: the arm fell > {LAG_GATE:g} rad behind its command {self.n_spells}x; the command "
                f"waited {self.n_steps * R.DT:.1f} s in all (longest {self.longest_s:.2f} s)"
                + (f", gave up waiting {self.gave_up}x (> {LAG_GATE_MAX_S:g} s)" if self.gave_up else ""))


GATE = LagGate()
_limiter_step = R.RateLimiter.step


def _gated_step(self, raw, q_meas, speed=1.0):
    """RateLimiter.step, ARM limiters only (gate_arm): while the previous command leads the
    measured arm by more than LAG_GATE, the command stays where it is (velocity 0)."""
    if getattr(self, "gate_arm", False) and self.k >= self.ANCHOR_STEPS and q_meas is not None:
        lead = float(np.max(np.abs(self.prev - np.asarray(q_meas, float))))
        if GATE.check(lead, time.monotonic()):
            self.desired = np.clip(np.asarray(raw, self.offset.dtype) * self.scale + self.offset, self.lo, self.hi)
            self.vel = np.zeros_like(self.vel)
            self.k += 1
            return self.prev.copy()
    return _limiter_step(self, raw, q_meas, speed)


def _gate_arm(make):
    def wrapped(*a, **k):
        L = make(*a, **k)
        if "arm" in L:
            L["arm"].gate_arm = True
        return L
    return wrapped


class _GatedTraj:
    """A scripted joint path that waits for the arm: the next point is only given while the
    last one sent leads the measured arm by <= LAG_GATE (toy_reach.Toy.stream plays it)."""

    def __init__(self, toy, traj):
        self.toy, self.traj = toy, list(traj)

    def __len__(self):
        return len(self.traj)

    def __iter__(self):
        io, last, i = self.toy.io, None, 0
        while i < len(self.traj):
            if last is not None:
                try:
                    lead = float(np.max(np.abs(np.asarray(last, float) - io.read().arm_q)))
                except Exception:
                    lead = 0.0
                if GATE.check(lead, time.monotonic()):
                    yield last
                    continue
            last = self.traj[i]
            i += 1
            yield last

_linear_joint_path_traj = T.joint_path_traj


def smooth_joint_path_traj(points, vmax, dt=R.DT, min_s=1.0):
    """toy_reach.joint_path_traj with the corners rounded. The toy's version runs one quintic
    along a polyline: at every waypoint each joint's velocity changes at once (the return
    path, simplify_path's points + HOME, and plan_home's via points) -- a jerk mid-motion.
    Here each inner corner is replaced by a quadratic Bezier from CORNER_BLEND_RAD before the
    waypoint to as far after it (at most 45% of either segment), sampled densely, so the
    direction changes continuously; the timing is the toy's (arc length, quintic, vmax).
    The body checks run on what this returns (path_check / plan_home call it), so the
    rounded path is the checked one."""
    pts = [np.asarray(p, float) for p in points]
    keep = [pts[0]]
    for q in pts[1:]:
        if np.max(np.abs(q - keep[-1])) > 1e-3:
            keep.append(q)
    if len(keep) <= 2:
        return _linear_joint_path_traj(keep, vmax, dt=dt, min_s=min_s)
    dense = [keep[0]]
    for i in range(1, len(keep) - 1):
        a, p, b = keep[i - 1], keep[i], keep[i + 1]
        l_in, l_out = float(np.linalg.norm(p - a)), float(np.linalg.norm(b - p))
        d = min(CORNER_BLEND_RAD, 0.45 * l_in, 0.45 * l_out)
        c0 = p - d * (p - a) / l_in
        c1 = p + d * (b - p) / l_out
        dense.append(c0)
        for u in np.linspace(0.0, 1.0, 13)[1:-1]:
            dense.append((1 - u) ** 2 * c0 + 2 * (1 - u) * u * p + u ** 2 * c1)
        dense.append(c1)
    dense.append(keep[-1])
    return _linear_joint_path_traj(dense, vmax, dt=dt, min_s=min_s)


FINISHING = False


def install_ctrl_c():
    """g1_launch.py ignores SIGINT while a toy runs (so the launcher itself survives the Ctrl-C),
    and a child inherits an ignored signal: Python then never installs its KeyboardInterrupt
    handler, and rclpy's own handler (it chains to the 'original' = ignore) only shut ROS down --
    Ctrl-C, and the launcher's SIGINT at shutdown, did nothing to the run (loopback, 2026-10-15).
    Restore the handler: the first Ctrl-C raises KeyboardInterrupt in the main thread; during
    finish() (hand-over, last voice line) it is ignored so the arm is handed back cleanly."""
    def handler(signum, frame):
        if FINISHING:
            sys.__stderr__.write("  (Ctrl-C ignored: the run is already ending)\n")
            return
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handler)


class _Tee:
    """stdout -> the run's toy_full.log (the terminal keeps only the short log). Until the
    run's folder exists, lines are buffered; if the run dies before that, they are shown."""

    def __init__(self):
        self.fh = None
        self.buf = []
        atexit.register(self._dump)

    def attach(self, path):
        self.fh = open(path, "a", buffering=1)
        self.fh.write("".join(self.buf))
        self.buf = []

    def _dump(self):
        if self.fh is None and self.buf:
            sys.__stdout__.write("".join(self.buf))

    def write(self, s):
        if self.fh is None:
            self.buf.append(s)
        else:
            self.fh.write(s)
        return len(s)

    def flush(self):
        if self.fh is not None:
            self.fh.flush()

    def isatty(self):
        return False


def _fast_dds(args):
    """Robot as a unicast DDS peer for this process (no 20-35 s discovery wait)."""
    if args.mock or getattr(args, "dds_default", False) or DDS_DEFAULT:
        return
    try:
        cfg = json.load(open(args.config or os.path.join(args.picker_dir, "apple_picker_config.json")))
        net_if = str(cfg.get("net_if") or (cfg.get("full_move2") or {}).get("net_if", ""))
    except Exception:
        net_if = ""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(args.picker_dir)), "g1_dual"))
    from fast_dds import use_fast_dds
    if use_fast_dds(getattr(args, "robot_ip", "192.168.123.164"), getattr(args, "spdp_s", 1.0), net_if):
        print("  DDS: robot as a unicast peer, announced every 1 s (fast discovery)")


class DualToyMixin:
    def __init__(self, args):
        _fast_dds(args)
        args.closer_cm = CLOSER_CM
        if not STRICT and not args.force:
            args.force = True
            self._envelope_note = True
        self._finished = False
        self._outcome = None
        self._closer_set = False
        super().__init__(args)
        node = getattr(self.io, "node", None)
        self.brief = BriefLog(os.path.join(self.log_dir, "brief.log"), node=node, echo=not VERBOSE)
        if isinstance(sys.stdout, _Tee):
            sys.stdout.attach(os.path.join(self.log_dir, "toy_full.log"))
            sys.__stdout__.write(f"  full toy output: {os.path.join(self.log_dir, 'toy_full.log')}\n")
        self.io.brief = self.brief
        self.watch.brief = self.brief
        # camera frames of this run -> <log_dir>/camera/NN (frame_recorder.py in live_locator)
        rec = getattr(self.io, "rec", None)
        if rec is not None and rec.enabled:
            rec.start(self.log_dir, type(self).__mro__[2].__name__)
            self.brief("INFO", f"camera recording while the arm moves -> {os.path.join(self.log_dir, 'camera')}")
        self.brief("INFO", f"{type(self).__mro__[2].__name__} [{self.io.name}] policy "
                           f"{os.path.basename(args.policy)} speed {args.speed}")
        if getattr(self, "_envelope_note", False):
            self.brief("INFO", "targets outside the training envelope run anyway (--strict_envelope to refuse)")

    # ---- the short log ----
    def say(self, msg, col=""):
        super().say(msg, col)
        kind = "STOP" if col == C_RED else "WARN" if col == C_YEL else "DONE" if col == T.C_GRN else "INFO"
        self.brief(kind, msg.strip())

    def set_phase(self, ph):
        super().set_phase(ph)
        self.brief("STEP", ph.lower().replace("_", " "), fold_s=0.0)
        if ph.startswith("MOVE_HOME"):
            self.voice("stand_clear")
        elif ph == "POLICY":
            self.voice("reaching")

    # ---- the robot's voice: the picker's recorded clips (apple_picker/audio_msgs_config.json) ----
    #   stand_clear "please stand clear, I am going to move my arms"   arm to home
    #   looking_for_fruit "I'm looking for fruit"                      sampling the camera at home
    #   ask_fruit "what kind of fruit should I harvest today"          READY: click the apple
    #   found_one "I found one, planning my move"                      apple chosen, waiting for GO
    #   reaching "reaching for it now"                                 the policy starts
    #   got_it / missed_it "I missed that one, let me try again"       REACHED / time limit
    #   need_help "I need help, please check my arm"                   any stop or error
    #   finish_my_work "I have finished my work here"                  back home, run over
    #   hasta_la_vista "Hasta la vista, baby" (sim2real/voice)         Ctrl-C, before the run ends
    def hasta_la_vista(self):
        """Easter egg, not a function of the run. Ctrl-C: the arm is lowered by the daemon's RESET
        (kp 0, light kd) and the clip plays from its own thread so that it ENDS as the arm settles,
        i.e. while it descends -- never at the Ctrl-C itself. Once per run."""
        if getattr(self, "_said_help", False):
            return
        self._said_help = True
        io = self.io

        def clip_s():
            try:
                import wave
                with wave.open(os.path.join(os.path.dirname(HERE), "voice", "hasta_la_vista_baby.wav")) as w:
                    return w.getnframes() / float(w.getframerate())
            except Exception:
                return 2.5

        def say_it(lower_s):
            def run():
                time.sleep(max(0.0, lower_s - clip_s() - 0.2))
                self.voice("hasta_la_vista", priority=True, max_age_s=0.0)
            threading.Thread(target=run, daemon=True, name="hasta_la_vista").start()
            self.brief("INFO", f"Ctrl-C: lowering the arm ({lower_s:g} s)", fold_s=0.0)
        try:
            io.lower_arm(on_descent=say_it)
        except Exception as e:
            self.brief("WARN", f"lowering the arm failed: {e}")

    def voice(self, key, **kw):
        speak = getattr(self.io, "speak", None)
        if speak is not None:
            speak(key, **kw)

    def plan_home(self, q_from, s):
        """The straight path first (one body check, ~10 ms); then the paths home remembered from
        earlier runs that started near here (home_paths.py, each re-checked from where the arm
        IS); the toy's full search of ~120 candidate paths (~10 s with nothing streamed) only
        when none of those is clear -- and what it finds is remembered for next time."""
        traj = T.joint_path_traj([np.asarray(q_from, float), R.HOME_Q], self.args.move_speed)
        if self.path_check(traj, s)[0]:
            return (), "straight"
        w = float(s.waist_q[0])
        try:
            for via, d, e in HOME_PATHS.candidates(q_from, w):
                tr = T.joint_path_traj([np.asarray(q_from, float), *via, R.HOME_Q], self.args.move_speed)
                if self.path_check(tr, s)[0]:
                    HOME_PATHS.used(e)
                    self.brief("INFO", f"path home: a remembered one (stored {e.get('t', '?')}, start "
                                       f"{d:.2f} rad from here, used {e.get('used', 0)}x)", fold_s=0.0)
                    return via, f"remembered ({e.get('name') or 'via'}, start {d:.2f} rad away)"
        except Exception as ex:
            self.brief("WARN", f"remembered paths home unusable ({ex}): searching")
        self.brief.wait("searching a path home around the body (straight one collides)")
        try:
            via, why = super().plan_home(q_from, s)
        finally:
            self.brief.wait(None)
        if via:
            try:
                HOME_PATHS.store(q_from, w, via, str(why).split(":")[0])
                self.brief("INFO", f"path home found and remembered ({len(HOME_PATHS.entries)} stored)", fold_s=0.0)
            except Exception:
                pass
        return via, why

    # ---- which apple: the one clicked on the dashboard ----
    def plan_from_detections(self):
        """Sample the tracked fruit at home (as the toys do), then take the one nearest to the
        dashboard click. No click yet: wait for one (ENTER = the nearest fruit instead)."""
        a, io = self.args, self.io
        tracks, n = [], 0
        t_start = io.now()
        self.brief("STEP", f"looking at the fruit for {a.detect_s:.0f} s (arm at home)")
        self.voice("looking_for_fruit")

        def sample():
            nonlocal n
            err = self.hold(R.HOME_Q, 0.2, label="hold home" if io.commands else "")
            if err:
                return err
            s = io.read()
            fr = self.frames(s)
            n += 1
            for p_t, rad in io.detection_targets():
                p = fr.torso_R @ np.asarray(p_t, float) + fr.torso_pos
                tr = next((t for t in tracks if np.linalg.norm(np.mean(t["p"][-20:], 0) - p) < 0.03), None)
                if tr is None:
                    tracks.append({"p": [p], "r": [rad]})
                else:
                    tr["p"].append(p)
                    tr["r"].append(rad)
            return None
        while io.now() - t_start < a.detect_s:
            err = sample()
            if err:
                return err
        click = getattr(io, "click", None)
        if click is None and a.pick is None:
            self.brief.wait("CLICK the apple on the dashboard camera (ENTER here = the nearest one)")
            self.voice("ask_fruit", max_age_s=0.0)          # READY to choose: never dropped as stale
            try:
                while getattr(io, "click", None) is None:
                    err = sample()
                    if err:
                        return err
                    r, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if r:
                        sys.stdin.readline()
                        break
            finally:
                self.brief.wait(None)
            click = getattr(io, "click", None)
        if not tracks:
            return "no fruit tracked (live_locator + fruits_to_torso running? the dashboard camera shows detections?)"
        s = io.read()
        fr = self.frames(s)
        means = [np.mean(t["p"][-20:], 0) for t in tracks]
        if click is not None:
            p_click = fr.torso_R @ click[0] + fr.torso_pos
            i = int(np.argmin([np.linalg.norm(m - p_click) for m in means]))
            d = float(np.linalg.norm(means[i] - p_click))
            if d > 0.10:
                return f"no tracked fruit within 10 cm of the click (nearest {d * 100:.0f} cm): click again, nearer"
            why = f"clicked ({d * 100:.1f} cm from the click)"
        elif a.pick is not None and a.pick < len(tracks):
            i, why = a.pick, f"--pick {a.pick}"
        else:
            sh = self.shoulder0(s)
            i = int(np.argmin([np.linalg.norm(m - sh) for m in means]))
            why = "nearest to the shoulder (no click)"
        radii = [r for r in tracks[i]["r"] if r]
        a.apple_radius = float(np.median(radii)) if radii else a.apple_radius
        self.brief("INFO", f"apple {np.round(means[i], 3).tolist()} (pelvis), r {a.apple_radius * 1000:.0f} mm: {why}")
        self.plan(s, apple_pelvis=means[i])
        self.voice("found_one")
        return None

    # ---- "go closer": the policy is told the apple is further along the approach ----
    def closer_m(self):
        v = getattr(self.io, "closer_cm", None)
        v = self.args.closer_cm if v is None else v
        return float(np.clip(v, 0.0, 15.0)) / 100.0

    def plan(self, s, apple_pelvis=None):
        super().plan(s, apple_pelvis=apple_pelvis)
        self.apple_true = None if self.apple is None else self.apple.copy()
        self.target_true = None if getattr(self, "target", None) is None else self.target.copy()
        self._closer = 0.0
        self.apply_closer()

    def apply_closer(self):
        """(Re)place the apple the policy sees: the real one + 'closer' along pelvis -> apple."""
        if getattr(self, "apple_true", None) is None:
            return
        d = self.closer_m()
        if abs(d - getattr(self, "_closer", 0.0)) < 1e-4 and self._closer_set:
            return
        u = np.array([self.apple_true[0], self.apple_true[1], 0.0])
        u /= max(np.linalg.norm(u), 1e-6)
        cal = np.zeros(3) if CALIB is None else CALIB.delta(self.apple_true)
        self.apple = self.apple_true + d * u + cal
        if self.target_true is not None:
            self.target = self.target_true + d * u + cal
        if CALIB is not None and not self._closer_set:
            self._calib_path = CALIB.path
            self.brief("INFO", f"calibration {CALIB.describe()}: the apple fed is shifted "
                               f"{np.round(cal * 100, 1).tolist()} cm (pelvis)", fold_s=0.0)
        self._closer, self._closer_set = d, True
        self.brief("INFO", "closer = 0: the policy sees the real apple" if d == 0 else
                   f"closer = {d * 100:.1f} cm: the policy is told the apple is {d * 100:.1f} cm further, "
                   f"so it should end {d * 100:.1f} cm closer to the real one", fold_s=0.0)

    def policy_loop(self):
        d = self.closer_m()
        self.gate_report("before the policy")
        if ONBOARD:
            from onboard_link import run_policy_onboard
            out = run_policy_onboard(self)
        else:
            out = super().policy_loop()
        self.gate_report("policy")
        o = str(out[0])
        if "Ctrl-C" in o:
            pass                                   # the easter egg waits for the lowering (finish)
        elif o.startswith("REACHED"):
            self.voice("got_it")
        elif o.startswith("TIME LIMIT"):
            self.voice("missed_it")
        else:
            self.voice("need_help", priority=True, max_age_s=0.0)
            self._said_help = True
        try:
            if getattr(self, "apple_true", None) is not None:
                fr = self.frames(self.io.read())
                v = self.apple_true - fr.palm_ref
                gap = float(np.linalg.norm(v)) - self.args.apple_radius
                n_gap = float(v @ fr.palm_normal) - self.args.apple_radius - R.PALM_TOUCH_HEIGHT
                lat = float(np.linalg.norm(v - (v @ fr.palm_normal) * fr.palm_normal))
                self.brief("INFO", f"palm {gap * 100:+.1f} cm from the REAL apple's surface"
                                   + (f" (policy aimed {d * 100:.1f} cm past it)" if d > 0 else "")
                                   + f"; along the palm normal palm surface -> apple surface {n_gap * 100:+.1f} cm, "
                                     f"apple centre {lat * 100:.1f} cm off the normal", fold_s=0.0)
        except Exception:
            pass
        return out

    # ---- the lag gate on scripted moves ----
    def stream(self, traj, *args, **kw):
        if LAG_GATE > 0 and getattr(self.io, "commands", False) and len(traj) > 1 \
                and np.ptp(np.asarray(traj, float), axis=0).max() > 1e-4:
            traj = _GatedTraj(self, traj)
        return super().stream(traj, *args, **kw)

    def gate_report(self, what):
        msg = GATE.summary()
        if msg:
            self.brief("WARN" if GATE.gave_up or GATE.n_steps * R.DT > 1.0 else "INFO", f"{what}: {msg}", fold_s=0.0)
        GATE.reset()

    # ---- one GO ----
    def confirm(self, prompt) -> bool:
        if "RUN THE POLICY" not in prompt and "RUN THE CHAIN" not in prompt:
            print(f"{C_CYN}{prompt.strip()} [auto]{C_OFF}", flush=True)
            return True
        if self.args.mock or (self.args.yes and getattr(self.io, "loopback", False)):
            print(f"{C_CYN}{prompt.strip()} [auto: mock / loopback --yes]{C_OFF}", flush=True)
            return True
        return self.wait_go()

    def wait_go(self) -> bool:
        io = self.io
        t0 = time.monotonic()
        self.brief.wait("GO: ENTER in the launcher terminal, or GO on the dashboard")
        sys.__stdout__.write(f"{C_CYN}  READY. ENTER (or GO on the dashboard) = run the policy;  "
                             f"'q' + ENTER = abort{C_OFF}\n")
        sys.__stdout__.flush()
        q_hold = self.q_sent if self.q_sent is not None else R.HOME_Q
        try:
            while True:
                if io.estop_why is not None:
                    return False
                if getattr(io, "go_t", None) is not None and io.go_t > t0:
                    return True
                self.apply_closer()                   # the dashboard's 'closer' may change while waiting
                r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if r:
                    return sys.stdin.readline().strip().lower() != "q"
                err = self.stream([q_hold] * 5, check=False)   # keep the arm held at kp/kd of the policy
                if err:
                    self.say(f"  while waiting for GO: {err}", C_RED)
                    return False
        finally:
            self.brief.wait(None)

    def ask_retreat(self) -> bool:
        out, sys.stdout = sys.stdout, sys.__stdout__       # the question must reach the terminal
        try:
            return super().ask_retreat()
        finally:
            sys.stdout = out

    # ---- the hand-over to v20's hold, without a jump ----
    def handover_to_hold(self):
        """After "bye" v20 holds the LAST target at kp 400 / kd 12; under the stream's kp 100 the
        arm sags g/100 below that target, so at the switch it snapped UP (seen at home 2026-10-15:
        hold_latch at q_cmd_last = HOME after every run). Here the stiffness ramps to v20's over
        HANDOVER_S while the target moves so that the arm stays where it is: at kp k the target is
        q_sent - g (1/kp0 - 1/k) (= q_sent at k = kp0). v20 then latches the last one at its own
        gains: no step in target or stiffness."""
        io = self.io
        if (not getattr(io, "commands", False) or not getattr(io, "_entered", False) or io.estop_why is not None
                or self.q_sent is None or not getattr(io, "rl_mine", False)):
            return
        if getattr(io, "rl_switch", False):
            self.brief("INFO", "ARM MODE switch on RL: the robot keeps holding the arm at the policy's gains "
                               "(no hand-over to v20)", fold_s=0.0)
            return
        kp0, kd0 = float(self.args.kp), float(self.args.kd)
        s = io.read()
        g = self.kin.gravity_torque(s.waist_q, s.arm_q, named=T.hand_named(s))
        q_c = np.asarray(self.q_sent, float).copy()
        n = max(1, int(HANDOVER_S / R.DT))
        t_next = io.now()
        q = q_c
        for k in range(1, n + 1):
            if io.estop_why is not None:
                return
            f = T.quintic(k / n)
            kp = kp0 + f * (V20_HOLD_KP - kp0)
            q = np.clip(q_c - g * (1.0 / kp0 - 1.0 / kp), self.lo, self.hi)
            io.send_arm(q, kp=kp, kd=kd0 + f * (V20_HOLD_KD - kd0))
            t_next += R.DT
            io.sleep_until(t_next)
        self.q_sent = q
        self.brief("INFO", f"stiffness handed to the robot's hold (kp {kp0:g} -> {V20_HOLD_KP:g} in {HANDOVER_S:g} s, "
                           f"target moved {np.max(np.abs(q - q_c)):.3f} rad so the arm stays put)", fold_s=0.0)

    # ---- always a log, whatever ends the run ----
    def finish(self, outcome, restore_mode=True):
        if self._finished:
            return self._outcome
        self._finished = True
        self._outcome = outcome
        global FINISHING
        FINISHING = True
        o = str(outcome)
        ctrl_c = "Ctrl-C" in o
        if ctrl_c:
            pass
        elif o.startswith(("REACHED", "TIME LIMIT", "HOME")):
            self.voice("finish_my_work", max_age_s=0.0)
        elif not getattr(self, "_said_help", False):
            self.voice("need_help", priority=True, max_age_s=0.0)
        try:
            self.handover_to_hold()
        except Exception as e:
            self.brief("WARN", f"hand-over to the robot's hold failed: {e}")
        try:
            if hasattr(self.io, "speech_done"):
                self.io.leave_rl()                    # the robot holds the arm while the last line plays
                if ctrl_c:
                    self.hasta_la_vista()             # lowers the arm; the clip plays during the descent
                self.io.speech_done(8.0)              # ... heard before the process ends
        except Exception:
            pass
        n_keep = getattr(self.io, "n_keep", 0)
        if n_keep:
            self.brief("INFO", f"keep-alive repeated the last arm target {n_keep}x while the toy computed "
                               f"(silences > {1000 * 0.06:.0f} ms filled; the daemon holds after 250 ms)", fold_s=0.0)
        out = super().finish(outcome, restore_mode=restore_mode)
        o = str(outcome)
        kind = "DONE" if o.startswith(("REACHED", "HOME", "shadow")) else "END" if o.startswith("TIME LIMIT") else "STOP"
        self.brief(kind, o, fold_s=0.0)
        if not VERBOSE:
            sys.__stdout__.write(f"   log: {self.log_dir}\n")
        return out

    def run(self):
        try:
            return super().run()
        except SystemExit as e:
            if not self._finished and e.code not in (None, 0):
                self.finish(f"STOPPED: {e.code}")
            raise


def mix(cls):
    return type("Dual" + cls.__name__, (DualToyMixin, cls), {})


def main():
    global STRICT, VERBOSE
    install_ctrl_c()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in ("reach", "single", "visible", "apple", "grasp",
                                                                   "chain", "v1", "tree", "calib"):
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    kind, rest = argv[0], argv[1:]
    if kind == "visible":                       # an AppleReachVisible policy through the single toy
        kind, rest = "single", rest + ["--visible"]
    if "--strict_envelope" in rest:
        STRICT = True
        rest.remove("--strict_envelope")
    if "--verbose" in rest:
        VERBOSE = True
        rest.remove("--verbose")
    global CLOSER_CM
    if "--closer_cm" in rest:
        i = rest.index("--closer_cm")
        CLOSER_CM = float(rest[i + 1])
        del rest[i:i + 2]
    global LAG_GATE
    if "--lag_gate" in rest:
        i = rest.index("--lag_gate")
        LAG_GATE = max(0.0, float(rest[i + 1]))
        del rest[i:i + 2]
    if LAG_GATE > 0:
        R.RateLimiter.step = _gated_step
        R.make_limiters = _gate_arm(R.make_limiters)
        if hasattr(R, "make_grasp_limiters"):
            R.make_grasp_limiters = _gate_arm(R.make_grasp_limiters)
    global CALIB
    if "--calib" in rest:
        i = rest.index("--calib")
        if kind not in ("single", "calib"):
            sys.exit("--calib is for the single reach (single / calib)")
        import reach_calib
        CALIB = reach_calib.ReachCalib(rest[i + 1])
        del rest[i:i + 2]
    global ONBOARD
    if "--onboard" in rest:
        if kind != "single":
            sys.exit("--onboard runs the single reach only (single)")
        ONBOARD = True
        rest.remove("--onboard")
    stop_cm = 1.0
    if "--stop_cm" in rest:
        i = rest.index("--stop_cm")
        if kind != "tree":
            sys.exit("--stop_cm is for the tree reach (tree)")
        stop_cm = float(rest[i + 1])
        del rest[i:i + 2]
    if not VERBOSE and not any(x in ("-h", "--help") for x in rest):
        sys.stdout = _Tee()
    # a raw rsl_rl checkpoint (model_N.pt) -> its TorchScript export (exported/policy_N.pt)
    for i, x in enumerate(rest):
        if x == "--policy" and i + 1 < len(rest):
            import re as _re
            m = _re.match(r"model_(\d+)\.pt$", os.path.basename(rest[i + 1]))
            if m:
                exp = os.path.join(os.path.dirname(rest[i + 1]), "exported", f"policy_{m.group(1)}.pt")
                if os.path.exists(exp):
                    sys.__stdout__.write(f"  --policy {rest[i + 1]} is a training checkpoint: using its export {exp}\n")
                    rest[i + 1] = exp
                else:
                    sys.exit(f"{rest[i + 1]} is a training checkpoint, not a TorchScript export, and {exp} does not "
                             f"exist: export it (scripts/export_policy_jit.py) or pass the exported .pt")
    T.PickerIO = DualIO
    T.Watch = LightWatch
    T.joint_path_traj = smooth_joint_path_traj
    if kind not in ("single", "calib") and "--dds_default" in rest:      # the launcher adds it in --loopback
        global DDS_DEFAULT
        DDS_DEFAULT = True
        rest.remove("--dds_default")
    if kind == "single":
        import toy_reach_single as S
        S.ToySingle = mix(S.ToySingle)
        sys.argv = [sys.argv[0]] + rest
        return S.main()
    if kind == "calib":
        import calib_reach as CR
        CR.CalibToy = mix(CR.CalibToy)
        sys.argv = [sys.argv[0]] + rest
        return CR.main()
    if kind == "chain":
        import toy_chain as CH
        CH.ChainToy = mix(CH.ChainToy)
        sys.argv = [sys.argv[0]] + rest
        return CH.main()
    defaults = {"apple": {"--target": ["detection"], "--waist": ["follow"]},
                "grasp": {"--target": ["detection"], "--waist": ["follow"]}}.get(kind, {})
    if kind == "v1":
        import reach_v1
        reach_v1.install(R)                       # this process only: reach_io.py is not changed
        if "--waist" in rest and rest[rest.index("--waist") + 1] != "hold":
            sys.exit("v1 is an arm-only policy (7 outputs): --waist hold only")
        # hand open: AppleGrasp-v1 holds the hand at its default (open) pose during the reach
        defaults = {"--target": ["detection"], "--waist": ["hold"], "--hand": ["open"],
                    "--policy": [reach_v1.default_policy(os.path.dirname(SIM2REAL))]}
    if kind == "tree":
        if "--waist" in rest and rest[rest.index("--waist") + 1] != "hold":
            sys.exit("tree is an arm-only policy (7 outputs): --waist hold only")
        for f in ("--touch", "--after_reach", "--reach_tol"):
            if f in rest:
                sys.exit(f"tree sets {f} itself (palm force 0, frozen at --stop_cm): use --stop_cm")
        # toy_reach freezes the arm (REACHED) when gap - GRASP_STANDOFF <= reach_tol, gap = |apple - palm
        # point| - radius; the palm's collider meets the apple at gap = PALM_TOUCH_HEIGHT, so the palm
        # surface is stop_cm from the apple at reach_tol = stop_cm + PALM_TOUCH_HEIGHT - GRASP_STANDOFF
        tol = stop_cm / 100.0 + R.PALM_TOUCH_HEIGHT - R.GRASP_STANDOFF
        # AppleReachTree-v1 holds the hand at its default (open: thumb_1 0.6, the rest 0); the touch
        # input is the palm sensor, always 0 here (no tactile on the real palm); ring = all free (reach_io)
        defaults = {"--target": ["detection"], "--waist": ["hold"], "--hand": ["open"], "--touch": ["zero"],
                    "--after_reach": ["freeze"], "--reach_tol": [f"{tol:.4f}"],
                    "--policy": [os.path.join(os.path.dirname(SIM2REAL), "logs", "rsl_rl", "apple_reach_tree",
                                              "2026-10-05_01-33-22", "exported", "policy_2000.pt")]}
    for flag, val in defaults.items():
        if flag not in rest:
            rest = rest + [flag] + val
    if kind == "grasp" and "--grasp" not in rest:
        rest.append("--grasp")
    T.Toy = mix(T.Toy)
    sys.argv = [sys.argv[0]] + rest
    return T.main()


if __name__ == "__main__":
    main()
