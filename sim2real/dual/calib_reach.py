#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""calib_reach.py -- record where the single reach SHOULD have ended (run_toy.py calib).

The single reach toy (toy_reach_single.py, through the dual daemon like run_toy.py single)
runs as usual: the dashboard-clicked apple, GO, the policy. When the policy stops, the
arm is held there and a CALIBRATION session starts in the launcher terminal:

  jog (default)   keys move the palm point in the TORSO frame, the arm solved by IK
                  (sim2real/reach_calib.solve_palm_ik, near the current posture), every
                  step a slow joint move checked against the body cover
      w / s   forward / back      a / d   left / right      r / f   up / down
      i / k   tilt the palm down / up (about torso y)       j / l   turn left / right (torso z)
      u / o   roll (about torso x)                            [ / ]   waist yaw -1 / +1 deg
      1 / 5   fine (1 mm, 1 deg) / coarse (5 mm, 3 deg) steps
  g       FLOAT (only with --allow_float): the arm goes soft (kp 30) and follows a firm push
          by hand (reach_calib.SlipSpring: gravity carried by the PD offset, no drift); g again
          = stiff where it is. EXPERIMENTAL: the daemon's contact guard may freeze the arm.
  SPACE   record: this pose is where the palm should have ended -> one line in
          --calib_log (default logs/sim2real/calib_reach/trials.jsonl), then back to where the
          policy stopped and the toy's usual return home
  z       back to where the policy stopped (keep calibrating)
  q       end without recording            h   this help

Then: python3 sim2real/fit_reach_calib.py  ->  logs/sim2real/reach_calib.json, applied with
run_toy.py single ... --calib logs/sim2real/reach_calib.json (the apple fed to the policy
is shifted so it ends where you marked).

  python3 sim2real/dual/run_toy.py calib --policy P --speed 0.3 --target detection
  python3 sim2real/dual/run_toy.py calib --policy P --mock --speed 0.5 --target 0.35 -0.20 0.05 \\
      --calib_keys "wwd5r1 " --yes                     (no robot: scripted keys)
Own options: --allow_float, --calib_log PATH, --calib_keys "KEYS" (scripted, for tests),
--jog_vmax RAD_S (0.15), --float_kp (30), --float_kd (3), --float_deadband (0.04).
"""
from __future__ import annotations

import inspect
import json
import math
import os
import select
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM2REAL = os.path.dirname(HERE)
sys.path[:0] = [HERE, SIM2REAL]
import reach_calib as RC  # noqa: E402
import toy_reach as T  # noqa: E402
import toy_reach_single as S  # noqa: E402

R = T.R
C_CYN, C_YEL, C_RED, C_GRN, C_OFF = T.C_CYN, T.C_YEL, T.C_RED, T.C_GRN, T.C_OFF
REPO = os.path.dirname(SIM2REAL)
DEFAULT_LOG = os.path.join(REPO, "logs", "sim2real", "calib_reach", "trials.jsonl")

#: own options (taken out of argv before toy_reach_single parses the rest)
OPTS = {"--allow_float": False, "--calib_log": DEFAULT_LOG, "--calib_keys": None, "--jog_vmax": 0.15,
        "--float_kp": 30.0, "--float_kd": 3.0, "--float_deadband": 0.04}


def take_opts(argv: list[str]) -> tuple[dict, list[str]]:
    o, rest, i = dict(OPTS), [], 0
    while i < len(argv):
        x = argv[i]
        if x == "--allow_float":
            o[x] = True
        elif x in OPTS:
            o[x] = type(OPTS[x])(argv[i + 1]) if OPTS[x] is not None else argv[i + 1]
            i += 1
        else:
            rest.append(x)
        i += 1
    return o, rest


CAL = dict(OPTS)


class _Keys:
    """Single key presses from the terminal (cbreak), or a scripted string (--calib_keys)."""

    def __init__(self, scripted: str | None):
        self.script = list(scripted) if scripted is not None else None
        self.old = None
        if self.script is None and sys.stdin.isatty():
            import termios
            import tty
            self.old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def get(self) -> str | None:
        if self.script is not None:
            return self.script.pop(0) if self.script else "q"
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not r:
            return None
        c = sys.stdin.read(1)
        return c if c else None

    def close(self):
        if self.old is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)
            self.old = None


class CalibToy(S.ToySingle):
    def out(self, msg, col=""):
        """To the terminal even when run_toy keeps the full output in a file."""
        line = f"{col}{msg}{C_OFF if col else ''}"
        if sys.stdout is not sys.__stdout__:       # run_toy keeps the full output in a file (toy_full.log)
            print(line, flush=True)
        sys.__stdout__.write(line + "\n")
        sys.__stdout__.flush()

    def policy_loop(self):
        outcome, stopped = super().policy_loop()
        if stopped or self.io.estop_why is not None or self.q_sent is None:
            return outcome, stopped
        try:
            note = self.calib_session()
        except Exception as e:                     # the arm is held where it is by the toy's run()
            import traceback
            traceback.print_exc()
            note = f"calibration session ERROR {type(e).__name__}: {e}"
            return outcome + "; " + note, True
        return outcome + "; " + note, stopped

    # ---------- the session ----------
    def _pose(self, waist3, q):
        return RC.palm_pose(self.kin, waist3, q, self._named)

    def calib_session(self):
        a, io = self.args, self.io
        self.set_phase("CALIB")
        self.ctx["doing"] = "calibration: move the palm to where it should have ended"
        s = io.read()
        # settle first (the policy may stop mid-motion, e.g. at its time limit): the end pose is the still one
        err = self.stream([np.asarray(self.q_sent, float)] * int(1.0 / R.DT), check=False,
                          waist_cmd=float(s.waist_q[0]))
        if err:
            return f"calibration: {err}"
        s = io.read()
        self._named = T.hand_named(s)
        w_rest = (float(s.waist_q[1]), float(s.waist_q[2]))
        q_end_cmd = np.asarray(self.q_sent, float).copy()
        w_end = float(s.waist_q[0])
        end = {"q_cmd": q_end_cmd, "q_meas": s.arm_q.copy(), "waist": w_end, "fr": self.frames(s)}
        self._end = end
        q_cmd, w_cmd = q_end_cmd.copy(), w_end
        fine = True
        moves, mode_used = 0, "jog"
        keys = _Keys(CAL["--calib_keys"])
        self.out(f"\n== CALIBRATION: the policy stopped. Move the palm to where it SHOULD be, SPACE to record. "
                 f"h = keys. ==", C_CYN)
        self.help()
        try:
            while True:
                if io.estop_why is not None:
                    return f"calibration ended by E-STOP ({io.estop_why})"
                err = self.stream([q_cmd], check=False, waist_cmd=w_cmd)
                if err:
                    return f"calibration hold: {err}"
                k = keys.get()
                if k is None:
                    continue
                if k == "h":
                    self.help()
                elif k in "15":
                    fine = k == "1"
                    self.out(f"  steps: {'1 mm / 1 deg' if fine else '5 mm / 3 deg'}")
                elif k in "wsadrf":
                    step = 0.001 if fine else 0.005
                    v = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0),
                         "r": (0, 0, 1), "f": (0, 0, -1)}[k]
                    q_cmd, ok = self.jog(q_cmd, w_cmd, w_rest, dp_torso=step * np.array(v, float))
                    moves += ok
                elif k in "ikjluo":
                    ang = math.radians(1.0 if fine else 3.0)
                    v = {"i": (0, 1, 0), "k": (0, -1, 0), "j": (0, 0, 1), "l": (0, 0, -1),
                         "u": (1, 0, 0), "o": (-1, 0, 0)}[k]
                    q_cmd, ok = self.jog(q_cmd, w_cmd, w_rest, drot_torso=ang * np.array(v, float))
                    moves += ok
                elif k in "[]":
                    w_new = float(np.clip(w_cmd + math.radians(-1.0 if k == "[" else 1.0),
                                          -R.WAIST_YAW_LIMIT, R.WAIST_YAW_LIMIT))
                    err = self.stream([q_cmd] * 25, label="calib waist", waist_cmd=w_new)
                    if err:
                        return f"calibration waist step: {err}"
                    w_cmd = w_new
                    moves += 1
                    self.report(q_cmd, w_cmd, w_rest, end)
                elif k == "g":
                    if not CAL["--allow_float"]:
                        self.out("  float mode is off (--allow_float to enable it)", C_YEL)
                        continue
                    q_new, err = self.float_mode(w_cmd, keys)
                    if err:
                        return f"calibration float: {err}"
                    q_cmd, mode_used = q_new, "float" if mode_used == "jog" else mode_used
                    moves += 1
                    self.report(q_cmd, w_cmd, w_rest, end)
                elif k == "z":
                    err = self.go_back(q_cmd, w_cmd, end)
                    if err:
                        return f"calibration back to the policy's end: {err}"
                    q_cmd, w_cmd = end["q_cmd"].copy(), end["waist"]
                    self.out("  back where the policy stopped")
                elif k == " ":
                    note = self.record(end, q_cmd, w_cmd, moves, mode_used)
                    err = self.go_back(q_cmd, w_cmd, end)
                    return note + ("" if not err else f"; back to the policy's end: {err}")
                elif k == "q":
                    err = self.go_back(q_cmd, w_cmd, end)
                    return "calibration: nothing recorded" + ("" if not err else f"; {err}")
        finally:
            keys.close()

    def help(self):
        self.out("  w/s fwd/back  a/d left/right  r/f up/down  (torso frame)   i/k tilt  j/l turn  u/o roll\n"
                 "  [ ] waist -/+1 deg   1/5 fine/coarse   " + ("g float   " if CAL["--allow_float"] else "")
                 + "SPACE record   z back to the policy's end   q quit   h help")

    def report(self, q_cmd, w_cmd, w_rest, end):
        s = self.io.read()
        fr = self.frames(s)
        d = fr.palm_ref - end["fr"].palm_ref
        Rt = end["fr"].torso_R
        rot = math.degrees(np.linalg.norm(RC.R_to_rotvec(fr.palm_R @ end["fr"].palm_R.T)))
        self.out(f"  palm moved {np.round(Rt.T @ d * 100, 1).tolist()} cm (torso axes at the policy's end), "
                 f"rotated {rot:.1f} deg, waist {math.degrees(float(s.waist_q[0]) - end['waist']):+.1f} deg")

    def jog(self, q_cmd, w_cmd, w_rest, dp_torso=None, drot_torso=None):
        """One IK step from the COMMANDED pose; refused if IK misses or the path hits the body."""
        a = self.args
        waist3 = np.array([w_cmd, *w_rest])
        p, Rm = self._pose(waist3, q_cmd)
        Rt = self.frames(self.io.read()).torso_R
        p_goal = p + (Rt @ dp_torso if dp_torso is not None else 0.0)
        R_goal = (RC.rotvec_to_R(Rt @ drot_torso) @ Rm) if drot_torso is not None else Rm
        q_new, _, pe, re = RC.solve_palm_ik(self.kin, waist3, q_cmd, p_goal, R_goal, self.lo, self.hi, self._named)
        if pe > 0.003 or re > math.radians(1.5):
            self.out(f"  refused: IK misses by {pe * 1000:.1f} mm / {math.degrees(re):.1f} deg (joint limit?)", C_YEL)
            return q_cmd, False
        if float(np.max(np.abs(q_new - q_cmd))) > 0.3:
            self.out("  refused: that step needs a joint jump > 0.3 rad", C_YEL)
            return q_cmd, False
        traj = T.joint_path_traj([q_cmd, q_new], CAL["--jog_vmax"], min_s=0.3)
        ok, c0, worst, c_end, bad, note = self.path_check(traj, self.io.read())
        if not ok:
            self.out(f"  refused: the step goes into the body cover ({worst * 100:+.1f} cm)", C_RED)
            return q_cmd, False
        err = self.stream(traj, label="calib jog", waist_cmd=w_cmd)
        if err:
            self.out(f"  jog stopped: {err}", C_RED)
            return self.q_sent.copy(), False
        self.report(q_new, w_cmd, w_rest, self._end)
        return q_new, True

    def float_mode(self, w_cmd, keys):
        """Soft arm, push it by hand; 'g' again = stiff where it is."""
        io, a = self.io, self.args
        kp_f, kd_f = CAL["--float_kp"], CAL["--float_kd"]
        has_gains = "kp" in inspect.signature(io.send_arm).parameters
        s = io.read()
        spring = RC.SlipSpring(s.arm_q, kp=kp_f, kd=kd_f, deadband=CAL["--float_deadband"], dt=R.DT)
        self.out(f"  FLOAT: kp {kp_f:g} -- push the hand where it should be; 'g' = stiff again. "
                 f"Moving faster than 1 rad/s ends it.", C_YEL)
        BC = self.watch.BC
        t_next, n = io.now(), 0
        # soften gradually (1 s), the target moving with it so the arm stays put
        for k in range(1, int(1.0 / R.DT) + 1):
            f = T.quintic(k / int(1.0 / R.DT))
            kp = a.kp + f * (kp_f - a.kp)
            s = io.read()
            g = self.kin.gravity_torque(s.waist_q, s.arm_q, named=T.hand_named(s))
            q = np.clip(spring.q_ref + g / kp, self.lo, self.hi)
            self._send(q, w_cmd, kp, a.kd + f * (kd_f - a.kd), has_gains)
            t_next += R.DT
            io.sleep_until(t_next)
        while True:
            if io.estop_why is not None:
                return None, f"E-STOP ({io.estop_why})"
            s = io.read()
            g = self.kin.gravity_torque(s.waist_q, s.arm_q, named=T.hand_named(s))
            q = np.clip(spring.step(s.arm_q, g), self.lo, self.hi)
            self._send(q, w_cmd, kp_f, kd_f, has_gains)
            n += 1
            if float(np.max(np.abs(s.arm_dq))) > 1.0:
                self.out("  FLOAT ended: the arm moved faster than 1 rad/s", C_RED)
                break
            if n % 5 == 0:
                c = float(BC.self_clearance("right", s.arm_q, waist=float(s.waist_q[0]),
                                            q_fingers=None if s.hand_q is None else list(s.hand_q)))
                if c < BC.DEFAULT_MARGIN_M:
                    self.out(f"\a  WARNING: arm {c * 100:+.1f} cm in the body cover -- move it away", C_RED)
            k = keys.get()
            if k == "g":
                break
            t_next += R.DT
            io.sleep_until(t_next)
        # stiff again where it is: kp back to the policy's over 1 s, target = where it is + its sag
        t_next = io.now()
        for k in range(1, int(1.0 / R.DT) + 1):
            f = T.quintic(k / int(1.0 / R.DT))
            kp = kp_f + f * (a.kp - kp_f)
            s = io.read()
            g = self.kin.gravity_torque(s.waist_q, s.arm_q, named=T.hand_named(s))
            q = np.clip(spring.q_ref + g / kp, self.lo, self.hi)
            self._send(q, w_cmd, kp, kd_f + f * (a.kd - kd_f), has_gains)
            t_next += R.DT
            io.sleep_until(t_next)
        self.q_sent = q.copy()
        self.out("  stiff again")
        return q, None

    def _send(self, q, w, kp, kd, has_gains):
        if has_gains:
            self.io.send_arm(q, waist=w, kp=kp, kd=kd)
        else:                                      # --mock: no gains in the fake robot's stream
            self.io.send_arm(q, waist=w)
        self.q_sent = np.asarray(q, float).copy()

    def go_back(self, q_cmd, w_cmd, end):
        """To where the policy stopped (the toy's return path starts there)."""
        if float(np.max(np.abs(q_cmd - end["q_cmd"]))) < 1e-4 and abs(w_cmd - end["waist"]) < 1e-4:
            return None
        traj = T.joint_path_traj([q_cmd, end["q_cmd"]], CAL["--jog_vmax"] * 2, min_s=0.5)
        ok, c0, worst, c_end, bad, note = self.path_check(traj, self.io.read())
        if not ok:
            return f"the way back to the policy's end goes into the body cover ({worst * 100:+.1f} cm): held here"
        if abs(w_cmd - end["waist"]) > 1e-4:
            err = self.stream([q_cmd] * int(1.0 / R.DT), label="calib waist back", waist_cmd=end["waist"])
            if err:
                return err
        return self.stream(traj, label="calib back", waist_cmd=end["waist"])

    def record(self, end, q_cmd, w_cmd, moves, mode_used):
        a = self.args
        s = self.io.read()
        fr = self.frames(s)
        fe = end["fr"]
        rot = float(np.linalg.norm(RC.R_to_rotvec(fr.palm_R @ fe.palm_R.T)))
        apple_true = getattr(self, "apple_true", None)
        apple_true = self.apple if apple_true is None else apple_true
        waist3_end = np.array([end["waist"], float(s.waist_q[1]), float(s.waist_q[2])])
        p_end_cmd, _ = self._pose(waist3_end, end["q_cmd"])
        trial = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "robot": self.io.name,
            "policy": os.path.abspath(a.policy), "policy_sha": RC.file_sha1(a.policy),
            "single_version": getattr(a, "single_version", 1), "speed": a.speed,
            "apple_true": np.asarray(apple_true, float).tolist(), "apple_fed": np.asarray(self.apple, float).tolist(),
            "apple_radius": a.apple_radius, "closer_cm": 100.0 * float(getattr(self, "_closer", 0.0)),
            "calib_applied": getattr(self, "_calib_path", None),
            "q_end_cmd": end["q_cmd"].tolist(), "q_end_meas": end["q_meas"].tolist(),
            "waist_end_deg": math.degrees(end["waist"]),
            "palm_end": fe.palm_ref.tolist(), "palm_end_R": fe.palm_R.tolist(), "palm_end_cmd": p_end_cmd.tolist(),
            "q_ideal_cmd": np.asarray(q_cmd, float).tolist(), "q_ideal_meas": s.arm_q.tolist(),
            "waist_ideal_deg": math.degrees(float(s.waist_q[0])),
            "palm_ideal": fr.palm_ref.tolist(), "palm_ideal_R": fr.palm_R.tolist(),
            "torso_R_end": fe.torso_R.tolist(),
            "rot_err_deg": math.degrees(rot), "moves": int(moves), "mode": mode_used,
        }
        d = RC.trial_delta(trial)
        trial["shift_needed_cm"] = (d * 100).tolist()
        RC.append_trial(CAL["--calib_log"], trial)
        RC.append_trial(os.path.join(self.log_dir, "calib_trial.jsonl"), trial)
        msg = (f"calibration RECORDED: palm off by {np.round((fr.palm_ref - fe.palm_ref) * 100, 1).tolist()} cm, "
               f"{math.degrees(rot):.1f} deg; the apple fed should shift {np.round(d * 100, 1).tolist()} cm "
               f"-> {CAL['--calib_log']}")
        self.out("  " + msg, C_GRN)
        return msg


def main():
    global CAL
    CAL, rest = take_opts(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    args = S.parse_args()
    toy = CalibToy(args)
    if args.real and not getattr(toy.io, "loopback", False):
        print(f"{C_RED}REAL ROBOT. Someone at the robot with the remote / emergency stop, the area "
              f"in front of the right arm clear.{C_OFF}")
    out = toy.run()
    sys.exit(0 if out and ("RECORDED" in out or out.startswith("shadow")) else 1)


if __name__ == "__main__":
    main()
