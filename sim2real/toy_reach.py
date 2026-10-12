#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
SIM2REAL TOY TEST 1 -- the reach policy alone drives the real G1's right arm
(and, with --waist follow, its waist) to ONE reachable point, through
g1_fruits_picker's robot interface, with the safety watch running every step.

What runs where
  laptop (this script): the reach policy (CPU), its observation (reach_io.py:
    pinocchio FK on the g1_fruits_picker URDF -- checked against Isaac Lab by
    validate_reach_io.py), the rate limiter at --speed, every safety check; and,
    reused from g1_fruits_picker/apple_picker (imported, never modified): the
    robot app (full_move2: rt/lowstate, Dex3 transport), HandController,
    RemoteArmController (the arm daemon's protocol, waist ramps), the
    DetectionManager (--target detection), the self-collision cover
    (body_collision) and the dashboard's topics (/apple_picker/status, ...).
  robot: arm_mpc_server.py in exec_mode "qref", playing the joint targets
    streamed here on rt/arm_ctrl/qref at 50 Hz. Its own guards stay on (1.2 rad
    jump -> freeze, stream older than 0.25 s -> hold, gains clamped to its own).

Sequence: offline checks (target inside this policy's training envelope and
probed workspace; nothing else commanding the arm) -> connect -> [waist to 0]
-> arm to HOME (reach_io.HOME_Q: inside the policy's start distribution, the most
room to the legs and torso; slow, joint space, path checked against the
body) -> hand cup -> ENTER -> the policy at --speed until the palm settles on
the target (or time limit / safety stop) -> hold -> back HOME
along the way it came -> hand open.

Run inside humble-box with conda g1_dex3_sim2real. Robot side: only pane 1 of
launch_apple_picker.sh (ssh_robot_launch.sh: cameras + arm_mpc_server), plus
live_locator / fruits_to_torso for --target detection, and the dashboard if
wanted. NOT apple_picker_sm.py (two commanders).

  cd ~/dex3_rl_manipulation
  python3 sim2real/toy_reach.py --mock   --speed 0.3 --target 0.25 -0.25 0.31   # no robot at all
  python3 sim2real/toy_reach.py --shadow --speed 0.3 --target 0.25 -0.25 0.31   # robot connected, nothing moves
  python3 sim2real/toy_reach.py          --speed 0.3 --target 0.25 -0.25 0.31   # the test
  python3 sim2real/toy_reach.py          --speed 0.3 --target detection

--speed s in (0, 1] (required): the reach policy's rate limiter runs with its
velocity cap x s and acceleration cap x s^2 -- the trained motion, 1/s times
slower (envs/mdp/actions.py speed_scale; same formula, checked in sim).

--target X Y Z: torso frame (g1_fruits_picker's: torso_link, as
/fruits_detected_torso and palm_monitor.py report) at the waist the robot has
when the script starts; --frame pelvis for the pelvis frame. By default the
point is where the palm's grasp point should end (--target_is palm); a virtual
apple of --apple_radius is placed just beyond it, along the approach the
training trees give, and its contact force is fed to the policy as the palm
sensor reading (--touch virtual). --target_is apple: the point is the apple
centre (as a detection is).

Safety, every step (WARN -> SLOW -> STOP): body self-collision of the measured,
commanded and 0.25 s look-ahead arm + hand (body_collision: torso, legs);
measured joint and palm speed vs what --speed allows; the palm leaving the
region a reach from its start to the apple uses (--fence) or moving away from
the apple (--regress) -- these defaults are the policy's own sim envelope plus
margin; pushing through the virtual apple;
commanded targets clipped to the picker's joint limits; the daemon's mode,
freeze and the lowstate age; motor temperature; and envs/safety/monitor.py
(command stream, external torque = tau_est - pinocchio RNEA, stuck joint;
the external torque only WARNs unless --resid_act: not calibrated on the robot).
--safety on: SLOW multiplies the speed by --safety_slow, STOP holds the arm
where it is (gravity-compensated target) and offers to retreat along its path;
--safety monitor: report only. Ctrl-C = STOP.

EMERGENCY STOP (g1_fruits_picker estop_link / onboard/estop.py): this script sends the
robot a heartbeat; if it dies, hangs or loses the link the robot stops by itself (arms
frozen then damped, hands opened then damped, standing). The dashboard's STOP button,
`python3 estop_cli.py stop` or the robot's own stop end the run at once: nothing more is
sent, no retreat. Clear with `python3 estop_cli.py reset` (apple_picker/) and rerun.

Logs: logs/sim2real/toy_reach/<time>_<mode>/ (run.npz: every step; meta.json).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import reach_io as R
import tactile_io as TIO  # noqa: E402

PICKER_DIR = os.path.expanduser("~/g1_fruits_picker/apple_picker")
#: the daemon's enable blend after an exec_mode switch (arm_mpc_server ENABLE_RAMP_S)
DAEMON_ENABLE_RAMP_S = 2.0
#: the daemon freezes a side when |q_ref - q_meas| exceeds this (ARM_MPC_QJUMP_RAD)
DAEMON_JUMP_RAD = 1.2
LOOKAHEAD_S = 0.25
ARM_RATED = [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0]

C_RED, C_YEL, C_GRN, C_CYN, C_OFF = "\033[1;31m", "\033[1;33m", "\033[1;32m", "\033[1;36m", "\033[0m"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", type=float, default=None,
                   help="speed factor (0, 1] of the reach policy: velocity cap x s, acceleration cap x s^2")
    p.add_argument("--target", nargs="+", default=None,
                   help="X Y Z (see --frame / --target_is), or 'detection' (required unless --home_only)")
    p.add_argument("--frame", choices=["torso", "pelvis"], default="torso")
    p.add_argument("--target_is", choices=["palm", "apple"], default="palm",
                   help="palm: where the palm's grasp point should end (default); apple: the apple centre")
    p.add_argument("--apple_radius", type=float, default=0.040, help="virtual apple radius [m] (trees: 0.032-0.048)")
    p.add_argument("--approach_deg", type=float, default=None,
                   help="approach heading of the palm in the pelvis frame [deg] (default: as the training trees)")
    p.add_argument("--policy", default=R.POLICY_DEFAULT)
    m = p.add_mutually_exclusive_group()
    m.add_argument("--mock", action="store_true", help="no robot: a kinematic stand-in (offline test of the script)")
    m.add_argument("--shadow", action="store_true", help="robot connected, read only: nothing is commanded")
    p.add_argument("--waist", choices=["hold", "follow"], default="follow",
                   help="follow (default): the policy turns the torso as trained -- its rate-limited waist "
                        "target is streamed with the arm (daemon v22: qref slot 12; an older daemon gets "
                        "waist ramps); hold: the waist stays at 0, the policy's waist output is ignored")
    p.add_argument("--waist_limit_deg", type=float, default=45.0)
    p.add_argument("--hand", choices=["cup", "open", "keep"], default="cup",
                   help="cup: the reach pre-shape the policy was trained with (default)")
    p.add_argument("--touch", choices=["virtual", "zero"], default="virtual",
                   help="palm force input: the virtual apple's contact (default) or always 0")
    p.add_argument("--kp", type=float, default=R.ARM_KP, help="arm stiffness sent with the targets (sim: 100)")
    p.add_argument("--kd", type=float, default=R.ARM_KD, help="arm damping (sim: 10)")
    p.add_argument("--time_limit", type=float, default=None, help="s of policy (default 8 s / speed)")
    p.add_argument("--reach_tol", type=float, default=0.010,
                   help="m: REACHED when the palm's grasp point is within this of the touch distance")
    p.add_argument("--after_reach", choices=["freeze", "policy"], default="freeze",
                   help="freeze: hold the arm where it reached (default -- a virtual apple cannot push back, "
                        "and in training the real apple's contact is what stopped the palm); policy: keep the "
                        "policy running (a REAL apple placed at the point)")
    p.add_argument("--settle_s", type=float, default=0.5, help="--after_reach policy: s within --reach_tol and still")
    p.add_argument("--hold_s", type=float, default=1.5, help="s to hold after reaching, before returning")
    p.add_argument("--retreat_window_s", type=float, default=1.5,
                   help="s of the arm's own path retraced first on the way back (then straight home)")
    p.add_argument("--return_to", choices=["start", "none"], default="start")
    p.add_argument("--open_roll", type=float, default=-0.40,
                   help="first move of the run: abduct the right shoulder roll to this angle [rad] (negative = "
                        "outward) so the arm leaves the leg, then go home; 0 = straight home")
    p.add_argument("--move_speed", type=float, default=0.35,
                   help="rad/s peak joint speed of the scripted moves (home, back)")
    p.add_argument("--detect_s", type=float, default=3.0,
                   help="--target detection: seconds of detections sampled at home before choosing")
    p.add_argument("--detect_spread", type=float, default=0.015,
                   help="m: a fruit whose sampled positions spread more than this is not offered")
    p.add_argument("--pick", type=int, default=None,
                   help="--target detection: the fruit # to reach (default: the nearest reachable one)")
    p.add_argument("--mock_apples", default=None,
                   help="--mock: fruit the stand-in camera 'sees', torso frame, 'x,y,z;x,y,z' (3 mm noise)")
    p.add_argument("--home_only", action="store_true",
                   help="only move the arm HOME (waist to 0), hold it there and exit -- e.g. at the start of a "
                        "session or after an E-STOP reset")
    p.add_argument("--reach_scale", type=float, default=1.25,
                   help="the policy's envelope is judged as F times larger (the real arm reaches further than "
                        "the sim's probed map); 1 = exactly the training envelope. Outside the training range "
                        "the policy is untested: the safety watch still applies")
    g = p.add_argument_group("grasp (toy_grasp.py)")
    g.add_argument("--grasp", action="store_true",
                   help="after the reach, hand off to the GRASP policy (hand-only, 44 in / 14 out): close the "
                        "fingers on the fruit, hold, release, back home")
    g.add_argument("--grasp_policy", default=R.GRASP_POLICY_DEFAULT)
    g.add_argument("--grip_n", type=float, default=6.0,
                   help="N: the grip-force COMMAND (trained 0.45-0.75 x the 20-40 N damage force = 11-19 N; the "
                        "pad force is uncalibrated -- start low)")
    g.add_argument("--grip_stop_n", type=float, default=20.0,
                   help="N: the estimated pad force of any finger above this opens the hand")
    g.add_argument("--grasp_s", type=float, default=4.0, help="s of grasp policy before it is judged")
    g.add_argument("--grasp_hold_s", type=float, default=2.0, help="s the grip is kept (policy running) after it formed")
    g.add_argument("--handoff_s", type=float, default=0.5, help="s the arm holds at the reach target before the hand-off")
    g.add_argument("--grasp_speed", type=float, default=None, help="speed factor of the grasp limiters (default --speed)")
    g.add_argument("--grasp_arm_free", action="store_true",
                   help="unlock the shoulder/elbow (scale 0.05) once a grip formed, as in training; default: only "
                        "the wrist is free (as in training before the grip)")
    g.add_argument("--grasp_wrist_max", type=float, default=0.15,
                   help="rad: the wrist targets stay within this of the hand-off pose (sim: up to 0.2-0.4 against a "
                        "compliant stem; at the arm's kp 100 each 0.1 rad blocked is 10 N*m)")
    g.add_argument("--grasp_move_stop", type=float, default=0.04,
                   help="m: the palm moving further than this from the hand-off point stops the grasp")
    g.add_argument("--tactile_calib", default=None,
                   help="tactile_calibrate.py's file (default: logs/sim2real/tactile_calib.json when it exists)")
    g.add_argument("--tactile_gain", type=float, default=3000.0,
                   help="raw Dex3 pressure counts per Newton (the sim's placeholder: tactile_layout."
                        "SIM_FORCE_TO_RAW_GAIN; uncalibrated on the real hand)")
    g.add_argument("--tactile_top", type=int, default=3,
                   help="the pad force is the mean of this many strongest pad rises above their baseline")
    g.add_argument("--tactile_map", default="pool",
                   help="'pool' (every pad feeds every finger: the grouping of the nine press_sensor_state "
                        "entries is not documented) or 'thumb=0,1;index=2,3;middle=4,5;palm=6,7,8' (entry numbers)")
    p.add_argument("--force", action="store_true", help="run a target outside the policy's training envelope")
    p.add_argument("--shadow_s", type=float, default=5.0, help="--shadow: seconds of read-only policy")
    p.add_argument("--realtime", action="store_true", help="--mock: run in real time (default: as fast as possible)")
    p.add_argument("--yes", action="store_true",
                   help="no confirmations: --mock, --shadow, or a loopback ('lo') test robot only -- ignored on "
                        "the robot's network")
    s = p.add_argument_group("safety")
    s.add_argument("--safety", choices=["on", "monitor"], default="on",
                   help="on: SLOW / STOP act; monitor: report only (the daemon / lowstate checks still stop)")
    s.add_argument("--safety_slow", type=float, default=0.5, help="speed factor x this while the watch says SLOW")
    s.add_argument("--resid_scale", type=float, default=1.0, help="x the external-torque thresholds (0.12 / 0.20 / "
                   "0.30 of rated torque, sustained)")
    s.add_argument("--resid_act", action="store_true",
                   help="let the external-torque layer SLOW / STOP too (default: it only WARNs -- its thresholds "
                        "are sim-calibrated, contact and no-contact overlap there; calibrate on the robot first)")
    s.add_argument("--clear_warn", type=float, default=0.010, help="m, body_collision clearance -> WARN")
    s.add_argument("--clear_stop", type=float, default=-0.005,
                   help="m, -> STOP (sphere covers; -0.011 = the calibrated contact line)")
    # Defaults from the policy itself (validate_reach_io.py data, 124 sim episodes at speed 1 / 0.5 / 0.3):
    # palm speed max per episode 1.62 / 0.99 / 0.84 m/s; joint speed 1.99 / 1.42 / 1.03 rad/s; palm beyond
    # the ball spanned by start palm -> apple at most 0.23 m; distance to the apple grew at most 0.13 m.
    s.add_argument("--fence", type=float, default=0.25,
                   help="m: palm this far outside the ball whose diameter is start palm -> apple -> STOP")
    s.add_argument("--regress", type=float, default=0.20,
                   help="m: palm this much FURTHER from the apple than at the start -> STOP")
    s.add_argument("--palm_speed_stop", type=float, default=None, help="m/s (default 2.0 x speed + 0.35)")
    s.add_argument("--joint_speed_stop", type=float, default=None, help="rad/s (default 1.8 x 2.0 x speed + 0.5)")
    # ACCELERATION (3-step mean of the measured joint / palm velocity change). The policy's
    # own sim maximum, 124 episodes: joint 10.6 / 14.0 / 21.5 rad/s^2 and palm 8.4 / 8.9 /
    # 12.6 m/s^2 at speed 0.3 / 0.5 / 1; a free-falling arm is ~30 rad/s^2 at the shoulder.
    s.add_argument("--joint_acc_stop", type=float, default=None, help="rad/s^2 (default 1.6 x (8 + 14 x speed))")
    s.add_argument("--palm_acc_stop", type=float, default=None, help="m/s^2 (default 1.8 x (7 + 6 x speed))")
    s.add_argument("--move_acc_stop", type=float, default=12.0,
                   help="rad/s^2: joint acceleration that stops a SCRIPTED move (home, back)")
    s.add_argument("--temp_warn", type=float, default=80.0)
    s.add_argument("--temp_stop", type=float, default=95.0)
    s.add_argument("--freeze_s", type=float, default=0.5)
    p.add_argument("--app_hz", type=float, default=100.0,
                   help="loop rate of the picker's robot app in THIS process (hand commands; the picker "
                        "itself runs 250): every Python thread here competes with the 50 Hz control loop")
    p.add_argument("--picker_dir", default=PICKER_DIR)
    p.add_argument("--config", default=None, help="apple_picker_config.json (default: the picker's)")
    p.add_argument("--urdf", default=R.URDF_DEFAULT)
    p.add_argument("--log_dir", default=None)
    a = p.parse_args()
    if a.home_only:
        a.speed = a.speed or 0.3                    # unused: no policy runs
        a.target = a.target or ["0", "0", "0"]
    if a.speed is None or a.target is None:
        p.error("--speed and --target are required (except with --home_only)")
    if not 0.0 < a.speed <= 1.0:
        p.error("--speed must be in (0, 1]")
    if a.target != ["detection"] and len(a.target) != 3:
        p.error("--target takes X Y Z or 'detection'")
    if a.target == ["detection"]:
        if a.mock and not a.mock_apples:
            p.error("--target detection with --mock needs --mock_apples")
        a.target_is = "apple"
    a.real = not (a.mock or a.shadow)
    a.time_limit = a.time_limit or 8.0 / a.speed
    a.palm_speed_stop = a.palm_speed_stop or (2.0 * a.speed + 0.35)
    a.joint_speed_stop = a.joint_speed_stop or (1.8 * R.ARM_MAX_VELOCITY * a.speed + 0.5)
    a.joint_acc_stop = a.joint_acc_stop or (1.6 * (8.0 + 14.0 * a.speed))
    a.palm_acc_stop = a.palm_acc_stop or (1.8 * (7.0 + 6.0 * a.speed))
    return a


# ─────────────────────────────────────────────
#  Robot I/O
# ─────────────────────────────────────────────

@dataclass
class Snap:
    """One reading of the robot (radians, N*m, pelvis-attached joint frames)."""
    arm_q: np.ndarray
    arm_dq: np.ndarray
    arm_tau: np.ndarray
    waist_q: np.ndarray                      # yaw, roll, pitch
    waist_dq: np.ndarray
    arm_temp: np.ndarray | None = None
    hand_q: np.ndarray | None = None         # Dex3 order (R.DEX3_ORDER)
    hand_dq: np.ndarray | None = None
    pads: np.ndarray | None = None           # every press_sensor_state element, raw counts (9 entries x 12)
    tactile: float | None = None             # raw Dex3 pressure mean (not a force)
    age: float = 0.0                         # s since the lowstate sample
    mode: str | None = None                  # daemon exec_mode, right arm
    frozen: bool | None = None
    daemon_age: float | None = None
    t: float = 0.0                           # when it was read (the io's clock)


def hand_named(s: Snap) -> dict:
    if s.hand_q is None:
        return {}
    return {f"right_hand_{n}_joint": float(v) for n, v in zip(R.DEX3_ORDER, s.hand_q)}


def quintic(f):
    f = min(max(f, 0.0), 1.0)
    return f ** 3 * (10 - 15 * f + 6 * f * f)


class MockIO:
    """A kinematic stand-in: the arm follows the streamed target through the sim's
    PD (kp, kd) on a unit-inertia joint, the waist plays daemon ramps; tau_est =
    the model torque (no external load). Simulated clock."""

    name = "mock"
    commands = True

    def __init__(self, args, kin: R.G1Kinematics):
        self.args, self.kin = args, kin
        self.t = 0.0
        self.q = np.array([-0.5, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0])       # forearm forward, as a picker standby
        self.dq = np.zeros(7)
        self.w = np.array([0.04, 0.0, R.WAIST_DEFAULT[2]])
        self.wd = np.zeros(3)
        self.q_ref, self.mode, self.frozen, self.ramp = None, "mpc", False, None
        self.hand = np.array([math.radians(30), 0, 0, 0, 0, 0, 0], float)
        self.hand_tgt = self.hand.copy()
        self.hand_dq = np.zeros(7)
        self.pad_base = np.full(9, 100000.0)
        self.kp, self.kd = args.kp, args.kd
        self.estop_why = None

    def connect(self):
        pass

    def now(self):
        return self.t

    def read(self) -> Snap:
        tau = self.kin.gravity_torque(self.w, self.q)
        return Snap(arm_q=self.q.copy(), arm_dq=self.dq.copy(), arm_tau=tau, waist_q=self.w.copy(),
                    waist_dq=self.wd.copy(), arm_temp=np.full(7, 40.0), hand_q=self.hand.copy(),
                    hand_dq=self.hand_dq.copy(), pads=self._pads(), tactile=0.0, age=0.002, mode=self.mode,
                    frozen=self.frozen, daemon_age=0.05, t=self.t)

    def _pads(self):
        """Stand-in tactile: a fruit in the closed fingers (index/middle bases + thumb flexed)
        presses the pads; 2 N per 0.1 rad past the contact angle, 3000 counts per N."""
        close = max(0.0, self.hand[4] - 1.45) + max(0.0, self.hand[3] - 0.5) + max(0.0, -self.hand[2] - 0.7)
        n = 20.0 * close
        p = np.full(108, 100000.0) + np.random.normal(0.0, 30.0, 108)
        p[:72] += 3000.0 * n
        return p

    def hand_cmd(self, q):
        self.hand_tgt = np.asarray(q, float).copy()

    def _physics(self, dt):
        if self.mode == "qref" and self.q_ref is not None and not self.frozen:
            if np.max(np.abs(self.q_ref - self.q)) > DAEMON_JUMP_RAD:
                self.frozen = True
            # unit-inertia PD with the arm's own gravity compensated: tracking only
            for _ in range(4):
                h = dt / 4
                acc = self.kp * (self.q_ref - self.q) - self.kd * self.dq
                acc = np.clip(acc, -60.0, 60.0)
                self.dq += acc * h
                self.q += self.dq * h
        else:
            self.dq *= 0.0
        # hand: a rate-limited follow of its target (thumb_0 locked at its rest)
        step = 3.0 * dt
        new = self.hand + np.clip(self.hand_tgt - self.hand, -step, step)
        new[0] = self.hand[0]
        self.hand_dq = (new - self.hand) / max(dt, 1e-6)
        self.hand = new
        if self.ramp is not None:
            w0, w1, t0, d = self.ramp
            f = (self.t - t0) / d
            new = w0 + (w1 - w0) * quintic(f)
            self.wd[0] = (new - self.w[0]) / dt
            self.w[0] = new
            if f >= 1.0:
                self.ramp = None
                self.wd[0] = 0.0

    def sleep_until(self, t):
        while self.t + 1e-9 < t:
            dt = min(R.DT, t - self.t)
            self._physics(dt)
            self.t += dt
            if self.args.realtime:
                time.sleep(dt)

    def exec_mode(self, mode):
        self.mode = mode
        self.frozen = False

    qref_waist = True
    waist_src = "qref"

    def send_arm(self, q, dq=None, waist=None, waist_dq=0.0):
        self.q_ref = np.asarray(q, float).copy()
        if waist is not None:                       # as the v22 daemon: rate-limited toward it
            self.ramp = None
            step = 1.5 * R.DT
            new = self.w[0] + float(np.clip(waist - self.w[0], -step, step))
            self.wd[0] = (new - self.w[0]) / R.DT
            self.w[0] = new

    def waist_to(self, yaw, duration):
        self.ramp = (float(self.w[0]), float(yaw), self.t, max(0.5, float(duration)))

    def hand_set(self, kind):
        if kind == "cup":
            for i, n in enumerate(R.DEX3_ORDER):
                if i:
                    self.hand[i] = self.hand_tgt[i] = R.REACH_HAND_PRESHAPE[f"right_hand_{n}_joint"]
        elif kind == "open":
            self.hand_tgt[1:] = 0.0

    def detection_info(self) -> str:
        return "mock"

    def detection_targets(self):
        """--mock_apples, seen from the torso at the current waist, 3 mm noise."""
        if not self.args.mock_apples:
            return []
        out = []
        fr0 = self.kin.frames([0.0, self.w[1], self.w[2]], self.q)          # waist-0 torso
        fr = self.kin.frames(self.w, self.q)
        for txt in self.args.mock_apples.split(";"):
            p0 = np.array([float(v) for v in txt.split(",")])
            p_pel = fr0.torso_R @ p0 + fr0.torso_pos                        # fixed to the world
            p_now = fr.torso_R.T @ (p_pel - fr.torso_pos) + np.random.normal(0, 0.003, 3)
            out.append((p_now, 0.04))
        return out

    def publish(self, doc, target_torso):
        pass

    def close(self, restore_mode=True):
        pass


class PickerIO:
    """The real robot through g1_fruits_picker/apple_picker (imported, not modified).
    shadow=True: read only -- the robot app's writer thread is never started, the
    hand output is disabled and nothing is sent to the arm daemon."""

    name = "robot"

    def __init__(self, args, shadow: bool):
        self.args, self.shadow = args, shadow
        self.commands = not shadow
        self.name = "shadow" if shadow else "robot"
        pd = os.path.abspath(args.picker_dir)
        if not os.path.isdir(pd):
            sys.exit(f"g1_fruits_picker not found at {pd} (--picker_dir)")
        sys.path.insert(0, pd)
        import _paths  # noqa: F401  (puts ../common on sys.path)
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import String
        import apple_picker_sm as SM

        self.SM, self.String, self.Node = SM, String, Node
        cfg_path = args.config or os.path.join(pd, "apple_picker_config.json")
        raw = SM.load_json(cfg_path)
        raw["__cfg_dir"] = os.path.dirname(os.path.abspath(cfg_path))
        cfg = SM.AppConfig.from_dict(raw)
        cfg.arm_rate_hz = float(args.app_hz)        # this process only (see --app_hz)
        # The control loop shares the GIL with the robot app's threads (500 Hz lowstate, both
        # Dex3 states, its own loop): measured on the robot 2026-10-12, 16-19 ms of each 20 ms
        # step and 4-15% late steps. A short switch interval hands the GIL back sooner.
        sys.setswitchinterval(0.001)
        if str(cfg.mpc.arm_backend).lower() != "onboard":
            sys.exit(f"{cfg_path}: mpc.arm_backend is '{cfg.mpc.arm_backend}' -- this test needs the onboard "
                     "daemon (the laptop must never write rt/arm_sdk itself)")
        self.cfg_path = cfg_path
        #: the DDS interface the picker's robot app binds: 'lo' = a test stand-in, never the robot
        self.net_if = str(cfg.net_if or (cfg.full_move2 or {}).get("net_if", ""))
        self.loopback = self.net_if == "lo"
        if self.loopback:
            self.name = "loopback test robot" if not shadow else "shadow (loopback)"
        rclpy.init()
        self.rclpy = rclpy

        class ToyNode(SM.ApplePickerNode):
            """The picker node's own init steps, without its state machine."""

            def __init__(self, cfg, shadow):
                Node.__init__(self, "rl_toy_reach")
                self.cfg = cfg
                self._init_fullmove_config()
                self._init_diagnostics()
                self._init_ros()                      # DetectionManager (/fruits_detected_torso)
                # the picker's wait_for_target_fruit waits for a voice / dashboard label
                # choice before accepting ANY fruit; this test takes the configured labels
                labels = tuple(cfg.filter.accepted_labels)
                if labels:
                    self.detection.set_accepted_labels(labels)
                # The picker's detection box (torso frame, x 0.04-0.46 m ...) is sized for its
                # table; a tree fruit further out is dropped before it is tracked. This test
                # judges reachability itself (the policy's envelope + map), so the box here
                # is only "in front of the robot".
                import dataclasses
                self.detection.workspace = dataclasses.replace(
                    self.detection.workspace, x_min=0.0, x_max=1.5, y_min=-1.0, y_max=1.0,
                    z_min_det=-0.8, z_max=1.2)
                self._init_hardware()                 # full_move2 robot app + HandController
                self.app.set_arm_output_enabled(False)   # never a second rt/arm_sdk writer
                if shadow:
                    self.hand.set_hand_output_enabled(False)
                self._init_hand_transport()
                self._init_gains_and_logging()
                self._init_left_arm()                 # RemoteArmController (onboard backend)

        self.node = ToyNode(cfg, shadow)
        self.app, self.hand, self.arm = self.node.app, self.node.hand, self.node.arm
        self.pub_status = self.node.create_publisher(String, SM.STATUS_TOPIC, 10)
        self.pub_target = self.node.create_publisher(String, "/apple_picker/grasp_target", 10)
        self.pub_agri = self.node.create_publisher(String, "AgriHumanoid_state", 10)
        # EMERGENCY STOP: heartbeat to the robot + its stop state + the dashboard's button
        from estop_link import EStopLink
        self.estop_why = None
        self.estop = EStopLink("rl_toy_reach", on_estop=self._on_estop, heartbeat=not shadow,
                               watch_state=not shadow).start()
        self.node.create_subscription(String, getattr(SM, "ESTOP_TOPIC", "/apple_picker/estop"),
                                      self._on_estop_topic, 10)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        threading.Thread(target=self.ex.spin, daemon=True).start()

        # the daemon's own state (frozen / exec_mode are not kept by RemoteArmController)
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdMsg
        self._dstate, self._dstate_t = {}, None
        self._dsub = ChannelSubscriber("rt/arm_ctrl/state", String_)
        self._dsub.Init(self._on_daemon_state, 10)
        self._qpub = None
        if not shadow:
            self._qpub = ChannelPublisher("rt/arm_ctrl/qref", LowCmd_)
            self._qpub.Init()
            self._cmd = LowCmdMsg()
        self.streaming = False

    def _on_estop_topic(self, msg):
        try:
            d = json.loads(msg.data or "{}")
        except Exception:
            d = {}
        self.estop.trigger(str(d.get("why") or "dashboard STOP button")
                           + (f" ({d['client']})" if d.get("client") else ""), "dashboard")

    def _on_estop(self, why, origin):
        """Once per E-STOP: nothing more from this process -- the robot freezes, damps,
        opens the hands; this laptop stops its hand writer and asks the loco to stand."""
        self.estop_why = f"{why} [{origin}]"
        print(f"\n{C_RED}  !!! E-STOP: {self.estop_why}{C_OFF}", flush=True)
        if self.commands:
            try:
                self.arm.estop_latch()                 # the picker's controller sends nothing more either
            except Exception:
                pass
        try:
            self.hand.set_hand_output_enabled(False)
        except Exception:
            pass

        def _loco():
            try:
                self.app.loco.StopMove()
            except Exception:
                pass
        if self.commands:
            threading.Thread(target=_loco, daemon=True).start()

    def _on_daemon_state(self, msg):
        try:
            d = json.loads(msg.data)
            self._dstate, self._dstate_t = d.get("right") or {}, time.monotonic()
            self._dfull = d
        except Exception:
            pass

    def log(self, msg):
        print(f"  [{self.name}] {msg}", flush=True)

    def connect(self):
        self.log("waiting for rt/lowstate ...")
        if not self.app.wait_ready(timeout_s=8.0):
            sys.exit("no rt/lowstate -- check the cable / network (student_setup/LOCAL_SETUP_this_laptop.md)")
        if not self.shadow:
            self.app.start()                      # the hand writer (arm output stays disabled)
            if self.args.hand != "keep":
                self.node._set_hand_output_enabled(True, reason="rl_toy_reach")
                self.node._startup_force_open_hands()
        self.log("waiting for the arm daemon (rt/arm_ctrl/state) ...")
        if not self.arm.wait_ready(timeout_s=60.0 if self.shadow else 90.0):   # discovery: ~20-35 s
            if self.shadow:
                self.log(f"{C_YEL}arm daemon not seen -- shadow continues without it{C_OFF}")
            else:
                sys.exit("arm daemon not up: start pane 1 of launch_apple_picker.sh (ssh_robot_launch.sh)")
        t0 = time.monotonic()
        while self._dstate_t is None and time.monotonic() - t0 < 2.0:      # our own state listener too
            time.sleep(0.05)
        es = self.estop.daemon_estop or {}
        d = getattr(self, "_dfull", {}) or {}
        print(f"  [{self.name}] daemon: right arm mode '{(d.get('right') or {}).get('exec_mode')}', "
              f"qref waist {'yes' if d.get('qref_waist') else 'NO (old build)'}, E-STOP "
              f"{'ACTIVE: ' + str(es.get('why')) if es.get('active') else ('off' if es else 'n/a (old build)')}",
              flush=True)
        if es.get("active") and not self.shadow:
            sys.exit(f"the robot is in E-STOP ({es.get('why')} [{es.get('src')}]): fix the cause, "
                     f"`python3 estop_cli.py reset` in apple_picker/, rerun")
        if self.estop.daemon_estop is None and not self.shadow:
            print(f"  {C_YEL}[{self.name}] the arm daemon reports no E-STOP block: it is not the v21-estop "
                  f"build -- the robot will NOT stop by itself if this laptop is lost. Deploy "
                  f"onboard/arm_mpc_server.py + onboard/estop.py.{C_OFF}")

    def now(self):
        return time.monotonic()

    def sleep_until(self, t):
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)

    def read(self) -> Snap:
        st = self.app.low_state
        ms = st.motor_state
        arm_i = [R.MOTOR_IDX[n] for n in R.RIGHT_ARM_JOINTS]
        w_i = [R.MOTOR_IDX[n] for n in R.WAIST_JOINTS]
        g = lambda idx, f: np.array([float(getattr(ms[i], f, 0.0)) for i in idx])
        temp = np.array([self.SM._first_float(getattr(ms[i], "temperature", None)) or float("nan") for i in arm_i])
        rx = getattr(self.app, "_low_state_rx_mono", None)
        hq = self.hand.hand_finger_q("right")
        try:
            tac = float(self.app.get_rh_tactile_mean())
        except Exception:
            tac = None
        d = self._dstate
        hdq = pads = None
        try:
            hs = self.app.right_hand_state
            if hs is not None and len(hs.motor_state) >= 7:
                hdq = np.array([float(getattr(hs.motor_state[i], "dq", 0.0)) for i in range(7)])
            if self.args.grasp:
                pads = TIO.flat_from_handstate(hs)
        except Exception:
            pass
        return Snap(arm_q=g(arm_i, "q"), arm_dq=g(arm_i, "dq"), arm_tau=g(arm_i, "tau_est"),
                    waist_q=g(w_i, "q"), waist_dq=g(w_i, "dq"), arm_temp=temp,
                    hand_q=None if hq is None else np.asarray(hq, float), hand_dq=hdq, pads=pads, tactile=tac,
                    age=(time.monotonic() - rx) if rx else 0.0,
                    mode=d.get("exec_mode"), frozen=d.get("frozen"),
                    daemon_age=None if self._dstate_t is None else time.monotonic() - self._dstate_t,
                    t=time.monotonic())

    def exec_mode(self, mode):
        if self.commands and self.estop_why is None:
            self.arm.set_exec_mode("right", mode)

    @property
    def qref_waist(self) -> bool:
        """The daemon takes the waist yaw in the qref stream (v22-estop and later)."""
        return bool(getattr(self, "_dfull", {}).get("qref_waist"))

    @property
    def waist_src(self):
        """Who drives the daemon's waist: 'qref' only while OUR stream (slot 12) is arriving
        fresh -- the proof that the daemon's reader has matched this process's writer."""
        return getattr(self, "_dfull", {}).get("waist_src")

    def send_arm(self, q, dq=None, waist=None, waist_dq=0.0):
        if not self.commands or self.estop_why is not None:
            return                                  # never a command after an E-STOP
        dq = np.zeros(7) if dq is None else dq
        for j, n in enumerate(R.RIGHT_ARM_JOINTS):
            m = self._cmd.motor_cmd[R.MOTOR_IDX[n]]
            m.mode = 1
            m.q, m.dq, m.tau = float(q[j]), float(dq[j]), 0.0
            m.kp, m.kd = float(self.args.kp), float(self.args.kd)
        w = self._cmd.motor_cmd[R.MOTOR_IDX["waist_yaw_joint"]]
        if waist is not None and self.qref_waist:
            w.mode, w.q, w.dq, w.tau, w.kp, w.kd = 1, float(waist), float(waist_dq), 0.0, 1.0, 0.0   # kp > 0 = commanded
        else:
            w.mode, w.kp = 0, 0.0                   # not commanded: the daemon holds the waist
        self._qpub.Write(self._cmd)
        self.streaming = True

    def waist_to(self, yaw, duration):
        if self.commands and self.estop_why is None:
            self.arm.ensure_waist(float(yaw), duration_s=max(0.5, float(duration)), force=True)

    def hand_set(self, kind):
        if not self.commands or kind == "keep" or self.estop_why is not None:
            return
        if kind == "open":
            self.hand.hand_open("right")
            return
        # the cup, joint by joint through full_move2 (thumb_0 is runtime-locked there)
        for i, n in enumerate(R.DEX3_ORDER):
            if i == 0:
                continue
            self.app.set_hand_joint_deg("right", i, math.degrees(R.REACH_HAND_PRESHAPE[f"right_hand_{n}_joint"]))

    def hand_cmd(self, q_dex3):
        """Joint targets [rad] for the seven Dex3 joints (R.DEX3_ORDER). thumb_0 is runtime-locked
        by the picker (30 deg) and is not commanded."""
        if not self.commands or self.estop_why is not None:
            return
        for i in range(1, 7):
            self.app.set_hand_joint_deg("right", i, math.degrees(float(q_dex3[i])))

    def detection_info(self) -> str:
        """The tracker's own account of the last message (what it kept, what it dropped, why)."""
        try:
            return str(self.node.detection.last_det_info)
        except Exception:
            return "?"

    def detection_targets(self):
        snap = self.node.detection.get_snapshot()
        rad = list(snap.radii) + [None] * len(snap.apples_all)
        return [(np.asarray(p, float), rad[i]) for i, p in enumerate(snap.apples_all)]

    def publish(self, doc, target_torso):
        try:
            m = self.String()
            m.data = json.dumps(doc, separators=(",", ":"))
            self.pub_status.publish(m)
            t = self.String()
            t.data = json.dumps({} if target_torso is None else
                                {"x": float(target_torso[0]), "y": float(target_torso[1]),
                                 "z": float(target_torso[2]), "side": "right"})
            self.pub_target.publish(t)
            b = self.node._collect_body_motor_states()
            for it in b:
                it["idx"] = it.get("i")
            a = self.String()
            a.data = json.dumps({"body_states": b, "body_motors": b}, separators=(",", ":"))
            self.pub_agri.publish(a)
        except Exception as e:
            print(f"  [status] publish failed: {e}")

    def close(self, restore_mode=True):
        try:
            self.estop.bye()                        # a clean exit is not an emergency
        except Exception:
            pass
        try:
            if self.commands and restore_mode and self.estop_why is None:
                # stop streaming first (the daemon latches a hold at the last command),
                # then hand the side back to the default mode for the picker
                time.sleep(0.3)
                t_end = time.monotonic() + 4.0
                while time.monotonic() < t_end:         # one send can be lost: until the daemon says so
                    self.arm.set_exec_mode("right", "mpc")
                    time.sleep(0.3)
                    if self._dstate.get("exec_mode") == "mpc":
                        break
        except Exception:
            pass
        try:
            if not self.shadow:
                self.app.stop()
        except Exception:
            pass
        try:
            self.ex.shutdown()
            self.Node.destroy_node(self.node)
            self.rclpy.shutdown()
        except Exception:
            pass


# ─────────────────────────────────────────────
#  Safety watch
# ─────────────────────────────────────────────

class Watch:
    """Every check of one control step -> (level, reasons). Levels as
    envs/safety/monitor.py: OK < WARN < SLOW < STOP."""

    def __init__(self, args, kin):
        import importlib.util as ilu
        spec = ilu.spec_from_file_location("safety_monitor", os.path.join(REPO, "envs", "safety", "monitor.py"))
        sm = ilu.module_from_spec(spec)
        sys.modules["safety_monitor"] = sm
        spec.loader.exec_module(sm)
        import torch
        torch.set_num_threads(1)             # tiny tensors: a thread pool only adds waits (see reach_io)
        self.torch, self.sm = torch, sm
        # external-torque thresholds; without --resid_act they are moved out of reach
        # so the monitor's levels come from the command stream and the stuck test, and
        # the residual is judged here, as WARN only (see step)
        k = args.resid_scale
        self.resid_thr = (0.12 * k, 0.20 * k, 0.30 * k)
        m = 1.0 if args.resid_act else 1e3
        self.mon = sm.SafetyMonitor(sm.SafetyCfg(dt=R.DT, rated=list(ARM_RATED), resid_warn=0.12 * k * m,
                                                 resid_slow=0.20 * k * m, resid_stop=0.30 * k * m), 1)
        self.args, self.kin = args, kin
        sys.path.insert(0, os.path.abspath(args.picker_dir))
        import _paths  # noqa: F401
        import body_collision as BC
        self.BC = BC
        self.dq_prev = None
        self.t_prev = None
        self.palm_v = 0.0
        self.palm_prev = None
        self.acc_hist = []                   # last measured joint accelerations
        self.palm_v_hist = []                # last palm velocity vectors
        self.peak_acc = 0.0
        self.peak_palm_acc = 0.0
        self.last_print = {}
        self.peak_resid = np.zeros(7)
        self.min_clear = float("inf")

    def reset(self):
        self.mon.reset()
        self.dq_prev = None
        self.t_prev = None
        self.palm_prev = None
        self.palm_v = 0.0
        self.acc_hist, self.palm_v_hist = [], []

    def clearance(self, q, waist, hand_q):
        return float(self.BC.self_clearance("right", np.asarray(q, float), waist=float(waist),
                                            q_fingers=None if hand_q is None else list(hand_q)))

    def step(self, s: Snap, fr: R.Frames, q_cmd, v_cmd, speed, ctx: dict):
        a = self.args
        sm = self.sm
        out = []                                           # (level, reason)
        # plumbing: the daemon and the data
        if s.age > 0.1:
            out.append((sm.STOP, f"lowstate {s.age * 1000:.0f} ms old"))
        if ctx.get("commanding"):
            if s.frozen:
                out.append((sm.STOP, "the daemon FROZE the arm (|q_ref - q| > 1.2 rad)"))
            if s.mode is not None and s.mode != "qref":
                out.append((sm.STOP, f"daemon exec_mode is '{s.mode}', not qref"))
            if s.daemon_age is not None and s.daemon_age > 1.0:
                out.append((sm.WARN, f"daemon state {s.daemon_age:.1f} s old"))
        # speeds
        jv = float(np.max(np.abs(s.arm_dq)))
        jv_warn = 1.3 * R.ARM_MAX_VELOCITY * speed + 0.3
        if jv > a.joint_speed_stop:
            out.append((sm.STOP, f"joint speed {jv:.2f} rad/s > {a.joint_speed_stop:.2f}"))
        elif jv > jv_warn:
            out.append((sm.WARN, f"joint speed {jv:.2f} rad/s (> {jv_warn:.2f})"))
        # rates from the MEASURED time between two readings: a late step divided by the
        # nominal 20 ms reads as a spike (a 39 ms step doubled the palm speed estimate)
        dt = R.DT if self.t_prev is None else max(0.005, s.t - self.t_prev)
        self.t_prev = s.t
        ctx["dt"] = dt
        if dt > 2.0 * R.DT:
            out.append((sm.WARN, f"control step {dt * 1000:.0f} ms late (50 Hz = 20 ms)"))
        if self.palm_prev is not None:
            pv = (fr.palm_ref - self.palm_prev) / dt
            v = float(np.linalg.norm(pv))
            self.palm_v = 0.5 * self.palm_v + 0.5 * v
            self.palm_v_hist = (self.palm_v_hist + [pv])[-4:]
        self.palm_prev = fr.palm_ref.copy()
        # ACCELERATION: 3-step mean of the measured change (as calibrated in sim)
        if self.dq_prev is not None:
            self.acc_hist = (self.acc_hist + [(s.arm_dq - self.dq_prev) / dt])[-3:]
        if len(self.acc_hist) == 3:
            ja = np.abs(np.mean(self.acc_hist, axis=0))
            j = int(np.argmax(ja))
            self.peak_acc = max(self.peak_acc, float(ja[j]))
            ctx["joint_acc"] = round(float(ja[j]), 1)
            if ja[j] > a.joint_acc_stop:
                out.append((sm.STOP, f"joint ACCELERATION {ja[j]:.1f} rad/s^2 at {R.RIGHT_ARM_JOINTS[j][6:-6]} "
                                     f"> {a.joint_acc_stop:.1f}"))
            elif ja[j] > 0.75 * a.joint_acc_stop:
                out.append((sm.WARN, f"joint acceleration {ja[j]:.1f} rad/s^2 at {R.RIGHT_ARM_JOINTS[j][6:-6]}"))
        if len(self.palm_v_hist) == 4:
            pa = float(np.linalg.norm(np.mean(np.diff(self.palm_v_hist, axis=0), axis=0))) / dt
            self.peak_palm_acc = max(self.peak_palm_acc, pa)
            ctx["palm_acc"] = round(pa, 1)
            if pa > a.palm_acc_stop:
                out.append((sm.STOP, f"palm ACCELERATION {pa:.1f} m/s^2 > {a.palm_acc_stop:.1f}"))
            elif pa > 0.75 * a.palm_acc_stop:
                out.append((sm.WARN, f"palm acceleration {pa:.1f} m/s^2"))
        pv_warn = 1.5 * a.speed + 0.1
        if self.palm_v > a.palm_speed_stop:
            out.append((sm.STOP, f"palm speed {self.palm_v:.2f} m/s > {a.palm_speed_stop:.2f}"))
        elif self.palm_v > pv_warn:
            out.append((sm.WARN, f"palm speed {self.palm_v:.2f} m/s (> {pv_warn:.2f})"))
        wv = abs(float(s.waist_dq[0]))
        if wv > 1.3 * R.WAIST_MAX_VELOCITY * speed + 0.2:
            out.append((sm.WARN, f"waist speed {wv:.2f} rad/s"))
        # self-collision: where the arm is, where it is sent, where it is heading
        w = float(s.waist_q[0])
        c_meas = self.clearance(s.arm_q, w, s.hand_q)
        if ctx.get("commanding"):
            # the 0.25 s look-ahead of the commanded target (the command itself lies between
            # the measured pose and it: two checks, not three -- each costs GIL time)
            q_la = np.clip(q_cmd + v_cmd * LOOKAHEAD_S, ctx["lo"], ctx["hi"])
            c_cmd = self.clearance(q_la, w, s.hand_q)
        else:                                   # shadow: the arm does not follow, judge where it is
            c_cmd = float("inf")
        ctx["clear"] = min(c_meas, c_cmd)
        self.min_clear = min(self.min_clear, ctx["clear"])
        if min(c_meas, c_cmd) < a.clear_stop:
            which = "measured" if c_meas <= c_cmd else "commanded / look-ahead"
            out.append((sm.STOP, f"BODY COLLISION risk: {which} clearance {min(c_meas, c_cmd) * 100:+.1f} cm"))
        elif min(c_meas, c_cmd) < a.clear_warn:
            out.append((sm.WARN, f"close to the body: clearance {min(c_meas, c_cmd) * 100:+.1f} cm"))
        # the palm stays in the region a reach from here to there uses, does not run away
        # from the apple, and does not push through it
        if ctx.get("segment") is not None:
            p0, p1 = ctx["segment"]
            mid, rad = 0.5 * (p0 + p1), 0.5 * float(np.linalg.norm(p1 - p0))
            off = float(np.linalg.norm(fr.palm_ref - mid)) - rad
            ctx["off_path"] = off
            if off > a.fence:
                out.append((sm.STOP, f"palm {off * 100:.0f} cm outside the reach region (fence {a.fence * 100:.0f})"))
            elif off > 0.7 * a.fence:
                out.append((sm.WARN, f"palm {off * 100:.0f} cm outside the reach region"))
            grow = float(np.linalg.norm(fr.palm_ref - p1)) - float(np.linalg.norm(p0 - p1))
            if grow > a.regress:
                out.append((sm.STOP, f"palm {grow * 100:.0f} cm FURTHER from the apple than at the start"))
            elif grow > 0.6 * a.regress:
                out.append((sm.WARN, f"palm {grow * 100:.0f} cm further from the apple than at the start"))
        gap = ctx.get("gap")
        if gap is not None:
            if gap < -0.02:
                out.append((sm.STOP, f"palm {-gap * 100:.1f} cm INSIDE the virtual apple"))
            elif gap < -0.005:
                out.append((sm.WARN, f"pressing into the virtual apple ({-gap * 100:.1f} cm)"))
        # joint limits of the picker (the target was clipped)
        if ctx.get("clipped"):
            out.append((sm.WARN, "target clipped to the picker's joint limits: " + ", ".join(ctx["clipped"])))
        # motors
        if s.arm_temp is not None and np.isfinite(s.arm_temp).any():
            tmax = float(np.nanmax(s.arm_temp))
            if tmax > a.temp_stop:
                out.append((sm.STOP, f"motor temperature {tmax:.0f} C"))
            elif tmax > a.temp_warn:
                out.append((sm.WARN, f"motor temperature {tmax:.0f} C"))
        # envs/safety/monitor.py: command stream, external torque, stuck
        ddq = np.zeros(7) if self.dq_prev is None else (s.arm_dq - self.dq_prev) / dt
        self.dq_prev = s.arm_dq.copy()
        tau_model = self.kin.gravity_torque(s.waist_q, s.arm_q, s.arm_dq, ddq, hand_named(s))
        T = lambda x: self.torch.as_tensor(np.asarray(x, np.float32)).view(1, -1)
        lvl = int(self.mon.update(T(s.arm_q), T(s.arm_dq), T(q_cmd if ctx.get("commanding") else s.arm_q),
                                  T(s.arm_tau), T(tau_model))[0])
        r_f = self.mon.r_f[0].numpy()
        resid = np.abs(r_f) / np.maximum(np.asarray(ARM_RATED), 1.0 / 0.12)    # as the monitor normalizes
        self.peak_resid = np.maximum(self.peak_resid, np.abs(r_f))
        ctx["resid"] = round(float(resid.max()), 3)
        j = int(np.argmax(resid))
        where = f"external torque {r_f[j]:+.1f} N*m at {R.RIGHT_ARM_JOINTS[j][6:-6]} ({resid[j] * 100:.0f}% of rated)"
        if lvl > sm.OK:
            out.append((lvl, f"watchdog {sm.LEVEL_NAMES[lvl]} ({sm.reasons(int(self.mon.reason[0]))}; {where})"))
        if not a.resid_act and resid[j] > self.resid_thr[0]:
            would = "STOP" if resid[j] > self.resid_thr[2] else "SLOW" if resid[j] > self.resid_thr[1] else "WARN"
            out.append((sm.WARN, f"{where} -- would {would} with --resid_act"))
        level = max([l for l, _ in out], default=sm.OK)
        return level, out

    def print_new(self, out, t):
        for lvl, why in out:
            key = why.split(" ")[0] + why.split(" ")[1] if len(why.split(" ")) > 1 else why
            if t - self.last_print.get(key, -9.0) >= 1.0:
                self.last_print[key] = t
                col = C_RED if lvl >= self.sm.STOP else C_YEL
                print(f"{col}  t={t:6.2f}s {self.sm.LEVEL_NAMES[lvl].upper():5s} {why}{C_OFF}", flush=True)


# ─────────────────────────────────────────────
#  Scripted joint moves (home, back)
# ─────────────────────────────────────────────

def joint_path_traj(points, vmax, dt=R.DT, min_s=1.0):
    """Waypoints -> per-tick joint targets along the polyline (arc length in the
    joint max-norm), quintic time scaling with peak joint speed vmax."""
    pts = [np.asarray(p, float) for p in points]
    keep = [pts[0]]
    for q in pts[1:]:
        if np.max(np.abs(q - keep[-1])) > 1e-3:
            keep.append(q)
    if len(keep) == 1:
        return [keep[0]] * int(min_s / dt)
    seg = np.array([np.max(np.abs(b - a)) for a, b in zip(keep[:-1], keep[1:])])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1])
    T = max(min_s, 1.875 * L / max(vmax, 1e-3))
    out = []
    for k in range(int(math.ceil(T / dt)) + 1):
        s = quintic(k * dt / T) * L
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(keep) - 2))
        f = (s - cum[i]) / max(seg[i], 1e-9)
        out.append(keep[i] + np.clip(f, 0.0, 1.0) * (keep[i + 1] - keep[i]))
    return out


def simplify_path(qs, tol=0.05):
    """Keep only the points where the path turns by more than `tol` rad (any joint)
    from the straight line between the kept neighbours: drops the limiter's small
    back-and-forth (a retrace of every wiggle took a minute), keeps the shape."""
    qs = [np.asarray(q, float) for q in qs]
    if len(qs) <= 2:
        return qs

    def rdp(a, b):
        if b <= a + 1:
            return [a]
        u = qs[b] - qs[a]
        f = np.array([np.clip((qs[i] - qs[a]) @ u / max(u @ u, 1e-12), 0, 1) for i in range(a + 1, b)])
        dev = [np.max(np.abs(qs[a + 1 + k] - (qs[a] + f[k] * u))) for k in range(len(f))]
        k = int(np.argmax(dev))
        if dev[k] <= tol:
            return [a]
        m = a + 1 + k
        return rdp(a, m) + rdp(m, b)
    idx = rdp(0, len(qs) - 1) + [len(qs) - 1]
    return [qs[i] for i in idx]


# ─────────────────────────────────────────────
#  Tactile: the real Dex3 pads -> the nine sim taxels [N]
# ─────────────────────────────────────────────

class PadTactile:
    """The real Dex3 pads -> the nine sim taxel forces [N] (thumb x2, middle x2, index x2, palm
    x3). With a calibration file (tactile_calibrate.py: which of the 108 pressure elements
    answer at each named place, and counts per newton) every taxel is the summed rise of its
    place's elements over the baseline taken at the start of the grasp, divided by the gain.
    Without one, a stopgap: the nine press_sensor_state entries' means, 'pool' (every entry
    feeds every finger: the mean of the --tactile_top strongest rises) or an explicit
    'thumb=0,1;index=2,3;middle=4,5;palm=6,7,8' entry grouping, at --tactile_gain counts/N."""

    def __init__(self, gain: float, top: int, mapping: str, calib=None):
        self.gain, self.top = float(gain), int(top)
        self.calib = calib
        if calib is not None:
            self.gain = calib.gain
        self.groups = None
        if calib is None and mapping != "pool":
            self.groups = {}
            for part in mapping.split(";"):
                k, v = part.split("=")
                self.groups[k.strip()] = [int(x) for x in v.split(",")]
        self.base = None
        self.noise = None
        self.samples = []

    def _view(self, flat):
        if flat is None:
            return None
        a = np.asarray(flat, float)
        return a if self.calib is not None else TIO.entry_means(a)

    def add_baseline(self, flat):
        v = self._view(flat)
        if v is not None and len(v) >= 9:
            self.samples.append(v)

    def finish_baseline(self) -> bool:
        if len(self.samples) < 5:
            return False
        self.base, self.noise = TIO.baseline(self.samples)
        return bool(np.any(np.isfinite(self.base)))

    def taxels(self, flat):
        """(9 taxel forces [N] in R.TAXEL_FINGER order, the strongest finger taxel [N], the
        per-element / per-entry rises in N for the log)."""
        v = self._view(flat)
        if self.base is None or v is None or len(v) != len(self.base):
            return np.zeros(9), 0.0, np.zeros(9 if self.calib is None else 1)
        if self.calib is not None:
            # zero = the baseline of this grasp's start + 4 sigma of each element's own noise
            zero = self.base + 4.0 * np.nan_to_num(self.noise)
            r = np.maximum(np.nan_to_num(v - zero, nan=0.0, posinf=0.0, neginf=0.0), 0.0) / self.gain
            t = self.calib.taxels(v, zero)
            return t, float(t[:6].max()), r
        r = np.maximum(np.nan_to_num(v - self.base, nan=0.0, posinf=0.0, neginf=0.0), 0.0) / self.gain
        if self.groups is None:
            f = float(np.sort(r)[::-1][:max(1, self.top)].mean())
            return np.array([f] * 6 + [0.0] * 3), f, r
        out = np.zeros(9)
        place = {"thumb": [0, 1], "middle": [2, 3], "index": [4, 5], "palm": [6, 7, 8]}
        for k, idx in place.items():
            e = [i for i in self.groups.get(k, []) if 0 <= i < 9]
            out[idx] = float(np.mean(r[e])) if e else 0.0
        return out, float(out[:6].max()), r


# ─────────────────────────────────────────────
#  The test
# ─────────────────────────────────────────────

class Toy:
    def __init__(self, args):
        self.args = args
        self.kin = R.G1Kinematics(args.urdf)
        self.pol = R.ReachPolicy(args.policy)
        self.layout = self.pol.layout
        self.gpol = R.GraspPolicy(args.grasp_policy) if args.grasp else None
        self.grasp_rows = []
        self.tcal = None
        if args.grasp:
            path = args.tactile_calib or (TIO.DEFAULT_CALIB if os.path.exists(TIO.DEFAULT_CALIB) else None)
            if path:
                self.tcal = TIO.TactileCalib.load(path)
                print(f"  tactile calibration {path}: " + ", ".join(
                    f"{k} {len(v)}" for k, v in self.tcal.locations.items()) + f" elements, "
                    f"{self.tcal.gain:g} counts/N")
            else:
                print(f"{C_YEL}  no tactile calibration (python3 sim2real/tactile_calibrate.py): the pads are "
                      f"read as nine entry means at {args.tactile_gain:g} counts/N{C_OFF}")
        if self.layout == "tree" and args.waist == "follow":
            sys.exit(f"{args.policy} is an arm-only policy: --waist follow needs a waist policy")
        sys.path.insert(0, os.path.abspath(args.picker_dir))
        import _paths  # noqa: F401
        import grasp_orientation_opt as G
        A = G.ARMS["right"]
        # the picker's joint limits AND the policy's training clip
        self.lo = np.maximum(np.asarray(A.LO, float), R.ARM_TARGET_LO)
        self.hi = np.minimum(np.asarray(A.HI, float), R.ARM_TARGET_HI)
        self.waist_lim = min(math.radians(args.waist_limit_deg), R.WAIST_YAW_LIMIT, float(G.WAIST_LIMIT))
        self.io = MockIO(args, self.kin) if args.mock else PickerIO(args, shadow=args.shadow)
        self.watch = Watch(args, self.kin)
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = args.log_dir or os.path.join(REPO, "logs", "sim2real", "toy_reach", f"{stamp}_{self.io.name.replace(' ', '_')}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.rows = []
        self.events = []
        self.phase, self.phase_t0 = "INIT", 0.0
        self.status_t = -1.0
        self.halted = None
        self.target_torso = None
        self.ctx = {}
        self.t0 = 0.0
        self.apple = self.target = None
        self.q_sent = None
        self.path_cmd = []
        self.reach_info = None

    # ---------- small helpers ----------
    def say(self, msg, col=""):
        print(f"{col}{msg}{C_OFF if col else ''}", flush=True)
        self.events.append({"t": round(self.io.now() - self.t0, 2), "msg": msg.strip()})
        self.events = self.events[-12:]

    def confirm(self, prompt) -> bool:
        if self.args.yes and (not self.args.real or getattr(self.io, "loopback", False)):
            print(f"{C_CYN}{prompt} [--yes]{C_OFF}")
            return True
        if not self.args.real and self.args.mock:
            print(f"{C_CYN}{prompt} [mock: yes]{C_OFF}")
            return True
        try:
            ans = input(f"{C_CYN}{prompt} [ENTER = yes, anything else = abort] {C_OFF}")
        except EOFError:
            return False
        return ans.strip() == ""

    def set_phase(self, ph):
        self.phase, self.phase_t0 = ph, self.io.now()

    def frames(self, s):
        return self.kin.frames(s.waist_q, s.arm_q, hand_named(s))

    def to_torso(self, p_pelvis, s):
        fr = self.frames(s)
        return fr.torso_R.T @ (np.asarray(p_pelvis) - fr.torso_pos)

    def status(self, s, extra=""):
        now = self.io.now()
        if now - self.status_t < 0.2:
            return
        self.status_t = now
        c = self.ctx
        target = None if self.apple is None else self.to_torso(self.apple, s)
        doc = {
            "t": round(time.time(), 3), "state": f"RL_TOY_REACH:{self.phase}",
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
                   "gap_mm": None if c.get("gap") is None else round(1000 * (c["gap"] - R.GRASP_STANDOFF), 1),
                   "touch_N": c.get("touch"), "clearance_cm": None if c.get("clear") is None
                   else round(100 * c["clear"], 1), "palm_speed": round(self.watch.palm_v, 3),
                   "off_path_cm": None if c.get("off_path") is None else round(100 * c["off_path"], 1),
                   "external_torque_frac": c.get("resid"), "safety": c.get("level_name", "ok"),
                   "warnings": c.get("warnings", [])},
        }
        self.io.publish(doc, target)

    # ---------- plan ----------
    def shoulder0(self, s):
        """The right shoulder at waist yaw 0 (the envelopes and maps are relative to it)."""
        return self.kin.frames([0.0, s.waist_q[1], s.waist_q[2]], R.ARM_DEFAULT).shoulder_pos

    def plan_from_detections(self):
        """The camera's fruit (g1_fruits_picker: live_locator -> fruits_to_torso -> the
        picker's DetectionManager, which keeps tracks across waist turns), sampled for
        --detect_s while the arm holds at home; a fruit is offered only if it was seen in
        most samples and its position held still (--detect_spread). Positions are taken
        to the PELVIS frame per sample (fixed to the world while the robot stands)."""
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
            yaw = math.radians(a.approach_deg) if a.approach_deg is not None else R.approach_yaw_default(mean, sh)
            chk = self.check(R.grasp_target_from_apple(mean, r, yaw), sh, quiet=True)
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

    def plan(self, s, apple_pelvis=None):
        a = self.args
        fr = self.frames(s)
        sh = self.shoulder0(s)
        if apple_pelvis is not None:
            p, a.target_is = np.asarray(apple_pelvis, float), "apple"
        elif a.target == ["detection"]:               # shadow: whatever the tracker has right now
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
            yaw = math.radians(a.approach_deg) if a.approach_deg is not None else R.approach_yaw_default(apple, sh)
            target = R.grasp_target_from_apple(apple, r, yaw)
        else:
            target = p
            yaw = math.radians(a.approach_deg) if a.approach_deg is not None else R.approach_yaw_default(target, sh)
            apple = R.apple_from_grasp_target(target, r, yaw)
        self.apple, self.target, self.approach_yaw, self.shoulder = apple, target, yaw, sh
        print(f"  palm grasp point target (pelvis): {np.round(target, 3).tolist()}   "
              f"{'apple centre (detected)' if apple_pelvis is not None else 'virtual apple centre'}: "
              f"{np.round(apple, 3).tolist()} (r {r * 1000:.0f} mm), "
              f"approach {math.degrees(yaw):+.0f} deg")
        chk = self.check(target, sh)
        if not chk["ok"]:
            if a.force:
                self.say(f"target outside the envelope -- running anyway (--force)", C_YEL)
            else:
                sys.exit("target outside this policy's training envelope (--force to run it anyway)")

    def check(self, target, sh, quiet=False):
        env = R.ENVELOPE_WAIST if self.args.waist == "follow" else R.ENVELOPE_ARM
        # --reach_scale F: the real arm reaches further than the sim envelope, so the target is
        # judged as if it were F times closer to the shoulder (distance, height, map cell). The
        # policy still gets the true target.
        F = max(self.args.reach_scale, 1e-3)
        sh_a = np.asarray(sh, float)
        t_eff = sh_a + (np.asarray(target, float) - sh_a) / F
        chk = R.reach_check(t_eff, sh, env)
        if F != 1.0:
            plain = R.reach_check(target, sh, env)          # the scale only ever EXTENDS the envelope
            if plain["ok"] and not chk["ok"]:
                chk, t_eff, F = plain, np.asarray(target, float), 1.0
            else:
                chk["r"] *= F
                chk["dz"] *= F
        if not quiet:
            other = R.reach_check(t_eff, sh, R.ENVELOPE_ARM if env is R.ENVELOPE_WAIST else R.ENVELOPE_WAIST)
            name = "waist" if env is R.ENVELOPE_WAIST else "arm-only"
            print(f"  from the shoulder: {chk['r']:.3f} m, azimuth {chk['az_deg']:+.0f} deg, "
                  f"height {chk['dz']:+.3f} m"
                  + (f"   (envelope judged x{F:g}: as if {chk['r'] / F:.3f} m)" if F != 1.0 else ""))
            print(f"  {name} envelope ({os.path.basename(env['map'])}): "
                  + (f"{C_GRN}inside{C_OFF}" if chk["ok"] else f"{C_RED}{'; '.join(chk['why'])}{C_OFF}"))
            if not chk["ok"] and other["ok"] and self.args.waist == "hold":
                print(f"  {C_YEL}...but inside the WAIST envelope: try --waist follow{C_OFF}")
        return chk

    # ---------- scripted moves ----------
    def stream(self, traj, waist_to=None, label="", check=True, waist_cmd=None):
        """Play per-tick joint targets with the light checks: plumbing, and (check) the
        measured joint SPEED and ACCELERATION against what a scripted move at
        --move_speed can produce -- an arm that speeds up beyond that is falling or
        being pushed: the move stops and the arm is held where it is."""
        io, a = self.io, self.args
        if waist_to is not None:
            io.waist_to(waist_to, len(traj) * R.DT)
        v_stop = 2.0 * a.move_speed + 0.6
        dq_prev, t_prev, acc_hist = None, None, []
        t_next = io.now()
        for q in traj:
            if io.estop_why is not None:
                return f"E-STOP: {io.estop_why}"
            s = io.read()
            if s.age > 0.1 or (s.frozen and io.commands):
                return f"{label}: {'lowstate stale' if s.age > 0.1 else 'daemon froze the arm'}"
            if io.commands and label and s.mode is not None and s.mode != "qref":
                return f"{label}: the daemon is in exec_mode '{s.mode}', not qref -- it is not following"
            if check and label:
                if dq_prev is not None:
                    acc_hist = (acc_hist + [(s.arm_dq - dq_prev) / max(0.005, s.t - t_prev)])[-3:]
                dq_prev, t_prev = s.arm_dq.copy(), s.t
                jv = float(np.max(np.abs(s.arm_dq)))
                ja = float(np.max(np.abs(np.mean(acc_hist, axis=0)))) if len(acc_hist) == 3 else 0.0
                if jv > v_stop or ja > a.move_acc_stop:
                    why = (f"joint speed {jv:.2f} rad/s > {v_stop:.2f}" if jv > v_stop else
                           f"joint ACCELERATION {ja:.1f} rad/s^2 > {a.move_acc_stop:.1f}")
                    print(f"{C_RED}  STOP in '{label}': {why} -- holding the arm where it is{C_OFF}", flush=True)
                    self.halted = f"{label}: {why}"
                    self.stream([self.gravity_hold(s)] * int(0.5 / R.DT), check=False)
                    return f"{label}: SAFETY STOP ({why})"
            io.send_arm(q, waist=waist_cmd, waist_dq=0.0)
            self.q_sent = np.asarray(q, float).copy()
            self.status(s)
            t_next += R.DT
            io.sleep_until(t_next)
        return None

    def hold(self, q, seconds, label=""):
        return self.stream([q] * max(1, int(seconds / R.DT)), label=label)

    #: a path may start inside the conservative body cover where the arm already IS; it
    #: may not go deeper than that start by more than this, and must end clear
    PATH_NO_WORSE_M = 0.003

    def path_check(self, traj, s):
        """(ok, start, worst, end, first bad index, note) for a scripted joint path,
        judged with g1_fruits_picker's body cover (body_collision: torso, legs, hand).

        THE ARM IS WHERE IT IS. The covers are conservative (a sphere reading of -1.1 cm
        is the calibrated contact line, and the leg cover is fitted to a nominal stance):
        measured on the robot 2026-10-12, the arm resting with the elbow at 1.2 rad read
        -2.5 cm against the LEGS while touching nothing. Refusing every path that starts
        there would refuse every start. So: a waypoint passes if it is clear of the
        contact line OR no deeper than where the path starts (+ PATH_NO_WORSE_M), and the
        path must END clear."""
        BC = self.watch.BC
        w = float(s.waist_q[0])
        qf = None if s.hand_q is None else list(s.hand_q)
        pts = list(traj[::5]) + [traj[-1]]
        c = [float(BC.self_clearance("right", np.asarray(q, float), waist=w, q_fingers=qf)) for q in pts]
        line = BC.DEFAULT_MARGIN_M
        floor = min(line, c[0] - self.PATH_NO_WORSE_M)
        bad = next((i for i, v in enumerate(c) if v < floor), None)
        end_ok = c[-1] >= line
        note = ""
        if c[0] < line:
            note = (f"starts {c[0] * 100:+.1f} cm in the body cover where the arm already is "
                    f"(contact line {line * 100:+.1f}: the cover is conservative), and "
                    + ("moves away from it" if bad is None else "goes DEEPER"))
        return bad is None and end_ok, c[0], min(c), c[-1], (bad if bad is not None else
                                                          (None if end_ok else len(c) - 1)), note

    def plan_home(self, q_from, s):
        """Joint-space path from where the arm IS to HOME, from a small library plus a
        seeded random search of 1-2 via points (shoulder abducted away from the legs,
        elbow/wrist on the way), every candidate judged by path_check's rule with the
        picker's body cover. Picks the shortest clear one that spends the least time
        inside the cover. Returns (via tuple or None, text)."""
        a = self.args
        BC = self.watch.BC
        w = float(s.waist_q[0])
        qf = None if s.hand_q is None else list(s.hand_q)
        lo, hi, H = self.lo, self.hi, R.HOME_Q
        q_from = np.asarray(q_from, float)
        line = BC.DEFAULT_MARGIN_M

        def opened(q, roll):
            q = np.array(q, float)
            q[1] = float(np.clip(min(q[1], roll), lo[1], hi[1]))
            return q
        cands = [((), "straight")]
        rolls = [r for r in (a.open_roll, -0.6, -0.8) if r < 0.0] or [-0.4]
        for r in rolls:
            cands.append(((opened(q_from, r),), f"open to roll {r:+.2f}, then home"))
            cands.append(((opened(q_from, r), opened(H, r)), f"open {r:+.2f}, home pose abducted, then home"))
        rng = np.random.default_rng(1)
        for _ in range(60):
            f = rng.uniform(0.2, 0.8)
            qa = q_from + f * (H - q_from) + rng.normal(0.0, 0.12, 7)
            qa = np.clip(opened(qa, rng.uniform(-0.9, -0.3)), lo, hi)
            cands.append(((qa,), "random via"))
            qb = np.clip(opened(q_from, rng.uniform(-0.9, -0.3)) + rng.normal(0.0, 0.05, 7), lo, hi)
            cands.append(((qb, qa), "random via x2"))
        best = None
        for via, name in cands:
            traj = joint_path_traj([q_from, *via, H], a.move_speed)
            pts = list(traj[::5]) + [traj[-1]]
            c = np.array([float(BC.self_clearance("right", np.asarray(q, float), waist=w, q_fingers=qf)) for q in pts])
            floor = min(line, c[0] - self.PATH_NO_WORSE_M)
            if np.any(c < floor) or c[-1] < line:
                continue
            length = sum(float(np.max(np.abs(b - x))) for x, b in zip([q_from, *via], [*via, H]))
            score = length + 1.0 * float(np.mean(c < line)) - 2.0 * min(float(c.min()), 0.05) * (c[0] >= line)
            if best is None or score < best[0]:
                best = (score, via, name, float(c.min()), float(c[0]), float(c[-1]))
        if best is None:
            return None, "every candidate path (%d) goes deeper into the body cover or ends inside it" % len(cands)
        _, via, name, cmin, c0, c1 = best
        return via, f"{name}: clearance start {c0 * 100:+.1f} / worst {cmin * 100:+.1f} / end {c1 * 100:+.1f} cm"

    def move_to(self, q_from, q_goal, label, waist_to=None, via=()):
        s = self.io.read()
        traj = joint_path_traj([q_from, *via, q_goal], self.args.move_speed)
        ok, c0, worst, c_end, bad, note = self.path_check(traj, s)
        print(f"  {label}: {len(traj) * R.DT:.1f} s, max joint travel "
              f"{np.max(np.abs(q_goal - q_from)):.2f} rad, body clearance start {c0 * 100:+.1f} / worst "
              f"{worst * 100:+.1f} / end {c_end * 100:+.1f} cm"
              + ("" if ok else f"  {C_RED}COLLIDES at waypoint {bad}{C_OFF}"))
        if note:
            print(f"  {C_YEL if ok else C_RED}  {note}{C_OFF}")
        if not ok:
            return f"{label}: the path collides with the body"
        if not self.confirm(f"  -> {label} now?"):
            return f"{label}: aborted by the operator"
        self.set_phase(label.upper().replace(" ", "_"))
        return self.stream(traj, waist_to=waist_to, label=label)

    # ---------- the run ----------
    def run(self):
        a, io = self.args, self.io
        self.apple = None
        print(f"{C_CYN}== toy reach [{io.name}]: {os.path.basename(a.policy)} ({self.layout}, "
              f"{self.pol.n_in} in / {self.pol.n_out} out), speed {a.speed}, waist {a.waist}, "
              f"safety {a.safety} =={C_OFF}")
        if a.real:
            r = subprocess.run(["pgrep", "-f", "apple_picker_sm.py"], capture_output=True, text=True)
            if r.stdout.strip():
                sys.exit(f"apple_picker_sm.py is running (pid {r.stdout.split()}): stop it -- one commander only")
        io.connect()
        self.t0 = io.now()
        s = io.read()
        print(f"  arm q {np.round(s.arm_q, 3).tolist()}  waist (yaw, roll, pitch) "
              f"{np.round(np.degrees(s.waist_q), 1).tolist()} deg  daemon mode {s.mode}")
        if a.waist == "follow" and self.layout == "waist" and not a.shadow:
            if getattr(io, "qref_waist", False):
                print("  waist: FOLLOW -- the policy's waist target streamed with the arm (daemon qref waist)")
            else:
                self.say("waist: FOLLOW through set_waist ramps (the daemon has no qref waist: deploy v22) -- "
                         "approximate", C_YEL)
        if abs(s.waist_q[2] - R.WAIST_DEFAULT[2]) > 0.08:
            self.say(f"waist pitch {math.degrees(s.waist_q[2]):+.1f} deg (training: "
                     f"{math.degrees(R.WAIST_DEFAULT[2]):+.1f}) -- the palm ends tilted by the difference", C_YEL)
        fr = self.frames(s)
        print(f"  palm grasp point now (pelvis) {np.round(fr.palm_ref, 3).tolist()}, "
              f"(torso) {np.round(self.to_torso(fr.palm_ref, s), 3).tolist()}")
        detect = a.target == ["detection"]
        if not a.home_only and not (detect and not a.shadow):
            self.plan(s)             # (a detection is planned at home, once the arm is out of view)
        outcome = None
        try:
            if a.shadow:
                return self.shadow_run()
            # 1. the side to qref; arm HOME; waist to 0
            # the daemon's hold is stiffer (kp 400) than the sim's 100 we stream: hold the
            # measured pose PLUS the sag kp 100 needs, so the hand does not drop at the switch
            q0 = self.gravity_hold(s)
            err = self.enter_qref(q0)
            if err:
                return self.finish(err)
            t_mode = io.now()
            # from wherever the arm is (resting on the leg, stopped mid-reach, ...): the best
            # collision-checked joint path home
            via, why = self.plan_home(q0, s)
            if via is None:
                return self.finish(f"no clear path home from here ({why}): move the arm by hand away "
                                   "from the body, or turn it with the picker")
            print(f"  path home: {why}")
            err = self.move_to(q0, R.HOME_Q, "move home" if not via else f"move home (via {len(via)})", via=via)
            if err:
                return self.finish(err)
            # the policy starts at waist ~0 (training: +-0.1 rad); turned with the arm at the
            # home pose, checked hand/arm vs the legs over the whole turn
            s = io.read()
            if abs(s.waist_q[0]) > math.radians(2.0):
                c = self.watch.BC.leg_clearance("right", R.HOME_Q, waist=(float(s.waist_q[0]), 0.0))
                dur = max(1.2, abs(float(s.waist_q[0])) / 0.4)
                print(f"  waist at {math.degrees(s.waist_q[0]):+.1f} deg -> 0 in {dur:.1f} s (arm/hand vs legs "
                      f"over the turn {c * 100:+.1f} cm)")
                if c < a.clear_stop:
                    return self.finish("waist not at 0, and turning it with the arm at home is not clear")
                if not self.confirm("  -> turn the waist to 0 now?"):
                    return self.finish("waist not at 0 (operator)")
                io.waist_to(0.0, dur)
                err = self.hold(R.HOME_Q, dur + 0.3, label="hold home (waist turn)")
                if err:
                    return self.finish(err)
            if a.home_only:
                self.set_phase("HOME")
                self.ctx["doing"] = "holding the arm at home"
                err = self.hold(R.HOME_Q, max(1.0, DAEMON_ENABLE_RAMP_S + 0.3 - (io.now() - t_mode)), label="hold home")
                if a.hand == "open":
                    io.hand_set("open")
                s = io.read()
                return self.finish(err or f"HOME: arm at home (q err {np.round(s.arm_q - R.HOME_Q, 3).tolist()} rad); "
                                          "the daemon keeps holding it")
            # 2. hand cup, settle (and let the daemon's enable blend run out)
            if a.hand != "keep":
                io.hand_set(a.hand)
                if a.hand == "cup":
                    print("  hand: the reach cup (index/middle tips curled 80 deg, thumb in a little); thumb_0 "
                          "stays at the picker's runtime lock (30 deg; the sim's cup has 0)")
            settle = max(1.0, DAEMON_ENABLE_RAMP_S + 0.3 - (io.now() - t_mode))
            err = self.hold(R.HOME_Q, settle, label="hold home")
            if err:
                return self.finish(err)
            s = io.read()
            e_home = s.arm_q - R.HOME_Q
            print(f"  at home: q err {np.round(e_home, 3).tolist()} "
                  f"(gravity sag under kp {a.kp:g} is expected; the sim sags too)")
            if a.grasp and a.hand == "cup" and s.hand_q is not None:
                want = np.array([R.REACH_HAND_PRESHAPE[f"right_hand_{n}_joint"] for n in R.DEX3_ORDER])
                dev = np.asarray(s.hand_q) - want
                dev[0] = 0.0                               # thumb_0 is locked by the picker
                print(f"  hand cup: measured {np.round(s.hand_q, 2).tolist()} vs commanded "
                      f"{np.round(want, 2).tolist()} rad (R.DEX3_ORDER)")
                if io.commands and float(np.max(np.abs(dev))) > 0.45:
                    j = int(np.argmax(np.abs(dev)))
                    return self.finish(f"the hand did not reach the commanded cup: {R.DEX3_ORDER[j]} is "
                                       f"{dev[j]:+.2f} rad off. The grasp policy's joint convention or the hand's "
                                       "output is wrong -- not going on (check with the picker's hand tools)")
            if io.commands and float(np.max(np.abs(e_home))) > 0.25:
                j = int(np.argmax(np.abs(e_home)))
                return self.finish(f"the arm did NOT reach home ({R.RIGHT_ARM_JOINTS[j][6:-6]} "
                                   f"{e_home[j]:+.2f} rad off; daemon mode '{s.mode}'): it is not following "
                                   "the stream -- stopping")
            if detect:
                err = self.plan_from_detections()
                if err:
                    return self.finish(err)
            if not self.confirm(f"  -> RUN THE POLICY (speed {a.speed}, up to {a.time_limit:.0f} s)?"):
                return self.finish("aborted by the operator before the policy")
            # 3. the policy
            outcome, stopped = self.policy_loop()
            if io.estop_why is not None:            # the robot owns the stop: send nothing more
                return self.finish(outcome if outcome.startswith("E-STOP") else f"E-STOP: {io.estop_why}")
            # 4. hold where it reached, then back
            if outcome.startswith("REACHED") and a.after_reach == "freeze":
                self.set_phase("HOLD")
                self.ctx["doing"] = "holding where the palm reached the target"
                self.hold(self.q_sent, a.handoff_s if a.grasp else a.hold_s)
            # 4b. the grasp policy, if asked: the fingers close on the fruit, the arm stays
            if a.grasp and outcome.startswith("REACHED") and not stopped and io.estop_why is None:
                if not self.confirm(f"  -> HAND OFF to the grasp policy (close the fingers, grip command "
                                    f"{a.grip_n:g} N, stop at {a.grip_stop_n:g} N)?"):
                    outcome += "; grasp aborted by the operator"
                else:
                    g_out, g_stopped, held = self.grasp_phase()
                    col = C_GRN if held else C_RED if g_stopped else C_YEL
                    self.say(f"  grasp: {g_out}", col)
                    outcome += "; " + g_out
                    stopped = stopped or g_stopped
                    if io.estop_why is None:
                        # let go (the fruit stays on its tree), arm held where it is
                        self.set_phase("RELEASE")
                        self.ctx["doing"] = "opening the hand"
                        io.hand_set("open")
                        self.hold(self.gravity_hold(io.read()) if g_stopped else self.q_sent, 1.5)
            if io.estop_why is not None:
                return self.finish(outcome if outcome.startswith("E-STOP") else f"E-STOP: {io.estop_why}")
            s = io.read()
            if stopped:
                q_hold = self.gravity_hold(s)
                if a.waist == "follow":
                    io.waist_to(float(s.waist_q[0]), 0.5)
                self.hold(q_hold, a.freeze_s)
                if not self.ask_retreat():
                    self.hold(q_hold, 0.5)
                    return self.finish(outcome, restore_mode=True)
            if a.return_to == "start":
                path = self.return_path(s)
                traj = joint_path_traj(path, a.move_speed * (0.6 if stopped else 1.0))
                self.set_phase("RETURN")
                self.ctx["doing"] = "back home along the way it came"
                print(f"  returning along the path: {len(traj) * R.DT:.1f} s")
                err = self.stream(traj, waist_to=0.0 if a.waist == "follow" else None, label="return")
                if err:
                    return self.finish(outcome + "; " + err)
                self.hold(R.HOME_Q, 0.5)
            if a.hand != "keep":
                io.hand_set("open")
            return self.finish(outcome)
        except KeyboardInterrupt:
            if io.estop_why is None:
                s = io.read()
                self.hold(self.gravity_hold(s), 0.5)
            return self.finish("operator Ctrl-C (arm held where it was)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                s = io.read()
                self.hold(self.gravity_hold(s), 0.5)
            except Exception:
                pass
            return self.finish(f"ERROR {type(e).__name__}: {e} (arm held where it was)")

    def return_path(self, s):
        """Back along the COMMANDED path (the measured one sits a gravity sag lower) for
        --retreat_window_s -- out of whatever it touched, the way it came -- then
        straight HOME if that is clear of the body, else the whole way
        back along the path."""
        n = int(self.args.retreat_window_s / R.DT)
        back = [self.q_sent] + self.path_cmd[::-1][:n]
        direct = joint_path_traj([back[-1], R.HOME_Q], 1.0)
        ok, _, worst, _, _, _ = self.path_check(direct, s)
        if ok:
            return simplify_path(back) + [R.HOME_Q]
        print(f"  {C_YEL}straight back home is not clear ({worst * 100:+.1f} cm): retracing the "
              f"whole path{C_OFF}")
        return simplify_path([self.q_sent] + self.path_cmd[::-1] + [R.HOME_Q], tol=0.08)

    def enter_qref(self, q_hold, timeout_s=45.0):
        """Switch the right arm to the daemon's exec_mode "qref" and WAIT until the daemon
        proves it is receiving THIS process's stream. A command or a stream sent before the
        daemon has discovered this process is lost (a fresh process takes ~20-35 s): the
        arm would not move. 'mode == qref' alone is not proof -- a previous run can leave
        the daemon in qref, and then the stream of this run is lost while the mode already
        reads right. The proof is the daemon's waist source: it reads 'qref' only while
        slot 12 of our stream arrives fresh, so the stream carries the waist's CURRENT
        yaw (no motion; the daemon rate-limits from where the waist stands)."""
        io = self.io
        if not io.commands:
            return None
        s0 = io.read()
        w0 = float(s0.waist_q[0])
        proof = bool(getattr(io, "qref_waist", False))
        if s0.mode == "qref":
            print("  daemon was left in exec_mode qref by an earlier run: waiting for proof that it "
                  "receives this stream" if proof else "  daemon was left in exec_mode qref by an earlier run")
        t0 = io.now()
        t_send = -9.0
        while io.now() - t0 < timeout_s:
            if io.now() - t_send > 0.5:
                io.exec_mode("qref")
                t_send = io.now()
            err = self.stream([q_hold] * 5, check=False, waist_cmd=w0 if proof else None)
            if err:
                return err
            if io.read().mode == "qref" and (not proof or io.waist_src == "qref"):
                print(f"  daemon in exec_mode qref{' and receiving the stream' if proof else ''} "
                      f"after {io.now() - t0:.1f} s")
                self.hold(q_hold, 0.3)
                return None
        return (f"the arm daemon did not confirm exec_mode qref + our stream in {timeout_s:.0f} s (mode "
                f"'{io.read().mode}', waist source '{io.waist_src}'): nothing would move. Is it running / "
                f"discovered? estop_cli.py status")

    def gravity_hold(self, s):
        """Hold where the arm IS: the measured q plus the gravity sag the PD needs
        (so it neither drops nor keeps pushing into whatever stopped it)."""
        g = self.kin.gravity_torque(s.waist_q, s.arm_q, named=hand_named(s))
        return np.clip(s.arm_q + np.clip(g / max(self.args.kp, 1.0), -0.1, 0.1), self.lo, self.hi)

    def ask_retreat(self) -> bool:
        if self.io.estop_why is not None:
            return False
        if self.args.mock or (self.args.yes and getattr(self.io, "loopback", False)):
            print(f"{C_CYN}  retreat along the path? [{'mock' if self.args.mock else '--yes'}: yes]{C_OFF}")
            return True
        print(f"{C_CYN}  The arm is HELD where it stopped. ENTER = retreat slowly along its path to the "
              f"home pose; 'h' + ENTER = keep holding and exit.{C_OFF}")
        s = self.io.read()
        q = self.gravity_hold(s)
        while True:          # keep the hold stream alive while waiting for the answer
            self.stream([q] * int(0.2 / R.DT), check=False)
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                return sys.stdin.readline().strip().lower() != "h"

    def policy_loop(self):
        a, io, W = self.args, self.io, self.watch
        sm = W.sm
        L = R.make_limiters(self.layout)
        prev_raw = np.zeros(self.pol.n_out)
        s = io.read()
        fr = self.frames(s)
        self.ctx.update(segment=(fr.palm_ref.copy(), self.apple.copy()), lo=self.lo, hi=self.hi, commanding=True)
        W.reset()
        self.path_cmd = []
        self.set_phase("POLICY")
        self.ctx["doing"] = "reach policy driving the arm"
        settled_since = reached_t = None
        stopped, outcome = False, None
        waist_sent, waist_sent_t = 0.0, -9.0
        level = sm.OK
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
                touch = R.virtual_touch_force(fr, self.apple, a.apple_radius) if a.touch == "virtual" else 0.0
                st = R.ReachState(arm_q=s.arm_q, arm_dq=s.arm_dq, waist_q=s.waist_q, waist_dq=s.waist_dq,
                                  apple=self.apple, touch_force=touch)
                obs, _ = R.build_obs(self.layout, self.kin, st, prev_raw, fr)
                raw = self.pol(obs)
                slow = a.safety == "on" and level >= sm.SLOW
                speed = a.speed * (a.safety_slow if slow else 1.0)
                tgt = L["arm"].step(raw[:7], s.arm_q, speed)
                q_cmd = np.clip(tgt, self.lo, self.hi)
                hit = np.abs(q_cmd - tgt) > 1e-6
                if hit.any():                             # keep the limiter on what is really sent
                    L["arm"].prev = q_cmd.copy()
                    L["arm"].vel[hit] = 0.0
                clipped = [R.RIGHT_ARM_JOINTS[j][6:-6] for j in np.flatnonzero(hit)]
                w_des = w_cmd = float("nan")
                if self.layout == "waist":
                    w_cmd = float(np.clip(L["waist"].step(raw[7:8], s.waist_q[:1], speed)[0],
                                          -self.waist_lim, self.waist_lim))
                    w_des = float(np.clip(L["waist"].desired[0], -self.waist_lim, self.waist_lim))
                    now = io.now()
                    if (a.waist == "follow" and not io.qref_waist and abs(w_des - waist_sent) > math.radians(2.0)
                            and now - waist_sent_t > 0.3):
                        dur = max(0.5, 1.875 * abs(w_des - float(s.waist_q[0])) / (R.WAIST_MAX_VELOCITY * speed))
                        io.waist_to(w_des, dur)
                        waist_sent, waist_sent_t = w_des, now
                self.ctx.update(gap=gap, touch=round(touch, 2), clipped=clipped, speed_now=speed)
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
                    # plumbing failures stop even with --safety monitor/off
                    hard = [w for l, w in out if l >= sm.STOP and (w.startswith("lowstate") or "daemon" in w)]
                    if hard:
                        stopped, outcome = True, "STOP: " + "; ".join(hard)
                        self.halted = outcome
                        break
                stream_waist = a.waist == "follow" and self.layout == "waist" and io.qref_waist
                p4 = tw()
                io.send_arm(q_cmd, waist=w_cmd if stream_waist else None,
                            waist_dq=float(L["waist"].vel[0]) if stream_waist else 0.0)
                prof["send"] += tw() - p4
                p5 = tw()
                self.q_sent = q_cmd.copy()
                prev_raw = raw
                self.path_cmd.append(q_cmd.copy())
                # reached: the palm's grasp point at the touch distance (freeze), or resting
                # there with the arm still (--after_reach policy, a real apple)
                err = gap - R.GRASP_STANDOFF
                still = float(np.max(np.abs(s.arm_dq))) < 0.08 * max(a.speed, 0.3) + 0.04
                if a.after_reach == "freeze":
                    arrived = err <= a.reach_tol
                else:
                    if abs(err) < a.reach_tol and still:
                        settled_since = settled_since if settled_since is not None else t
                    else:
                        settled_since = None
                    arrived = settled_since is not None and t - settled_since >= a.settle_s
                if arrived and reached_t is None:
                    reached_t = t
                    tilt = math.degrees(math.asin(np.clip(fr.palm_normal[2], -1, 1)))
                    head = math.degrees(math.atan2(fr.palm_normal[1], fr.palm_normal[0]) - self.approach_yaw)
                    self.reach_info = dict(t=t, err_mm=err * 1000, tilt_deg=tilt, heading_deg=head,
                                           palm=fr.palm_ref.tolist(), q=s.arm_q.tolist())
                    self.say(f"  REACHED at t={t:.2f} s: palm gap error {err * 1000:+.1f} mm, palm normal "
                             f"{tilt:+.0f} deg from horizontal (trained: within 20), {head:+.0f} deg from the "
                             f"approach (trained: within 45)", C_GRN)
                    if a.after_reach == "freeze":
                        outcome = f"REACHED in {t:.2f} s (gap error {err * 1000:+.1f} mm)"
                        break
                    self.set_phase("HOLD")
                    self.ctx["doing"] = "policy holding the touch"
                self.rows.append(dict(
                    t=t, obs=obs, raw=raw, q_cmd=q_cmd, q=s.arm_q, dq=s.arm_dq, tau=s.arm_tau,
                    waist=s.waist_q, waist_dq=s.waist_dq, waist_des=w_des, waist_cmd=w_cmd, palm=fr.palm_ref,
                    normal=fr.palm_normal,
                    gap=gap, touch=touch, speed=speed, level=level, clear=self.ctx.get("clear", np.nan),
                    resid=self.ctx.get("resid", np.nan), off_path=self.ctx.get("off_path", np.nan),
                    tactile=np.nan if s.tactile is None else s.tactile, age=s.age,
                    compute=io.now() - t_tick))
                if t - t_print >= 0.5:
                    t_print = t
                    tilt = math.degrees(math.asin(np.clip(fr.palm_normal[2], -1, 1)))
                    head = math.degrees(math.atan2(fr.palm_normal[1], fr.palm_normal[0]) - self.approach_yaw)
                    print(f"  t={t:5.2f}s gap {err * 1000:+6.1f} mm  touch {touch:4.1f} N  palm v "
                          f"{W.palm_v:.2f} m/s  clear {self.ctx.get('clear', np.nan) * 100:+5.1f} cm  "
                          f"tilt {tilt:+4.0f} heading {head:+4.0f} deg  waist "
                          f"{math.degrees(s.waist_q[0]):+5.1f} (cmd {math.degrees(w_cmd):+5.1f})  "
                          f"[{sm.LEVEL_NAMES[level]}{' SLOW' if slow else ''}]", flush=True)
                self.status(s)
                prof["status+log"] += tw() - p5
                if reached_t is not None and t - reached_t >= a.hold_s:
                    outcome = f"REACHED in {reached_t:.2f} s, held by the policy {a.hold_s:.1f} s " \
                              f"(gap error {err * 1000:+.1f} mm)"
                    break
                if t >= a.time_limit:
                    outcome = f"TIME LIMIT {a.time_limit:.0f} s (gap error {err * 1000:+.1f} mm)"
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

    def make_tactile(self):
        a = self.args
        return PadTactile(a.tactile_gain, a.tactile_top, a.tactile_map, self.tcal)

    def grasp_phase(self):
        """The hand-only grasp policy, from where the reach left the palm at the fruit:
        close the fingers on it (arm frozen at the hand-off pose except the wrist, as in
        training before a grip), judge the grip from the pads, keep it --grasp_hold_s,
        return (outcome, stopped, held). Nothing pulls: the arm does not move away."""
        a, io, W = self.args, self.io, self.watch
        sm = W.sm
        spd = a.grasp_speed or a.speed
        L = R.make_grasp_limiters()
        s = io.read()
        fr = self.frames(s)
        # arm: offset = the target the reach was commanding (not the default pose); hand: from where
        # the fingers are (the reach cup)
        L["arm"].offset = self.q_sent.copy()
        L["arm"].prev, L["arm"].vel, L["arm"].k = self.q_sent.copy(), np.zeros(7), L["arm"].ANCHOR_STEPS
        hq_art = R.hand_dex3_to_art(s.hand_q if s.hand_q is not None else
                                    np.array([R.REACH_HAND_PRESHAPE[f"right_hand_{n}_joint"] for n in R.DEX3_ORDER]))
        hq_art[R.HAND_ART_ORDER.index("thumb_0")] = R.REACH_HAND_PRESHAPE["right_hand_thumb_0_joint"]   # locked at 30 deg
        anchor = self.q_sent.copy()
        L["hand"].prev, L["hand"].vel, L["hand"].k = hq_art.copy(), np.zeros(7), L["hand"].ANCHOR_STEPS
        tact = self.make_tactile()
        feats = R.TactileFeatures()
        # tactile baseline: the pads at rest, the arm held
        self.set_phase("GRASP_BASE")
        self.ctx["doing"] = "tactile baseline (hand still)"
        t_b = io.now()
        for _ in range(int(0.8 / R.DT)):
            if io.estop_why is not None:
                return f"E-STOP: {io.estop_why}", True, False
            s = io.read()
            tact.add_baseline(s.pads)
            io.send_arm(self.q_sent)
            t_b += R.DT
            io.sleep_until(t_b)
        if not tact.finish_baseline():
            self.say("no tactile data from the hand (press_sensor_state): the grip is judged from the "
                     "fingers only and the force input is 0", C_YEL)
        else:
            ok_el = np.isfinite(tact.base) & (tact.base > TIO.PRESENT_ABOVE)
            print(f"  tactile baseline: {int(ok_el.sum())} of {len(tact.base)} readings report, "
                  f"{np.nanmedian(tact.base[ok_el]) if ok_el.any() else float('nan'):.0f} counts at rest, noise "
                  f"{np.nanmax(tact.noise[ok_el]) / tact.gain if ok_el.any() else float('nan'):.2f} N max")
        palm0 = fr.palm_ref.copy()
        W.reset()
        self.ctx.update(segment=None, gap=None, commanding=True, lo=self.lo, hi=self.hi)
        self.set_phase("GRASP")
        self.ctx["doing"] = "grasp policy closing the fingers"
        prev_raw = np.zeros(7)
        t_start = io.now()
        t_next = t_start
        t_print = -1.0
        held_since = None
        unlocked = False
        grip_since = None
        outcome, stopped, held = None, False, False
        peak = 0.0
        level = sm.OK
        try:
            while True:
                if io.estop_why is not None:
                    return f"E-STOP: {io.estop_why}", True, False
                s = io.read()
                fr = self.frames(s)
                t = io.now() - t_start
                taxels, pooled, rises = tact.taxels(s.pads)
                slip, bal, lvl = feats.step(taxels)
                hq = np.array(s.hand_q if s.hand_q is not None else R.hand_art_to_dex3(L["hand"].prev), float)
                hdq = np.zeros(7) if s.hand_dq is None else np.asarray(s.hand_dq, float)
                # thumb_0 is runtime-locked at 30 deg by the picker; the policy believes its own command
                hq[0] = float(R.hand_art_to_dex3(L["hand"].prev)[0])
                hdq[0] = 0.0
                obs = R.build_grasp_obs(hq, hdq, slip, bal, lvl, a.grip_n, R.apple_in_palm_frame(fr, self.apple),
                                        prev_raw, R.gravity_in_palm(fr))
                raw = self.gpol(obs)
                arm_raw = raw[:7].copy()
                if not (unlocked and a.grasp_arm_free):
                    arm_raw[:4] = 0.0                       # shoulder + elbow frozen until a grip (training: unlock_with_grip)
                slow = a.safety == "on" and level >= sm.SLOW
                spd_now = spd * (a.safety_slow if slow else 1.0)
                tgt_arm = L["arm"].step(arm_raw, s.arm_q, spd_now)
                lim = np.array([0.05] * 4 + [a.grasp_wrist_max] * 3)
                tgt_arm = np.clip(tgt_arm, np.maximum(anchor - lim, self.lo), np.minimum(anchor + lim, self.hi))
                hit = np.abs(tgt_arm - L["arm"].prev) > 1e-9
                if hit.any():                              # keep the limiter on what is really sent
                    L["arm"].prev = tgt_arm.copy()
                    L["arm"].vel[hit] = 0.0
                tgt_hand_art = L["hand"].step(raw[7:14], None, spd_now)
                lo_h = np.array([R.HAND_LIMITS[n][0] for n in R.HAND_ART_ORDER])
                hi_h = np.array([R.HAND_LIMITS[n][1] for n in R.HAND_ART_ORDER])
                hand_dex = R.hand_art_to_dex3(np.clip(tgt_hand_art, lo_h, hi_h))
                level, out = W.step(s, fr, tgt_arm, L["arm"].vel, a.speed, self.ctx)
                self.ctx["level_name"] = sm.LEVEL_NAMES[level]
                self.ctx["warnings"] = [w for _, w in out][:4]
                W.print_new(out, t)
                moved = float(np.linalg.norm(fr.palm_ref - palm0))
                peak = max(peak, pooled)
                why = None
                if level >= sm.STOP and a.safety == "on":
                    why = "SAFETY STOP: " + "; ".join(w for l, w in out if l >= sm.STOP)
                elif moved > a.grasp_move_stop:
                    why = f"the palm moved {moved * 100:.1f} cm from the hand-off point (> {a.grasp_move_stop * 100:.0f})"
                elif pooled > a.grip_stop_n:
                    why = f"estimated pad force {pooled:.1f} N > {a.grip_stop_n:.0f} N"
                if why:
                    self.halted = why
                    return why, True, False
                io.send_arm(tgt_arm)
                io.hand_cmd(hand_dex)
                self.q_sent = tgt_arm.copy()
                prev_raw = raw[7:14].copy()
                # grip: two finger groups loaded (> 0.5 N) for 0.3 s (the training's latch is tactile too)
                n_on = int(np.sum(lvl > 0.5))
                grip_now = n_on >= 2
                grip_since = (grip_since if grip_since is not None else t) if grip_now else None
                gripped = grip_since is not None and t - grip_since >= 0.3
                if gripped and not unlocked:
                    unlocked = True
                    self.say(f"  grip formed at t={t:.2f} s: finger forces {np.round(lvl, 1).tolist()} N "
                             f"(thumb, index, middle; command {a.grip_n:g} N)", C_GRN)
                held_since = (held_since if held_since is not None else t) if gripped else None
                self.grasp_rows.append(dict(t=t, obs=obs, raw=raw, hand=hq, tgt_hand=hand_dex, tgt_arm=tgt_arm,
                                            taxels=taxels, rises=rises, pads=np.nan_to_num(
                                                np.full(1, np.nan) if s.pads is None else s.pads, nan=-1.0).astype(np.float32),
                                            moved=moved, level=level))
                if t - t_print >= 0.5:
                    t_print = t
                    print(f"  t={t:4.1f}s  finger N (thumb, index, middle) {np.round(lvl, 1).tolist()}  index_1 "
                          f"{hq[4]:+.2f} middle_1 {hq[6]:+.2f} thumb_2 {hq[2]:+.2f} rad  palm moved "
                          f"{moved * 1000:4.1f} mm  [{sm.LEVEL_NAMES[level]}{' SLOW' if slow else ''}]", flush=True)
                self.status(s, extra="GRASP")
                if held_since is not None and t - held_since >= a.grasp_hold_s:
                    outcome, held = (f"GRASPED: held {a.grasp_hold_s:.1f} s, pad force peak {peak:.1f} N, finger "
                                     f"forces {np.round(lvl, 1).tolist()} N"), True
                    break
                if t >= a.grasp_s + (a.grasp_hold_s if held_since is not None else 0.0) and held_since is None:
                    outcome = (f"NO GRIP in {a.grasp_s:.0f} s: finger forces {np.round(lvl, 1).tolist()} N, "
                               f"index_1 {hq[4]:+.2f} middle_1 {hq[6]:+.2f} rad (peak pad force {peak:.1f} N)")
                    break
                t_next += R.DT
                if io.now() - t_next > R.DT:
                    t_next = io.now()
                io.sleep_until(t_next)
        except KeyboardInterrupt:
            return "OPERATOR STOP (Ctrl-C)", True, False
        return outcome, False, held

    def shadow_run(self):
        """Read-only: the policy on the real robot's state, nothing sent. The arm
        does not move, so this checks the observation, FK, target, collision and
        external-torque model against the robot at its current pose."""
        a, io, W = self.args, self.io, self.watch
        L = R.make_limiters(self.layout)
        prev_raw = np.zeros(self.pol.n_out)
        s = io.read()
        fr = self.frames(s)
        self.ctx.update(segment=(fr.palm_ref.copy(), self.apple.copy()), lo=self.lo, hi=self.hi, commanding=False)
        W.reset()
        self.set_phase("SHADOW")
        n = int(a.shadow_s / R.DT)
        t_next = io.now()
        first = None
        res = []
        for k in range(n):
            s = io.read()
            fr = self.frames(s)
            touch = R.virtual_touch_force(fr, self.apple, a.apple_radius) if a.touch == "virtual" else 0.0
            obs, _ = R.build_obs(self.layout, self.kin, R.ReachState(s.arm_q, s.arm_dq, s.waist_q, s.waist_dq,
                                                                     self.apple, touch), prev_raw, fr)
            raw = self.pol(obs)
            q_cmd = np.clip(L["arm"].step(raw[:7], s.arm_q, a.speed), self.lo, self.hi)
            self.ctx["gap"] = float(np.linalg.norm(self.apple - fr.palm_ref)) - a.apple_radius
            level, out = W.step(s, fr, q_cmd, L["arm"].vel, a.speed, self.ctx)
            W.print_new(out, k * R.DT)
            prev_raw = raw
            first = first if first is not None else (obs.copy(), raw.copy())
            g = self.kin.gravity_torque(s.waist_q, s.arm_q, named=hand_named(s))
            res.append((s.arm_tau, g))
            self.status(s, extra="SHADOW: nothing is commanded")
            t_next += R.DT
            io.sleep_until(t_next)
        obs0, raw0 = first
        np.set_printoptions(precision=3, suppress=True, linewidth=140)
        print("\n  first observation:")
        names = [("arm q - default", 0, 7), ("arm dq", 7, 14), ("apple - palm", 14, 17), ("palm quat", 17, 21),
                 ("prev raw", 21, 28), ("touch", 28, 29), ("ring", 29, 37)] + \
                ([("waist q, dq, prev raw", 37, 40)] if self.layout == "waist" else [])
        for nm, i, j in names:
            print(f"    {nm:22s} {obs0[i:j]}")
        print(f"  first raw action {raw0}\n  -> first joint targets would move the arm toward "
              f"{np.round(np.clip(R.ARM_DEFAULT + 0.5 * raw0[:7], self.lo, self.hi), 3).tolist()} (before the limiter)")
        tau = np.array([r[0] for r in res])
        g = np.array([r[1] for r in res])
        print("  external-torque model at rest (tau_est - RNEA gravity), per joint [N*m]:")
        for j, n in enumerate(R.RIGHT_ARM_JOINTS):
            d = tau[:, j] - g[:, j]
            print(f"    {n[6:-6]:16s} tau_est {tau[:, j].mean():+6.2f}  model {g[:, j].mean():+6.2f}  "
                  f"residual {d.mean():+6.2f} +- {d.std():.2f}  ({abs(d.mean()) / ARM_RATED[j] * 100:4.1f}% of rated; "
                  f"WARN at {12 * a.resid_scale:.0f}%)")
        print("  (the daemon holds the arm with ITS gains here: the residual is the model error / friction "
              "the watchdog will see at rest)")
        if a.grasp and self.gpol is not None:
            self.shadow_grasp()
        return self.finish("shadow run done (nothing was commanded)")

    def shadow_grasp(self):
        """Read-only look at the grasp inputs: the hand, the pads (baseline + noise), and what the
        grasp policy would do if the palm were at the fruit (pocket 0, nothing touching)."""
        a, io = self.args, self.io
        tact = self.make_tactile()
        t_b = io.now()
        s = io.read()
        for _ in range(int(1.0 / R.DT)):
            s = io.read()
            tact.add_baseline(s.pads)
            t_b += R.DT
            io.sleep_until(t_b)
        print("\n  grasp inputs (shadow):")
        if s.hand_q is None:
            print("    no hand state (rt/dex3/right/state)")
            return
        np.set_printoptions(precision=3, suppress=True, linewidth=150)
        print(f"    hand q  (thumb_0,1,2 index_0,1 middle_0,1) {np.round(s.hand_q, 2).tolist()} rad")
        want = np.array([R.REACH_HAND_PRESHAPE[f"right_hand_{n}_joint"] for n in R.DEX3_ORDER])
        print(f"    reach cup (commanded in a run)             {np.round(want, 2).tolist()} rad  "
              f"(thumb_0 is locked by the picker at 30 deg)")
        if tact.finish_baseline():
            ok_el = np.isfinite(tact.base) & (tact.base > TIO.PRESENT_ABOVE)
            print(f"    pads: {int(ok_el.sum())} of {len(tact.base)} readings report, "
                  f"{np.nanmedian(tact.base[ok_el]) if ok_el.any() else float('nan'):.0f} counts at rest; noise "
                  f"{float(np.nanmax(tact.noise[ok_el])) / tact.gain if ok_el.any() else float('nan'):.2f} N max "
                  f"({tact.gain:g} counts/N, {'calibrated places' if tact.calib is not None else 'entry means'})")
        else:
            print("    no pad data (press_sensor_state): the force input would be 0")
        fr = self.frames(s)
        taxels, pooled, _ = tact.taxels(s.pads)
        feats = R.TactileFeatures()
        slip, bal, lvl = feats.step(taxels)
        hq = np.array(s.hand_q, float)
        hq[0] = R.REACH_HAND_PRESHAPE["right_hand_thumb_0_joint"]
        obs = R.build_grasp_obs(hq, np.zeros(7), slip, bal, lvl, a.grip_n, np.zeros(3) + R.GRASP_CENTER_LOCAL,
                                np.zeros(7), R.gravity_in_palm(fr))
        raw = self.gpol(obs)
        L = R.make_grasp_limiters()
        art = R.hand_dex3_to_art(hq)
        L["hand"].prev, L["hand"].vel, L["hand"].k = art.copy(), np.zeros(7), L["hand"].ANCHOR_STEPS
        tgt = R.hand_art_to_dex3(L["hand"].step(raw[7:14], None, a.grasp_speed or a.speed))
        print(f"    grasp policy, fruit in the pocket, grip command {a.grip_n:g} N:")
        print(f"      raw hand action (articulation order {R.HAND_ART_ORDER}): {np.round(raw[7:14], 2).tolist()}")
        print(f"      first hand step toward (R.DEX3_ORDER): {np.round(tgt, 3).tolist()} rad; arm raw "
              f"{np.round(raw[:7], 2).tolist()} (wrist x0.05, clamped {a.grasp_wrist_max} rad, shoulder/elbow frozen)")

    def finish(self, outcome, restore_mode=True):
        a = self.args
        try:
            self.io.close(restore_mode=restore_mode)
        except Exception as e:
            print(f"  close: {e}")
        meta = {"outcome": outcome, "mode": self.io.name, "args": {k: v for k, v in vars(a).items()},
                "policy": a.policy, "layout": self.layout,
                "target_pelvis": None if getattr(self, "target", None) is None else self.target.tolist(),
                "apple_pelvis": None if getattr(self, "apple", None) is None else self.apple.tolist(),
                "approach_yaw_deg": math.degrees(getattr(self, "approach_yaw", 0.0)),
                "events": self.events, "reach": self.reach_info, "min_clearance_m": self.watch.min_clear,
                "peak_joint_acc": self.watch.peak_acc, "peak_palm_acc": self.watch.peak_palm_acc,
                "peak_external_torque_Nm": self.watch.peak_resid.tolist(),
                "joint_limits": [self.lo.tolist(), self.hi.tolist()],
                "git": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()}
        with open(os.path.join(self.log_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        if self.grasp_rows:
            keys = self.grasp_rows[0].keys()
            np.savez_compressed(os.path.join(self.log_dir, "grasp.npz"),
                                **{k: np.array([r[k] for r in self.grasp_rows]) for k in keys})
        if self.rows:
            keys = self.rows[0].keys()
            np.savez_compressed(os.path.join(self.log_dir, "run.npz"),
                                **{k: np.array([r[k] for r in self.rows]) for k in keys})
        print(f"\n{C_CYN}== {outcome}{C_OFF}")
        if self.rows:
            per = np.diff([r["t"] for r in self.rows]) if len(self.rows) > 1 else np.array([R.DT])
            comp = np.array([r["compute"] for r in self.rows])
            late = int(np.sum(per > 1.5 * R.DT))
            print(f"   {len(self.rows)} policy steps ({late} late > 30 ms); period {np.median(per) * 1000:.1f} ms median / "
                  f"{per.max() * 1000:.1f} ms max; compute {np.median(comp) * 1000:.1f} ms median / "
                  f"{comp.max() * 1000:.1f} ms max; min body clearance {self.watch.min_clear * 100:+.1f} cm; "
                  f"peak external torque {np.round(self.watch.peak_resid, 1).tolist()} N*m; peak joint acc "
                  f"{self.watch.peak_acc:.1f} rad/s^2 (stop {self.args.joint_acc_stop:.1f}), palm acc "
                  f"{self.watch.peak_palm_acc:.1f} m/s^2 (stop {self.args.palm_acc_stop:.1f})")
        if getattr(self, "prof", None) and self.rows:
            n = len(self.rows)
            print("   step time per part (ms, mean): " + ", ".join(f"{k} {v / n * 1000:.1f}" for k, v in self.prof.items()))
        print(f"   log: {self.log_dir}")
        return outcome


def main():
    args = parse_args()
    toy = Toy(args)
    if args.real and not getattr(toy.io, "loopback", False):
        print(f"{C_RED}REAL ROBOT. Someone at the robot with the remote / emergency stop, the area "
              f"in front of the right arm clear.{C_OFF}")
    out = toy.run()
    sys.exit(0 if out and (out.startswith("REACHED") or out.startswith("shadow")) else 1)


if __name__ == "__main__":
    main()
