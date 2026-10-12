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
import reach_io as R  # noqa: E402

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
    p.add_argument("--waist", choices=["hold", "follow"], default="hold",
                   help="hold: waist stays at 0, the policy's waist output is ignored (default); "
                        "follow: the policy's waist target is sent as daemon waist ramps")
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
    p.add_argument("--move_speed", type=float, default=0.35,
                   help="rad/s peak joint speed of the scripted moves (home, back)")
    p.add_argument("--home_only", action="store_true",
                   help="only move the arm HOME (waist to 0), hold it there and exit -- e.g. at the start of a "
                        "session or after an E-STOP reset")
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
    s.add_argument("--temp_warn", type=float, default=80.0)
    s.add_argument("--temp_stop", type=float, default=95.0)
    s.add_argument("--freeze_s", type=float, default=0.5)
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
        if a.mock:
            p.error("--target detection needs the robot (perception)")
        a.target_is = "apple"
    a.real = not (a.mock or a.shadow)
    a.time_limit = a.time_limit or 8.0 / a.speed
    a.palm_speed_stop = a.palm_speed_stop or (2.0 * a.speed + 0.35)
    a.joint_speed_stop = a.joint_speed_stop or (1.8 * R.ARM_MAX_VELOCITY * a.speed + 0.5)
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
    tactile: float | None = None             # raw Dex3 pressure mean (not a force)
    age: float = 0.0                         # s since the lowstate sample
    mode: str | None = None                  # daemon exec_mode, right arm
    frozen: bool | None = None
    daemon_age: float | None = None


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
                    tactile=0.0, age=0.002, mode=self.mode, frozen=self.frozen, daemon_age=0.05)

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

    def send_arm(self, q, dq=None):
        self.q_ref = np.asarray(q, float).copy()

    def waist_to(self, yaw, duration):
        self.ramp = (float(self.w[0]), float(yaw), self.t, max(0.5, float(duration)))

    def hand_set(self, kind):
        if kind == "cup":
            for i, n in enumerate(R.DEX3_ORDER):
                if i:
                    self.hand[i] = R.REACH_HAND_PRESHAPE[f"right_hand_{n}_joint"]
        elif kind == "open":
            self.hand[1:] = 0.0

    def detection_targets(self):
        return []

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
        return Snap(arm_q=g(arm_i, "q"), arm_dq=g(arm_i, "dq"), arm_tau=g(arm_i, "tau_est"),
                    waist_q=g(w_i, "q"), waist_dq=g(w_i, "dq"), arm_temp=temp,
                    hand_q=None if hq is None else np.asarray(hq, float), tactile=tac,
                    age=(time.monotonic() - rx) if rx else 0.0,
                    mode=d.get("exec_mode"), frozen=d.get("frozen"),
                    daemon_age=None if self._dstate_t is None else time.monotonic() - self._dstate_t)

    def exec_mode(self, mode):
        if self.commands and self.estop_why is None:
            self.arm.set_exec_mode("right", mode)

    def send_arm(self, q, dq=None):
        if not self.commands or self.estop_why is not None:
            return                                  # never a command after an E-STOP
        dq = np.zeros(7) if dq is None else dq
        for j, n in enumerate(R.RIGHT_ARM_JOINTS):
            m = self._cmd.motor_cmd[R.MOTOR_IDX[n]]
            m.mode = 1
            m.q, m.dq, m.tau = float(q[j]), float(dq[j]), 0.0
            m.kp, m.kd = float(self.args.kp), float(self.args.kd)
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
                self.arm.set_exec_mode("right", "mpc")
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
        self.palm_v = 0.0
        self.palm_prev = None
        self.last_print = {}
        self.peak_resid = np.zeros(7)
        self.min_clear = float("inf")

    def reset(self):
        self.mon.reset()
        self.dq_prev = None
        self.palm_prev = None
        self.palm_v = 0.0

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
        if self.palm_prev is not None:
            v = float(np.linalg.norm(fr.palm_ref - self.palm_prev)) / R.DT
            self.palm_v = 0.5 * self.palm_v + 0.5 * v
        self.palm_prev = fr.palm_ref.copy()
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
            q_la = np.clip(q_cmd + v_cmd * LOOKAHEAD_S, ctx["lo"], ctx["hi"])
            c_cmd = min(self.clearance(q_cmd, w, s.hand_q), self.clearance(q_la, w, s.hand_q))
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
        ddq = np.zeros(7) if self.dq_prev is None else (s.arm_dq - self.dq_prev) / R.DT
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
#  The test
# ─────────────────────────────────────────────

class Toy:
    def __init__(self, args):
        self.args = args
        self.kin = R.G1Kinematics(args.urdf)
        self.pol = R.ReachPolicy(args.policy)
        self.layout = self.pol.layout
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
    def plan(self, s):
        a = self.args
        fr = self.frames(s)
        # the shoulder at waist yaw 0 (the envelopes and maps are relative to it)
        sh = self.kin.frames([0.0, s.waist_q[1], s.waist_q[2]], R.ARM_DEFAULT).shoulder_pos
        if a.target == ["detection"]:
            cands = self.io.detection_targets()
            if not cands:
                sys.exit("no fruit detected (/fruits_detected_torso: are live_locator + fruits_to_torso running?)")
            best = None
            print("  detected fruit (torso frame):")
            for i, (p, rad) in enumerate(cands):
                apple = fr.torso_R @ p + fr.torso_pos
                r = rad or a.apple_radius
                yaw = math.radians(a.approach_deg) if a.approach_deg is not None else R.approach_yaw_default(apple, sh)
                chk = self.check(R.grasp_target_from_apple(apple, r, yaw), sh, quiet=True)
                print(f"    #{i}: {np.round(p, 3).tolist()}  r {r * 1000:.0f} mm  -> "
                      f"{'REACHABLE' if chk['ok'] else 'no: ' + '; '.join(chk['why'])}")
                if chk["ok"] and (best is None or chk["r"] < best[2]):
                    best = (p, r, chk["r"])
            if best is None:
                sys.exit("no detected fruit inside this policy's envelope")
            p_in, a.apple_radius = best[0], best[1]
            print(f"  -> chosen {np.round(p_in, 3).tolist()}")
        else:
            p_in = np.array([float(v) for v in a.target])
        p = fr.torso_R @ p_in + fr.torso_pos if (a.frame == "torso" or a.target == ["detection"]) else p_in
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
              f"virtual apple centre: {np.round(apple, 3).tolist()} (r {r * 1000:.0f} mm), "
              f"approach {math.degrees(yaw):+.0f} deg")
        chk = self.check(target, sh)
        if not chk["ok"]:
            if a.force:
                self.say(f"target outside the envelope -- running anyway (--force)", C_YEL)
            else:
                sys.exit("target outside this policy's training envelope (--force to run it anyway)")

    def check(self, target, sh, quiet=False):
        env = R.ENVELOPE_WAIST if self.args.waist == "follow" else R.ENVELOPE_ARM
        chk = R.reach_check(target, sh, env)
        if not quiet:
            other = R.reach_check(target, sh, R.ENVELOPE_ARM if env is R.ENVELOPE_WAIST else R.ENVELOPE_WAIST)
            name = "waist" if env is R.ENVELOPE_WAIST else "arm-only"
            print(f"  from the shoulder: {chk['r']:.3f} m, azimuth {chk['az_deg']:+.0f} deg, "
                  f"height {chk['dz']:+.3f} m")
            print(f"  {name} envelope ({os.path.basename(env['map'])}): "
                  + (f"{C_GRN}inside{C_OFF}" if chk["ok"] else f"{C_RED}{'; '.join(chk['why'])}{C_OFF}"))
            if not chk["ok"] and other["ok"] and self.args.waist == "hold":
                print(f"  {C_YEL}...but inside the WAIST envelope: try --waist follow{C_OFF}")
        return chk

    # ---------- scripted moves ----------
    def stream(self, traj, waist_to=None, label=""):
        """Play per-tick joint targets with the light checks (collision, plumbing)."""
        io = self.io
        if waist_to is not None:
            io.waist_to(waist_to, len(traj) * R.DT)
        t_next = io.now()
        for q in traj:
            if io.estop_why is not None:
                return f"E-STOP: {io.estop_why}"
            s = io.read()
            if s.age > 0.1 or (s.frozen and io.commands):
                return f"{label}: {'lowstate stale' if s.age > 0.1 else 'daemon froze the arm'}"
            io.send_arm(q)
            self.q_sent = np.asarray(q, float).copy()
            self.status(s)
            t_next += R.DT
            io.sleep_until(t_next)
        return None

    def hold(self, q, seconds):
        return self.stream([q] * max(1, int(seconds / R.DT)))

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

    def move_to(self, q_from, q_goal, label, waist_to=None):
        s = self.io.read()
        traj = joint_path_traj([q_from, q_goal], self.args.move_speed)
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
        if abs(s.waist_q[2] - R.WAIST_DEFAULT[2]) > 0.08:
            self.say(f"waist pitch {math.degrees(s.waist_q[2]):+.1f} deg (training: "
                     f"{math.degrees(R.WAIST_DEFAULT[2]):+.1f}) -- the palm ends tilted by the difference", C_YEL)
        fr = self.frames(s)
        print(f"  palm grasp point now (pelvis) {np.round(fr.palm_ref, 3).tolist()}, "
              f"(torso) {np.round(self.to_torso(fr.palm_ref, s), 3).tolist()}")
        if not a.home_only:
            self.plan(s)
        outcome = None
        try:
            if a.shadow:
                return self.shadow_run()
            # 1. the side to qref; arm HOME; waist to 0
            # the daemon's hold is stiffer (kp 400) than the sim's 100 we stream: hold the
            # measured pose PLUS the sag kp 100 needs, so the hand does not drop at the switch
            q0 = self.gravity_hold(s)
            io.exec_mode("qref")
            t_mode = io.now()
            err = self.hold(q0, 0.3)
            if err:
                return self.finish(err)
            err = self.move_to(q0, R.HOME_Q, "move home")
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
                err = self.hold(R.HOME_Q, dur + 0.3)
                if err:
                    return self.finish(err)
            if a.home_only:
                self.set_phase("HOME")
                self.ctx["doing"] = "holding the arm at home"
                err = self.hold(R.HOME_Q, max(1.0, DAEMON_ENABLE_RAMP_S + 0.3 - (io.now() - t_mode)))
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
            err = self.hold(R.HOME_Q, settle)
            if err:
                return self.finish(err)
            s = io.read()
            print(f"  at home: q err {np.round(s.arm_q - R.HOME_Q, 3).tolist()} "
                  f"(gravity sag under kp {a.kp:g} is expected; the sim sags too)")
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
                self.hold(self.q_sent, a.hold_s)
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
            self.hold(q, 0.2)
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
        try:
            while True:
                t_tick = io.now()
                if io.estop_why is not None:
                    stopped, outcome = True, f"E-STOP: {io.estop_why}"
                    self.halted = outcome
                    break
                s = io.read()
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
                w_des = float("nan")
                if self.layout == "waist":
                    L["waist"].step(raw[7:8], s.waist_q[:1], speed)
                    w_des = float(np.clip(L["waist"].desired[0], -self.waist_lim, self.waist_lim))
                    now = io.now()
                    if (a.waist == "follow" and abs(w_des - waist_sent) > math.radians(2.0)
                            and now - waist_sent_t > 0.3):
                        dur = max(0.5, 1.875 * abs(w_des - float(s.waist_q[0])) / (R.WAIST_MAX_VELOCITY * speed))
                        io.waist_to(w_des, dur)
                        waist_sent, waist_sent_t = w_des, now
                self.ctx.update(gap=gap, touch=round(touch, 2), clipped=clipped, speed_now=speed)
                level, out = W.step(s, fr, q_cmd, L["arm"].vel, a.speed, self.ctx)
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
                io.send_arm(q_cmd)
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
                    waist=s.waist_q, waist_dq=s.waist_dq, waist_des=w_des, palm=fr.palm_ref, normal=fr.palm_normal,
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
                          f"{math.degrees(s.waist_q[0]):+5.1f} (policy {math.degrees(w_des):+5.1f})  "
                          f"[{sm.LEVEL_NAMES[level]}{' SLOW' if slow else ''}]", flush=True)
                self.status(s)
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
        return self.finish("shadow run done (nothing was commanded)")

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
                "peak_external_torque_Nm": self.watch.peak_resid.tolist(),
                "joint_limits": [self.lo.tolist(), self.hi.tolist()],
                "git": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()}
        with open(os.path.join(self.log_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        if self.rows:
            keys = self.rows[0].keys()
            np.savez_compressed(os.path.join(self.log_dir, "run.npz"),
                                **{k: np.array([r[k] for r in self.rows]) for k in keys})
        print(f"\n{C_CYN}== {outcome}{C_OFF}")
        if self.rows:
            per = np.diff([r["t"] for r in self.rows]) if len(self.rows) > 1 else np.array([R.DT])
            comp = np.array([r["compute"] for r in self.rows])
            print(f"   {len(self.rows)} policy steps; period {np.median(per) * 1000:.1f} ms median / "
                  f"{per.max() * 1000:.1f} ms max; compute {np.median(comp) * 1000:.1f} ms median / "
                  f"{comp.max() * 1000:.1f} ms max; min body clearance {self.watch.min_clear * 100:+.1f} cm; "
                  f"peak external torque {np.round(self.watch.peak_resid, 1).tolist()} N*m")
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
