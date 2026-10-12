#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
TEST ONLY: a stand-in for the G1 + arm_mpc_server on loopback DDS, so
toy_reach.py's robot path (g1_fruits_picker's node, full_move2, the daemon
protocol, the qref message layout) can be exercised without the robot.

Publishes rt/lowstate (500 Hz), rt/dex3/right/state (100 Hz) and
rt/arm_ctrl/state (10 Hz, the daemon's JSON); plays rt/arm_ctrl/qref and
rt/arm_ctrl/cmd (exec_mode, set_waist, stop, estop, estop_reset) with the daemon's
rules -- including its E-STOP (onboard/estop.py): heartbeat watch on
rt/arm_ctrl/heartbeat (--hb_timeout), freeze then damped, commands ignored while
latched, an "estop" block in the state -- and: 2 s
enable blend after a mode switch, stream older than 0.25 s -> hold the last
command, |q_ref - q| > 1.2 rad -> freeze. The arm is a unit-inertia PD with the
streamed gains and tau_est = pinocchio gravity (no load); hand commands are not
simulated (the hand reads as its last pose).

Run both with the same loopback DDS (inside humble-box, conda g1_dex3_sim2real):
  python3 sim2real/fake_robot.py --net-if lo &
  python3 sim2real/toy_reach.py --speed 0.5 --target 0.25 -0.25 0.31 --config <copy with "net_if": "lo">
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_io as R  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--net-if", default="lo")
ap.add_argument("--duration", type=float, default=600.0)
ap.add_argument("--hb_timeout", type=float, default=1.0)
ap.add_argument("--freeze_s", type=float, default=1.5)
args = ap.parse_args()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_, unitree_hg_msg_dds__LowState_  # noqa: E402
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowCmd_, LowState_  # noqa: E402

ChannelFactoryInitialize(0, args.net_if)
kin = R.G1Kinematics()
ARM = [R.MOTOR_IDX[n] for n in R.RIGHT_ARM_JOINTS]
lock = threading.Lock()
S = dict(q=np.array([-0.5, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0]), dq=np.zeros(7), waist=np.array([0.03, 0.0, 0.1]),
         wdq=0.0, mode="mpc", frozen=False, qref=None, kp=np.full(7, 400.0), kd=np.full(7, 12.0), qref_t=0.0,
         t_en=0.0, q_en=None, hold=None, ramp=None, last_cmd=None)
S["hold"] = S["q"].copy()
E = dict(active=False, why="", src="", t0=0.0, freeze_q=None, hb={})


def estop(why, src):
    if E["active"]:
        return
    E.update(active=True, why=why, src=src, t0=time.monotonic(), freeze_q=S["q"].copy())
    S["qref"], S["ramp"] = None, None
    print(f"[fake] E-STOP ({src}): {why}", flush=True)


def on_hb(msg):
    d = json.loads(msg.data)
    with lock:
        if d.get("bye"):
            if E["hb"].pop(d.get("src"), None) is not None:
                print(f"[fake] heartbeat '{d.get('src')}' signed off", flush=True)
        else:
            if d.get("src") not in E["hb"]:
                print(f"[fake] heartbeat '{d.get('src')}' watched", flush=True)
            E["hb"][d.get("src")] = time.monotonic()


def on_qref(msg):
    with lock:
        if S["mode"] != "qref" or E["active"]:
            return
        mc = msg.motor_cmd
        S["qref"] = np.array([mc[i].q for i in ARM])
        S["kp"] = np.clip([mc[i].kp for i in ARM], 0, 450.0)
        S["kd"] = np.clip([mc[i].kd for i in ARM], 0, 12.0)
        S["qref_t"] = time.monotonic()


def on_cmd(msg):
    d = json.loads(msg.data)
    c = d.get("cmd")
    with lock:
        if c == "estop":
            estop(d.get("why", ""), d.get("src", ""))
            return
        if c == "estop_reset":
            if E["active"]:
                E["active"], E["hb"] = False, {}
                S["mode"], S["qref"], S["hold"], S["last_cmd"] = "mpc", None, S["q"].copy(), S["q"].copy()
                print("[fake] E-STOP reset", flush=True)
            return
        if E["active"]:
            print(f"[fake] '{c}' ignored: E-STOP latched", flush=True)
            return
        if c == "exec_mode" and d.get("side") == "right":
            S["mode"] = d.get("mode")
            S["qref"], S["t_en"], S["q_en"], S["hold"] = None, time.monotonic(), S["q"].copy(), None
            print(f"[fake] exec_mode {S['mode']}", flush=True)
        elif c == "set_waist":
            yaw = max(-0.8, min(0.8, float(d.get("yaw", 0.0))))
            S["ramp"] = (float(S["waist"][0]), yaw, time.monotonic(), max(0.5, float(d.get("duration_s", 2.0))))
            print(f"[fake] set_waist {yaw:+.3f} in {S['ramp'][3]:.1f} s", flush=True)
        elif c == "stop":
            S["mode"], S["qref"] = "mpc", None


ChannelSubscriber("rt/arm_ctrl/qref", LowCmd_).Init(on_qref, 10)
ChannelSubscriber("rt/arm_ctrl/cmd", String_).Init(on_cmd, 10)
ChannelSubscriber("rt/arm_ctrl/heartbeat", String_).Init(on_hb, 50)
pub_ls = ChannelPublisher("rt/lowstate", LowState_)
pub_ls.Init()
pub_hs = ChannelPublisher("rt/dex3/right/state", HandState_)
pub_hs.Init()
pub_st = ChannelPublisher("rt/arm_ctrl/state", String_)
pub_st.Init()
ls, hs = unitree_hg_msg_dds__LowState_(), unitree_hg_msg_dds__HandState_()
for i, v in enumerate([math.radians(30), 0, 0, 0, 0, 0, 0]):
    hs.motor_state[i].q = v
    hs.motor_state[i].mode = 1

dt, k = 0.002, 0
t_end = time.monotonic() + args.duration
print(f"[fake] G1 stand-in on '{args.net_if}': lowstate 500 Hz, daemon state 10 Hz", flush=True)
while time.monotonic() < t_end:
    t0 = time.monotonic()
    with lock:
        if not E["active"]:
            lost = [(s_, t0 - t_) for s_, t_ in E["hb"].items() if t0 - t_ > args.hb_timeout]
            if lost:
                estop(f"heartbeat from '{lost[0][0]}' lost for {lost[0][1]:.2f} s", "watchdog")
        q, dq = S["q"], S["dq"]
        fresh = (S["mode"] == "qref" and S["qref"] is not None and t0 - S["qref_t"] < 0.25 and not S["frozen"]
                 and not E["active"])
        if fresh:
            a = min(1.0, (t0 - S["t_en"]) / 2.0)
            tgt = (1 - a) * S["q_en"] + a * S["qref"]
            if np.max(np.abs(tgt - q)) > 1.2:
                S["frozen"] = True
                print("[fake] FREEZE: |q_ref - q| > 1.2 rad", flush=True)
            kp, kd = S["kp"], S["kd"]
            S["last_cmd"] = tgt.copy()
            S["hold"] = None
        elif E["active"]:
            if t0 - E["t0"] < args.freeze_s:
                tgt, kp, kd = E["freeze_q"], np.full(7, 400.0), np.full(7, 12.0)
            else:
                tgt, kp, kd = q, np.zeros(7), np.full(7, 4.0)       # damped
        else:
            if S["hold"] is None:
                S["hold"] = (S["last_cmd"] if S["last_cmd"] is not None else q).copy()
            tgt, kp, kd = S["hold"], np.full(7, 400.0), np.full(7, 12.0)
        acc = np.clip(kp * (tgt - q) - kd * dq, -60, 60)
        S["dq"] = dq + acc * dt
        S["q"] = q + S["dq"] * dt
        if S["ramp"] is not None:
            w0, w1, tr, d = S["ramp"]
            f = min(1.0, (t0 - tr) / d)
            new = w0 + (w1 - w0) * f ** 3 * (10 - 15 * f + 6 * f * f)
            S["wdq"] = (new - S["waist"][0]) / dt
            S["waist"][0] = new
            if f >= 1.0:
                S["ramp"], S["wdq"] = None, 0.0
        tau = kin.gravity_torque(S["waist"], S["q"])
        for j, i in enumerate(ARM):
            m = ls.motor_state[i]
            m.mode, m.q, m.dq, m.tau_est = 1, float(S["q"][j]), float(S["dq"][j]), float(tau[j])
            m.temperature = [40, 38]
        for j, n in enumerate(R.WAIST_JOINTS):
            m = ls.motor_state[R.MOTOR_IDX[n]]
            m.mode, m.q, m.dq = 1, float(S["waist"][j]), float(S["wdq"] if j == 0 else 0.0)
        ls.tick = k
    pub_ls.Write(ls)
    if k % 5 == 0:
        pub_hs.Write(hs)
    if k % 50 == 0:
        with lock:
            now_ = time.monotonic()
            es = {"active": E["active"], "phase": ("off" if not E["active"] else "freeze"
                                                   if now_ - E["t0"] < args.freeze_s else "damped"),
                  "why": E["why"], "src": E["src"], "since_s": round(now_ - E["t0"], 1) if E["active"] else None,
                  "heartbeats": {s_: round(now_ - t_, 2) for s_, t_ in E["hb"].items()},
                  "hb_timeout_s": args.hb_timeout}
            st = {"ts": time.time(), "waist_yaw": round(float(S["waist"][0]), 4), "waist_ramp": S["ramp"] is not None,
                  "estop": es,
                  "right": {"enabled": False, "frozen": S["frozen"], "exec_mode": S["mode"],
                            "q_meas": [round(float(v), 4) for v in S["q"]], "grasp_active": False},
                  "left": {"enabled": False, "frozen": False, "exec_mode": "mpc", "q_meas": [0.0] * 7}}
        pub_st.Write(String_(data=json.dumps(st)))
    k += 1
    time.sleep(max(0.0, dt - (time.monotonic() - t0)))
