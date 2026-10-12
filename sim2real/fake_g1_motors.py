#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
TEST ONLY: the G1's MOTORS on loopback DDS, under the REAL arm daemon
(sim2real/daemon_loopback.py runs g1_fruits_picker's onboard/arm_mpc_server.py
against this). No daemon logic here -- unlike fake_robot.py, which emulates it.

Publishes rt/lowstate (500 Hz) and rt/dex3/{left,right}/state (100 Hz); plays
rt/arm_sdk (per-motor PD: tau = kp (q* - q) + kd (dq* - dq) + tau_ff on the arms
and the waist, inertia --inertia) and rt/dex3/<side>/cmd (each finger relaxes
toward its target at kp/kd per second; kp 0 = it stays). With no rt/arm_sdk for
0.1 s the arms hold where they are (the robot's own controller taking them
back). No gravity: a damped arm stays where it is instead of sinking.

  python3 sim2real/fake_g1_motors.py --net-if lo --log motors.csv
"""
import argparse
import math
import threading
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--net-if", default="lo")
ap.add_argument("--duration", type=float, default=600.0)
ap.add_argument("--inertia", type=float, default=0.5, help="kg m^2 per arm / waist joint")
ap.add_argument("--friction", type=float, default=3.0, help="N m s/rad viscous friction per joint")
ap.add_argument("--log", default=None, help="CSV of t, right arm q, its kp, hand q, waist, kd/tau of joint 22 (10 Hz)")
ap.add_argument("--blackout_at", type=float, default=None, help="s: stop publishing rt/lowstate here ...")
ap.add_argument("--blackout_s", type=float, default=2.0, help="... for this long (commands are still played)")
ap.add_argument("--press_file", default=None,
                help="TEST: file holding a place index 0-6 (tactile_io.LOCATIONS) whose pad elements 0-3 are pressed")
ap.add_argument("--start_arm", default=None, help="7 comma-separated rad: where the right arm starts")
args = ap.parse_args()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_, unitree_hg_msg_dds__LowState_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_, PressSensorState_  # noqa: E402

ChannelFactoryInitialize(0, args.net_if)
N = 29
DRIVEN = list(range(12, 29))                 # waist (12-14) + both arms (15-28)
lock = threading.Lock()
q = np.zeros(N)
q[12:15] = [0.03, 0.0, 0.1]
q[15:22] = [-0.5, 0.2, -0.1, 1.0, 0.0, 0.0, 0.0]          # left arm
q[22:29] = [-0.5, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0]          # right arm
if args.start_arm:
    q[22:29] = [float(v) for v in args.start_arm.split(",")]
dq = np.zeros(N)
cmd = {"q": q.copy(), "dq": np.zeros(N), "kp": np.zeros(N), "kd": np.zeros(N), "tau": np.zeros(N), "t": -1.0}
hold_q = q.copy()
hand = {s: {"q": np.array([math.radians(30), 0, 0, 0, 0, 0, 0], float), "tgt": None, "kp": np.zeros(7),
            "kd": np.ones(7)} for s in ("left", "right")}


def on_sdk(msg):
    mc = msg.motor_cmd
    with lock:
        for i in DRIVEN:
            cmd["q"][i], cmd["dq"][i], cmd["kp"][i] = mc[i].q, mc[i].dq, mc[i].kp
            cmd["kd"][i], cmd["tau"][i] = mc[i].kd, mc[i].tau
        cmd["t"] = time.monotonic()


def on_hand(side, msg):
    with lock:
        h = hand[side]
        h["tgt"] = np.array([msg.motor_cmd[i].q for i in range(7)])
        h["kp"] = np.array([msg.motor_cmd[i].kp for i in range(7)])
        h["kd"] = np.array([max(msg.motor_cmd[i].kd, 1e-3) for i in range(7)])


ChannelSubscriber("rt/arm_sdk", LowCmd_).Init(on_sdk, 10)
for s in ("left", "right"):
    ChannelSubscriber(f"rt/dex3/{s}/cmd", HandCmd_).Init(lambda m, s=s: on_hand(s, m), 10)
pub_ls = ChannelPublisher("rt/lowstate", LowState_)
pub_ls.Init()
pub_hs = {s: ChannelPublisher(f"rt/dex3/{s}/state", HandState_) for s in ("left", "right")}
for p in pub_hs.values():
    p.Init()
ls = unitree_hg_msg_dds__LowState_()
hs = {s: unitree_hg_msg_dds__HandState_() for s in ("left", "right")}
for _s in hs.values():          # nine pad entries of 12 readings, ~100000 counts at rest (as the real hand)
    _s.press_sensor_state = [PressSensorState_([100000.0] * 12, [30.0] * 12, 0, 0) for _ in range(9)]
logf = open(args.log, "w") if args.log else None
if logf:
    logf.write("t," + ",".join(f"q{i}" for i in range(22, 29)) + "," + ",".join(f"kp{i}" for i in range(22, 29))
               + "," + ",".join(f"hand{i}" for i in range(7)) + ",waist,kd22,tau22,blackout\n")

dt, k = 0.002, 0
T0 = time.monotonic()
t_end = T0 + args.duration
print(f"[motors] G1 motors on '{args.net_if}': playing rt/arm_sdk + rt/dex3/*/cmd", flush=True)
while time.monotonic() < t_end:
    t0 = time.monotonic()
    blackout = (args.blackout_at is not None and args.blackout_at <= t0 - T0 < args.blackout_at + args.blackout_s)
    with lock:
        live = t0 - cmd["t"] < 0.1
        if live:
            tau = cmd["kp"] * (cmd["q"] - q) + cmd["kd"] * (cmd["dq"] - dq) + cmd["tau"]
            hold_q[:] = q
        else:                                # no arm_sdk: the robot's controller holds the arms
            tau = 400.0 * (hold_q - q) - 12.0 * dq
        acc = np.zeros(N)
        tau = tau - args.friction * dq                  # viscous joint friction
        acc[DRIVEN] = np.clip(tau[DRIVEN] / args.inertia, -200, 200)
        dq += acc * dt
        q += dq * dt
        for s, h in hand.items():
            if h["tgt"] is not None:
                rate = np.minimum(h["kp"] / h["kd"] * dt, 1.0)
                h["q"] += rate * (h["tgt"] - h["q"])
        for i in range(N):
            m = ls.motor_state[i]
            m.mode, m.q, m.dq, m.tau_est = 1, float(q[i]), float(dq[i]), float(tau[i] if live else 0.0)
            m.temperature = [40, 38]
        ls.tick = k
        if k % 5 == 0:
            for s, h in hand.items():
                for i in range(7):
                    hs[s].motor_state[i].q = float(h["q"][i])
                    hs[s].motor_state[i].mode = 1
                # a fruit in the closed fingers loads the pads: 3000 counts per 0.1 rad past contact
                qh = h["q"]
                rise = 30000.0 * (max(0.0, qh[4] - 1.45) + max(0.0, qh[6] - 1.45)) if s == "right" else 0.0
                press = -1
                if args.press_file and s == "right":
                    try:
                        press = int(open(args.press_file).read().strip() or -1)
                    except (OSError, ValueError):
                        press = -1
                for e in range(9):
                    vals = [100000.0 + (rise if e < 6 else 0.0)] * 12
                    if e == press:
                        vals[:4] = [100000.0 + 9000.0] * 4
                    hs[s].press_sensor_state[e].pressure = vals
        if logf and k % 50 == 0:
            logf.write(f"{t0 - T0:.3f}," + ",".join(f"{v:.4f}" for v in q[22:29]) + ","
                       + ",".join(f"{v:.1f}" for v in (cmd["kp"][22:29] if live else np.full(7, -1.0))) + ","
                       + ",".join(f"{v:.3f}" for v in hand["right"]["q"])
                       + f",{q[12]:.4f},{cmd['kd'][22] if live else -1:.1f},{cmd['tau'][22] if live else 0:.2f},"
                       f"{int(blackout)}\n")
            logf.flush()
    blackout = (args.blackout_at is not None and args.blackout_at <= t0 - T0 < args.blackout_at + args.blackout_s)
    if not blackout:
        pub_ls.Write(ls)
    if k % 5 == 0:
        for s in ("left", "right"):
            pub_hs[s].Write(hs[s])
    k += 1
    time.sleep(max(0.0, dt - (time.monotonic() - t0)))
