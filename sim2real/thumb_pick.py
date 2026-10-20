#!/usr/bin/env python3
"""Pick the Dex3 RIGHT thumb's rotation (thumb_0) by eye, on the robot: type an angle, close, look, open.

The angle you settle on is the RL runs' default (toy_reach --thumb_deg; the original test keeps its 30 deg).
The URDF says 0 deg closes the thumb tip into the index-middle gap (30 deg lands it on the middle finger), but
the real motor may sit off its zero -- this is how to find the real one.

Run it ALONE: the launcher (g1_launch) / a toy must not be running (two writers on rt/dex3/right/cmd).

    python3 sim2real/thumb_pick.py                  # net-if from apple_picker_config.json
    python3 sim2real/thumb_pick.py --thumb_kp 2.0

Keys (then ENTER):
    <number>   thumb_0 to that angle [deg] (-60 .. 60), the other joints stay as they are
    c          close the hand (the picker's close for thumb_1/2, index, middle), thumb_0 kept
    o          open the hand (the picker's open), thumb_0 kept
    q          open and quit -- prints the --thumb_deg to use
Gains: thumb_0 --thumb_kp (default 1.5: the picker's 0.5 does not move it), the rest the picker's (0.5 / 0.75).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np

AP = os.path.expanduser("~/g1_fruits_picker/apple_picker")
sys.path.insert(0, AP)
import dex3_joint_probe as P  # noqa: E402  (its Hand: RIS frames built as full_move2 builds them)

CFG = os.path.join(AP, "apple_picker_config.json")


def busy():
    for pat in (r"^python3 -u [^ ]*apple_picker/g1_launch[.]py", r"^[^ ]*python[^ ]* .*dual/run_toy[.]py"):
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-if", default=None)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--thumb_kp", type=float, default=1.5)
    ap.add_argument("--move_s", type=float, default=1.0, help="each move takes this long (s)")
    a = ap.parse_args()
    if busy():
        sys.exit("the launcher / a toy is running: stop it first (two writers on the hand)")
    cfg = json.load(open(CFG))
    net_if = a.net_if or (cfg.get("full_move2") or {}).get("net_if", "")
    pick = cfg["pick"]
    OPEN = np.array(pick["hand_open_target_right"], float)
    CLOSE = np.array(pick["hand_close_target_right"], float)
    kp_cfg = pick.get("hand_kp", {"thumb": 0.5, "index": 0.75, "middle": 0.75})
    if not isinstance(kp_cfg, dict):
        kp_cfg = {"thumb": float(kp_cfg), "index": float(kp_cfg), "middle": float(kp_cfg)}
    KP = np.array([a.thumb_kp, kp_cfg.get("thumb", 0.5), kp_cfg.get("thumb", 0.5), kp_cfg.get("index", 0.75),
                   kp_cfg.get("index", 0.75), kp_cfg.get("middle", 0.75), kp_cfg.get("middle", 0.75)], float)
    h = P.Hand("right", net_if, a.domain, kp=0.5, kd=0.2)

    def send(q):                                  # one frame, per-joint kp
        cmd = h._mk()
        for i in range(7):
            mc = cmd.motor_cmd[i]
            mc.mode = P.FM.pack_ris_mode(i, status=0x01, timeout=P.FM.DEX3_ARK_TIMEOUT_BIT)
            mc.q, mc.dq, mc.tau, mc.kp, mc.kd = float(q[i]), 0.0, 0.0, float(KP[i]), 0.2
        h.pub.Write(cmd)

    target = h.q().copy()                         # start where it is: nothing jumps
    lock = threading.Lock()
    alive = True

    def writer():                                 # 50 Hz stream of the current target
        while alive:
            with lock:
                q = target.copy()
            send(q)
            time.sleep(0.02)
    threading.Thread(target=writer, daemon=True).start()

    def move_to(q_goal):
        with lock:
            q0 = target.copy()
        t0 = time.monotonic()
        while (f := (time.monotonic() - t0) / a.move_s) < 1.0:
            s = f * f * (3 - 2 * f)
            with lock:
                target[:] = q0 + s * (q_goal - q0)
            time.sleep(0.02)
        with lock:
            target[:] = q_goal
        time.sleep(0.6)
        q = h.q()
        print(f"  measured: thumb_0 {math.degrees(q[0]):+.1f} deg (commanded {math.degrees(q_goal[0]):+.1f}); "
              f"modes {h.modes()}")
        if abs(q[0] - q_goal[0]) > math.radians(4):
            print("  thumb_0 did not reach it: try a higher --thumb_kp (max ~3), or help it by hand once")

    th = float(target[0])
    print(f"thumb_0 now {math.degrees(th):+.1f} deg; kp {KP.round(2).tolist()}. Opening the hand first.")
    q = OPEN.copy(); q[0] = th
    move_to(q)
    closed = False
    try:
        while True:
            k = input("angle [deg] / c / o / q > ").strip().lower()
            if k == "q":
                break
            if k in ("c", "o"):
                closed = k == "c"
                q = (CLOSE if closed else OPEN).copy(); q[0] = th
                move_to(q)
                continue
            try:
                th = math.radians(float(np.clip(float(k), -60.0, 60.0)))
            except ValueError:
                print("  a number, c, o or q")
                continue
            with lock:
                q = target.copy()
            q[0] = th
            move_to(q)
    except (KeyboardInterrupt, EOFError):
        pass
    q = OPEN.copy(); q[0] = th
    move_to(q)
    alive = False
    time.sleep(0.1)
    print(f"\nfor the RL runs: --thumb_deg {math.degrees(th):.0f}   (tell Claude to make it the default)")


if __name__ == "__main__":
    main()
