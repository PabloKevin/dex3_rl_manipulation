#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
TEST ONLY: g1_fruits_picker's REAL arm daemon (onboard/arm_mpc_server.py, with
its E-STOP, onboard/estop.py) on loopback DDS, without its MPC solvers -- the
command protocol, exec_mode "qref", set_waist, the E-STOP and the 90 Hz publish
loop run exactly as on the Jetson; "mpc"/"path" targets are not solved. Run it
over sim2real/fake_g1_motors.py:

  python3 sim2real/fake_g1_motors.py --net-if lo &
  python3 sim2real/daemon_loopback.py --net-if lo
"""
import argparse
import os
import signal
import sys
import threading
import time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--net-if", default="lo")
ap.add_argument("--picker_dir", default=os.path.expanduser("~/g1_fruits_picker/apple_picker"))
a = ap.parse_args()
if a.net_if != "lo":
    sys.exit("daemon_loopback.py is a loopback test: --net-if lo only (never next to the real robot)")
os.environ["ARM_MPC_PUB_PROC"] = "0"           # publish from this process (no child, no sudo)
od = os.path.join(a.picker_dir, "onboard")
sys.path[:0] = [od, a.picker_dir, os.path.join(os.path.dirname(a.picker_dir), "common")]
import arm_mpc_server as A  # noqa: E402

srv = A.ArmMpcServer(0, a.net_if, publish=True)


def _on_shutdown(signum, _frame):
    if srv._damp_t0 is None:
        srv._damp_t0 = time.monotonic()
        print(f"[daemon-loopback] signal {signum} -> damped release", flush=True)


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, _on_shutdown)
srv.hw.wait_for_state(timeout=10.0)
srv.estop.start()
threading.Thread(target=srv._state_loop, daemon=True).start()
print("[daemon-loopback] arm_mpc_server (no solvers) up: cmd / qref / estop / state / pub loop", flush=True)
try:
    srv._pub_loop()
finally:
    srv._run = False
    print("[daemon-loopback] exit", flush=True)
