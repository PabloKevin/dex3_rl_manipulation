#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
SIM2REAL TOY TEST 3 -- REACH and GRASP a real apple the camera sees, with both
policies: the waist reach (AppleReachSpindle-v3: right arm + waist yaw) puts the
palm at the fruit, then the hand-only grasp policy (AppleGraspSpindle-v2, model
1499: 44 inputs -> arm 7 + hand 7) closes the fingers on it.

This is toy_apple.py (camera -> detections -> nearest reachable fruit -> reach, same
robot path, safety watch, E-STOP, home pose) plus, after the reach:

  hand-off   the palm holds at the reach target for --handoff_s (0.5 s: the sim's
             "converged" hand-off), the pads' baseline is taken (0.8 s);
  GRASP      the grasp policy drives the fingers (hand targets = thumb pre-shape +
             0.3 x action, rate limited like the sim's) and the wrist (+0.05 x action);
             shoulder and elbow stay frozen at the hand-off pose until a grip is
             felt, as in training (--grasp_arm_free to unlock them after it).
             Its inputs: finger joints, the pad rises (converted to newtons), the
             grip-force command --grip_n, the fruit in the palm frame (from the
             detection fixed at the choice), gravity in the palm frame;
  judged     GRASPED = two finger groups loaded and the fingers closed for 0.3 s,
             kept --grasp_hold_s; NO GRIP after --grasp_s;
  release    the hand opens, the arm goes back home along its path. NOTHING PULLS
             the fruit: the arm stays where it is while the hand grips.

STOPS (hand opens, arm held, then the usual retreat question): any safety STOP of the
reach watch (collision, acceleration, ...), the palm moving > --grasp_move_stop from the
hand-off point, an estimated pad force > --grip_stop_n, the E-STOP.

TACTILE: run sim2real/tactile_calibrate.py ONCE first (read-only; you press each named
place of the hand, optionally one place on a scale): it records which of the hand's 108
pressure elements answer where and the counts per newton, into logs/sim2real/tactile_calib.json,
which this test loads by itself. Without it the pads are read as nine entry means at the
sim's placeholder 3000 counts/N (approximate; the toy says so at start). Start with the
default low command (--grip_n 6). The log (grasp.npz) has every pad reading.

Terminals: as toy_apple.py (1 robot launch, 2 live_locator, 3 fruits_to_torso, 4 dashboard
for STOP), then in 5 (once: python3 sim2real/tactile_calibrate.py --net-if enp62s0):
  python3 sim2real/toy_grasp.py --speed 0.3 --shadow     # lists the fruit, nothing moves
  python3 sim2real/toy_grasp.py --speed 0.3              # reach + grasp
  python3 sim2real/toy_grasp.py --speed 0.3 --pick 1

Offline: python3 sim2real/toy_grasp.py --mock --speed 0.5 --mock_apples "0.33,-0.21,0.28"

Every toy_reach.py option applies (python3 sim2real/toy_reach.py -h).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy_reach  # noqa: E402

DEFAULTS = {"--target": ["detection"], "--waist": ["follow"]}

if __name__ == "__main__":
    if any(x in ("-h", "--help") for x in sys.argv[1:]):
        print(__doc__)
    argv = sys.argv[1:]
    for flag, val in DEFAULTS.items():
        if flag not in argv:
            argv = argv + [flag] + val
    if "--grasp" not in argv:
        argv = argv + ["--grasp"]
    sys.argv = [sys.argv[0]] + argv
    toy_reach.main()
