#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Apple harvest state machine — sends joint targets to the running sim via DDS.

Run in a SECOND terminal while run_env.py is already running:
    python3 scripts/harvest_apple.py

Harvest motion stages (agronomically correct sequence):
  1. HOME    — neutral ready pose
  2. REACH   — extend arm toward apple, hand open
  3. CLOSE   — close fingers around apple
  4. ROLL    — rotate wrist 90 deg (key detachment motion — more effective than pull)
  5. PULL    — small downward retraction to break stem
  6. REST    — bring apple to comfortable hold position

Tuning workflow:
  1. Use tune_arm.py to find good REACH and CLOSE values interactively
  2. Press 'p' in tune_arm.py to save to ~/arm_targets.txt
  3. Copy those values into the TARGETS dict below
  4. Run this script to test the full sequence
  5. Repeat without restarting run_env.py

Press Ctrl+C at any time to abort and return robot to default pose.
Press Enter at the end to run the sequence again.
"""

import json
import time
import sys
import threading

# ── DDS setup ────────────────────────────────────────────────────────────────
try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    ChannelFactoryInitialize(1)
    publisher = ChannelPublisher("rt/arm_tune/targets", String_)
    publisher.Init()
    DDS_OK = True
    print("[DDS] Connected to rt/arm_tune/targets")
except Exception as e:
    print(f"[DDS] OFFLINE: {e}")
    print("  Check: pip install unitree_sdk2py")
    DDS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  JOINT TARGETS
#
#  ARM joints (7):
#    [0] shoulder_pitch   positive = forward/up
#    [1] shoulder_roll    negative = outward (right arm)
#    [2] shoulder_yaw
#    [3] elbow            negative = extend, positive = bend
#    [4] wrist_roll       ← KEY for harvest: 1.57 = 90 deg roll
#    [5] wrist_pitch
#    [6] wrist_yaw
#
#  HAND joints (7) — Dex3 right hand:
#    [0] j0  spread         range [-1.05,  1.05]
#    [1] j1  base flex      range [-1.05,  0.742]
#    [2] j2  index base     range [-1.75,  0.0]   negative to close
#    [3] j3  index tip      range [ 0.0,   1.57]
#    [4] j4  middle tip     range [ 0.0,   1.75]
#    [5] j5  thumb rotation range [ 0.0,   1.57]
#    [6] j6  thumb tip      range [ 0.0,   1.75]  (may be mimic joint)
#
#  All values are radians RELATIVE TO DEFAULT POSE (use_default_offset=True).
#
#  *** TUNE THESE USING tune_arm.py BEFORE RUNNING THIS SCRIPT ***
# ─────────────────────────────────────────────────────────────────────────────

TARGETS = {
    "HOME": {
        "arm":  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        "hand": [0.0,  0.0,  0.0,  0.0,  0.0,  0.7,  0.0],
        "duration": 1.5,
        "desc": "Neutral ready pose",
    },
    "REACH": {
        # Tune with tune_arm.py until wrist is at apple position, hand open
        "arm":  [-1.8, 0.1, 0.0, -0.8, 1.0, -0.5, -1.1],
        "hand": [0.0,  0.0,  0.0,  0.0,  0.0,  0.7,  0.0],
        "duration": 2.5,
        "desc": "Reaching toward apple with open hand",
    },
    "CLOSE": {
        # Same arm as REACH. Tune hand with tune_arm.py 'c' preset then adjust
        "arm":  [-1.8, 0.1, 0.0, -0.8, 1.0, -0.5, -1.1],
        "hand": [1.05, 0.742, 0.0, 1.67, 1.05, 0.7, -1.4],
        "duration": 1.5,
        "desc": "Closing hand around apple",
    },
    "ROLL": {
        # CRITICAL STAGE: wrist_roll = 1.57 rad (90 degrees)
        # Research shows wrist rotation is the primary detachment mechanism
        # for apples — more effective than pure pull force
        "arm":  [-1.8, 0.1, 0.0, -0.8, 2.57, -0.5, -1.1],
        "hand": [1.05, 0.742, 0.0, 1.67, 1.05, 0.7, -1.4],
        "duration": 2.0,
        "desc": "Rolling wrist 90 deg — primary detachment motion",
    },
    "PULL": {
        # After roll: small downward + backward motion to finish breaking stem
        # Reduce shoulder_pitch slightly, increase elbow bend
        "arm":  [-0.5, 0.1, 0.0, -0.8, 2.57, -0.5, -1.1],
        "hand": [1.05, 0.742, 0.0, 1.67, 1.05, 0.7, -1.4],
        "duration": 1.5,
        "desc": "Pulling apple downward to snap stem",
    },
    "REST": {
        # Comfortable holding position with apple secured
        "arm":  [-0.5, 0.1, 0.0, -0.8, 2.57, -0.5, -1.1],
        "hand": [1.05, 0.742, 0.0, 1.67, 1.05, 0.7, -1.4],
        "duration": 2.0,
        "desc": "Rest position — apple held securely",
    },
}

STAGE_ORDER = ["HOME", "REACH", "CLOSE", "ROLL", "PULL", "REST"]
CONTROL_HZ  = 50      # must match sim: dt=0.005, decimation=4 → 50 Hz
STEP_DT     = 1.0 / CONTROL_HZ


# ─────────────────────────────────────────────────────────────────────────────

def smoothstep(t: float) -> float:
    """5th-order smooth interpolation — no overshoot, smooth start/end."""
    t = max(0.0, min(1.0, t))
    return 6*t**5 - 15*t**4 + 10*t**3


def interp(a: list, b: list, t: float) -> list:
    s = smoothstep(t)
    return [a[i] + (b[i] - a[i]) * s for i in range(len(a))]


def publish(arm: list, hand: list) -> None:
    if not DDS_OK:
        return
    try:
        msg = String_(data=json.dumps({"arm": arm, "hand": hand}))
        publisher.Write(msg)
    except Exception:
        pass


def send_default() -> None:
    """Return robot to default pose (send twice for reliability)."""
    publish([0.0]*7, [0.0]*7)
    time.sleep(0.05)
    publish([0.0]*7, [0.0]*7)


def run_sequence(stop_event: threading.Event) -> bool:
    """
    Execute the full harvest sequence.
    Returns True if completed, False if interrupted.
    """
    prev_arm  = [0.0] * 7
    prev_hand = [0.0] * 7

    for stage_name in STAGE_ORDER:
        if stop_event.is_set():
            return False

        stage    = TARGETS[stage_name]
        tgt_arm  = stage["arm"]
        tgt_hand = stage["hand"]
        n_steps  = max(1, int(stage["duration"] * CONTROL_HZ))

        print(f"\n  ── {stage_name}: {stage['desc']}")
        if stage_name == "ROLL":
            print(f"     ⭮  wrist_roll target: {tgt_arm[4]:.2f} rad "
                  f"({tgt_arm[4]*57.3:.0f}°)")

        t0 = time.perf_counter()
        for i in range(n_steps):
            if stop_event.is_set():
                return False

            t    = (i + 1) / n_steps
            arm  = interp(prev_arm,  tgt_arm,  t)
            hand = interp(prev_hand, tgt_hand, t)
            publish(arm, hand)

            # Precise sleep to maintain control rate
            elapsed = time.perf_counter() - t0
            target  = (i + 1) * STEP_DT
            sleep_t = target - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        prev_arm  = list(tgt_arm)
        prev_hand = list(tgt_hand)
        print(f"     ✓ done  ({stage['duration']:.1f}s)")

    return True


def print_summary():
    total = sum(TARGETS[s]["duration"] for s in STAGE_ORDER)
    print("\n" + "="*60)
    print("APPLE HARVEST SEQUENCE")
    print("="*60)
    t = 0.0
    for name in STAGE_ORDER:
        s = TARGETS[name]
        print(f"  {t:4.1f}s  {name:<8}  ({s['duration']:.1f}s)  {s['desc']}")
        t += s["duration"]
    print(f"  {'':>4}  {'TOTAL':<8}  ({total:.1f}s)")
    print("="*60)
    print("Key: ROLL stage (wrist_roll = 1.57 rad) is the primary")
    print("     detachment mechanism — twists stem before pulling.")
    print("="*60)


def main():
    print_summary()
    print("\nDDS:", "CONNECTED ✓" if DDS_OK else "OFFLINE ✗")
    print("Sim topic: rt/arm_tune/targets")
    print("run_env.py must be running in another terminal.")
    print("\nPress Enter to start, Ctrl+C to abort.\n")

    if not DDS_OK:
        print("Cannot run — DDS offline.")
        return

    stop_event = threading.Event()

    try:
        input()   # wait for Enter
        while True:
            stop_event.clear()
            print("Starting harvest sequence...")
            completed = run_sequence(stop_event)

            if completed:
                print("\n" + "="*40)
                print("✅ Sequence complete!")
                print("   If stem broke: apple is free, held in REST pose.")
                print("   If stem didn't break: reduce BREAK_FORCE in tree_utils.py")
                print("   or adjust ROLL angle (currently "
                      f"{TARGETS['ROLL']['arm'][4]*57.3:.0f}°)")
                print("="*40)
            else:
                print("\nSequence interrupted.")

            print("\nPress Enter to run again, Ctrl+C to quit.")
            input()

    except KeyboardInterrupt:
        print("\n\nAborted.")
    finally:
        stop_event.set()
        print("Returning to default pose...")
        send_default()
        print("Done.")


if __name__ == "__main__":
    main()