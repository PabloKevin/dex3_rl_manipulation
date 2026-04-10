#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Interactive joint tuner — sends arm + hand targets to the running sim via DDS.

Run in a SECOND terminal while run_env.py is already running:
    python3 scripts/tune_arm.py

Controls:
    ↑ / ↓          select joint
    ← / →          decrement / increment selected joint
    [ / ]          halve / double the step size
    r              reset ALL joints to zero (default pose)
    o              open hand (joints 7-13 → 0)
    c              close hand (joints 7-13 → preset values)
    h              reset arm only (joints 0-6 → 0)
    p              print current targets as Python list (copy-paste into scripts)
    q / Ctrl+C     quit

No Isaac Lab needed — runs with plain python3 in your conda env.
Requires: unitree_sdk2py  (pip install unitree_sdk2py)
"""

import curses
import json
import os
import time
import sys

# ── DDS setup ────────────────────────────────────────────────────────────────
try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    ChannelFactoryInitialize(1)
    publisher = ChannelPublisher("rt/arm_tune/targets", String_)
    publisher.Init()
    DDS_OK = True
except Exception as e:
    print(f"[WARN] DDS unavailable: {e}")
    print("  Targets will display but NOT be sent to the sim.")
    print("  Install unitree_sdk2py or check channel settings.")
    DDS_OK = False
    time.sleep(2)

# ── Joint metadata ────────────────────────────────────────────────────────────
# (name, min_rad, max_rad, default_action)
ARM_JOINTS = [
    ("shoulder_pitch", -3.14,  3.14, 0.0),
    ("shoulder_roll",  -3.14,  3.14, 0.0),
    ("shoulder_yaw",   -3.14,  3.14, 0.0),
    ("elbow",          -3.14,  3.14, 0.0),
    ("wrist_roll",     -3.14,  3.14, 0.0),
    ("wrist_pitch",    -3.14,  3.14, 0.0),
    ("wrist_yaw",      -3.14,  3.14, 0.0),
]

# Dex3 right hand limits from SDK (right hand)
HAND_JOINTS = [
    ("hand_j0",  -1.05,  1.05, 0.0),
    ("hand_j1",  -1.05,  0.742, 0.0),
    ("hand_j2",  -1.75,  0.0,  0.0),
    ("hand_j3",   0.0,   1.57, 0.0),
    ("hand_j4",   0.0,   1.75, 0.0),
    ("hand_j5",   0.0,   1.57, 0.0),
    ("hand_j6",   -1.75,   0.0, 0.0),
]

ALL_JOINTS = ARM_JOINTS + HAND_JOINTS   # 14 total

# Preset for fully closed hand (tune these for your URDF)
HAND_CLOSE_PRESET = [0.9, 0.5, -1.5, 1.5, 1.5, 1.5, 1.5]

STEP_SIZES   = [0.02, 0.05, 0.10, 0.20, 0.50]
DEFAULT_STEP = 2   # index into STEP_SIZES → 0.10 rad


def publish(arm_vals, hand_vals):
    if not DDS_OK:
        return
    try:
        msg = String_(data=json.dumps({"arm": arm_vals, "hand": hand_vals}))
        publisher.Write(msg)
    except Exception as e:
        pass   # don't crash the UI on a transient DDS error


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def draw_ui(stdscr, values, selected, step_idx):
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    arm_vals  = values[:7]
    hand_vals = values[7:]

    title = "── ARM + HAND JOINT TUNER ──"
    stdscr.addstr(0, max(0, (w - len(title)) // 2), title, curses.A_BOLD)

    step = STEP_SIZES[step_idx]
    stdscr.addstr(1, 2, f"Step: {step:.2f} rad   [ ] to change   p=print   q=quit",
                  curses.color_pair(3))

    # ── Arm joints ───────────────────────────────────────────────────────
    stdscr.addstr(3, 2, "RIGHT ARM  (action = delta from default pose, radians)",
                  curses.A_UNDERLINE)
    for i, (name, lo, hi, _) in enumerate(ARM_JOINTS):
        val  = arm_vals[i]
        attr = curses.color_pair(2) | curses.A_BOLD if i == selected else curses.color_pair(1)
        bar  = _bar(val, lo, hi, width=20)
        marker = "▶" if i == selected else " "
        stdscr.addstr(4 + i, 2,
            f"{marker} [{i}] {name:<17}  {val:+7.3f} rad  {bar}  [{lo:+.2f} … {hi:+.2f}]",
            attr)

    # ── Hand joints ──────────────────────────────────────────────────────
    stdscr.addstr(12, 2, "RIGHT HAND  (radians from open=0)",
                  curses.A_UNDERLINE)
    for i, (name, lo, hi, _) in enumerate(HAND_JOINTS):
        idx  = i + 7
        val  = hand_vals[i]
        attr = curses.color_pair(2) | curses.A_BOLD if idx == selected else curses.color_pair(1)
        bar  = _bar(val, lo, hi, width=20)
        marker = "▶" if idx == selected else " "
        stdscr.addstr(13 + i, 2,
            f"{marker} [{idx}] {name:<17}  {val:+7.3f} rad  {bar}  [{lo:+.2f} … {hi:+.2f}]",
            attr)

    # ── Controls ─────────────────────────────────────────────────────────
    row = 21
    stdscr.addstr(row,   2, "↑↓ select   ←→ adjust   r=reset all   h=reset arm", curses.color_pair(3))
    stdscr.addstr(row+1, 2, "o=open hand   c=close hand   p=print targets", curses.color_pair(3))

    dds_str = "DDS: CONNECTED ✓" if DDS_OK else "DDS: OFFLINE (display only)"
    dds_col = curses.color_pair(2) if DDS_OK else curses.color_pair(4)
    stdscr.addstr(row+2, 2, dds_str, dds_col)

    stdscr.refresh()


def _bar(val, lo, hi, width=20):
    """Simple ASCII progress bar."""
    span = hi - lo if hi != lo else 1.0
    frac = (val - lo) / span
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def main(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE,  -1)   # normal
    curses.init_pair(2, curses.COLOR_GREEN,  -1)   # selected / connected
    curses.init_pair(3, curses.COLOR_CYAN,   -1)   # help text
    curses.init_pair(4, curses.COLOR_RED,    -1)   # offline
    stdscr.keypad(True)
    stdscr.timeout(50)   # 50 ms non-blocking read → ~20 Hz UI refresh

    values   = [0.0] * 14
    selected = 0
    step_idx = DEFAULT_STEP
    last_publish = 0.0

    # Send initial zeros immediately
    publish(values[:7], values[7:])

    while True:
        draw_ui(stdscr, values, selected, step_idx)

        key = stdscr.getch()

        if key == curses.KEY_UP:
            selected = (selected - 1) % 14

        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % 14

        elif key in (curses.KEY_RIGHT, ord('+')):
            lo, hi = ALL_JOINTS[selected][1], ALL_JOINTS[selected][2]
            values[selected] = clamp(values[selected] + STEP_SIZES[step_idx], lo, hi)
            publish(values[:7], values[7:])

        elif key in (curses.KEY_LEFT, ord('-')):
            lo, hi = ALL_JOINTS[selected][1], ALL_JOINTS[selected][2]
            values[selected] = clamp(values[selected] - STEP_SIZES[step_idx], lo, hi)
            publish(values[:7], values[7:])

        elif key == ord('['):
            step_idx = max(0, step_idx - 1)

        elif key == ord(']'):
            step_idx = min(len(STEP_SIZES) - 1, step_idx + 1)

        elif key == ord('r'):
            values = [0.0] * 14
            publish(values[:7], values[7:])

        elif key == ord('h'):
            for i in range(7):
                values[i] = 0.0
            publish(values[:7], values[7:])

        elif key == ord('o'):
            for i in range(7, 14):
                values[i] = 0.0
            publish(values[:7], values[7:])

        elif key == ord('c'):
            for i, v in enumerate(HAND_CLOSE_PRESET):
                values[7 + i] = v
            publish(values[:7], values[7:])

        elif key == ord('p'):
            arm_str  = [round(v, 3) for v in values[:7]]
            hand_str = [round(v, 3) for v in values[7:]]
            with open("joint_targets/index.txt", "r") as f:
                index = f.read()
                f.close()
            with open("joint_targets/index.txt", "w") as f:
                f.write(str(int(index) + 1))
                f.close()
            save_path = f"joint_targets/arm_targets_{index}.txt"
            with open(save_path, "w") as f:
                f.write(f"ARM_REACH  = {arm_str}\n")
                f.write(f"HAND_CLOSE = {hand_str}\n")
            import sys as _sys
            print(f"\nARM_REACH  = {arm_str}\nHAND_CLOSE = {hand_str}",
                  file=_sys.stderr)
            stdscr.addstr(23, 2,
                f"  Saved to {save_path}  ",
                curses.color_pair(2))
            stdscr.refresh()
            time.sleep(1.5)

        elif key in (ord('q'), 27):   # q or Escape
            break

    # On exit: send zeros so the robot returns to default pose
    publish([0.0]*7, [0.0]*7)


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    finally:
        # Restore robot to default pose on exit
        if DDS_OK:
            try:
                msg = String_(data=json.dumps({"arm": [0.0]*7, "hand": [0.0]*7}))
                publisher.Write(msg)
            except Exception:
                pass
        print("\nTuner exited. Robot returned to default pose.")
        print("Check /tmp/arm_targets.txt for saved targets (if you pressed 'p').")