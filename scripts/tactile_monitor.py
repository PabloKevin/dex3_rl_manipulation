#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Dex3-1 RIGHT hand tactile sensor viewer — terminal 2.

Live-plots the 9 tactile taxels (2 on thumb, 2 on index, 2 on middle finger,
3 on palm) on a simplified schematic of the hand, updating in real time
while scripts/run_env.py is running in terminal 1.

Usage:
  Terminal 1: ~/IsaacLab/isaaclab.sh -p scripts/run_env.py
  Terminal 2: python3 scripts/tactile_monitor.py

This is a PLAIN PYTHON script — it does NOT launch Isaac Sim, only
unitree_sdk2py (DDS) + matplotlib, mirroring reach_apple_ik.py's "terminal 2
DDS client" pattern.

Sim2real note: this script subscribes to the exact message type/topic a real
Dex3-1 hand publishes (unitree_hg HandState_.press_sensor_state on
rt/dex3/right/state) and decodes it with the same convention the real grasp
controller uses (raw/10000.0, 30000 = "no data" — see tactile_layout.py for
where that convention comes from). Point ChannelFactoryInitialize() at the
robot's network interface instead of the local sim and this script keeps
working unmodified against real hardware.
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.tactile_layout as tactile_layout

parser = argparse.ArgumentParser(description="Dex3-1 right hand tactile viewer")
parser.add_argument("--domain", type=int, default=1,
                     help="DDS channel factory domain id — must match run_env.py (default 1).")
parser.add_argument("--vmax", type=float, default=3.0,
                     help="Pressure value (decoded units, raw/10000) that maps to the top of "
                          "the color scale. Tune to taste — see tactile_layout.SIM_FORCE_TO_RAW_GAIN.")
parser.add_argument("--hz", type=float, default=15.0, help="Plot refresh rate.")
parser.add_argument("--stale_after", type=float, default=1.0,
                     help="Seconds since the last message before the viewer flags the feed as stale.")
args_cli = parser.parse_args()

# ── DDS setup ─────────────────────────────────────────────────────────────
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    ChannelFactoryInitialize(args_cli.domain)
    DDS_OK = True
except Exception as e:
    print(f"[DDS] OFFLINE: {e}")
    print("Check that unitree_sdk2py is installed in this Python environment "
          "(same one run_env.py's conda env uses).")
    DDS_OK = False

_state_lock = threading.Lock()
_latest_raw = [tactile_layout.INVALID_RAW] * tactile_layout.NUM_RAW_SLOTS
_last_msg_time = None  # time.monotonic() of the last received message, None = never


def _hand_state_callback(msg):
    global _last_msg_time
    try:
        pressure = list(msg.press_sensor_state[0].pressure)
        with _state_lock:
            global _latest_raw
            _latest_raw = pressure
            _last_msg_time = time.monotonic()
    except Exception as e:
        print(f"[DDS] tactile parse error: {e}")


if DDS_OK:
    try:
        sub = ChannelSubscriber(tactile_layout.DDS_TOPIC_HAND_STATE_RIGHT, HandState_)
        sub.Init(_hand_state_callback, 1)
        print(f"[DDS] Subscribed to {tactile_layout.DDS_TOPIC_HAND_STATE_RIGHT}")
        print("Waiting for run_env.py to start publishing tactile data "
              "(make sure it's running, and both scripts use the same --domain)...")
    except Exception as e:
        print(f"[DDS] Could not subscribe: {e}")
        DDS_OK = False


# ── Plot ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, FancyBboxPatch

# Sequential "blue" ramp, light->dark (dataviz skill reference palette).
_SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "tactile_pressure",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
_NO_DATA_COLOR = "#c3c2b7"   # text-secondary token — flags "no reading" without alarming red
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_SURFACE = "#fcfcfb"
_HAND_OUTLINE = "#d8d6cf"

FINGER_LABELS = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "palm": "Palm"}


def _draw_hand_outline(ax):
    """A deliberately simplified schematic (not anatomically exact) — just
    enough for a viewer to see which cluster of taxels is which."""
    # Palm
    ax.add_patch(FancyBboxPatch(
        (0.32, 0.28), 0.34, 0.28, boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1.5, edgecolor=_HAND_OUTLINE, facecolor="#f4f3ef", zorder=0))
    # Index finger (vertical capsule)
    ax.add_patch(FancyBboxPatch(
        (0.25, 0.55), 0.11, 0.40, boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1.5, edgecolor=_HAND_OUTLINE, facecolor="#f4f3ef", zorder=0))
    # Middle finger (vertical capsule, slightly longer)
    ax.add_patch(FancyBboxPatch(
        (0.48, 0.55), 0.11, 0.44, boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1.5, edgecolor=_HAND_OUTLINE, facecolor="#f4f3ef", zorder=0))
    # Thumb (angled capsule via rotated ellipse)
    ax.add_patch(Ellipse(
        (0.77, 0.28), width=0.34, height=0.15, angle=-35,
        linewidth=1.5, edgecolor=_HAND_OUTLINE, facecolor="#f4f3ef", zorder=0))
    ax.text(0.49, 0.14, "Dex3-1 RIGHT hand (simplified, top-down)",
            ha="center", va="center", fontsize=8, color=_TEXT_SECONDARY)


def main():
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.05, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_hand_outline(ax)

    fig.suptitle("Dex3-1 Right Hand — Tactile Sensors", fontsize=14,
                 fontweight="bold", color=_TEXT_PRIMARY, y=0.98)
    status_text = fig.text(0.5, 0.945, "", ha="center", fontsize=9.5)

    xs = [t.xy[0] for t in tactile_layout.TAXELS]
    ys = [t.xy[1] for t in tactile_layout.TAXELS]
    norm = Normalize(vmin=0.0, vmax=args_cli.vmax)
    # NOTE: scatter's own facecolors are always set explicitly via set_color()
    # below (per-taxel: the pressure ramp, or a flat gray for "no data") rather
    # than through scatter's own cmap/data-driven mapping — literal per-point
    # colors aren't uniform magnitude data, so the colorbar is driven by its
    # own ScalarMappable(_SEQUENTIAL_BLUE, norm) instead of this scatter.
    scatter = ax.scatter(xs, ys, s=650, c=[_NO_DATA_COLOR] * len(xs),
                          edgecolors=_TEXT_PRIMARY, linewidths=1.3, zorder=3)

    value_labels = []
    for taxel in tactile_layout.TAXELS:
        ax.text(taxel.xy[0], taxel.xy[1] - 0.055, taxel.name, ha="center", va="top",
                 fontsize=7, color=_TEXT_SECONDARY, zorder=3)
        lbl = ax.text(taxel.xy[0], taxel.xy[1], "--", ha="center", va="center",
                       fontsize=7.5, color=_TEXT_PRIMARY, fontweight="bold", zorder=4)
        value_labels.append(lbl)

    # Finger group legend (identity is via text, not color — color encodes magnitude)
    for finger, (lx, ly) in {"thumb": (0.90, 0.30), "index": (0.10, 0.80),
                              "middle": (0.68, 0.90), "palm": (0.28, 0.35)}.items():
        ax.text(lx, ly, FINGER_LABELS[finger], ha="center", va="center",
                 fontsize=9, color=_TEXT_SECONDARY, style="italic", zorder=3)

    # Palm taxels share one physical link in sim (see tactile_layout.py) —
    # flagged directly on the plot rather than silently implied.
    ax.text(0.49, 0.09, "Note: sim palm taxels (palm_0..2) currently share one "
            "contact-force reading (single palm link).",
            ha="center", va="center", fontsize=7, color=_TEXT_SECONDARY, style="italic")

    sm = ScalarMappable(norm=norm, cmap=_SEQUENTIAL_BLUE)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.045, pad=0.03)
    cbar.set_label("Pressure (decoded units, raw/10000)", color=_TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors=_TEXT_SECONDARY)

    def update(_frame):
        with _state_lock:
            raw = list(_latest_raw)
            last_t = _last_msg_time
        colors = []
        for i, taxel in enumerate(tactile_layout.TAXELS):
            value = tactile_layout.decode_pressure(raw[taxel.raw_index]) if taxel.raw_index < len(raw) else None
            if value is None:
                colors.append(_NO_DATA_COLOR)
                value_labels[i].set_text("--")
                value_labels[i].set_color(_TEXT_PRIMARY)
            else:
                nv = norm(value)
                colors.append(_SEQUENTIAL_BLUE(nv))
                value_labels[i].set_text(f"{value:.2f}")
                # Dark ink on the pale end of the ramp, white ink once the fill
                # gets dark enough to need it — keeps the number legible either way.
                value_labels[i].set_color("white" if nv >= 0.55 else _TEXT_PRIMARY)
        scatter.set_color(colors)

        now = time.monotonic()
        if not DDS_OK:
            status_text.set_text("● DDS offline — see terminal for error")
            status_text.set_color("#e34948")
        elif last_t is None:
            status_text.set_text(f"● waiting for run_env.py on {tactile_layout.DDS_TOPIC_HAND_STATE_RIGHT} ...")
            status_text.set_color("#eda100")
        elif now - last_t > args_cli.stale_after:
            status_text.set_text(f"● stale — no message for {now - last_t:.1f}s")
            status_text.set_color("#eda100")
        else:
            status_text.set_text("● connected — live")
            status_text.set_color("#1baf7a")
        return [scatter, status_text, *value_labels]

    anim = FuncAnimation(fig, update, interval=1000.0 / args_cli.hz, blit=False, cache_frame_data=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


if __name__ == "__main__":
    main()
