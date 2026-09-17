# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Dex3-1 RIGHT hand tactile taxel layout — the single source of truth shared by:
  - scripts/run_env.py        (sim side: computes + DDS-publishes these taxels)
  - scripts/tactile_monitor.py (terminal-2 viewer: subscribes + plots them)

Deliberately has ZERO Isaac Lab / torch / matplotlib imports so both a
full Isaac Sim process and a plain-python terminal-2 viewer can import it
identically.

Where this layout comes from
-----------------------------
The real Dex3-1 hand reports tactile data as `unitree_hg::msg::HandState.
press_sensor_state`, a sequence of `PressSensorState_` entries each carrying
`pressure: float32[12]`. Unitree's own Dex3 driver never documents the taxel
layout, but the real closed-loop grasp controller shipped in
unitree_g1_dex3_stack/src/dex3_controller.cpp (handStateCallback) hardcodes
the only known-working index mapping, decoding EVERY entry in the sequence
the same way:
    pressure[0], pressure[1]  -> thumb
    pressure[2], pressure[3]  -> middle
    pressure[4], pressure[5]  -> index
    pressure[6], pressure[7], pressure[8] -> palm
    pressure[9..11]           -> unused (always the "no data" sentinel)
  value  = pressure[idx] / PRESSURE_SCALE
  "no data" sentinel = INVALID_RAW (30000)
That gives 9 real taxels total: 2 on the thumb, 2 on the index, 2 on the
middle finger, 3 on the palm — this module mirrors that exactly so a script
written against it decodes a REAL Dex3-1 HandState message unmodified
(only the DDS domain/network would need to change to point at real hardware).

Sim-side approximation (documented, not hidden)
------------------------------------------------
Isaac Lab's PhysX ContactSensor reports one net force per rigid-body LINK,
not per discrete taxel pad. The Dex3-1 right hand has exactly one link per
finger phalanx that could plausibly carry a pad (thumb_1/thumb_2, index_0/
index_1, middle_0/middle_1) so those map 1:1 onto the 2-taxel-per-finger
real layout. The palm is a SINGLE link in sim, so all 3 palm taxels read the
same net force value — there is no finer spatial resolution available
without subdividing the palm collision mesh into separate bodies, which is
out of scope here. This is flagged in the viewer (palm taxels share one
reading) rather than faked as independent.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Real-hardware wire-format constants (unitree_hg PressSensorState_) ──────
PRESSURE_SCALE = 10000.0   # decoded_value = raw_pressure / PRESSURE_SCALE
INVALID_RAW = 30000.0      # firmware's "no data for this slot" sentinel
NUM_RAW_SLOTS = 12         # PressSensorState_.pressure is float32[12]

# ── Sim-only: contact-force (N) -> raw-count approximation ──────────────────
# No real force/pressure calibration data is available for the Dex3-1 pads,
# so this is a documented, tunable placeholder (not a measured constant):
# it just needs to spread a plausible grasp force range (~0-10 N) across the
# 0..INVALID_RAW headroom so the live plot is legible. Tune freely.
SIM_FORCE_TO_RAW_GAIN = 3000.0   # raw_count = min(force_N * GAIN, INVALID_RAW - 1)


@dataclass(frozen=True)
class Taxel:
    name: str            # short id, e.g. "thumb_1"
    finger: str          # "thumb" | "index" | "middle" | "palm"
    raw_index: int        # index into PressSensorState_.pressure[0:12]
    link_name: str         # sim body/link name this taxel's force is read from
    xy: tuple[float, float]  # position on the schematic hand diagram, axes-fraction (0..1)


# Order matches the real controller's decode order (thumb, middle, index, palm).
# xy positions lay out a simplified TOP-DOWN view of the right hand: palm
# roughly centered-low, thumb offset to the +x side, index/middle extending
# upward (see scripts/tactile_monitor.py for the outline this overlays).
TAXELS: list[Taxel] = [
    Taxel("thumb_0",  "thumb",  0, "right_hand_thumb_1_link",  (0.72, 0.34)),
    Taxel("thumb_1",  "thumb",  1, "right_hand_thumb_2_link",  (0.82, 0.22)),
    Taxel("middle_0", "middle", 2, "right_hand_middle_0_link", (0.53, 0.72)),
    Taxel("middle_1", "middle", 3, "right_hand_middle_1_link", (0.53, 0.90)),
    Taxel("index_0",  "index",  4, "right_hand_index_0_link",  (0.33, 0.68)),
    Taxel("index_1",  "index",  5, "right_hand_index_1_link",  (0.30, 0.87)),
    Taxel("palm_0",   "palm",   6, "right_hand_palm_link",     (0.50, 0.42)),
    Taxel("palm_1",   "palm",   7, "right_hand_palm_link",     (0.40, 0.50)),
    Taxel("palm_2",   "palm",   8, "right_hand_palm_link",     (0.56, 0.52)),
]

# Distinct sim link names the ContactSensor needs to expose (order-independent).
SIM_LINK_NAMES: list[str] = sorted({t.link_name for t in TAXELS})

DDS_TOPIC_HAND_STATE_RIGHT = "rt/dex3/right/state"   # matches real Dex3-1 SDK topic


def decode_pressure(raw: float) -> float | None:
    """Mirror dex3_controller.cpp's decode: None means 'no data' (sentinel)."""
    if raw is None or raw >= INVALID_RAW:
        return None
    return raw / PRESSURE_SCALE


def force_to_raw(force_n: float) -> float:
    """Sim-side encode: Newtons -> the same raw-count units the real firmware
    would put on the wire, so decode_pressure() above round-trips identically
    whether the message came from run_env.py or a real Dex3-1 hand."""
    return min(max(force_n, 0.0) * SIM_FORCE_TO_RAW_GAIN, INVALID_RAW - 1.0)
