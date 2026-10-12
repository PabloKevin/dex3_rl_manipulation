# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
The real Dex3-1 pads, as the grasp policy needs them (pure numpy, shared by
tactile_calibrate.py and toy_reach.py).

HandState_.press_sensor_state is a list of PressSensorState_ (nine on the real hand),
each with pressure[12]: 108 raw elements, about 100000 counts at rest on the ones that
report, the rest at a sentinel. Which element sits where is not documented, so
tactile_calibrate.py has the operator press each named place and records the elements
that answer. The toy then builds the nine TAXELS the policy was trained on
(scripts/tactile_layout.TAXELS: thumb x2, middle x2, index x2, palm x3) from those
elements, in newtons through a counts-per-newton gain.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

#: places the operator presses, in the order of the policy's taxels (thumb, middle, index, palm)
LOCATIONS = ["thumb_prox", "thumb_dist", "middle_prox", "middle_dist", "index_prox", "index_dist", "palm"]
LOCATION_HELP = {
    "thumb_prox": "the THUMB, the segment nearer the palm (its inner face / pad)",
    "thumb_dist": "the THUMB TIP pad",
    "middle_prox": "the MIDDLE finger, the segment nearer the palm (inner face)",
    "middle_dist": "the MIDDLE finger TIP pad",
    "index_prox": "the INDEX finger, the segment nearer the palm (inner face)",
    "index_dist": "the INDEX finger TIP pad",
    "palm": "the PALM (the flat part, centre)",
}
#: the policy's nine taxels -> the place each reads (the palm is one link in the sim: three equal taxels)
TAXEL_LOCATION = ["thumb_prox", "thumb_dist", "middle_prox", "middle_dist", "index_prox", "index_dist",
                  "palm", "palm", "palm"]
#: the sim's counts-per-newton placeholder (tactile_layout.SIM_FORCE_TO_RAW_GAIN)
DEFAULT_GAIN = 3000.0
#: readings at or below this are "absent" (g1_fruits_picker: values > 30001 are pads that report)
PRESENT_ABOVE = 30001.0
DEFAULT_CALIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "logs", "sim2real", "tactile_calib.json")


def flat_from_handstate(hs) -> np.ndarray | None:
    """All pressure elements of one HandState_, in message order (nan where unreadable)."""
    ps = getattr(hs, "press_sensor_state", None) if hs is not None else None
    if ps is None:
        return None
    out = []
    for e in ps:
        v = getattr(e, "pressure", None)
        if v is None:
            continue
        try:
            out.extend(float(x) for x in v)
        except TypeError:
            out.append(float(v))
    return np.array(out, float) if out else None


def entry_means(flat: np.ndarray, per_entry: int = 12) -> np.ndarray:
    """Legacy view: the mean of the reporting elements of each entry (the picker's own
    per-finger pressure), nan for an entry that reports nothing."""
    a = np.asarray(flat, float)
    n = len(a) // per_entry
    out = np.full(n, np.nan)
    for i in range(n):
        v = a[i * per_entry:(i + 1) * per_entry]
        v = v[np.isfinite(v) & (v > PRESENT_ABOVE)]
        if v.size:
            out[i] = v.mean()
    return out


def baseline(samples) -> tuple[np.ndarray, np.ndarray]:
    a = np.array(samples, float)
    with np.errstate(all="ignore"):
        return np.nanmedian(a, axis=0), np.nanstd(a, axis=0)


def present(base: np.ndarray) -> np.ndarray:
    return np.isfinite(base) & (base > PRESENT_ABOVE)


def window_rise(window, base: np.ndarray) -> np.ndarray:
    """Per element: the 90th percentile over the window of (reading - baseline), >= 0."""
    a = np.array(window, float) - base
    with np.errstate(all="ignore"):
        r = np.nanpercentile(a, 90, axis=0)
    return np.nan_to_num(np.maximum(r, 0.0))


def responders(rise: np.ndarray, base: np.ndarray, noise: np.ndarray, k_sigma: float = 8.0,
               min_counts: float = 400.0) -> dict[int, float]:
    """{element: rise} of the elements that answered: above k sigma of their own noise and
    min_counts, and reporting at rest."""
    thr = np.maximum(k_sigma * np.nan_to_num(noise), min_counts)
    ok = present(base) & (rise > thr)
    return {int(i): float(rise[i]) for i in np.flatnonzero(ok)}


class TactileCalib:
    """locations: {name: [element index, ...]}; gain: counts per newton (of the SUM of a
    location's element rises); base/noise: the resting values when it was made."""

    def __init__(self, locations=None, gain=DEFAULT_GAIN, base=None, noise=None, n_elements=None, notes=""):
        self.locations = {k: list(v) for k, v in (locations or {}).items()}
        self.gain = float(gain)
        self.base = None if base is None else np.asarray(base, float)
        self.noise = None if noise is None else np.asarray(noise, float)
        self.n_elements = n_elements
        self.notes = notes

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"locations": self.locations, "gain_counts_per_N": self.gain,
                       "base": None if self.base is None else np.nan_to_num(self.base, nan=-1).tolist(),
                       "noise": None if self.noise is None else np.nan_to_num(self.noise, nan=-1).tolist(),
                       "n_elements": self.n_elements, "notes": self.notes,
                       "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "TactileCalib":
        with open(path) as f:
            d = json.load(f)
        return cls(d["locations"], d.get("gain_counts_per_N", DEFAULT_GAIN), d.get("base"), d.get("noise"),
                   d.get("n_elements"), d.get("notes", ""))

    def force(self, flat: np.ndarray, base: np.ndarray) -> dict[str, float]:
        """Newtons per named location from one reading against `base`."""
        d = np.nan_to_num(np.asarray(flat, float) - base, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.maximum(d, 0.0)
        out = {}
        for k, idx in self.locations.items():
            idx = [i for i in idx if i < len(d)]
            out[k] = float(d[idx].sum() / self.gain) if idx else 0.0
        return out

    def taxels(self, flat: np.ndarray, base: np.ndarray) -> np.ndarray:
        """The policy's nine taxel forces [N] (a place that was not calibrated reads like its
        finger's other segment, or 0 when the whole finger is missing)."""
        f = self.force(flat, base)
        twin = {"thumb_prox": "thumb_dist", "thumb_dist": "thumb_prox", "middle_prox": "middle_dist",
                "middle_dist": "middle_prox", "index_prox": "index_dist", "index_dist": "index_prox"}
        out = []
        for loc in TAXEL_LOCATION:
            if self.locations.get(loc):
                out.append(f[loc])
            elif self.locations.get(twin.get(loc, "")):
                out.append(f[twin[loc]])
            else:
                out.append(0.0)
        return np.array(out)
