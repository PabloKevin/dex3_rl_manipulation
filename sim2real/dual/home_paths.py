"""home_paths.py -- remembered joint paths HOME for the dual toys (run_toy.py plan_home).

The toy's search for a path home around the body (toy_reach.Toy.plan_home: ~120 candidate
via-point paths, each checked against the picker's body cover) takes ~10 s with the hand
behind the leg -- the same few starts come back run after run. Every path the search finds
is stored here (start pose, waist, via points); the next time the arm starts near a stored
start, the stored via points are tried first, nearest start first, and each one is RE-CHECKED
with the same rule from where the arm really is (path_check) before it is used. A stored path
is never trusted blindly: it only saves the search.

File: home_paths.json beside this module (a list, newest last, at most MAX_ENTRIES).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "home_paths.json")
#: rad (max over the 7 joints): a stored start this close to the current one is a candidate
NEAR_RAD = 0.6
#: rad: a new path from a start this close to a stored one replaces it
SAME_RAD = 0.05
#: rad: waist yaw difference allowed between the stored start and now
WAIST_RAD = 0.15
MAX_TRY = 6
MAX_ENTRIES = 200


class HomePaths:
    def __init__(self, path: str = PATH):
        self.path = path
        self.entries = []
        try:
            with open(path) as f:
                self.entries = [e for e in json.load(f) if "q_from" in e and "via" in e]
        except Exception:
            self.entries = []

    def candidates(self, q_from, waist: float):
        """Stored via tuples, nearest start first (within NEAR_RAD and WAIST_RAD)."""
        q = np.asarray(q_from, float)
        scored = []
        for e in self.entries:
            d = float(np.max(np.abs(np.asarray(e["q_from"], float) - q)))
            if d <= NEAR_RAD and abs(float(e.get("waist", 0.0)) - waist) <= WAIST_RAD:
                scored.append((d, e))
        scored.sort(key=lambda x: x[0])
        return [(tuple(np.asarray(v, float) for v in e["via"]), d, e) for d, e in scored[:MAX_TRY]]

    def store(self, q_from, waist: float, via, name: str = "") -> None:
        q = np.asarray(q_from, float)
        keep = [e for e in self.entries
                if float(np.max(np.abs(np.asarray(e["q_from"], float) - q))) > SAME_RAD]
        keep.append({"q_from": np.round(q, 4).tolist(), "waist": round(float(waist), 4),
                     "via": [np.round(np.asarray(v, float), 4).tolist() for v in via],
                     "name": name, "t": time.strftime("%Y-%m-%d %H:%M:%S"), "used": 0})
        self.entries = keep[-MAX_ENTRIES:]
        self._save()

    def used(self, entry) -> None:
        entry["used"] = int(entry.get("used", 0)) + 1
        entry["t_used"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.entries, f, indent=0)
            os.replace(tmp, self.path)
        except Exception:
            pass
