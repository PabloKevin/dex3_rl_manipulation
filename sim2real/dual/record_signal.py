"""record_signal.py -- tell the laptop's camera recorder when this toy is active.

g1_fruits_picker/apple_picker/frame_recorder.py (inside live_locator) records the camera frames
the laptop already receives while a toy is ACTIVE: from the moment the commanded arm target
starts changing (any phase: home path, policy, return, lowering) or an apple is clicked on the
dashboard, until STOP_AFTER_S (10 s) after the last such moment, plus a 5 s pre-roll.

This side only keeps the time of the last activity (two float assignments in send_arm's path);
a daemon thread writes <G1_LAUNCH_CTL>/record_ctl.json at most every WRITE_EVERY_S while that
time moves, never from the control loop:
    {"run_dir": <this toy's log folder>, "last_activity": <time.time()>, "pid": ..., "toy": ...}
The recordings land in <run_dir>/camera/NN/. No control folder (the toy was not started by
g1_launch.py, and APPLE_PICKER_RECORD_CTL is unset) or APPLE_PICKER_RECORD=0: nothing at all.
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

CTL_NAME = "record_ctl.json"
#: a joint target change above this (rad) counts as motion (the toys hold a fixed target when still)
MOTION_EPS = 2e-3
WRITE_EVERY_S = 0.5


class RecordSignal:
    def __init__(self):
        off = os.environ.get("APPLE_PICKER_RECORD", "1").strip().lower() in ("0", "false", "no", "off")
        ctl = os.environ.get("APPLE_PICKER_RECORD_CTL") or os.environ.get("G1_LAUNCH_CTL")
        self.path = None if off or not ctl or not os.path.isdir(ctl) else os.path.join(ctl, CTL_NAME)
        self.run_dir = None
        self.toy = None
        self._last_q = None
        self._activity = 0.0          # time.time() of the last motion / click
        self._written = 0.0
        if self.path is not None:
            threading.Thread(target=self._loop, daemon=True, name="record_signal").start()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def start(self, run_dir: str, toy: str) -> None:
        self.run_dir, self.toy = os.path.abspath(run_dir), toy

    def arm_target(self, q) -> None:
        """send_arm's hook: the commanded arm target (7)."""
        if self.path is None:
            return
        if self._last_q is None:                  # the first target: nothing to compare with yet
            self._last_q = q
        elif float(np.max(np.abs(q - self._last_q))) > MOTION_EPS:
            self._activity = time.time()
            self._last_q = q

    def poke(self) -> None:
        """Activity without motion (an apple clicked on the dashboard)."""
        if self.path is not None:
            self._activity = time.time()

    def _write(self) -> None:
        tmp = self.path + f".{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"run_dir": self.run_dir, "last_activity": self._activity, "pid": os.getpid(),
                       "toy": self.toy}, f)
        os.replace(tmp, self.path)
        self._written = self._activity

    def _loop(self) -> None:
        while True:
            time.sleep(WRITE_EVERY_S)
            if self.run_dir is not None and self._activity > self._written:
                try:
                    self._write()
                except OSError:
                    pass
