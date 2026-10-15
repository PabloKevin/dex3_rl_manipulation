"""brief_log.py -- the short log: one line per event, repeats folded, written as it happens
(a killed run still leaves its lines) to <run>/brief.log, and published on ROS
/apple_picker/brief for the dashboard's "Now" panel.

  12:03:04 [WAIT] the arm daemon (RL mode) ... 1.2 s
  12:03:05 [MOVE] policy started, 50 Hz
  12:03:09 [WARN] palm 14 cm further from the apple than at the start
  12:03:11 [DONE] REACHED in 5.8 s
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

TOPIC = "/apple_picker/brief"
_COL = {"STOP": "\033[1;31m", "WARN": "\033[1;33m", "DONE": "\033[1;32m", "WAIT": "\033[1;36m"}


def _colour(kind: str, line: str) -> str:
    c = _COL.get(kind)
    return f"{c}{line}\033[0m" if c and sys.__stdout__.isatty() else line


class BriefLog:
    def __init__(self, path: str | None = None, node=None, echo: bool = False):
        self.path = path
        self.echo = echo
        self._fh = None
        self._lock = threading.Lock()
        self._last = {}            # msg -> (t, n)
        self._pub = None
        self._String = None
        self.waiting = None        # (what, since) shown on the dashboard
        self.lines = []
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._fh = open(path, "a", buffering=1)
        if node is not None:
            self.attach(node)

    def attach(self, node) -> None:
        try:
            from std_msgs.msg import String
            self._String = String
            self._pub = node.create_publisher(String, TOPIC, 10)
        except Exception:
            self._pub = None

    def __call__(self, kind: str, msg: str, fold_s: float = 2.0) -> None:
        now = time.monotonic()
        with self._lock:
            t, n = self._last.get(msg, (-1e9, 0))
            if now - t < fold_s:
                self._last[msg] = (t, n + 1)
                return
            if n:
                self._emit("INFO", f"(previous line x{n + 1})")
            self._last[msg] = (now, 0)
            self._emit(kind, msg)

    def _emit(self, kind: str, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} [{kind}] {msg}"
        self.lines = (self.lines + [line])[-30:]
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
            except Exception:
                pass
        if self.echo:
            try:
                sys.__stdout__.write(_colour(kind, line) + "\n")
                sys.__stdout__.flush()
            except Exception:
                pass
        self._publish(kind, msg)

    def wait(self, what: str | None) -> None:
        """What the run is waiting for now (None = nothing): a dashboard field."""
        if what != (self.waiting[0] if self.waiting else None):
            self.waiting = None if what is None else (what, time.monotonic())
            if what:
                self("WAIT", what, fold_s=0.0)
            else:
                self._publish("INFO", "")

    def _publish(self, kind: str, msg: str) -> None:
        if self._pub is None:
            return
        try:
            m = self._String()
            w = None if self.waiting is None else {"what": self.waiting[0],
                                                   "for_s": round(time.monotonic() - self.waiting[1], 1)}
            m.data = json.dumps({"t": round(time.time(), 2), "kind": kind, "msg": msg, "waiting": w,
                                 "lines": self.lines[-10:]}, separators=(",", ":"))
            self._pub.publish(m)
        except Exception:
            pass
