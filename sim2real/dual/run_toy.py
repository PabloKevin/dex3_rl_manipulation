#!/usr/bin/env python3
"""run_toy.py -- the sim2real toys through the dual daemon's RL mode. The toy files are
imported unchanged; three of their pieces are swapped here:

  robot IO    PickerIO -> DualIO     (rt/arm_rl/*: RL mode, waist in the stream, heartbeat)
  safety      Watch -> LightWatch    (no per-step collision model: the daemon guards contact)
  prompts     one GO before the policy (ENTER here, or GO on the dashboard); the rest auto

  python3 sim2real/dual/run_toy.py single --policy P --speed 0.3 --target detection
  python3 sim2real/dual/run_toy.py reach  --policy P --speed 0.3 --target X Y Z
  python3 sim2real/dual/run_toy.py apple|grasp|chain ...     (as toy_apple / toy_grasp / toy_chain)

Own options (everything else goes to the toy):
  --verbose           the toy's full output in this terminal (default: only the short log here,
                      the full output in <run>/toy_full.log)
  --strict_envelope   refuse a target outside the policy's training envelope (default: run it,
                      with a warning -- an imprecise reach beats no motion)
The old toys (sim2real/toy_*.py) still talk to v20 directly and keep working.
"""
from __future__ import annotations

import atexit
import json
import os
import select
import sys
import time

import numpy as np

HERE =os.path.dirname(os.path.abspath(__file__))
SIM2REAL = os.path.dirname(HERE)
sys.path[:0] = [HERE, SIM2REAL]
import toy_reach as T  # noqa: E402
from brief_log import BriefLog  # noqa: E402
from dual_io import DualIO  # noqa: E402
from light_watch import LightWatch  # noqa: E402

R = T.R
C_CYN, C_OFF, C_YEL, C_RED = T.C_CYN, T.C_OFF, T.C_YEL, T.C_RED
STRICT = False
VERBOSE = False


class _Tee:
    """stdout -> the run's toy_full.log (the terminal keeps only the short log). Until the
    run's folder exists, lines are buffered; if the run dies before that, they are shown."""

    def __init__(self):
        self.fh = None
        self.buf = []
        atexit.register(self._dump)

    def attach(self, path):
        self.fh = open(path, "a", buffering=1)
        self.fh.write("".join(self.buf))
        self.buf = []

    def _dump(self):
        if self.fh is None and self.buf:
            sys.__stdout__.write("".join(self.buf))

    def write(self, s):
        if self.fh is None:
            self.buf.append(s)
        else:
            self.fh.write(s)
        return len(s)

    def flush(self):
        if self.fh is not None:
            self.fh.flush()

    def isatty(self):
        return False


def _fast_dds(args):
    """Robot as a unicast DDS peer for this process (no 20-35 s discovery wait)."""
    if args.mock or getattr(args, "dds_default", False):
        return
    try:
        cfg = json.load(open(args.config or os.path.join(args.picker_dir, "apple_picker_config.json")))
        net_if = str(cfg.get("net_if") or (cfg.get("full_move2") or {}).get("net_if", ""))
    except Exception:
        net_if = ""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(args.picker_dir)), "g1_dual"))
    from fast_dds import use_fast_dds
    if use_fast_dds(getattr(args, "robot_ip", "192.168.123.164"), getattr(args, "spdp_s", 1.0), net_if):
        print("  DDS: robot as a unicast peer, announced every 1 s (fast discovery)")


class DualToyMixin:
    def __init__(self, args):
        _fast_dds(args)
        if not STRICT and not args.force:
            args.force = True
            self._envelope_note = True
        self._finished = False
        self._outcome = None
        super().__init__(args)
        node = getattr(self.io, "node", None)
        self.brief = BriefLog(os.path.join(self.log_dir, "brief.log"), node=node, echo=not VERBOSE)
        if isinstance(sys.stdout, _Tee):
            sys.stdout.attach(os.path.join(self.log_dir, "toy_full.log"))
            sys.__stdout__.write(f"  full toy output: {os.path.join(self.log_dir, 'toy_full.log')}\n")
        self.io.brief = self.brief
        self.watch.brief = self.brief
        self.brief("INFO", f"{type(self).__mro__[2].__name__} [{self.io.name}] policy "
                           f"{os.path.basename(args.policy)} speed {args.speed}")
        if getattr(self, "_envelope_note", False):
            self.brief("INFO", "targets outside the training envelope run anyway (--strict_envelope to refuse)")

    # ---- the short log ----
    def say(self, msg, col=""):
        super().say(msg, col)
        kind = "STOP" if col == C_RED else "WARN" if col == C_YEL else "DONE" if col == T.C_GRN else "INFO"
        self.brief(kind, msg.strip())

    def set_phase(self, ph):
        super().set_phase(ph)
        self.brief("STEP", ph.lower().replace("_", " "), fold_s=0.0)

    def plan_home(self, q_from, s):
        """The straight path first (one body check, ~10 ms); the toy's full search of ~120
        candidate paths (~10 s with nothing streamed) only when the straight one collides."""
        traj = T.joint_path_traj([np.asarray(q_from, float), R.HOME_Q], self.args.move_speed)
        if self.path_check(traj, s)[0]:
            return (), "straight"
        self.brief.wait("searching a path home around the body (straight one collides)")
        try:
            return super().plan_home(q_from, s)
        finally:
            self.brief.wait(None)

    # ---- one GO ----
    def confirm(self, prompt) -> bool:
        if "RUN THE POLICY" not in prompt and "RUN THE CHAIN" not in prompt:
            print(f"{C_CYN}{prompt.strip()} [auto]{C_OFF}", flush=True)
            return True
        if self.args.mock or (self.args.yes and getattr(self.io, "loopback", False)):
            print(f"{C_CYN}{prompt.strip()} [auto: mock / loopback --yes]{C_OFF}", flush=True)
            return True
        return self.wait_go()

    def wait_go(self) -> bool:
        io = self.io
        t0 = time.monotonic()
        self.brief.wait("GO: ENTER in the launcher terminal, or GO on the dashboard")
        sys.__stdout__.write(f"{C_CYN}  READY. ENTER (or GO on the dashboard) = run the policy;  "
                             f"'q' + ENTER = abort{C_OFF}\n")
        sys.__stdout__.flush()
        q_hold = self.q_sent if self.q_sent is not None else R.HOME_Q
        try:
            while True:
                if io.estop_why is not None:
                    return False
                if getattr(io, "go_t", None) is not None and io.go_t > t0:
                    return True
                r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if r:
                    return sys.stdin.readline().strip().lower() != "q"
                err = self.stream([q_hold] * 5, check=False)   # keep the arm held at kp/kd of the policy
                if err:
                    self.say(f"  while waiting for GO: {err}", C_RED)
                    return False
        finally:
            self.brief.wait(None)

    def ask_retreat(self) -> bool:
        out, sys.stdout = sys.stdout, sys.__stdout__       # the question must reach the terminal
        try:
            return super().ask_retreat()
        finally:
            sys.stdout = out

    # ---- always a log, whatever ends the run ----
    def finish(self, outcome, restore_mode=True):
        if self._finished:
            return self._outcome
        self._finished = True
        self._outcome = outcome
        out = super().finish(outcome, restore_mode=restore_mode)
        o = str(outcome)
        kind = "DONE" if o.startswith(("REACHED", "HOME", "shadow")) else "END" if o.startswith("TIME LIMIT") else "STOP"
        self.brief(kind, o, fold_s=0.0)
        if not VERBOSE:
            sys.__stdout__.write(f"   log: {self.log_dir}\n")
        return out

    def run(self):
        try:
            return super().run()
        except SystemExit as e:
            if not self._finished and e.code not in (None, 0):
                self.finish(f"STOPPED: {e.code}")
            raise


def mix(cls):
    return type("Dual" + cls.__name__, (DualToyMixin, cls), {})


def main():
    global STRICT, VERBOSE
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in ("reach", "single", "apple", "grasp", "chain"):
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    kind, rest = argv[0], argv[1:]
    if "--strict_envelope" in rest:
        STRICT = True
        rest.remove("--strict_envelope")
    if "--verbose" in rest:
        VERBOSE = True
        rest.remove("--verbose")
    if not VERBOSE and not any(x in ("-h", "--help") for x in rest):
        sys.stdout = _Tee()
    # a raw rsl_rl checkpoint (model_N.pt) -> its TorchScript export (exported/policy_N.pt)
    for i, x in enumerate(rest):
        if x == "--policy" and i + 1 < len(rest):
            import re as _re
            m = _re.match(r"model_(\d+)\.pt$", os.path.basename(rest[i + 1]))
            if m:
                exp = os.path.join(os.path.dirname(rest[i + 1]), "exported", f"policy_{m.group(1)}.pt")
                if os.path.exists(exp):
                    sys.__stdout__.write(f"  --policy {rest[i + 1]} is a training checkpoint: using its export {exp}\n")
                    rest[i + 1] = exp
                else:
                    sys.exit(f"{rest[i + 1]} is a training checkpoint, not a TorchScript export, and {exp} does not "
                             f"exist: export it (scripts/export_policy_jit.py) or pass the exported .pt")
    T.PickerIO = DualIO
    T.Watch = LightWatch
    if kind == "single":
        import toy_reach_single as S
        S.ToySingle = mix(S.ToySingle)
        sys.argv = [sys.argv[0]] + rest
        return S.main()
    if kind == "chain":
        import toy_chain as CH
        CH.ChainToy = mix(CH.ChainToy)
        sys.argv = [sys.argv[0]] + rest
        return CH.main()
    defaults = {"apple": {"--target": ["detection"], "--waist": ["follow"]},
                "grasp": {"--target": ["detection"], "--waist": ["follow"]}}.get(kind, {})
    for flag, val in defaults.items():
        if flag not in rest:
            rest = rest + [flag] + val
    if kind == "grasp" and "--grasp" not in rest:
        rest.append("--grasp")
    T.Toy = mix(T.Toy)
    sys.argv = [sys.argv[0]] + rest
    return T.main()


if __name__ == "__main__":
    main()
