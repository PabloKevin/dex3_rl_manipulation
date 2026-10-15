"""dual_io.py -- toy_reach.PickerIO through the dual daemon's RL mode (g1_fruits_picker/g1_dual).

Same robot app, hands, detections and dashboard topics as PickerIO; only the ARM path
changes:
  exec_mode("qref") -> {"cmd":"rl_enter"} on rt/arm_rl/ctl (ack in ~10 ms, no 2 s blend)
  send_arm()        -> rt/arm_rl/cmd (arm + waist in ONE message; no set_waist ramps)
  heartbeat         -> {"cmd":"hb"} 20 Hz from its own thread; silence 0.5 s = the robot
                       goes back to ORIGINAL (v20 stiff hold)
  E-STOP            -> the dashboard / remote stop is the DAEMON's; this side sees it in
                       rt/arm_rl/state and stops sending
  close()           -> {"cmd":"bye"}: ORIGINAL again, for the picker
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import toy_reach as T  # noqa: E402

R = T.R
RL_CMD_TOPIC = "rt/arm_rl/cmd"
RL_CTL_TOPIC = "rt/arm_rl/ctl"
RL_STATE_TOPIC = "rt/arm_rl/state"
GO_TOPIC = "/apple_picker/go"
HB_HZ = 20.0


class DualIO(T.PickerIO):
    def __init__(self, args, shadow: bool):
        super().__init__(args, shadow)
        self.name = ("dual loopback" if self.loopback else "dual") + (" shadow" if shadow else "")
        self.src = f"toy-{os.getpid()}"
        self.brief = None                          # set by run_toy (BriefLog)
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdMsg
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        self._S = String_
        self._rl, self._rl_t = {}, None
        self._rl_sub = ChannelSubscriber(RL_STATE_TOPIC, String_)
        self._rl_sub.Init(self._on_rl, 10)
        self._hb_on = False
        self._entered = False
        self._estop_seen = False
        self.go_t = None
        self.node.create_subscription(self.String, GO_TOPIC, lambda m: setattr(self, "go_t", time.monotonic()), 10)
        if not shadow:
            self._ctl = ChannelPublisher(RL_CTL_TOPIC, String_)
            self._ctl.Init()
            self._rlpub = ChannelPublisher(RL_CMD_TOPIC, LowCmd_)
            self._rlpub.Init()
            self._rlcmd = LowCmdMsg()
            threading.Thread(target=self._hb_loop, daemon=True, name="rl_hb").start()

    # ---------------- the daemon's RL state ----------------
    def _on_rl(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        self._rl, self._rl_t = d, time.monotonic()
        if self._entered and not self._estop_seen and d.get("mode") in ("FREEZE", "RELEASE"):
            self._estop_seen = True
            e = d.get("estop") or {}
            self._on_estop(f"{e.get('why') or d.get('mode')}", str(e.get("src") or "robot"))
        if self.brief is not None and d.get("refused") and d.get("reply_to") == self.src:
            self.brief("WARN", f"RL mode refused by the robot: {d['refused']}")

    @property
    def rl_mine(self) -> bool:
        d = self._rl
        return (d.get("mode") == "RL" and d.get("owner") == self.src and self._rl_t is not None
                and time.monotonic() - self._rl_t < 1.0)

    @property
    def qref_waist(self) -> bool:
        return self._rl_t is not None               # the dual daemon takes the waist in the RL stream

    @property
    def waist_src(self):
        d = self._rl
        return "qref" if self.rl_mine and d.get("cmd_age") is not None and d["cmd_age"] < 0.25 else None

    def _ctl_send(self, d):
        if self.commands:
            d = dict(d, src=self.src)
            try:
                self._ctl.Write(self._S(data=json.dumps(d)))
            except Exception:
                pass

    def _hb_loop(self):
        while True:
            if self._hb_on:
                self._ctl_send({"cmd": "hb"})
            time.sleep(1.0 / HB_HZ)

    def _on_estop_topic(self, msg):
        """The dashboard's old /apple_picker/estop: forwarded to the daemon's RL E-STOP."""
        self._ctl_send({"cmd": "estop", "why": "dashboard STOP (/apple_picker/estop)"})

    # ---------------- PickerIO overrides ----------------
    def connect(self):
        b = self.brief or (lambda *_a, **_k: None)
        t0 = time.monotonic()
        b("WAIT", "rt/lowstate (robot motors)")
        if not self.app.wait_ready(timeout_s=8.0):
            sys.exit("no rt/lowstate -- check the cable / network (student_setup/LOCAL_SETUP_this_laptop.md)")
        if not self.shadow:
            self.app.start()
            if self.args.hand != "keep":
                self.node._set_hand_output_enabled(True, reason="rl_toy_dual")
                self.node._startup_force_open_hands()
        b("WAIT", "the dual arm daemon (rt/arm_rl/state)")
        t1 = time.monotonic()
        while self._rl_t is None and time.monotonic() - t1 < 25.0:
            if self.commands:
                self._ctl_send({"cmd": "ping"})
            time.sleep(0.1)
        if self._rl_t is None:
            if self.arm.wait_ready(timeout_s=1.0):
                sys.exit("the robot runs the plain v20 daemon (no RL mode): launch with --daemon dual "
                         "(apple_picker/g1_launch.sh), or use the original toys")
            sys.exit("no arm daemon answers: is it running? (apple_picker/g1_launch.sh starts it)")
        t2 = time.monotonic()
        while self.app.right_hand_state is None and time.monotonic() - t2 < 3.0:
            time.sleep(0.05)
        if self.app.right_hand_state is None:
            b("WARN", "no rt/dex3/right/state: the hand is not reporting (hand_q will be NaN)")
        d = self._rl
        b("INFO", f"robot ready in {time.monotonic() - t0:.1f}s: daemon {d.get('dual')} mode {d.get('mode')}, "
                  f"remote {'seen' if (d.get('remote') or {}).get('seen') else 'NOT seen (dashboard STOP only)'}")
        if d.get("mode") in ("FREEZE", "RELEASE"):
            sys.exit(f"the robot is in E-STOP ({(d.get('estop') or {}).get('why')}): press RESET on the dashboard")

    def read(self):
        s = super().read()
        mode = s.mode
        if self.commands and self._entered:
            mode = "qref" if self.rl_mine else str(self._rl.get("mode", "?")).lower()
        return dataclasses.replace(s, mode=mode)

    def exec_mode(self, mode):
        if not self.commands or self.estop_why is not None:
            return
        if mode == "qref":
            self._entered = True
            self._hb_on = True
            self._ctl_send({"cmd": "rl_enter"})
        else:
            self._ctl_send({"cmd": "rl_exit"})

    def clear_freeze(self, q_here, timeout_s=6.0) -> bool:
        """v20's jump-freeze is cleared by entering RL mode (the daemon resets it)."""
        t0 = time.monotonic()
        self.exec_mode("qref")
        while time.monotonic() - t0 < timeout_s:
            if self._dstate.get("frozen") is False:
                return True
            time.sleep(0.05)
        return False

    def send_arm(self, q, dq=None, waist=None, waist_dq=0.0):
        if not self.commands or self.estop_why is not None:
            return
        dq = np.zeros(7) if dq is None else dq
        c = self._rlcmd.motor_cmd
        for j, n in enumerate(R.RIGHT_ARM_JOINTS):
            m = c[R.MOTOR_IDX[n]]
            m.mode = 1
            m.q, m.dq, m.tau = float(q[j]), float(dq[j]), 0.0
            m.kp, m.kd = float(self.args.kp), float(self.args.kd)
        w = c[R.MOTOR_IDX["waist_yaw_joint"]]
        if waist is not None:
            w.mode, w.q, w.dq, w.tau, w.kp, w.kd = 1, float(waist), float(waist_dq), 0.0, 1.0, 0.0
        else:
            w.mode, w.kp = 0, 0.0
        self._rlpub.Write(self._rlcmd)
        self.streaming = True

    def close(self, restore_mode=True):
        self._hb_on = False
        if self.commands and self._entered and self.estop_why is None:
            for _ in range(3):                          # one lost packet must not leave RL mode
                self._ctl_send({"cmd": "bye"})
                time.sleep(0.05)
        super().close(restore_mode=False)
