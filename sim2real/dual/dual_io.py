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

import collections
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
sys.path.insert(0, HERE)
from record_signal import RecordSignal  # noqa: E402

R = T.R
RL_CMD_TOPIC = "rt/arm_rl/cmd"
RL_CTL_TOPIC = "rt/arm_rl/ctl"
RL_STATE_TOPIC = "rt/arm_rl/state"
GO_TOPIC = "/apple_picker/go"
SELECT_TOPIC = "/apple_picker/select_fruit"      # the dashboard's click (torso frame), as the picker uses it
CLOSER_TOPIC = "/apple_picker/reach_closer"      # {"cm": x} from the dashboard
STAGE_TOPIC = "/apple_picker/stage_radius"       # {"m": x} from the dashboard (the arm-pose toy's staging radius)
OFFSET_TOPIC = "/apple_picker/apple_offset"      # {"cm": [x, y, z]} from the dashboard (the arm-pose toy's goal offset)
HARVEST_CTL_TOPIC = "/apple_picker/harvest_ctl"  # {"cmd": "stop"} from the dashboard's END button (toy_harvest)
DETECTED_TOPIC = "/fruits_detected_torso"         # the detector's fruit, torso frame, with their picture boxes
CAMERA_INFO_TOPIC = "/unitree/color/camera_info"  # the head camera's picture size / intrinsics
HB_HZ = 20.0
VOICE_DIR = os.path.join(os.path.dirname(HERE), "voice")
#: speak() key -> WAV in sim2real/voice
EXTRA_CLIPS = {"hasta_la_vista": "hasta_la_vista_baby.wav"}
#: their volume_gain (the picker's clips play at 1.8): the easter egg is quieter
EXTRA_GAIN = {"hasta_la_vista": 0.9}
#: v20 grabs the arm with its own stiff hold (HOLD_KP 400 at its latch) when no target came for
#: QREF_STALE_S = 0.25 s; at the switch back to the stream's kp 100 the arm jumps by the sag
#: difference. Measured 2026-10-15 (session 15-27-04): 300-1474 ms gaps while the toy computed
#: body-clearance checks (run start, path home after a RELEASE, return start), each followed by
#: a v20 hold_latch. So the last target is repeated (zero-order hold, exactly what the daemon
#: plays between messages) whenever the main loop is silent for KEEP_GAP_S ...
KEEP_GAP_S = 0.06
#: ... but only this long after the main loop's last own send: a hung toy still loses the arm to
#: v20's hold, as before
KEEP_MAX_S = 5.0
#: arm_mpc_server.py (v20) HOLD_KP / HOLD_KD: the hold it latches after "bye"
V20_HOLD_KP = 400.0
#: dual_config.LOWER_S (the daemon's RESET lowering time; rl_mode.py)
LOWER_S = 2.0
V20_HOLD_KD = 12.0


class DualIO(T.PickerIO):
    def __init__(self, args, shadow: bool):
        super().__init__(args, shadow)
        self.name = ("dual loopback" if self.loopback else "dual") + (" shadow" if shadow else "")
        self.src = f"toy-{os.getpid()}"
        # the picker's own /apple_picker/target_fruit callback (inherited by the toy's node) speaks
        # through `audio`, which the toy never builds: a label message must not kill the ROS thread
        # The robot's voice: the picker's own player and clips (audio_msgs_config.json, robot speaker).
        # Shadow stays silent; any failure leaves a silent stand-in.
        self.audio_ok = False
        if not shadow and not hasattr(self.node, "audio"):
            try:
                self.node._init_audio()
                self.audio_ok = bool(getattr(self.node.audio, "enabled", False))
                if self.audio_ok:
                    # clips of this repo (sim2real/voice, 16 kHz mono 16-bit WAV, < 3 s), registered in
                    # this process only: the picker's audio_msgs_config.json is not touched
                    for key, fname in EXTRA_CLIPS.items():
                        path = os.path.join(VOICE_DIR, fname)
                        if os.path.isfile(path):
                            self.node.audio._msgs[key] = {"type": "wav", "file": path, "sample_rate_hz": 16000,
                                                          "channels": 1, "sample_width_bytes": 2,
                                                          "volume_gain": EXTRA_GAIN.get(key, 1.8)}
            except Exception as e:
                print(f"  audio: the picker's player failed ({e}): the toy stays silent")
        if not hasattr(self.node, "audio"):
            self.node.audio = type("NoAudio", (), {"say": lambda *_a, **_k: None,
                                                   "__getattr__": lambda _s, _n: (lambda *_a, **_k: None)})()
        self.brief = None                          # set by run_toy (BriefLog)
        #: the laptop's camera recorder (frame_recorder.py in live_locator): activity times only
        self.rec = RecordSignal()
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
        self._lowering = False                     # this process asked the daemon for the RESET lowering
        self.go_t = None
        self.node.create_subscription(self.String, GO_TOPIC, lambda m: setattr(self, "go_t", time.monotonic()), 10)
        #: the dashboard's click on a fruit: (torso-frame position, monotonic time) or None
        self.click = None
        self.node.create_subscription(self.String, SELECT_TOPIC, self._on_select, 10)
        #: "aim N cm past the apple" from the dashboard (None = not set there)
        self.closer_cm = None
        self.node.create_subscription(self.String, CLOSER_TOPIC, self._on_closer, 10)
        #: the dashboard's staging radius for the arm-pose toy, m (None = not set there)
        self.stage_radius = None
        self.node.create_subscription(self.String, STAGE_TOPIC, self._on_stage, 10)
        #: the dashboard's apple goal offset for the arm-pose toy, cm in the torso frame (None = not set there)
        self.apple_offset_cm = None
        self.node.create_subscription(self.String, OFFSET_TOPIC, self._on_offset, 10)
        #: monotonic time of the dashboard's last END (toy_harvest ends after the current step)
        self.harvest_stop_t = None
        self.node.create_subscription(self.String, HARVEST_CTL_TOPIC, self._on_harvest_ctl, 10)
        #: the detector's last boxes (toy_harvest: is the fruit whole in the picture?): (monotonic time,
        #: [(torso-frame xyz, [x1, y1, x2, y2] px)])
        self.det_boxes = (0.0, [])
        #: the detector's messages of the last ~2 s: [(monotonic t, [(torso xyz, box, class label)])]
        self.det_hist = collections.deque(maxlen=40)
        self.node.create_subscription(self.String, DETECTED_TOPIC, self._on_detected, 10)
        #: the head camera's {"w", "fx", "cx"} once heard (None = not yet)
        self.cam_info = None
        try:
            from sensor_msgs.msg import CameraInfo
            self.node.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_camera_info, 2)
        except Exception:
            pass
        self._tx_lock = threading.Lock()
        self._tx_t = self._main_t = 0.0
        self._tx_last = None                       # the last send_arm arguments (keep-alive repeats them)
        self.n_keep = 0                            # keep-alive repeats sent (logged by run_toy)
        if not shadow:
            self._ctl = ChannelPublisher(RL_CTL_TOPIC, String_)
            self._ctl.Init()
            self._rlpub = ChannelPublisher(RL_CMD_TOPIC, LowCmd_)
            self._rlpub.Init()
            self._rlcmd = LowCmdMsg()
            threading.Thread(target=self._hb_loop, daemon=True, name="rl_hb").start()
            threading.Thread(target=self._keep_loop, daemon=True, name="rl_keep").start()

    # ---------------- the daemon's RL state ----------------
    def _on_rl(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        self._rl, self._rl_t = d, time.monotonic()
        if d.get("mode") == "RL" and d.get("owner") == self.src:
            self._was_mine = True
        if (self._entered and getattr(self, "_was_mine", False) and not getattr(self, "_left", False)
                and not self._lowering and not self._estop_seen and d.get("mode") == "ORIGINAL"):
            # the dashboard's ARM MODE switch went to ORIGINAL (or the link was lost, switch off)
            self._estop_seen = True
            self._on_estop("the robot went back to ORIGINAL (ARM MODE switch on the dashboard, or link lost)",
                           "robot")
        if self._entered and not self._estop_seen and d.get("mode") in ("FREEZE", "RELEASE", "LOWER") \
                and not (self._lowering and d.get("mode") == "LOWER"):
            self._estop_seen = True
            e = d.get("estop") or {}
            self._on_estop(f"{e.get('why') or d.get('mode')}", str(e.get("src") or "robot"))
        if self.brief is not None and d.get("refused") and d.get("reply_to") == self.src:
            self.brief("WARN", f"RL mode refused by the robot: {d['refused']}")

    def _on_select(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        if d.get("clear"):
            self.click = None
        elif all(k in d for k in ("x", "y", "z")):
            self.click = (np.array([float(d["x"]), float(d["y"]), float(d["z"])]), time.monotonic())
            self.rec.poke()
            if self.brief is not None:
                self.brief("INFO", f"apple clicked on the dashboard at {np.round(self.click[0], 3).tolist()} (torso)")

    def _on_harvest_ctl(self, msg):
        try:
            if json.loads(msg.data).get("cmd") == "stop":
                self.harvest_stop_t = time.monotonic()
                if self.brief is not None:
                    self.brief("STOP", "END pressed on the dashboard: the harvest ends after the current step")
        except Exception:
            pass

    def _on_detected(self, msg):
        try:
            out = []
            for o in json.loads(msg.data).get("objects", []):
                c, bb = o.get("cartesian"), o.get("bbox")
                if c is not None and bb is not None and len(bb) >= 4:
                    out.append((np.array([float(c["x"]), float(c["y"]), float(c["z"])]), [float(v) for v in bb[:4]],
                                str(o.get("class_label", ""))))
            t = time.monotonic()
            self.det_boxes = (t, [(x, b) for x, b, _ in out])
            self.det_hist.append((t, out))
        except Exception:
            pass

    def _on_camera_info(self, msg):
        try:
            if msg.width > 0 and msg.k[0] > 0:
                self.cam_info = {"w": int(msg.width), "fx": float(msg.k[0]), "cx": float(msg.k[2])}
        except Exception:
            pass

    def _on_offset(self, msg):
        try:
            v = np.asarray(json.loads(msg.data)["cm"], float).reshape(3)
            if np.all(np.isfinite(v)):
                self.apple_offset_cm = v
        except Exception:
            pass

    def _on_stage(self, msg):
        try:
            self.stage_radius = float(json.loads(msg.data)["m"])
        except Exception:
            pass

    def _on_closer(self, msg):
        try:
            self.closer_cm = float(json.loads(msg.data)["cm"])
        except Exception:
            pass

    @property
    def rl_mine(self) -> bool:
        d = self._rl
        return (d.get("mode") == "RL" and d.get("owner") == self.src and self._rl_t is not None
                and time.monotonic() - self._rl_t < 1.0)

    @property
    def rl_switch(self) -> bool:
        """The dashboard's ARM MODE switch is on RL: the daemon keeps RL after this toy ends."""
        return self._rl.get("switch") == "rl"

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

    def speak(self, key, max_age_s=None, priority=False):
        """One of the picker's clips (audio_msgs_config.json key), queued; never blocks."""
        if not self.audio_ok or (self.estop_why is not None and not priority):
            return
        try:
            self.node.audio.say(key, priority=priority, max_age_s=max_age_s)
        except Exception:
            pass

    def speech_done(self, timeout_s):
        """Wait until the queue is empty and the clip playing has ended (or timeout_s). Idle
        must hold for 0.2 s: the worker takes a clip off the queue before it locks to play it."""
        if not self.audio_ok:
            return
        au = self.node.audio
        t_end = time.monotonic() + timeout_s
        idle_since = None
        while time.monotonic() < t_end:
            with au._q_lock:
                busy = bool(au._q) or au._play_lock.locked()
            if busy:
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= 0.2:
                return
            time.sleep(0.05)

    def _keep_loop(self):
        while True:
            time.sleep(0.02)
            now = time.monotonic()
            if (self._hb_on and self._entered and self.estop_why is None and self._tx_last is not None
                    and now - self._tx_t > KEEP_GAP_S and now - self._main_t < KEEP_MAX_S):
                self._write(*self._tx_last)
                self.n_keep += 1

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
        if d.get("mode") in ("FREEZE", "RELEASE", "LOWER"):
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

    def send_arm(self, q, dq=None, waist=None, waist_dq=0.0, kp=None, kd=None):
        """kp / kd: None = --kp / --kd (the policy's); only the hand-over to v20's hold changes them."""
        if not self.commands or self.estop_why is not None:
            return
        dq = np.zeros(7) if dq is None else np.asarray(dq, float).copy()
        kp = float(self.args.kp if kp is None else kp)
        kd = float(self.args.kd if kd is None else kd)
        self._main_t = time.monotonic()
        q = np.asarray(q, float).copy()
        self.rec.arm_target(q)
        self._write(q, dq, waist, waist_dq, kp, kd)

    def _write(self, q, dq, waist, waist_dq, kp, kd):
        with self._tx_lock:
            if not self.commands or self.estop_why is not None:
                return
            c = self._rlcmd.motor_cmd
            for j, n in enumerate(R.RIGHT_ARM_JOINTS):
                m = c[R.MOTOR_IDX[n]]
                m.mode = 1
                m.q, m.dq, m.tau = float(q[j]), float(dq[j]), 0.0
                m.kp, m.kd = kp, kd
            w = c[R.MOTOR_IDX["waist_yaw_joint"]]
            if waist is not None:
                w.mode, w.q, w.dq, w.tau, w.kp, w.kd = 1, float(waist), float(waist_dq), 0.0, 1.0, 0.0
            else:
                w.mode, w.kp = 0, 0.0
            self._rlpub.Write(self._rlcmd)
            self._tx_last = (q, dq, waist, waist_dq, kp, kd)
            self._tx_t = time.monotonic()
        self.streaming = True

    def leave_rl(self):
        """Hand the arm back to v20 (ORIGINAL). Idempotent: close() calls it too."""
        self._hb_on = False
        if self.commands and self._entered and self.estop_why is None and not getattr(self, "_left", False):
            self._left = True
            for _ in range(3):                          # one lost packet must not leave RL mode
                self._ctl_send({"cmd": "bye"})
                time.sleep(0.05)

    def lower_arm(self, on_descent=None, timeout_s=6.0):
        """The end of a Ctrl-C run: the daemon's RESET lowering (kp 0, light kd for LOWER_S, then held
        where it ends -- the same sink as the shutdown's damped release). on_descent(lower_s) is called
        once the arm starts going down (or at once when the daemon has no LOWER mode: an old deploy),
        with the seconds the lowering lasts. Returns when the arm is held again. Idempotent."""
        if not self.commands or getattr(self, "_lowered", False):
            if on_descent:
                on_descent(0.0)
            return False
        self._lowered = True
        self._lowering = True
        self._hb_on = False
        for _ in range(3):
            self._ctl_send({"cmd": "reset"})
            time.sleep(0.05)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.5 and self._rl.get("mode") != "LOWER":
            time.sleep(0.02)
        went = self._rl.get("mode") == "LOWER"
        if on_descent:
            on_descent(LOWER_S if went else 0.0)
        if went:
            while time.monotonic() - t0 < timeout_s and self._rl.get("mode") == "LOWER":
                time.sleep(0.05)
        return went

    def close(self, restore_mode=True):
        self.leave_rl()
        super().close(restore_mode=False)
