"""onboard_link.py -- the laptop end of the onboard reach runner (sim2real/onboard/rl_runner.py, on the Jetson).

run_toy.py --onboard (the single reach): everything before and after the policy stays on the laptop
(home path, detection, GO, return, voice, dashboard); the POLICY phase runs on the Jetson:

  1. the dashboard's ARM MODE switch -> RL (the daemon then keeps the arm, at the same gains, between owners)
  2. the toy says "bye": the daemon holds the arm where the toy left it (no owner)
  3. {"cmd":"start", ...} to the runner: it takes the arm (rl_enter) and runs the 50 Hz loop on the robot
  4. here, 50 Hz: the runner's hand phase -> the Dex3 (the hand stays the laptop's), heartbeat to the
     runner (silent laptop = the runner stops), the dashboard status, Ctrl-C / E-STOP -> "stop"
  5. the runner ends ("bye": the daemon holds the arm), the toy takes the arm back (rl_enter) and goes on.
The runner's per-step log stays on the Jetson (/tmp/rl_onboard/<id>.npz); fetch_log() copies it here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "onboard"))
import runner_config as RC  # noqa: E402

HB_HZ = 20.0
ROBOT = "unitree@192.168.123.164"
#: the runner's stdout on the robot (rl_onboard_robot.sh start)
RUNNER_LOG = "/tmp/rl_onboard.log"


def npz_name(policy_path: str) -> str:
    """The deployed name of a policy: <experiment>_<run>_<file>.npz (every export is 'policy.pt')."""
    p = os.path.abspath(policy_path)
    parts = p.split(os.sep)
    stem = os.path.splitext(parts[-1])[0]
    if "exported" in parts:
        i = parts.index("exported")
        return f"{parts[i - 2]}_{parts[i - 1]}_{stem}.npz"
    return f"{stem}.npz"


class OnboardLink:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        self._S = String_
        self.state, self.state_t = {}, None
        self.done = None
        self._pub = ChannelPublisher(RC.CTL_TOPIC, String_)
        self._pub.Init()
        self._sub = ChannelSubscriber(RC.STATE_TOPIC, String_)
        self._sub.Init(self._on_state, 10)
        self._hb = False
        threading.Thread(target=self._hb_loop, daemon=True, name="onboard_hb").start()

    def _on_state(self, msg):
        try:
            d = json.loads(str(msg.data))
        except Exception:
            return
        self.state, self.state_t = d, time.monotonic()
        if d.get("phase") == "DONE" or d.get("refused"):
            self.done = d

    def send(self, d):
        try:
            self._pub.Write(self._S(data=json.dumps(d)))
        except Exception:
            pass

    def _hb_loop(self):
        while True:
            if self._hb:
                self.send({"cmd": "hb"})
            time.sleep(1.0 / HB_HZ)

    def wait_idle(self, timeout_s: float = 3.0) -> dict:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self.send({"cmd": "ping"})
            time.sleep(0.2)
            if self.state.get("phase") == "IDLE" and time.monotonic() - (self.state_t or 0) < 1.0:
                return self.state
        return {}

    def start(self, d: dict):
        self.done = None
        self._hb = True
        self.send(dict(d, cmd="start"))

    def stop(self, why: str):
        for _ in range(3):
            self.send({"cmd": "stop", "why": why})
            time.sleep(0.02)

    def end(self):
        self._hb = False


def fetch_log(remote: str, local_dir: str) -> str:
    """scp the runner's episode log into the run folder (best effort, a few seconds)."""
    dst = os.path.join(local_dir, "onboard_" + os.path.basename(remote))
    if os.path.exists(remote):                       # loopback: the runner ran on this machine
        import shutil
        shutil.copy(remote, dst)
        return dst
    try:
        subprocess.run(["sshpass", "-p", os.environ.get("ROBOT_PASS", "123"), "scp", "-q", "-o",
                        "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=5",
                        f"{ROBOT}:{remote}", dst], timeout=15, check=True, capture_output=True)
        return dst
    except Exception:
        return ""


#: log copies running in the background (fetch_async): the run does not wait for scp between its steps
_FETCHES = []


def fetch_async(remote: str, local_dir: str, say=None) -> None:
    """fetch_log in a background thread (robot 2026-10-19: two scp per policy leg held every next step ~1 s;
    in loopback the robot copy waited out its 5 s timeout). join_fetches() collects them at the end."""
    def run():
        got = fetch_log(remote, local_dir)
        if got and say is not None:
            try:
                say("INFO", f"onboard log: {got}", fold_s=0.0)
            except Exception:
                pass
    th = threading.Thread(target=run, daemon=True, name="fetch_log")
    th.start()
    _FETCHES.append(th)


def join_fetches(timeout_s: float = 20.0) -> None:
    """Wait (at most timeout_s in all) for the background log copies; the runner's text log too."""
    t_end = time.monotonic() + timeout_s
    for th in list(_FETCHES):
        th.join(max(0.0, t_end - time.monotonic()))
    _FETCHES.clear()


def check_runner(toy, timeout_s: float = 20.0) -> str:
    """Right after the robot connects (before the arm moves): the runner answers and has the policy.
    -> '' or why not. The link is built here, so DDS discovery has the whole home phase too."""
    link = getattr(toy, "_onboard", None)
    if link is None:
        link = toy._onboard = OnboardLink()
    t0 = time.monotonic()
    st = link.wait_idle(timeout_s)
    if not st:
        return (f"the onboard runner does not answer in {timeout_s:.0f} s (DDS discovery, or it is not running: "
                f"sim2real/onboard/rl_onboard_robot.sh status)")
    name = npz_name(toy.args.policy)
    if name not in st.get("policies", []):
        return (f"{name} is not on the robot: sim2real/export_policy_npz.py --policy {toy.args.policy} --layout "
                f"{toy.layout}, then sim2real/onboard/deploy_rl_onboard.sh <that .npz>")
    toy.brief("INFO", f"onboard runner answers ({time.monotonic() - t0:.1f} s, {st.get('fk')} kinematics), "
                      f"policy {name} on the robot", fold_s=0.0)
    return ""


def run_policy_onboard(toy, extra=None, tick=None):
    """The POLICY phase on the Jetson -> (outcome, stopped), as policy_loop. extra: more request fields (the
    arm-pose toy's goals); tick: called every 20 ms while the runner drives (the arm-pose toy's grip settle)."""
    a, io = toy.args, toy.io
    say = toy.brief
    if toy.layout not in ("single", "single_tactile", "visible", "arm_pose"):
        return f"NOT STARTED: --onboard runs the single / visible reach and the arm-pose toy (layout {toy.layout})", True
    link = getattr(toy, "_onboard", None)
    if link is None:
        link = toy._onboard = OnboardLink()
    st = link.wait_idle(15.0)
    if not st:
        return "NOT STARTED: the onboard runner does not answer (DDS discovery, or it stopped)", True
    name = npz_name(a.policy)
    if name not in st.get("policies", []):
        return (f"NOT STARTED: {name} is not on the robot: sim2real/export_policy_npz.py --policy {a.policy} "
                f"--layout {toy.layout}, then sim2real/onboard/deploy_rl_onboard.sh <that .npz>"), True
    # 1. ARM MODE switch RL: the daemon keeps the arm between this toy and the runner
    if not io.rl_switch:
        say("INFO", "ARM MODE switch -> RL (the onboard runner takes the arm from the daemon's hold)", fold_s=0.0)
        t0 = time.monotonic()
        while not io.rl_switch and time.monotonic() - t0 < 3.0:
            io._ctl_send({"cmd": "switch", "to": "rl"})
            time.sleep(0.25)
        if not io.rl_switch:
            return "NOT STARTED: the daemon did not take the ARM MODE switch to RL", True
    # 2. the toy lets go: the daemon holds the arm where the toy's last target put it
    io.leave_rl()
    t0 = time.monotonic()
    while io._rl.get("owner") is not None and time.monotonic() - t0 < 1.5:
        time.sleep(0.02)
    # 3. start the runner
    ep_id = time.strftime("%Y%m%d_%H%M%S")
    version = int(getattr(a, "visible_version", 0) or 0) if toy.layout == "visible" \
        else int(getattr(a, "single_version", 1))
    req = {"id": ep_id, "layout": toy.layout, "policy": name, "version": version,
           "apple": np.asarray(toy.apple, float).tolist(), "apple_radius": a.apple_radius, "speed": a.speed,
           "kp": a.kp, "kd": a.kd, "time_limit": a.time_limit, "reach_tol": a.reach_tol,
           "after_reach": a.after_reach, "settle_s": a.settle_s, "hold_s": a.hold_s, "waist": a.waist,
           "waist_lim": toy.waist_lim, "palm_touch": getattr(a, "palm_touch", "virtual"),
           "q_lo": np.asarray(toy.lo).tolist(), "q_hi": np.asarray(toy.hi).tolist(),
           "joint_speed_stop": a.joint_speed_stop, "temp_stop": a.temp_stop, "hand_ramp_s": a.hand_ramp_s}
    arm_pose = toy.layout == "arm_pose"
    if arm_pose:                              # toy_arm_pose.py: its goals (or the plan made on the robot from these)
        req.update(stage_radius=toy.stage_radius_now(), min_stage_back=a.min_stage_back, apple_offset=a.apple_offset,
                   stage_via=bool(a.stage_via), fingers=a.fingers, waypoint_timeout=a.waypoint_timeout)
    req.update(extra or {})
    toy.set_phase("POLICY")
    toy.ctx["doing"] = "single reach policy ON THE JETSON"
    say("STEP", f"policy on the Jetson (episode {ep_id}, {name})")
    if arm_pose and not req.get("goals"):
        toy.hand_cup()                        # the policy trained with the hand in the reach's cup
        say("INFO", f"staging radius {req['stage_radius']:.2f} m", fold_s=0.0)
    link.start(req)
    toy.sched.reset()
    outcome, stopped, res = None, False, None
    t_status = 0.0
    try:
        while True:
            time.sleep(0.02)
            if tick is not None:
                tick()
            d = link.done
            if d is not None and (d.get("id") == ep_id or d.get("reply_to") == ep_id):
                res = d
                break
            if io.estop_why is not None:
                link.stop(f"E-STOP: {io.estop_why}")
            s = link.state
            if arm_pose and s.get("id") == ep_id:
                if s.get("phase") == "PLAN" and not getattr(toy, "_plan_said", None) == ep_id:
                    toy._plan_said = ep_id
                    say("INFO", f"plan (robot): {s.get('plan')}", fold_s=0.0)
                for ev in s.get("events") or []:
                    if ev not in getattr(toy, "_events_said", set()):
                        toy._events_said = getattr(toy, "_events_said", set()) | {ev}
                        say("STEP", ev, fold_s=0.0)
                if s.get("pocket_cm") is not None:
                    toy.ctx.update(gap=float(s["pocket_cm"]) / 100.0)
            elif s.get("id") == ep_id and "hand_phase" in s:
                toy.sched.phase = float(s["hand_phase"])
                toy.sched.latched = toy.sched.phase > 0.0
                if a.hand == "schedule":
                    toy.send_hand()
                toy.ctx.update(gap=(s.get("gap_cm") or 0) / 100.0, seat=(s.get("seat_mm") or 0) / 1000.0)
            if time.monotonic() - (link.state_t or 0) > 1.5:
                link.stop("no runner state on the laptop")
                outcome, stopped = "STOP: the onboard runner went silent", True
                break
            if time.monotonic() - t_status > 0.2:
                t_status = time.monotonic()
                toy.status(io.read())
    except KeyboardInterrupt:
        link.stop("Ctrl-C on the laptop")
        t0 = time.monotonic()
        while link.done is None and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)
        res = link.done
        outcome, stopped = "OPERATOR STOP (Ctrl-C)", True
        toy.halted = outcome
    finally:
        link.end()
    if res is not None and outcome is None:
        outcome = res.get("outcome") or f"NOT STARTED: {res.get('refused')}"
        stopped = bool(res.get("stopped", True)) if res.get("outcome") else True
    if res is not None:
        if res.get("q_cmd"):
            toy.q_sent = np.asarray(res["q_cmd"], float)
        if res.get("reach"):
            toy.reach_info = res["reach"]
        say("INFO", f"onboard: {res.get('steps')} steps at {res.get('hz')} Hz, {res.get('late')} late, compute max "
                    f"{res.get('compute_ms_max')} ms", fold_s=0.0)
        if res.get("log"):
            fetch_async(res["log"], toy.log_dir, say)
    # the runner's own text log (robot /tmp/rl_onboard.log): once, at the end of the run (run_toy.finish)
    toy._fetch_runner_log = True
    if stopped and "OPERATOR" not in str(outcome):
        toy.halted = outcome
    # 5. the toy takes the arm back from the daemon's hold (same gains) for what follows
    if io.estop_why is None:
        io._left = False
        io._tx_last = None
        io.exec_mode("qref")
        t0 = time.monotonic()
        while not io.rl_mine and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)
    toy.ctx["commanding"] = False
    return outcome, stopped
