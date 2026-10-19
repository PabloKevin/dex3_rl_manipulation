#!/usr/bin/env python3
"""rl_runner.py -- the reach policy ON THE JETSON, as its own process beside the arm daemon.

Why: on the laptop the 50 Hz loop crossed the cable twice (rt/lowstate in, rt/arm_rl/cmd out) and
competed with everything else on the laptop. Here it reads rt/lowstate and writes the dual daemon's
RL command on the robot itself.

What it does NOT touch: the daemon (g1_fruits_picker/g1_dual, unchanged), v20, the motors directly
(only the daemon writes rt/arm_sdk), the hands (the laptop keeps the Dex3), the ORIGINAL picker.
While idle it sends NOTHING to the daemon: it only answers on its own topics.

  laptop (run_toy --onboard)                 Jetson
   ── rt/rl_onboard/ctl  {start ...} ──────►  rl_runner.py ── rt/arm_rl/ctl rl_enter / hb / bye ──► dual daemon
   ◄─ rt/rl_onboard/state {phase, hand ...}   50 Hz loop   ── rt/arm_rl/cmd (arm + waist)  ─────────►   (RL mode)
   ── rt/rl_onboard/ctl  {hb, stop, apple}──►              ◄─ rt/lowstate (polled, newest)
                                                           ◄─ rt/arm_rl/state (mode, owner)

One episode = the policy loop of sim2real/toy_reach_single.py (same reach_io.py code, deployed beside
this file): observation -> numpy policy -> rate limiter -> joint clip -> command, the hand schedule's
phase (sent to the laptop, which moves the hand), REACHED / TIME LIMIT / hold logic. It needs the
daemon's ARM MODE switch on RL (the laptop's toy sets it): taking the arm from the idle hold and
handing it back with "bye" then keep the same gains, no jump.

Stops (the episode ends, "bye", the daemon holds the arm at the last target, same gains):
  laptop silent > LAPTOP_TIMEOUT_S · rt/lowstate older than LOWSTATE_STALE_S · the daemon not in RL
  with this runner as owner (its E-STOP, the remote, a contact stop, the switch) · joint speed / motor
  temperature over the episode's limits · {"cmd":"stop"} · SIGTERM / SIGINT.

  python3 rl_runner.py --net-if eth0 [--urdf g1_29dof_with_hand.urdf] [--policy_dir policies]
Python 3.8 (the Jetson): numpy, pinocchio, unitree_sdk2py; no torch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]      # deployed: beside this file; in the repo: sim2real/
import reach_io as R  # noqa: E402  (dex3_rl_manipulation/sim2real/reach_io.py)
import runner_config as C  # noqa: E402
from np_policy import NumpyPolicy  # noqa: E402

#: reach_io's AppleReachSingle-v1 constants, to switch back after a v2 episode (use_single_v2 overwrites them)
SINGLE_V1 = {k: getattr(R, k) for k in R.SINGLE_V2}
ARM_IDX = [R.MOTOR_IDX[n] for n in R.RIGHT_ARM_JOINTS]
WAIST_IDX = [R.MOTOR_IDX[n] for n in R.WAIST_JOINTS]


def _reader_of(sub):
    ch = getattr(sub, "_ChannelSubscriber__channel", None)
    rd = getattr(ch, "_Channel__reader", None)
    return getattr(rd, "_Reader__reader", None)


class Lowstate:
    """rt/lowstate by polling (no Python in the DDS receive thread): the newest sample and the
    monotonic time it was taken."""

    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        self.msg, self.t = None, None
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init()
        self._reader = _reader_of(self._sub)
        if self._reader is None:
            raise RuntimeError("rt/lowstate: no DataReader inside the SDK subscriber (SDK changed?)")
        threading.Thread(target=self._loop, daemon=True, name="lowstate_poll").start()

    def _loop(self):
        from cyclonedds.internal import InvalidSample
        while True:
            try:
                got = [s for s in self._reader.take(N=32) if not isinstance(s, InvalidSample)]
            except Exception:
                got = []
            if got:
                self.msg, self.t = got[-1], time.monotonic()
            time.sleep(C.POLL_S)

    def age(self) -> float:
        return float("inf") if self.t is None else time.monotonic() - self.t


def make_kinematics(urdf: str, fk: str):
    """pinocchio (reach_io.G1Kinematics) when it imports, else np_fk (same frames, checked equal)."""
    if fk in ("auto", "pinocchio"):
        try:
            return R.G1Kinematics(urdf), "pinocchio"
        except Exception as e:
            if fk == "pinocchio":
                raise
            print(f"[rl_onboard] pinocchio unavailable ({type(e).__name__}: {e}): numpy kinematics", flush=True)
    from np_fk import NumpyKinematics
    return NumpyKinematics(urdf, R), "numpy"


class StopEpisode(Exception):
    pass


class Runner:
    def __init__(self, urdf: str, policy_dir: str, fk: str = "auto"):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        self.String_ = String_
        self.policy_dir = policy_dir
        self.kin, self.fk = make_kinematics(urdf, fk)
        self.policies = {}
        self.ls = Lowstate()
        self.lock = threading.Lock()
        self.pending = None               # a start request (dict) for the main thread
        self.stop_req = None              # why, while an episode runs
        self.laptop_t = None
        self.apple_update = None
        self.daemon, self.daemon_t = {}, None
        self.episode = None
        self.last = {}                    # the last episode's result (repeated in the idle state)
        self._run = True
        self.cmd_pub = ChannelPublisher(C.RL_CMD_TOPIC, LowCmd_)
        self.cmd_pub.Init()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.rl_ctl = ChannelPublisher(C.RL_CTL_TOPIC, String_)
        self.rl_ctl.Init()
        self.state_pub = ChannelPublisher(C.STATE_TOPIC, String_)
        self.state_pub.Init()
        self._pub_lock = threading.Lock()
        self._sub_d = ChannelSubscriber(C.RL_STATE_TOPIC, String_)
        self._sub_d.Init(self._on_daemon, 10)
        self._sub_c = ChannelSubscriber(C.CTL_TOPIC, String_)
        self._sub_c.Init(self._on_ctl, 10)

    # ---------------- DDS in ----------------
    def _on_daemon(self, msg):
        try:
            self.daemon, self.daemon_t = json.loads(str(msg.data)), time.monotonic()
        except Exception:
            pass

    def _on_ctl(self, msg):
        try:
            d = json.loads(str(msg.data))
        except Exception:
            return
        cmd = str(d.get("cmd", ""))
        self.laptop_t = time.monotonic()
        if cmd == "start":
            with self.lock:
                if self.episode is not None or self.pending is not None:
                    self.publish({"reply_to": d.get("id"), "refused": "an episode is already running"})
                    return
                self.pending = d
        elif cmd == "stop":
            self.stop_req = str(d.get("why") or "stop from the laptop")
        elif cmd == "apple":
            try:
                p = np.asarray(d.get("p"), float).reshape(3)
                if np.all(np.isfinite(p)):
                    self.apple_update = p
            except Exception:
                pass
        elif cmd == "ping":
            self.publish(self.idle_state())

    # ---------------- DDS out ----------------
    def publish(self, d):
        try:
            m = self.String_(data=json.dumps(d, separators=(",", ":")))
            with self._pub_lock:
                self.state_pub.Write(m)
        except Exception:
            pass

    def rl_send(self, d):
        d = dict(d, src=C.SRC)
        try:
            self.rl_ctl.Write(self.String_(data=json.dumps(d)))
        except Exception:
            pass

    def send_arm(self, q, kp, kd, waist=None, waist_dq=0.0):
        c = self.cmd.motor_cmd
        for j, i in enumerate(ARM_IDX):
            m = c[i]
            m.mode, m.q, m.dq, m.tau, m.kp, m.kd = 1, float(q[j]), 0.0, 0.0, float(kp), float(kd)
        w = c[R.MOTOR_IDX["waist_yaw_joint"]]
        if waist is not None:
            w.mode, w.q, w.dq, w.tau, w.kp, w.kd = 1, float(waist), float(waist_dq), 0.0, 1.0, 0.0
        else:
            w.mode, w.kp = 0, 0.0
        self.cmd_pub.Write(self.cmd)

    def idle_state(self):
        return {"phase": "IDLE", "runner": C.SRC, "fk": self.fk, "lowstate_age_ms": round(1000 * min(self.ls.age(), 9.999), 1),
                "daemon_mode": self.daemon.get("mode"), "daemon_owner": self.daemon.get("owner"),
                "policies": sorted(f for f in os.listdir(self.policy_dir) if f.endswith(".npz"))
                if os.path.isdir(self.policy_dir) else [], "last": self.last}

    # ---------------- main ----------------
    def serve(self):
        print(f"[rl_onboard] up ({self.fk} kinematics): idle, waiting for a start on {C.CTL_TOPIC} (sends nothing to the daemon "
              f"until then)", flush=True)
        t_state = 0.0
        while self._run:
            with self.lock:
                req, self.pending = self.pending, None
            if req is not None:
                self.run_episode(req)
            now = time.monotonic()
            if now - t_state >= 1.0 / C.STATE_HZ_IDLE:
                t_state = now
                self.publish(self.idle_state())
            time.sleep(0.01)

    def run_episode(self, req):
        ep = None
        try:
            ep = Episode(self, req)
        except Exception as e:
            self.publish({"reply_to": req.get("id"), "refused": f"{type(e).__name__}: {e}"})
            print(f"[rl_onboard] start refused: {e}", flush=True)
            return
        with self.lock:
            self.episode = ep
            self.stop_req = None
            self.apple_update = None
        try:
            ep.run()
        except Exception as e:
            ep.finish(f"RUNNER ERROR: {type(e).__name__}: {e}", stopped=True)
            traceback.print_exc()
        finally:
            for _ in range(3):                            # one lost packet must not keep the arm
                self.rl_send({"cmd": "bye"})
                time.sleep(0.02)
            self.last = ep.result
            self.publish(dict(ep.result, phase="DONE"))
            ep.save()
            with self.lock:
                self.episode = None
            print(f"[rl_onboard] episode {ep.id}: {ep.result.get('outcome')}", flush=True)


class Episode:
    def __init__(self, rn: Runner, d: dict):
        self.rn, self.d = rn, d
        self.id = str(d.get("id") or time.strftime("%Y%m%d_%H%M%S"))
        self.layout = str(d["layout"])
        if self.layout not in R.LAYOUT_DIMS:
            raise ValueError(f"unknown layout '{self.layout}'")
        name = os.path.basename(str(d["policy"]))
        path = os.path.join(rn.policy_dir, name)
        if path not in rn.policies:
            rn.policies[path] = NumpyPolicy(path, self.layout)
        self.pol = rn.policies[path]
        if (self.pol.n_in, self.pol.n_out) != R.LAYOUT_DIMS[self.layout]:
            raise ValueError(f"{name}: {self.pol.n_in}/{self.pol.n_out}, layout {self.layout} needs "
                             f"{R.LAYOUT_DIMS[self.layout]}")
        R.__dict__.update(R.SINGLE_V2 if int(d.get("single_version", 1)) == 2 else SINGLE_V1)
        self.apple = np.asarray(d["apple"], float).reshape(3)
        if not np.all(np.isfinite(self.apple)):
            raise ValueError("apple not finite")
        g = lambda k, v: d.get(k, v) if d.get(k) is not None else v  # noqa: E731
        self.radius = float(g("apple_radius", 0.04))
        self.speed = float(g("speed", 0.3))
        if not 0.05 <= self.speed <= 1.0:
            raise ValueError(f"speed {self.speed} outside 0.05..1")
        self.kp, self.kd = float(g("kp", R.ARM_KP)), float(g("kd", R.ARM_KD))
        self.time_limit = float(g("time_limit", 8.0 / self.speed))
        self.reach_tol = float(g("reach_tol", R.SINGLE_POS_TOL))
        self.after_reach = str(g("after_reach", "freeze"))
        self.settle_s, self.hold_s = float(g("settle_s", 0.5)), float(g("hold_s", 1.5))
        self.waist = str(g("waist", "hold"))
        self.waist_lim = float(g("waist_lim", R.WAIST_YAW_LIMIT))
        self.palm_touch = str(g("palm_touch", "virtual"))
        self.lo = np.asarray(g("q_lo", R.ARM_TARGET_LO), float)
        self.hi = np.asarray(g("q_hi", R.ARM_TARGET_HI), float)
        self.joint_speed_stop = float(g("joint_speed_stop", 1.8 * R.ARM_MAX_VELOCITY * self.speed + 0.5))
        self.temp_stop = float(g("temp_stop", C.TEMP_STOP_C))
        self.sched = R.SingleHandSchedule(ramp_s=float(g("hand_ramp_s", R.SINGLE_HAND_RAMP_S / self.speed)))
        self.L = R.make_limiters(self.layout)
        self.rows = []
        self.result = {"id": self.id, "outcome": None}
        self.reach_info = None

    # ---- robot state ----
    def read(self):
        rn = self.rn
        m = rn.ls.msg
        if m is None:
            raise StopEpisode("no rt/lowstate")
        age = rn.ls.age()
        if age > C.LOWSTATE_STALE_S:
            raise StopEpisode(f"rt/lowstate {age * 1000:.0f} ms old")
        ms = m.motor_state
        q = np.array([ms[i].q for i in ARM_IDX], float)
        dq = np.array([ms[i].dq for i in ARM_IDX], float)
        wq = np.array([ms[i].q for i in WAIST_IDX], float)
        wdq = np.array([ms[i].dq for i in WAIST_IDX], float)
        temp = np.array([max(ms[i].temperature[0], ms[i].temperature[1]) for i in ARM_IDX], float)
        return q, dq, wq, wdq, temp, age

    def check_daemon(self, need_owner=True):
        rn = self.rn
        d, t = rn.daemon, rn.daemon_t
        if t is None or time.monotonic() - t > C.DAEMON_STATE_STALE_S:
            raise StopEpisode("the dual daemon's state is not arriving")
        if d.get("mode") != "RL":
            raise StopEpisode(f"the daemon left RL (now {d.get('mode')}"
                              + (f": {(d.get('estop') or {}).get('why')}" if d.get("estop") else "") + ")")
        if need_owner and d.get("owner") != C.SRC:
            raise StopEpisode(f"the daemon's RL owner is now '{d.get('owner')}'")

    def acquire(self):
        rn = self.rn
        d = rn.daemon
        if rn.daemon_t is None or time.monotonic() - rn.daemon_t > C.DAEMON_STATE_STALE_S:
            raise StopEpisode("no dual daemon state (rt/arm_rl/state)")
        if d.get("switch") != "rl":
            raise StopEpisode("the ARM MODE switch is not on RL (needed: the hand-over keeps the gains)")
        if d.get("mode") != "RL" or d.get("owner") not in (None, C.SRC):
            raise StopEpisode(f"the daemon is in {d.get('mode')}, owner '{d.get('owner')}': the laptop must "
                              f"say bye first")
        t0 = time.monotonic()
        while time.monotonic() - t0 < C.ACQUIRE_TIMEOUT_S:
            rn.rl_send({"cmd": "rl_enter"})
            time.sleep(0.1)
            if rn.daemon.get("owner") == C.SRC and rn.daemon.get("mode") == "RL":
                return
        raise StopEpisode(f"the daemon did not hand over the arm (owner '{rn.daemon.get('owner')}')")

    # ---- the loop (toy_reach_single.ToySingle.policy_loop) ----
    def run(self):
        rn = self.rn
        rn.publish({"reply_to": self.id, "phase": "ACQUIRE"})
        try:
            self.acquire()
        except StopEpisode as e:
            self.finish(f"NOT STARTED: {e}", stopped=True)
            return
        dt = R.DT
        prev_raw = np.zeros(self.pol.n_out)
        self.sched.reset()
        reached_t = settled_since = None
        latched_t = None
        t_start = time.monotonic()
        t_next, t_state, t_hb = t_start, 0.0, 0.0
        n_late, compute_max = 0, 0.0
        q_cmd = w_cmd = None
        outcome, stopped = None, False
        try:
            while True:
                t_tick = time.monotonic()
                t = t_tick - t_start
                if rn.stop_req is not None:
                    raise StopEpisode(f"OPERATOR STOP: {rn.stop_req}")
                if rn.laptop_t is None or t_tick - rn.laptop_t > C.LAPTOP_TIMEOUT_S:
                    raise StopEpisode(f"the laptop went silent ({t_tick - (rn.laptop_t or 0):.2f} s)")
                self.check_daemon()
                if rn.apple_update is not None:
                    self.apple, rn.apple_update = rn.apple_update, None
                q, dq, wq, wdq, temp, age = self.read()
                fr = rn.kin.frames(wq, q)
                gap = float(np.linalg.norm(self.apple - fr.palm_ref)) - self.radius
                seat = R.single_seat_error(fr, self.apple, self.radius)
                was = self.sched.latched
                phase = self.sched.step(fr.palm_ref, self.apple)
                if self.sched.latched and not was:
                    latched_t = t
                touch = 0.0
                if self.layout == "single_tactile" and self.palm_touch == "virtual":
                    touch = float(R.virtual_touch_force(fr, self.apple, self.radius))
                st = R.ReachState(arm_q=q, arm_dq=dq, waist_q=wq, waist_dq=wdq, apple=self.apple,
                                  touch_force=touch, hand_phase=phase)
                obs, _ = R.build_obs(self.layout, rn.kin, st, prev_raw, fr)
                raw = self.pol(obs)
                tgt = self.L["arm"].step(raw[:7], q, self.speed)
                q_cmd = np.clip(tgt, self.lo, self.hi)
                hit = np.abs(q_cmd - tgt) > 1e-6
                if hit.any():                              # the limiter keeps what is really sent
                    self.L["arm"].prev = q_cmd.copy()
                    self.L["arm"].vel[hit] = 0.0
                w_cmd, w_dq = None, 0.0
                if "waist" in self.L and self.waist == "follow":
                    w_cmd = float(np.clip(self.L["waist"].step(raw[7:8], wq[:1], self.speed)[0],
                                          -self.waist_lim, self.waist_lim))
                    w_dq = float(self.L["waist"].vel[0])
                elif "waist" in self.L:
                    self.L["waist"].step(raw[7:8], wq[:1], self.speed)     # its state, as the toy keeps it
                jv = float(np.max(np.abs(dq)))
                if jv > self.joint_speed_stop:
                    raise StopEpisode(f"SAFETY STOP: joint speed {jv:.2f} rad/s > {self.joint_speed_stop:.2f}")
                if np.isfinite(temp).any() and float(np.nanmax(temp)) > self.temp_stop:
                    raise StopEpisode(f"SAFETY STOP: motor temperature {float(np.nanmax(temp)):.0f} C")
                rn.send_arm(q_cmd, self.kp, self.kd, w_cmd, w_dq)
                prev_raw = raw
                compute = time.monotonic() - t_tick
                compute_max = max(compute_max, compute)
                # reached (toy_reach_single): the seat within reach_tol, at once (freeze) or held still
                still = float(np.max(np.abs(dq))) < 0.08 * max(self.speed, 0.3) + 0.04
                if self.after_reach == "freeze":
                    arrived = seat <= self.reach_tol
                else:
                    if seat < self.reach_tol and still:
                        settled_since = settled_since if settled_since is not None else t
                    else:
                        settled_since = None
                    arrived = settled_since is not None and t - settled_since >= self.settle_s
                if arrived and reached_t is None:
                    reached_t = t
                    tilt = math.degrees(math.asin(float(np.clip(fr.palm_normal[2], -1, 1))))
                    self.reach_info = {"t": round(t, 3), "seat_err_mm": round(seat * 1000, 1),
                                       "surface_gap_mm": round(gap * 1000, 1), "tilt_deg": round(tilt, 1),
                                       "hand_phase": round(phase, 3), "palm": fr.palm_ref.round(4).tolist(),
                                       "q": q.round(4).tolist()}
                    if self.after_reach == "freeze":
                        outcome = f"REACHED in {t:.2f} s (seat error {seat * 1000:.1f} mm)"
                self.rows.append((t, obs, raw, q_cmd, q, dq, wq, np.nan if w_cmd is None else w_cmd,
                                  fr.palm_ref, seat, gap, phase, age, compute))
                now = time.monotonic()
                if now - t_hb >= 1.0 / C.DAEMON_HB_HZ:
                    t_hb = now
                    rn.rl_send({"cmd": "hb"})
                if now - t_state >= 1.0 / C.STATE_HZ_RUN or outcome is not None:
                    t_state = now
                    rn.publish({"phase": "HOLD" if reached_t is not None else "POLICY", "id": self.id,
                                "t": round(t, 3), "seat_mm": round(seat * 1000, 1), "gap_cm": round(gap * 100, 2),
                                "hand_phase": round(phase, 4), "hand_targets": self.sched.targets().round(4).tolist(),
                                "q": q.round(4).tolist(), "q_cmd": q_cmd.round(4).tolist(),
                                "waist_q": round(float(wq[0]), 4), "waist_cmd": w_cmd,
                                "palm": fr.palm_ref.round(4).tolist(), "lowstate_age_ms": round(age * 1000, 1),
                                "late": n_late, "compute_ms_max": round(compute_max * 1000, 2),
                                "clipped": [R.RIGHT_ARM_JOINTS[j][6:-6] for j in np.flatnonzero(hit)],
                                "reach": self.reach_info})
                if outcome is not None:
                    break
                if reached_t is not None and t - reached_t >= self.hold_s:
                    outcome = (f"REACHED in {reached_t:.2f} s, held by the policy {self.hold_s:.1f} s "
                               f"(seat error {seat * 1000:.1f} mm)")
                    break
                if t >= self.time_limit:
                    outcome = (f"TIME LIMIT {self.time_limit:.0f} s (seat error {seat * 1000:.1f} mm, hand "
                               f"{'closed at t=%.1f s' % latched_t if latched_t is not None else 'never closed'})")
                    break
                t_next += dt
                if time.monotonic() - t_next > dt:
                    n_late += 1
                    t_next = time.monotonic()
                time.sleep(max(0.0, t_next - time.monotonic()))
        except StopEpisode as e:
            outcome, stopped = str(e), True
        self.finish(outcome, stopped, q_cmd=q_cmd, n_late=n_late, compute_max=compute_max)

    def finish(self, outcome, stopped, q_cmd=None, n_late=0, compute_max=0.0):
        n = len(self.rows)
        dur = self.rows[-1][0] if n else 0.0
        self.result = {"id": self.id, "outcome": outcome, "stopped": bool(stopped), "steps": n,
                       "hz": round(n / dur, 1) if dur > 0 else None, "late": n_late,
                       "compute_ms_max": round(compute_max * 1000, 2), "reach": self.reach_info,
                       "q_cmd": None if q_cmd is None else np.asarray(q_cmd).round(5).tolist(),
                       "log": os.path.join(C.LOG_DIR, f"{self.id}.npz")}

    def save(self):
        if not self.rows:
            return
        try:
            os.makedirs(C.LOG_DIR, exist_ok=True)
            keys = ("t", "obs", "raw", "q_cmd", "q", "dq", "waist", "waist_cmd", "palm", "seat", "gap",
                    "hand_phase", "age", "compute")
            cols = list(zip(*self.rows))
            np.savez_compressed(self.result["log"], meta=json.dumps({"start": self.d, "result": self.result}),
                                **{k: np.asarray(v) for k, v in zip(keys, cols)})
        except Exception as e:
            print(f"[rl_onboard] log not saved: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="the reach policy on the Jetson (beside the dual daemon)")
    ap.add_argument("--net-if", default="eth0")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--urdf", default=os.path.join(HERE, "g1_29dof_with_hand.urdf"))
    ap.add_argument("--policy_dir", default=os.path.join(HERE, "policies"))
    ap.add_argument("--fk", choices=["auto", "pinocchio", "numpy"], default="auto")
    a = ap.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(a.domain_id, a.net_if)
    rn = Runner(a.urdf, a.policy_dir, a.fk)

    def _stop(signum, _frame):
        rn.stop_req = f"signal {signum}"
        rn._run = False
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _stop)
    rn.serve()
    print("[rl_onboard] exit", flush=True)


if __name__ == "__main__":
    main()
