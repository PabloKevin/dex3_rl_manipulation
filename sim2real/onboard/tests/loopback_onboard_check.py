#!/usr/bin/env python3
"""TEST ONLY (loopback 'lo', no robot): the onboard reach runner end to end.

Starts fake_g1_motors.py, the REAL dual daemon (--loopback) and rl_runner.py on 'lo', then plays the
laptop: ARM MODE switch RL, a "toy" owns the arm and lets go, the runner takes it, runs an episode,
gives it back. Checks: the hand-overs move nothing, ~50 Hz, the stops (laptop silence, dashboard
E-STOP), and replays the runner's log through the LAPTOP's code (TorchScript + pinocchio) to show the
onboard observation / action are the same numbers.

  ~/miniconda3/envs/g1_dex3_sim2real/bin/python sim2real/onboard/tests/loopback_onboard_check.py \
      --policy logs/rsl_rl/apple_reach_single_tactile/2026-10-16_16-30-26/exported/policy.pt --layout single_tactile
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONB = os.path.dirname(HERE)
SIM2REAL = os.path.dirname(ONB)
REPO = os.path.dirname(SIM2REAL)
PICKER = os.path.expanduser("~/g1_fruits_picker")
sys.path[:0] = [ONB, SIM2REAL, os.path.join(SIM2REAL, "dual")]

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, help="TorchScript policy.pt (exported to .npz here)")
ap.add_argument("--layout", default="single_tactile")
ap.add_argument("--version", type=int, default=2, help="the layout's version (single v1/v2, visible v1/v2)")
ap.add_argument("--fk", default="auto", choices=["auto", "pinocchio", "numpy"], help="the runner's kinematics")
a = ap.parse_args()

work = tempfile.mkdtemp(prefix="onboard_check_")
xml = os.path.join(work, "cdds_lo.xml")
open(xml, "w").write('<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS><Domain Id="any"><General><Interfaces>'
                     '<NetworkInterface name="lo" priority="default" multicast="default"/></Interfaces></General>'
                     '</Domain></CycloneDDS>')
env = dict(os.environ, CYCLONEDDS_URI="file://" + xml, PYTHONUNBUFFERED="1")
py = sys.executable
pol_dir = os.path.join(work, "policies")
os.makedirs(pol_dir)
from onboard_link import npz_name  # noqa: E402
NPZ = npz_name(a.policy)
subprocess.run([py, os.path.join(SIM2REAL, "export_policy_npz.py"), "--policy", a.policy, "--layout", a.layout,
                "--out", os.path.join(pol_dir, NPZ), "--n_check", "200"], check=True, env=env)
procs = []


def start(name, cmd):
    fh = open(os.path.join(work, name + ".log"), "w")
    procs.append((name, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)))


start("motors", [py, "-B", os.path.join(SIM2REAL, "fake_g1_motors.py"), "--net-if", "lo", "--duration", "600"])
time.sleep(1.5)
start("daemon", [py, "-B", os.path.join(PICKER, "g1_dual", "arm_mpc_server_dual.py"), "--net-if", "lo", "--loopback",
                 "--base", os.path.join(PICKER, "backup_pre_estop_2026-10-12", "onboard"),
                 "--extra-path", os.path.join(PICKER, "apple_picker", "onboard"),
                 "--urdf", os.path.join(PICKER, "robot_model_urdf", "g1_29dof_with_hand.urdf"),
                 "--log", os.path.join(work, "arm_dual.log")])
start("runner", [py, "-B", os.path.join(ONB, "rl_runner.py"), "--net-if", "lo", "--policy_dir", pol_dir,
                 "--urdf", os.path.join(PICKER, "robot_model_urdf", "g1_29dof_with_hand.urdf"), "--fk", a.fk])

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402

os.environ["CYCLONEDDS_URI"] = env["CYCLONEDDS_URI"]
ChannelFactoryInitialize(0, "lo")
import reach_io as R  # noqa: E402
from onboard_link import OnboardLink  # noqa: E402

S = {"rl": {}, "q": None, "qhist": []}
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({extra})" if extra else ""), flush=True)
    if not ok:
        fails.append(name)


def on_rl(m):
    S["rl"] = json.loads(m.data)


def on_ls(m):
    S["q"] = np.array([m.motor_state[i].q for i in range(22, 29)])
    S["w"] = np.array([m.motor_state[i].q for i in (12, 13, 14)])


ChannelSubscriber("rt/arm_rl/state", String_).Init(on_rl, 10)
ChannelSubscriber("rt/lowstate", LowState_).Init(on_ls, 1)
ctl = ChannelPublisher("rt/arm_rl/ctl", String_)
ctl.Init()
cmdp = ChannelPublisher("rt/arm_rl/cmd", LowCmd_)
cmdp.Init()
cmd = unitree_hg_msg_dds__LowCmd_()


def send(d, src="toy-test"):
    ctl.Write(String_(data=json.dumps(dict(d, src=src))))


def send_arm(q, kp=100.0, kd=10.0):
    for j, i in enumerate(range(22, 29)):
        m = cmd.motor_cmd[i]
        m.mode, m.q, m.dq, m.tau, m.kp, m.kd = 1, float(q[j]), 0.0, 0.0, kp, kd
    cmd.motor_cmd[12].mode, cmd.motor_cmd[12].kp = 0, 0.0
    cmdp.Write(cmd)


def wait(pred, t):
    t0 = time.time()
    while time.time() - t0 < t:
        if pred():
            return True
        time.sleep(0.02)
    return False


def toy_hold(seconds):
    """the 'toy' owns the arm and streams a hold at its current pose (as before GO)."""
    q0 = S["q"].copy()
    t0 = time.time()
    while time.time() - t0 < seconds:
        send({"cmd": "hb"})
        send_arm(q0)
        time.sleep(0.02)
    return q0


try:
    check("daemon up", wait(lambda: S["rl"].get("mode") == "ORIGINAL" and S["q"] is not None, 60.0),
          str(S["rl"].get("mode")))
    link = OnboardLink()
    st = link.wait_idle(10.0)
    check("runner idle + answers", bool(st), f"fk {st.get('fk')}, policies {st.get('policies')}")
    check("runner idle sends nothing to the daemon", S["rl"].get("mode") == "ORIGINAL")
    for _ in range(10):
        send({"cmd": "switch", "to": "rl"})
        time.sleep(0.1)
    check("switch RL", wait(lambda: S["rl"].get("switch") == "rl" and S["rl"].get("mode") == "RL", 3.0))
    send({"cmd": "rl_enter"})
    check("toy owns the arm", wait(lambda: S["rl"].get("owner") == "toy-test", 2.0))
    q_toy = toy_hold(1.5)
    for _ in range(3):
        send({"cmd": "bye"})
        time.sleep(0.03)
    check("toy let go (idle hold)", wait(lambda: S["rl"].get("owner") is None and S["rl"].get("mode") == "RL", 2.0))
    # an apple ~20 cm ahead of the palm, 5 cm up (pelvis frame)
    kin = R.G1Kinematics(os.path.join(PICKER, "robot_model_urdf", "g1_29dof_with_hand.urdf"))
    fr = kin.frames(S["w"], S["q"])
    apple = fr.palm_ref + np.array([0.20, -0.03, 0.05])
    req = {"id": "loop1", "layout": a.layout, "policy": NPZ, "version": a.version,
           "apple": apple.tolist(), "apple_radius": 0.04, "speed": 0.3, "time_limit": 8.0, "waist": "follow",
           "after_reach": "freeze"}
    q_before = S["q"].copy()
    link.start(req)
    check("runner owns the arm", wait(lambda: S["rl"].get("owner") == "rl_onboard", 3.0), str(S["rl"].get("owner")))
    # the first 0.1 s after the take-over: the arm must not jump (the runner starts from the held target)
    time.sleep(0.1)
    jump = float(np.max(np.abs(S["q"] - q_before)))
    check("take-over: no jump", jump < 0.02, f"{jump:.4f} rad in 0.1 s")
    check("episode done", wait(lambda: link.done is not None, 15.0), str(link.done and link.done.get("outcome")))
    res = link.done or {}
    link.end()
    print("   result:", json.dumps({k: res.get(k) for k in ("outcome", "steps", "hz", "late", "compute_ms_max")}))
    check("~50 Hz", res.get("hz") is not None and 45.0 <= res["hz"] <= 52.0, str(res.get("hz")))
    check("compute < 10 ms", (res.get("compute_ms_max") or 99) < 10.0, str(res.get("compute_ms_max")))
    check("bye: daemon idle hold", wait(lambda: S["rl"].get("owner") is None and S["rl"].get("mode") == "RL", 2.0))
    q_end = S["q"].copy()
    time.sleep(0.5)
    q_half = S["q"].copy()
    time.sleep(0.5)
    # the daemon holds the LAST command: an arm still moving at the end settles onto it, nothing jumps
    q_last = np.asarray(res.get("q_cmd") or q_end, float)
    drift = float(np.max(np.abs(S["q"] - q_half)))
    off = float(np.max(np.abs(S["q"] - q_last)))
    check("hand-back: arm settles on the last command", drift < 0.01 and off < 0.03,
          f"{drift:.4f} rad in the 2nd half second, {off:.4f} rad from the last command")
    moved = float(np.max(np.abs(q_end - q_toy)))
    check("the policy moved the arm", moved > 0.05, f"{moved:.3f} rad")

    # replay the runner's log through the LAPTOP code (TorchScript + pinocchio)
    log = np.load(os.path.join("/tmp/rl_onboard", "loop1.npz"), allow_pickle=False)
    import torch
    net = torch.jit.load(a.policy, map_location="cpu").eval()
    R.use_layout_constants(a.layout, a.version)
    sched = R.SingleHandSchedule(ramp_s=R.SINGLE_HAND_RAMP_S / 0.3)
    d_obs = d_raw = 0.0
    prev = np.zeros(log["raw"].shape[1])
    if a.layout == "arm_pose":                     # the plan as the runner made it, from the first step's pose
        import arm_pose_io as APIO
        APIO.use_constants()
        plan, _ = APIO.make_plan(kin.frames(log["waist"][0], log["q"][0]), apple)
        for k in range(len(log["t"])):
            frk = kin.frames(log["waist"][k], log["q"][k])
            obs = APIO.build_obs(log["q"][k], log["dq"][k], frk, plan[int(log["waypoint"][k])], prev)
            d_obs = max(d_obs, float(np.max(np.abs(obs - log["obs"][k]))))
            prev = log["raw"][k]
    for k in range(len(log["t"]) if a.layout != "arm_pose" else 0):
        w3 = log["waist"][k]
        frk = kin.frames(w3, log["q"][k])
        ph = sched.step(frk.palm_ref, apple)
        touch = float(R.virtual_touch_force(frk, apple, 0.04)) if a.layout == "single_tactile" else 0.0
        stk = R.ReachState(arm_q=log["q"][k], arm_dq=log["dq"][k], waist_q=w3, waist_dq=np.zeros(3), apple=apple,
                           touch_force=touch, hand_phase=ph)
        obs, _ = R.build_obs(a.layout, kin, stk, prev, frk)
        mask = np.ones_like(obs, bool)
        # the waist velocity is not in the log: skip its slot (the runner read it from lowstate)
        if a.layout in ("single", "single_tactile"):
            mask[-3] = False
        d_obs = max(d_obs, float(np.max(np.abs((obs - log["obs"][k])[mask]))))
        prev = log["raw"][k]
    for k in range(len(log["t"])):
        with torch.no_grad():
            raw = net(torch.as_tensor(log["obs"][k:k + 1].astype(np.float32)))[0].numpy()
        d_raw = max(d_raw, float(np.max(np.abs(raw - log["raw"][k]))))
    check("onboard obs = laptop obs", d_obs < 1e-4, f"max |diff| {d_obs:.1e} over {len(log['t'])} steps")
    check("onboard action = TorchScript", d_raw < 1e-4, f"max |diff| {d_raw:.1e}")

    # stop 1: the laptop goes silent
    link.start(dict(req, id="loop2"))
    wait(lambda: S["rl"].get("owner") == "rl_onboard", 3.0)
    time.sleep(0.5)
    link.end()                                          # no more heartbeats
    t0 = time.time()
    ok = wait(lambda: link.done is not None, 3.0)
    check("laptop silent -> runner stops", ok and "silent" in str(link.done.get("outcome")),
          f"{time.time() - t0:.2f} s: {link.done and link.done.get('outcome')}")
    # stop 2: the dashboard E-STOP (the daemon freezes; the runner must leave)
    link.start(dict(req, id="loop3"))
    wait(lambda: S["rl"].get("owner") == "rl_onboard", 3.0)
    time.sleep(0.5)
    send({"cmd": "estop", "why": "test", "id": 1}, src="dash-test")
    ok = wait(lambda: link.done is not None, 3.0)
    check("E-STOP -> FREEZE, runner stops", ok and S["rl"].get("mode") in ("FREEZE", "RELEASE"),
          f"{S['rl'].get('mode')}: {link.done and link.done.get('outcome')}")
    link.end()
    send({"cmd": "reset"}, src="dash-test")
    check("reset -> RL idle", wait(lambda: S["rl"].get("mode") == "RL" and S["rl"].get("owner") is None, 6.0),
          str(S["rl"].get("mode")))
    for _ in range(5):
        send({"cmd": "switch", "to": "original"})
        time.sleep(0.1)
    check("switch ORIGINAL", wait(lambda: S["rl"].get("mode") == "ORIGINAL", 4.0))
finally:
    for name, p in reversed(procs):
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
    time.sleep(3.0)
    for name, p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
    print(f"logs: {work}")
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
