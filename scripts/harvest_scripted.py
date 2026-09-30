#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Phase 3 (first approach) -- SCRIPTED harvest on top of the trained grasp.

Per env, a small state machine:
  REACH   frozen reach policy (inside the env, unchanged)
  GRASP   the trained grasp policy drives wrist + fingers; the arm stays
          frozen and the simulated stem never lets go, so the policy's
          grip_established observation stays 0 -- exactly as in the
          training episodes it knows (no retraining needed)
  READY   envs.mdp.harvest.GraspReadyDetector: closure + thumb pad
          pressure + settled fingers, held 0.5 s
  HARVEST grasp policy off; fingers HOLD their last joint targets (the hand
          PD keeps them there); a breakable stem (envs.mdp.harvest.
          BreakableStem) replaces the grasp phase's soft pin; the palm
          follows a lift-and-roll: a swing about the stem pivot plus a twist
          about the stem axis, tracked with differential IK on the 7 arm
          joints (wrist included)
  HOLD    keep the final pose for --hold_after s, then score the episode

Each env is assigned one swing direction from --modes (cycled), so one run
compares them: side+ / side- (sideways, either way), toward+ (toward the
robot and up), toward- (away).

Rounds: when every env is done, the round's table is printed and all envs
are reset (the reach hand-off runs again) for a new round, until --rounds
is reached (0 = forever). Press 'r' + Enter in this terminal at any time to
end the current round early and reset, without reloading Isaac Sim. A
running tally per swing direction is printed after each round.

Usage:
  python scripts/harvest_scripted.py --num_envs 4 --checkpoint logs/rsl_rl/apple_grasp_grasp/<run>/model_800.pt   (viewer, loops)
  python scripts/harvest_scripted.py --headless --rounds 1 --num_envs 16 --checkpoint ...                          (one table)
  python scripts/harvest_scripted.py --headless --policy <run>/exported/policy.pt
"""
import argparse
import os
import sys
import threading

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description="Scripted harvest on top of the grasp policy")
p.add_argument("--checkpoint", type=str, default=None, help="rsl_rl model_*.pt of a grasp run")
p.add_argument("--policy", type=str, default=None, help="JIT-exported grasp policy.pt (alternative to --checkpoint)")
p.add_argument("--num_envs", type=int, default=12)
p.add_argument("--modes", type=str, default="side+,side-,toward+,toward-")
p.add_argument("--swing_deg", type=float, default=90.0)
p.add_argument("--twist_deg", type=float, default=90.0)
p.add_argument("--duration", type=float, default=2.0, help="seconds for the swing+twist")
p.add_argument("--hold_after", type=float, default=1.0, help="seconds to hold the final pose")
p.add_argument("--grasp_timeout", type=float, default=5.0, help="seconds after hand-off to become READY")
p.add_argument("--rounds", type=int, default=0, help="rounds to run before exiting (0 = until the window is closed)")
p.add_argument("--trace", action="store_true", help="print per-step harvest diagnostics")
p.add_argument("--leave_margin", type=float, default=0.06, help="fruit this far from the grasp point = dropped")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if (args.checkpoint is None) == (args.policy is None):
    p.error("pass exactly one of --checkpoint / --policy")
app = AppLauncher(args).app

import torch
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import envs  # noqa: F401  (registers the gym tasks)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, axis_angle_from_quat
from envs.apple_grasp_grasp_env_cfg import AppleGraspGraspEnvCfg, GRASP_CENTER_LOCAL
from envs.mdp import harvest as hv
from envs.mdp.grasp_rewards import _max_link_apple_force

# ── env: the grasp task, with the stem never released and nothing ending episodes ──
cfg = AppleGraspGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 120.0
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
rp = cfg.events.reset_reach_grasp_phase.params
rp.update(stem_release_schedule=None, stem_release_time_range=None, release_on_latch=False, shaping_grace_s=None)
env = gym.make("AppleGraspGrasp-v0", cfg=cfg).unwrapped
N, dev, dt = env.num_envs, env.device, env.step_dt

# ── grasp policy ──
if args.policy:
    _jit = torch.jit.load(args.policy, map_location=dev).eval()
    policy = lambda: _jit(env.observation_manager.compute()["policy"])
else:
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from envs.agents.rsl_rl_ppo_cfg import AppleGraspGraspPPORunnerCfg
    acfg = AppleGraspGraspPPORunnerCfg()
    wrap = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
    runner = OnPolicyRunner(wrap, acfg.to_dict(), log_dir=None, device=dev)
    runner.load(args.checkpoint)
    _inf = runner.get_inference_policy(device=dev)
    policy = lambda: _inf(wrap.get_observations())

robot = env.scene["robot"]
apple = env.scene["apple"]
arm = env.action_manager.get_term("right_arm")
hand = env.action_manager.get_term("right_hand")
arm_dim = arm.action_dim
palm_id = robot.data.body_names.index("right_hand_palm_link")
sc = SceneEntityCfg("contact_forces"); sc.resolve(env.scene)

stem = hv.BreakableStem(env)
ready_det = hv.GraspReadyDetector(env, sc, hand._joint_ids)
ik = hv.DifferentialIK(robot, "right_hand_palm_link", arm._joint_ids)

# stem wrench every PHYSICS substep: piggyback on the hand term's apply_actions
_orig_apply = hand.apply_actions
def _apply_with_stem():
    _orig_apply()
    stem.apply()
hand.apply_actions = _apply_with_stem

modes = args.modes.split(",")
env_mode = [modes[i % len(modes)] for i in range(N)]

# ── per-env state (reset every round) ──
GRASP, HARVEST, HOLD, DONE = 0, 1, 2, 3


def new_round():
    global state, t_state, t_ready, hand_hold, palm0_p, palm0_q, swing_ax, break_t, break_swing, drop_t, max_in_hand, max_fruit_f, max_ik_pos, max_ik_rot, outcome
    state = torch.zeros(N, dtype=torch.long, device=dev)
    t_state = torch.zeros(N, device=dev)
    t_ready = torch.full((N,), float("nan"), device=dev)
    hand_hold = torch.zeros(N, hand.action_dim, device=dev)
    palm0_p = torch.zeros(N, 3, device=dev); palm0_q = torch.zeros(N, 4, device=dev)
    swing_ax = torch.zeros(N, 3, device=dev)
    break_t = torch.full((N,), float("nan"), device=dev); break_swing = torch.full((N,), float("nan"), device=dev)
    drop_t = torch.full((N,), float("nan"), device=dev)
    max_in_hand = torch.zeros(N, device=dev); max_fruit_f = torch.zeros(N, device=dev)
    max_ik_pos = torch.zeros(N, device=dev); max_ik_rot = torch.zeros(N, device=dev)
    outcome = ["" for _ in range(N)]
    stem.active[:] = False
    stem.broken[:] = False
    stem.break_cause[:] = 0
    for t in (stem.peak_pull, stem.peak_bend, stem.peak_twist):
        t[:] = 0.0
    ready_det.reset(slice(None))


new_round()
swing_rad, twist_rad, T = torch.deg2rad(torch.tensor(args.swing_deg)), torch.deg2rad(torch.tensor(args.twist_deg)), args.duration
c_local = torch.tensor(GRASP_CENTER_LOCAL, device=dev).expand(N, 3)


def fruit_to_hand():
    gp = robot.data.body_pos_w[:, palm_id] + quat_apply(robot.data.body_quat_w[:, palm_id], c_local)
    return torch.norm(apple.data.root_pos_w - gp, dim=-1)


fmt = lambda x, s="{:.2f}": "-" if x != x else s.format(x)


def report(rnd, early=False):
    print(f"\n===== ROUND {rnd}" + (" (ended early by 'r')" if early else "") + " =====")
    dmg = getattr(env, "_fruit_damage_force", torch.full((N,), float("nan"), device=dev))
    print("\n env mode     ready_s  stem   cause  at_swing  peak pull/bend/twist      held   drop_s  max_in_hand  max_fruit_F (dmg thr)  IK err pos/rot")
    for i in range(N):
        if not outcome[i] and int(state[i]) == GRASP:
            outcome[i] = "NOT READY (round cut)"
        elif not outcome[i] and int(state[i]) != DONE:
            outcome[i] = "INTERRUPTED"
        if not outcome[i]:
            held = torch.isnan(drop_t[i]).item()
            outcome[i] = ("BROKE+HELD" if stem.broken[i] else "HELD, STEM INTACT") if held else \
                         ("DROPPED after break" if stem.broken[i] else "DROPPED, stem intact")
        print(f" {i:3d} {env_mode[i]:8s} {fmt(t_ready[i].item()):>6s}  {'broke' if stem.broken[i] else 'intact':6s} "
              f"{hv.BREAK_CAUSES[int(stem.break_cause[i])]:6s} {fmt(break_swing[i].item(), '{:.0f}deg'):>7s}   "
              f"{stem.peak_pull[i]:5.1f}N {stem.peak_bend[i]:4.2f}Nm {stem.peak_twist[i]:4.2f}Nm   "
              f"{'yes' if torch.isnan(drop_t[i]) else 'NO':4s} {fmt(drop_t[i].item()):>6s}  {max_in_hand[i] * 100:6.1f} cm   "
              f"{max_fruit_f[i]:6.1f} N ({dmg[i]:.0f})       {max_ik_pos[i] * 100:4.1f} cm / {torch.rad2deg(max_ik_rot[i]):4.0f} deg"
              f"   -> {outcome[i]}")
    print("\nsummary:", dict(Counter(outcome)))
    for m in modes:
        ids = [i for i in range(N) if env_mode[i] == m]
        print(f"  {m:8s}: " + ", ".join(outcome[i] for i in ids))
        tally[m].update(outcome[i] for i in ids)
    print(f"\nrunning tally after {rnd} round(s):")
    for m in modes:
        print(f"  {m:8s}: " + ", ".join(f"{k} {v}" for k, v in tally[m].most_common()))
    sys.stdout.flush()


_reset_requested = False
_reset_lock = threading.Lock()


def _listen_terminal_reset():
    global _reset_requested
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if line.strip().lower() == "r":
            with _reset_lock:
                _reset_requested = True


threading.Thread(target=_listen_terminal_reset, daemon=True).start()

env.reset()
print(f"[harvest] robot fixed base: {robot.is_fixed_base}, jacobian shape {tuple(robot.root_physx_view.get_jacobians().shape)}, "
      f"palm body id {palm_id}, arm joint ids {arm._joint_ids}", flush=True)
print(f"[harvest] {N} envs, modes {modes}, swing {args.swing_deg} deg + twist {args.twist_deg} deg over {T} s", flush=True)
print("[harvest] press 'r' + Enter to end the round early and reset", flush=True)
from collections import Counter, defaultdict
tally = defaultdict(Counter)
rnd = 1
step = 0
while app.is_running():
    with torch.inference_mode():
        a = policy().clone()
        since_handoff = (env.episode_length_buf - env._grasp_handoff_step).float() * dt

        # GRASP -> READY -> HARVEST
        ready = ready_det.update() & (state == GRASP)
        timeout = (state == GRASP) & env._grasp_phase_active & (since_handoff > args.grasp_timeout)
        if bool(ready.any()):
            ids = ready.nonzero(as_tuple=True)[0]
            t_ready[ids] = since_handoff[ids]
            hand_hold[ids] = hand._processed_actions[ids]
            palm0_p[ids] = robot.data.body_pos_w[ids, palm_id]
            palm0_q[ids] = robot.data.body_quat_w[ids, palm_id]
            env._stem_released[ids] = True          # soft grasp pin off, arm unlocked
            stem.engage(ids)
            for i in ids.tolist():
                m = env_mode[i]
                swing_ax[i] = hv.swing_axis_for(m[:-1], apple.data.root_pos_w[i:i + 1], robot.data.root_pos_w[i:i + 1],
                                                1.0 if m.endswith("+") else -1.0)[0]
            state[ids] = HARVEST; t_state[ids] = 0.0
        for i in timeout.nonzero(as_tuple=True)[0].tolist():
            outcome[i] = "NOT READY"; state[i] = DONE

        # HARVEST / HOLD: override the policy's actions
        h = (state == HARVEST) | (state == HOLD)
        if bool(h.any()):
            frac = hv.smoothstep(t_state / T) * (state == HARVEST) + (state == HOLD).float()
            pos_d, quat_d = hv.harvest_palm_pose(palm0_p, palm0_q, stem.pivot, stem.axis, swing_ax,
                                                 swing_rad * frac, twist_rad * frac)
            q_des = ik.step(pos_d, quat_d, base=arm._processed_actions)
            raw_arm = (q_des - arm._anchor) / arm._scale
            raw_hand = (hand_hold - hand._grasp_offset) / hand._scale
            a[h, :arm_dim] = raw_arm[h]
            a[h, arm_dim:] = raw_hand[h]
            e_p, e_r = ik.pose_error(pos_d, quat_d)
            max_ik_pos = torch.where(h, torch.maximum(max_ik_pos, e_p.norm(dim=-1)), max_ik_pos)
            max_ik_rot = torch.where(h, torch.maximum(max_ik_rot, e_r.norm(dim=-1)), max_ik_rot)

        env.step(a)
        step += 1
        if args.trace and bool(h.any()):
            hi = h.nonzero(as_tuple=True)[0]
            e_p, e_r = ik.pose_error(pos_d, quat_d)
            dh = fruit_to_hand()
            for i in hi.tolist():
                if t_state[i] < 0.6 or (t_state[i] % 0.2) < dt:
                    print(f"   TRACE env{i} {env_mode[i]} st={int(state[i])} t={float(t_state[i]):.2f} swing={float(torch.rad2deg(swing_rad * frac[i])):.0f} "
                          f"pull={float(stem.pull[i]):.1f} bend={float(stem.bend[i]):.2f}({float(stem.bend_deg[i]):.0f}d) "
                          f"twist={float(stem.twist[i]):.2f}({float(stem.twist_deg[i]):.0f}d) broken={bool(stem.broken[i])} "
                          f"in_hand={float(dh[i])*100:.1f}cm ik_err={float(e_p[i].norm())*100:.1f}cm/{float(torch.rad2deg(e_r[i].norm())):.0f}d "
                          f"q-lim={[round(x,2) for x in (robot.data.joint_pos[i, arm._joint_ids] - robot.data.soft_joint_pos_limits[i, arm._joint_ids, 1]).tolist()]}",
                          flush=True)

        # bookkeeping (all on GPU, one sync per step at most)
        h = (state == HARVEST) | (state == HOLD)
        d_hand = fruit_to_hand()
        max_in_hand = torch.where(h, torch.maximum(max_in_hand, d_hand), max_in_hand)
        max_fruit_f = torch.where(h, torch.maximum(max_fruit_f, _max_link_apple_force(env, sc)), max_fruit_f)
        newly_broken = stem.broken & torch.isnan(break_t) & h
        break_t = torch.where(newly_broken, t_state, break_t)
        cur_swing = torch.rad2deg(swing_rad) * hv.smoothstep(t_state / T)
        break_swing = torch.where(newly_broken, cur_swing, break_swing)
        dropped = h & (d_hand > args.leave_margin) & torch.isnan(drop_t)
        drop_t = torch.where(dropped, t_state, drop_t)
        t_state = t_state + dt * h.float()
        to_hold = (state == HARVEST) & (t_state >= T)
        state = torch.where(to_hold, torch.full_like(state, HOLD), state)
        t_state = torch.where(to_hold, torch.zeros_like(t_state), t_state)
        finished = ((state == HOLD) & (t_state >= args.hold_after)) | (h & ~torch.isnan(drop_t))
        state = torch.where(finished, torch.full_like(state, DONE), state)

        if step % 50 == 0:
            print(f"  t={step * dt:5.1f}s  grasping={int((state == GRASP).sum())} harvesting={int((state == HARVEST).sum())} "
                  f"holding={int((state == HOLD).sum())} done={int((state == DONE).sum())}  "
                  f"stems broken={int(stem.broken.sum())}", flush=True)
            g = (state == GRASP) & env._grasp_phase_active
            if bool(g.any()):
                L = ready_det.last
                print(f"     in GRASP: closure {L['closure'][g].float().mean():.2f}  thumb>=min {L['thumb'][g].float().mean():.2f} "
                      f"(thumb {L['thumb_N'][g].median():.2f} N)  settled {L['settled'][g].float().mean():.2f} "
                      f"(finger span over window, median {L['max_dq'][g].median():.2f} rad)", flush=True)
        with _reset_lock:
            manual, _reset_requested = _reset_requested, False
        if manual or bool((state == DONE).all()):
            report(rnd, early=manual)
            if args.rounds and rnd >= args.rounds:
                break
            rnd += 1
            env.reset()
            new_round()
            step = 0
            print(f"\n>>> ROUND {rnd}: reset, reach hand-off running again <<<\n", flush=True)

sys.stdout.flush()
os._exit(0)

sys.stdout.flush()
os._exit(0)
