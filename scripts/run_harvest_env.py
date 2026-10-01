#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Phase 3 -- view the HARVEST task (AppleHarvest-Play-v1), interactively.

Every episode starts from a ready grasp out of the start bank
(scripts/collect_harvest_starts.py). Press 'r' + Enter to reset all envs
without reloading Isaac Sim. One line is printed per finished episode:
how it ended, whether / how the stem broke, and whether the fruit stayed
in the hand.

Usage:
  python scripts/run_harvest_env.py --checkpoint logs/rsl_rl/apple_grasp_harvest/<run>/model_XXX.pt
  python scripts/run_harvest_env.py --policy <run>/exported/policy.pt
  python scripts/run_harvest_env.py --zero_actions      (arm holds still: checks the start bank itself)
  add --headless --episodes 32 for a quick numeric summary without the viewer
"""
import argparse
import os
import sys
import threading

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description="Harvest task viewer")
p.add_argument("--checkpoint", type=str, default=None)
p.add_argument("--policy", type=str, default=None)
p.add_argument("--zero_actions", action="store_true")
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--episodes", type=int, default=0, help="stop after this many finished episodes (0 = never)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if sum(x is not None and x is not False for x in (args.checkpoint, args.policy, args.zero_actions or None)) != 1:
    p.error("pass exactly one of --checkpoint / --policy / --zero_actions")
app = AppLauncher(args).app

import torch
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import envs  # noqa: F401
from envs.apple_grasp_harvest_env_cfg import AppleGraspHarvestEnvCfg_PLAY
from envs.mdp import harvest as hv

cfg = AppleGraspHarvestEnvCfg_PLAY()
cfg.scene.num_envs = args.num_envs
env = gym.make("AppleHarvest-Play-v1", cfg=cfg).unwrapped
dev = env.device

if args.checkpoint:
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from envs.agents.rsl_rl_ppo_cfg import AppleGraspHarvestPPORunnerCfg
    acfg = AppleGraspHarvestPPORunnerCfg()
    wrap = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
    runner = OnPolicyRunner(wrap, acfg.to_dict(), log_dir=None, device=dev)
    runner.load(args.checkpoint)
    _inf = runner.get_inference_policy(device=dev)
    policy = lambda: _inf(wrap.get_observations())
elif args.policy:
    _jit = torch.jit.load(args.policy, map_location=dev).eval()
    policy = lambda: _jit(env.observation_manager.compute()["policy"])
else:
    policy = lambda: torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=dev)

_reset_requested = False
_lock = threading.Lock()


def _listen():
    global _reset_requested
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if line.strip().lower() == "r":
            with _lock:
                _reset_requested = True


threading.Thread(target=_listen, daemon=True).start()

env.reset()
stem = env._harvest_stem
tally = {}
done_eps = 0
print("[harvest] press 'r' + Enter to reset all envs", flush=True)
while app.is_running():
    with torch.inference_mode():
        with _lock:
            manual, _reset_requested = _reset_requested, False
        if manual:
            env.reset()
            print("\n>>> MANUAL RESET <<<\n", flush=True)
        a = policy()
        # stem state BEFORE the step: an ending episode is reset inside step()
        broken, cause = stem.broken.clone(), stem.break_cause.clone()
        peak_pull = stem.peak_pull.clone()
        rerr = getattr(env, "_harvest_radius_err", torch.zeros(env.num_envs, device=dev)).clone()
        _, _, term, trunc, _ = env.step(a)
        broken |= stem.broken & ~(term | trunc)
        ended = (term | trunc).nonzero(as_tuple=True)[0].tolist()
        for i in ended:
            why = [n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[i])]
            why = ",".join(why) or "?"
            held = why == "time_out"
            key = ("stem broke" if bool(broken[i]) else "stem intact") + (", fruit held" if held else f", {why}")
            tally[key] = tally.get(key, 0) + 1
            print(f"  env {i}: {why:16s} stem {'BROKE (' + hv.BREAK_CAUSES[int(cause[i])] + ')' if broken[i] else 'intact':14s} "
                  f"peak pull {float(peak_pull[i]):4.1f} N  radius mismatch {float(rerr[i]) * 1000:4.1f} mm  -> {key}", flush=True)
            done_eps += 1
        if ended:
            print("  tally: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1])), flush=True)
        if args.episodes and done_eps >= args.episodes:
            break
sys.stdout.flush()
os._exit(0)
