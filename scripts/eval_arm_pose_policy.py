#!/usr/bin/env python3
"""Compare exported arm-pose policies on one task (default AppleArmPose-Play-v4: the robot's shoulder-roll limit
+0.0524 rad, the v2 holds) -- the env's own goal stream, deterministic actions, --seconds of sim per policy.
Reported from the env's episode logs: goals reached (share), the pocket error at a goal's timeout, hold share in
tolerance, episodes ended by a body collision.

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/eval_arm_pose_policy.py --headless \\
        --policies logs/rsl_rl/apple_arm_pose/2026-10-19_13-31-55/exported/policy.pt \\
                   logs/rsl_rl/apple_arm_pose_v4/<run>/exported/policy.pt
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="AppleArmPose-Play-v4")
parser.add_argument("--policies", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seconds", type=float, default=90.0)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import envs  # noqa: E402,F401

KEYS = {"reached": "Curriculum/metrics/reached_share", "timeout_err_cm": "Curriculum/metrics/timeout_err_cm",
        "hold_in_tol": "Curriculum/metrics/hold_in_tol_share", "collision_end": "Episode_Termination/collision"}


def run(policy_path):
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    pol = torch.jit.load(policy_path, map_location=env.device).eval()
    obs, _ = env.reset()
    acc = {k: [] for k in KEYS}
    steps = int(args.seconds / env.step_dt)
    for _ in range(steps):
        with torch.inference_mode():
            act = pol(obs["policy"])
        obs, _, _, _, extras = env.step(act)
        log = extras.get("log", {}) if isinstance(extras, dict) else {}
        for k, key in KEYS.items():
            if key in log:
                v = log[key]
                acc[k].append(float(v.item() if hasattr(v, "item") else v))
    env.close()
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def main():
    rows = []
    for p in args.policies:
        r = run(p)
        r["policy"] = p
        rows.append(r)
        print(f"[eval] {p}: reached {r['reached'] * 100:.1f}%, timeout error {r['timeout_err_cm']:.1f} cm, hold in tol "
              f"{r['hold_in_tol'] * 100:.0f}%, collision ends {r['collision_end'] * 100:.1f}%", flush=True)
    out = os.path.join(REPO, "logs", "sim2real", f"eval_arm_pose_{args.task}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
    app.close()
