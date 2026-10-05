#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Warm start for AppleReachTree-v1 from a trained AppleReach-v1 (single-apple)
checkpoint: same network, observation grown 29 -> 37 (the 8-sector neighbour
clearance ring appended, envs/apple_reach_tree_env_cfg.py).

  - actor/critic first layers: the 29 trained input columns copied, the 8
    new ones ZERO -> the warm-started policy starts out acting exactly like
    the v1 reach and learns to use the ring from there;
  - observation normalizers: the 29 trained statistics kept, the new inputs
    start at mean 1 (= "free", what the ring reads most of the time), var
    0.05, and the sample count is cut so every statistic adapts to the tree
    within a few iterations (12 M samples froze them);
  - optimizer: fresh Adam state (the old moments have the wrong shapes),
    learning rate --lr; iteration counter back to 0.

Writes logs/rsl_rl/apple_reach_tree/<--name>/model_0.pt; train from it with
  python scripts/train.py --task AppleReachTree-v1 --headless --resume --load_run <--name> --checkpoint model_0.pt
Plain torch, no Isaac Sim needed.
"""
import argparse
import glob
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", default=os.path.join(REPO, "logs/rsl_rl/apple_grasp_reach/2026-10-01_19-17-35"),
               help="AppleReach-v1 run folder (its newest model_*.pt) or a .pt file; default = the reach the grasp task pins")
p.add_argument("--name", default="warmstart_v1")
p.add_argument("--new_inputs", type=int, default=8)
p.add_argument("--lr", type=float, default=1.0e-4)
p.add_argument("--min_std", type=float, default=0.3, help="exploration noise floor per action dim")
p.add_argument("--norm_count", type=float, default=1.0e5)
args = p.parse_args()

src = args.src
if os.path.isdir(src):
    src = max(glob.glob(os.path.join(src, "model_*.pt")), key=lambda f: int(f.rsplit("_", 1)[1][:-3]))
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
k = args.new_inputs
out = {}
for name, t in sd.items():
    if name in ("actor.0.weight", "critic.0.weight"):
        t = torch.cat([t, torch.zeros(t.shape[0], k)], dim=1)
    elif name.endswith("_obs_normalizer._mean"):
        t = torch.cat([t, torch.ones(1, k)], dim=1)
    elif name.endswith("_obs_normalizer._var"):
        t = torch.cat([t, torch.full((1, k), 0.05)], dim=1)
    elif name.endswith("_obs_normalizer._std"):
        t = torch.cat([t, torch.full((1, k), 0.05 ** 0.5)], dim=1)
    elif name.endswith("_obs_normalizer.count"):
        t = torch.tensor(args.norm_count, dtype=t.dtype)
    elif name == "std":
        t = t.clamp(min=args.min_std)
    out[name] = t
groups = ck["optimizer_state_dict"]["param_groups"]
for g in groups:
    g["lr"] = args.lr
dst_dir = os.path.join(REPO, "logs", "rsl_rl", "apple_reach_tree", args.name)
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, "model_0.pt")
torch.save({"model_state_dict": out, "optimizer_state_dict": {"state": {}, "param_groups": groups},
            "iter": 0, "infos": None}, dst)
print(f"[warmstart] {src}\n  -> {dst}  (inputs {sd['actor.0.weight'].shape[1]} -> {out['actor.0.weight'].shape[1]}, "
      f"std {[round(float(x), 2) for x in out['std']]}, lr {args.lr})")
