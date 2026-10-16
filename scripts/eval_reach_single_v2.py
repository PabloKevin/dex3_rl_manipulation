#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Deterministic evaluation of a reach policy in AppleReachSingle-v2
(envs/apple_reach_single_v2_env_cfg.py): how often the apple ends ENCLOSED in
the hand, how far short / pushed, fruit knocked off, fingertip depth spread,
palm tilt, energy, self-collision -- split by FEASIBLE targets and STRETCH
targets (drawn just beyond the measured workspace).

  ~/miniconda3/envs/env_isaaclab/bin/python scripts/eval_reach_single_v2.py --headless \\
      --policy logs/rsl_rl/apple_reach_single_v2/<run>/model_1999.pt --episodes 512
  --policy accepts an rsl_rl checkpoint (exported here on the fly) or a TorchScript export.
  --stretch_frac 0.5: more stretch apples than training's 10%, for that question.
  --task: AppleReachSingle-v2 (default); any 40-input single-layout policy runs (v1 for a baseline).
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--policy", required=True)
p.add_argument("--task", default="AppleReachSingle-v2")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--episodes", type=int, default=256)
p.add_argument("--stretch_frac", type=float, default=None, help="override the stretch share")
p.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_grasp_env_cfg import PALM_LOCAL_OFFSET, PALM_NORMAL_AXIS  # noqa: E402
from envs.mdp import reach_rewards as rr  # noqa: E402
from envs.mdp import single_reach as sr  # noqa: E402
from envs.mdp.observations import apple_force_per_link  # noqa: E402
from envs.mdp.self_collision import self_collision_depth  # noqa: E402


def load_policy(path: str, device: str):
    try:
        return torch.jit.load(path, map_location=device)
    except RuntimeError:
        from export_policy_jit import _Policy
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        return _Policy(sd).to(device).eval()


def main():
    cfg = gym.spec(args.task).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.stretch_frac is not None:
        cfg.events.randomize_tree.params["seed_envelope"]["stretch_frac"] = args.stretch_frac
    env = ManagerBasedRLEnv(cfg)
    dev = env.device
    policy = load_policy(args.policy, dev)
    palm = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
    palm.resolve(env.scene)
    apple = SceneEntityCfg("apple")
    sensor = SceneEntityCfg("contact_forces")
    tips = ["right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"]
    soft = ["right_hand_palm_link", "right_hand_index_0_link", "right_hand_middle_0_link",
            "right_hand_thumb_0_link", "right_hand_thumb_1_link"]
    N = env.num_envs
    z = lambda v=0.0: torch.full((N,), v, device=dev)
    acc = {"min_gap": z(1e3), "max_soft_N": z(), "tip_touch": z(), "max_disp": z(), "max_coll": z(),
           "load": z(), "steps": z()}
    last = {}
    rows = []
    obs, _ = env.reset()
    n_neigh0 = None
    tm = env.termination_manager
    with torch.inference_mode():
        while len(rows) < args.episodes:
            # per-step state BEFORE the step: what an episode that ends now ended in
            gap, lat = sr.enclosure_state(env, palm, apple, PALM_LOCAL_OFFSET, PALM_NORMAL_AXIS)
            last = {"gap": gap.clone(), "lat": lat.clone(), "disp": sr.target_displacement(env),
                    "spread": sr.fingertip_depth_spread(env, apple),
                    "tilt": torch.rad2deg(rr._palm_tilt(env, palm, PALM_NORMAL_AXIS)),
                    "heading": torch.rad2deg(rr._palm_normal_angle(env, palm, PALM_NORMAL_AXIS, apple)),
                    "stretch": getattr(env, "_target_stretch", torch.zeros(N, dtype=torch.bool, device=dev)).clone()}
            if hasattr(env, "_tree"):
                n_neigh = torch.stack([s.broken for s in env._tree.stems[1:]], 1).sum(-1).float()
            else:
                n_neigh = z()
            act = policy(obs["policy"])
            obs, _, term, trunc, _ = env.step(act)
            robot = env.scene["robot"]
            # closest the apple came to the pocket: shortfall along the normal (pressing in counts 0) + off-centre
            acc["min_gap"] = torch.minimum(acc["min_gap"], torch.hypot(last["gap"].clamp(min=0.0), last["lat"]))
            acc["max_soft_N"] = torch.maximum(acc["max_soft_N"], apple_force_per_link(env, sensor, soft).sum(-1))
            acc["tip_touch"] = torch.maximum(acc["tip_touch"], (apple_force_per_link(env, sensor, tips) > 0.2).any(-1).float())
            acc["max_disp"] = torch.maximum(acc["max_disp"], last["disp"])
            acc["max_coll"] = torch.maximum(acc["max_coll"], self_collision_depth(env).amax(-1))
            arm_ids = env.action_manager.get_term("right_arm")._joint_ids
            rated = torch.tensor([25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0], device=dev)
            acc["load"] += ((robot.data.applied_torque[:, arm_ids] / rated) ** 2).sum(-1)
            acc["steps"] += 1
            done = (term | trunc).nonzero().squeeze(-1).tolist()
            succ = tm.get_term("reach_success")
            knocked = tm.get_term("target_knocked_off") if "target_knocked_off" in tm.active_terms else z().bool()
            for i in done:
                rows.append({
                    "stretch": bool(last["stretch"][i]), "success": bool(succ[i]), "knocked": bool(knocked[i]),
                    "neigh_knocked": float(n_neigh[i]),
                    "gap_cm": float(last["gap"][i]) * 100, "lat_cm": float(last["lat"][i]) * 100,
                    "min_gap_cm": float(acc["min_gap"][i]) * 100, "disp_cm": float(acc["max_disp"][i]) * 100,
                    "soft_N": float(acc["max_soft_N"][i]), "tip_touch": float(acc["tip_touch"][i]),
                    "spread_cm": float(last["spread"][i]) * 100, "tilt_deg": float(last["tilt"][i]),
                    "heading_deg": float(last["heading"][i]), "coll_mm": float(acc["max_coll"][i]) * 1000,
                    "load": float(acc["load"][i] / acc["steps"][i].clamp(min=1)),
                    "time_s": float(acc["steps"][i]) * env.step_dt,
                })
                for k in acc:
                    acc[k][i] = 1e3 if k == "min_gap" else 0.0
            if len(rows) and len(rows) % 64 < len(done):
                print(f"  {len(rows)} episodes", flush=True)

    def report(name, rs):
        if not rs:
            print(f"\n{name}: none")
            return
        a = {k: np.array([r[k] for r in rs], float) for k in rs[0]}
        enclosed = (a["gap_cm"] <= 0.8) & (a["gap_cm"] >= -1.5) & (a["lat_cm"] < 2.0)
        print(f"\n{name}: {len(rs)} episodes")
        print(f"  success {a['success'].mean():.0%}   enclosed at the end {enclosed.mean():.0%}   "
              f"target knocked off {a['knocked'].mean():.0%}   neighbours knocked / ep {a['neigh_knocked'].mean():.2f}")
        q = lambda v: f"{np.percentile(v, 10):6.1f} {np.median(v):6.1f} {np.percentile(v, 90):6.1f}"
        print("  (p10 / median / p90)")
        print(f"  final gap (cm, <0 pressed in)  {q(a['gap_cm'])}     closest to the pocket {q(a['min_gap_cm'])}")
        print(f"  final lateral off-centre (cm)  {q(a['lat_cm'])}")
        print(f"  max target displacement (cm)   {q(a['disp_cm'])}")
        print(f"  max palm/base force (N)        {q(a['soft_N'])}     fingertip touch {a['tip_touch'].mean():.0%}")
        print(f"  fingertip depth spread (cm)    {q(a['spread_cm'])}")
        print(f"  palm tilt (deg)                {q(a['tilt_deg'])}     heading {q(a['heading_deg'])}")
        print(f"  self-collision depth (mm)      {q(a['coll_mm'])}     any {np.mean(a['coll_mm'] > 0):.0%}")
        print(f"  arm load sum (tau/rated)^2     {np.percentile(a['load'], 10):.3f} {np.median(a['load']):.3f} "
              f"{np.percentile(a['load'], 90):.3f}")
        print(f"  episode time (s)               {q(a['time_s'])}")

    print(f"\npolicy {args.policy}, task {args.task}")
    report("FEASIBLE targets", [r for r in rows if not r["stretch"]])
    report("STRETCH targets (beyond the measured workspace)", [r for r in rows if r["stretch"]])
    env.close()


if __name__ == "__main__":
    main()
    app.close()
