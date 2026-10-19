#!/usr/bin/env python3
"""Make an rsl_rl warm-start checkpoint (model_0.pt) from an EXPORTED policy
(scripts/play.py's exported/policy.pt: actor + input normalizer only), for a
task with the same input / output sizes, when the run's model_*.pt files are
not at hand (e.g. they stayed on the other PC).

The actor and its observation normalizer are copied from the export; the
critic is the task runner's fresh one (its normalizer gets the actor's
statistics when the critic sees the same inputs); the action noise std is set
to --std. Load it with train.py --resume --load_run <dir name> --checkpoint model_0.pt.

    python scripts/warmstart_from_jit.py --task AppleReachVisiblePrecise-v1 \\
        --jit envs/agents/best_weights/grasp_handoff_reach_visible.pt \\
        --out logs/rsl_rl/apple_reach_visible_v15/warmstart_v1/model_0.pt --std 0.3
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--task", required=True)
p.add_argument("--jit", required=True)
p.add_argument("--out", required=True)
p.add_argument("--std", type=float, default=0.3)
p.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import envs  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

spec = gym.spec(args.task)
env_cfg = spec.kwargs["env_cfg_entry_point"]()
env_cfg.scene.num_envs = args.num_envs
agent = spec.kwargs["rsl_rl_cfg_entry_point"]()
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg).unwrapped)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=env.unwrapped.device)
pol = runner.alg.policy

src = torch.jit.load(args.jit, map_location="cpu").state_dict()
actor = {k[len("actor."):]: v for k, v in src.items() if k.startswith("actor.")}
missing = pol.actor.load_state_dict(actor, strict=True)
norm = {k[len("normalizer."):]: v for k, v in src.items() if k.startswith("normalizer.")}
if norm:
    pol.actor_obs_normalizer.load_state_dict(norm)
    try:
        pol.critic_obs_normalizer.load_state_dict(norm)
        crit = "critic normalizer = actor's"
    except Exception as e:
        crit = f"critic normalizer fresh ({type(e).__name__})"
else:
    crit = "no normalizer in the export"
with torch.no_grad():
    if hasattr(pol, "std"):
        pol.std.fill_(args.std)
    elif hasattr(pol, "log_std"):
        pol.log_std.fill_(torch.log(torch.tensor(args.std)).item())
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
runner.current_learning_iteration = 0
runner.logger_type = getattr(runner, "logger_type", "tensorboard")   # unset without a log_dir; save() reads it
runner.save(os.path.abspath(args.out))

# check: the saved actor + normalizer reproduce the export
with torch.no_grad():
    x = torch.randn(64, actor["0.weight"].shape[1])
    ref = torch.jit.load(args.jit, map_location="cpu")(x)
    pol_cpu = pol.to("cpu")
    mine = pol_cpu.actor(pol_cpu.actor_obs_normalizer(x))
print(f"[warmstart] {args.out}: actor from {args.jit} (max |diff| vs the export {float((mine - ref).abs().max()):.2e}), "
      f"std {args.std}, {crit}, critic fresh", flush=True)
env.close()
app.close()
