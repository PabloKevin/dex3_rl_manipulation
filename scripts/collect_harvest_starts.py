#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Record the START BANK for the harvest task (AppleGraspHarvest-v0): ready
grasps produced by a trained grasp policy, one per episode.

Each env runs the normal reach -> grasp episode (grasp env, arm frozen, the
simulated stem never released, so the grasp policy sees exactly its training
conditions -- same setup as scripts/harvest_scripted.py). The moment
envs.mdp.harvest.GraspReadyDetector fires (closure + thumb pressure + fingers
settled, held 0.5 s), that env's state is saved and the env is reset to
produce another. Saved per state:
  joint_pos and joint_target (all robot joints: the commanded targets of the
  joints no action drives are NOT their defaults in the grasp env, and
  restoring defaults made the left arm and legs swing and shake the grasp
  loose), arm_target / hand_target (the COMMANDED
  joint targets, RIGHT_ARM_JOINTS / RIGHT_HAND_JOINTS order), apple pose,
  palm pose (positions relative to the env origin), apple radius, and the
  force the grasp phase's soft stem was applying (the harvest stem starts
  preloaded with it -- the grip is in equilibrium with that force).

Output: logs/harvest_starts/<grasp run>_<model>.pt, also copied to
logs/harvest_starts/latest.pt, which the harvest task loads by default.

Usage:
  python scripts/collect_harvest_starts.py --headless \
      --checkpoint logs/rsl_rl/apple_grasp_grasp/<run>/model_800.pt --num_states 1000
"""
import argparse
import os
import shutil
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description="Record ready-grasp start states for the harvest task")
p.add_argument("--checkpoint", type=str, required=True, help="rsl_rl model_*.pt of a grasp run")
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--num_states", type=int, default=1000)
p.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import envs  # noqa: F401
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from envs.apple_grasp_grasp_env_cfg import AppleGraspGraspEnvCfg, RIGHT_ARM_JOINTS, RIGHT_HAND_JOINTS
from envs.mdp import harvest as hv


def _collected(env):
    return env._collect_flag


cfg = AppleGraspGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
cfg.terminations.collected = DoneTerm(func=_collected)
cfg.events.reset_reach_grasp_phase.params.update(
    stem_release_schedule=None, stem_release_time_range=None, release_on_latch=False, shaping_grace_s=None)
env = gym.make("AppleGraspGrasp-v0", cfg=cfg).unwrapped
N, dev = env.num_envs, env.device
env._collect_flag = torch.zeros(N, dtype=torch.bool, device=dev)

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from envs.agents.rsl_rl_ppo_cfg import AppleGraspGraspPPORunnerCfg
acfg = AppleGraspGraspPPORunnerCfg()
wrap = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
runner = OnPolicyRunner(wrap, acfg.to_dict(), log_dir=None, device=dev)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=dev)

robot = env.scene["robot"]
apple = env.scene["apple"]
arm = env.action_manager.get_term("right_arm")
hand = env.action_manager.get_term("right_hand")
arm_idx = [list(arm._joint_names).index(n) for n in RIGHT_ARM_JOINTS]
hand_idx = [list(hand._joint_names).index(n) for n in RIGHT_HAND_JOINTS]
palm_id = robot.data.body_names.index("right_hand_palm_link")
sc = SceneEntityCfg("contact_forces"); sc.resolve(env.scene)
ready_det = hv.GraspReadyDetector(env, sc, hand._joint_ids)

rec = {k: [] for k in ("joint_pos", "joint_target", "arm_target", "hand_target", "apple_pos", "apple_quat", "palm_pos", "palm_quat",
                       "apple_radius", "stem_force")}
episodes = timeouts = 0
wrap.reset()
print(f"[collect] {N} envs, target {args.num_states} states from {args.checkpoint}", flush=True)
count = 0
while app.is_running() and count < args.num_states:
    with torch.inference_mode():
        a = policy(wrap.get_observations())
        env._collect_flag[:] = False
        ready = ready_det.update()
        if bool(ready.any()):
            ids = ready.nonzero(as_tuple=True)[0]
            o = env.scene.env_origins[ids]
            r = getattr(env, "_apple_radius", None)
            rec["joint_pos"].append(robot.data.joint_pos[ids].cpu())
            rec["joint_target"].append(robot.data.joint_pos_target[ids].cpu())
            rec["arm_target"].append(arm._processed_actions[ids][:, arm_idx].cpu())
            rec["hand_target"].append(hand._processed_actions[ids][:, hand_idx].cpu())
            rec["apple_pos"].append((apple.data.root_pos_w[ids] - o).cpu())
            rec["apple_quat"].append(apple.data.root_quat_w[ids].cpu())
            rec["palm_pos"].append((robot.data.body_pos_w[ids, palm_id] - o).cpu())
            rec["palm_quat"].append(robot.data.body_quat_w[ids, palm_id].cpu())
            rec["apple_radius"].append((r[ids] if isinstance(r, torch.Tensor) else torch.full((len(ids),), 0.04, device=dev)).cpu())
            # the grasp phase's soft stem (ReachHandoffJointPositionAction.
            # apply_actions): the policy's grip is in equilibrium WITH it, so
            # the harvest stem starts preloaded with the same force
            c = arm.cfg
            f = c.stem_stiffness * (env._grasp_handoff_apple_pos[ids] - apple.data.root_pos_w[ids]) \
                - c.stem_damping * apple.data.root_lin_vel_w[ids]
            f = f * (c.stem_max_force / f.norm(dim=-1, keepdim=True).clamp(min=c.stem_max_force))
            rec["stem_force"].append(f.cpu())
            env._collect_flag[ids] = True
            count += len(ids)
            print(f"  {count} / {args.num_states} states  (env {ids.tolist()})", flush=True)
        wrap.step(a)
        just_reset = env.episode_length_buf == 0
        if bool(just_reset.any()):
            rid = just_reset.nonzero(as_tuple=True)[0]
            ready_det.reset(rid)
            episodes += len(rid)
            timeouts += int(env.termination_manager.time_outs[rid].sum())

bank = {k: torch.cat(v)[: args.num_states] for k, v in rec.items()}
bank.update(
    joint_names=list(robot.data.joint_names), arm_joint_names=list(RIGHT_ARM_JOINTS),
    hand_joint_names=list(RIGHT_HAND_JOINTS), grasp_checkpoint=os.path.abspath(args.checkpoint),
)
run = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
model = os.path.splitext(os.path.basename(args.checkpoint))[0]
out = args.out or os.path.join(REPO, "logs", "harvest_starts", f"{run}_{model}.pt")
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(bank, out)
latest = os.path.join(os.path.dirname(out), "latest.pt")
if os.path.abspath(out) != os.path.abspath(latest):
    shutil.copyfile(out, latest)
print(f"[collect] saved {bank['joint_pos'].shape[0]} states -> {out} (and {latest}); "
      f"{episodes} episodes ended, {timeouts} of them timed out without a ready grasp", flush=True)
sys.stdout.flush()
os._exit(0)
