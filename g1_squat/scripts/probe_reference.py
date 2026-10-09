"""Run the squat env with ZERO actions (pure PD tracking of the stand<->squat reference).

Sanity check before training: managers wire up, the open-loop reference is roughly
balanced, which links touch in the squat (self-collision), torques vs limits.

  ~/miniconda3/envs/env_isaaclab/bin/python g1_squat/scripts/probe_reference.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seconds", type=float, default=14.0)
parser.add_argument("--arms", action="store_true", help="enable the random arm motion while squatting")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from g1_squat.mdp.observations import com_offset_b  # noqa: E402
from g1_squat.squat_env_cfg import G1SquatEnvCfg_PLAY  # noqa: E402


def main():
    cfg = G1SquatEnvCfg_PLAY()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.seconds + 5.0
    # deterministic schedule: stand 1 s, squat (2 s ramp + 5 s hold), stand up, stand
    cfg.commands.squat.initial_stand_range = (1.0, 1.0)
    cfg.commands.squat.squat_hold_range = (5.0, 5.0)
    cfg.commands.squat.stand_hold_range = (10.0, 10.0)
    cfg.commands.squat.abort_prob = 0.0
    if not args.arms:
        cfg.commands.arm_motion.amplitude_range = (0.0, 0.0)
        cfg.commands.arm_motion.load_force_range = (0.0, 0.0)
    env = ManagerBasedRLEnv(cfg)
    robot = env.scene["robot"]
    sq = env.command_manager.get_term("squat")
    feet = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
    feet.resolve(env.scene)
    sensor = env.scene["contact_forces"]
    lw_ids = sq.joint_ids
    names = robot.data.joint_names
    print("joints:", len(names), "bodies:", robot.num_bodies)
    print(sq)

    env.reset()
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    steps = int(args.seconds / env.step_dt)
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    peak_ratio = torch.zeros(len(lw_ids), device=env.device)
    for i in range(steps):
        _, rew, term, trunc, _ = env.step(actions)
        fell |= term
        tau = robot.data.computed_torque[:, lw_ids].abs()
        lim = robot.data.joint_effort_limits[:, lw_ids]
        peak_ratio = torch.maximum(peak_ratio, (tau / lim).max(dim=0)[0])
        if i % 25 == 0:
            com = com_offset_b(env, feet)
            forces = sensor.data.net_forces_w[0].norm(dim=-1)
            touching = [
                f"{sensor.body_names[b]}:{forces[b]:.0f}"
                for b in torch.nonzero(forces > 1.0).flatten().tolist()
                if "ankle_roll" not in sensor.body_names[b]
            ]
            print(
                f"t {i * env.step_dt:5.2f} tgt {int(sq.target[0])} s {sq.phase[0]:.2f} | pelvis "
                f"{sq.pelvis_height()[0]:.3f} ref {sq.ref_height[0]:.3f} | CoM x {com[:, 0].mean() * 100:+.1f} cm "
                f"(min {com[:, 0].min() * 100:+.1f} max {com[:, 0].max() * 100:+.1f}) | fell {int(fell.sum())}"
                f"/{env.num_envs} | rew {rew.mean():+.2f} | contacts env0 {touching}"
            )
    print("\npeak |tau|/limit per leg+waist joint:")
    for j, r in zip(sq.joint_names, peak_ratio.tolist()):
        print(f"  {j:28s} {r:.2f}")
    print("\nreward terms (last step, env mean):")
    for name, term_cfg in zip(env.reward_manager.active_terms, env.reward_manager._term_cfgs):
        val = term_cfg.func(env, **term_cfg.params) * term_cfg.weight
        print(f"  {name:20s} {val.mean().item():+.4f}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
