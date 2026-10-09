"""Hand-over test: Unitree velocity policy <-> squat policy, in the squat env.

Timeline (per env, all envs in parallel):
  VEL   stand 2 s, walk forward --walk_speed for 3 s, stand 2 s        (unitree_rl_lab v0 policy)
  SQUAT switch to the squat policy (toggle = stand) 1 s, toggle squat, hold --hold s with
        random arm motion + hand loads, toggle stand, settle --settle s
  VEL   switch back, stand 1 s, walk 3 s, stand 2 s
Reports falls (and in which phase), pelvis height, hand height above the ground while
squatting, foot slip, peak tilt, and the joint-target jump at each switch.

  ~/miniconda3/envs/env_isaaclab/bin/python g1_squat/scripts/switch_test.py --num_envs 16 [--headless]
  (no --headless to watch it; --checkpoint defaults to the latest g1_squat run)
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, default=None, help="squat model_*.pt (default: latest run)")
parser.add_argument("--hold", type=float, default=8.0, help="seconds held in the squat")
parser.add_argument("--settle", type=float, default=1.5, help="seconds standing before handing back")
parser.add_argument("--walk_speed", type=float, default=0.5)
parser.add_argument("--no_arms", action="store_true", help="keep arms at rest during the squat hold")
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--no_hands", action="store_true", help="use the hand-less G1 (the model v0 was trained on)")
parser.add_argument("--vel_only", action="store_true", help="only run the velocity policy (checks the v0 loader)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import time  # noqa: E402

import torch  # noqa: E402

G1_SQUAT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(G1_SQUAT_DIR))
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from g1_squat import robot_cfg as rc  # noqa: E402
from g1_squat.policies import UnitreeVelocityPolicy, load_squat_actor  # noqa: E402
from g1_squat.squat_env_cfg import G1SquatEnvCfg_PLAY  # noqa: E402


def latest_checkpoint() -> str:
    runs = sorted(glob.glob(os.path.join(G1_SQUAT_DIR, "logs", "rsl_rl", "g1_squat", "*")))
    for run in reversed(runs):
        models = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
        if models:
            return models[-1]
    raise FileNotFoundError("no g1_squat checkpoint; train first or pass --checkpoint")


def main():
    ckpt = None if args.vel_only else (args.checkpoint or latest_checkpoint())
    print(f"[switch_test] squat policy: {ckpt}")

    cfg = G1SquatEnvCfg_PLAY()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 1.0e6
    cfg.terminations.time_out = None
    cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)  # all walk along +x
    if args.no_hands:
        from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

        from g1_squat.robot_cfg import DEPLOY_ACTUATORS

        cfg.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", actuators=DEPLOY_ACTUATORS)
    if args.no_arms:
        cfg.commands.arm_motion.amplitude_range = (0.0, 0.0)
        cfg.commands.arm_motion.load_force_range = (0.0, 0.0)
    env = ManagerBasedRLEnv(cfg)
    dev = env.device
    robot = env.scene["robot"]
    sq = env.command_manager.get_term("squat")
    arms = env.command_manager.get_term("arm_motion")
    act_term = env.action_manager.get_term("legs_waist")

    squat_actor = load_squat_actor(ckpt).to(dev).eval() if ckpt else None
    vel = UnitreeVelocityPolicy(robot, dev)
    sdk_ids = vel.joint_ids
    hand_ids = arms.hand_ids
    feet_ids = sq.feet_ids
    wrist_ids, _ = robot.find_bodies(arms.cfg.palm_bodies, preserve_order=True)

    # --- control-mode plumbing ----------------------------------------------------------
    mode = {"vel": True}
    vel_targets = robot.data.default_joint_pos[:, sdk_ids].clone()
    orig_apply = act_term.apply_actions
    orig_write_arms = arms._write_targets

    def apply_actions():
        if mode["vel"]:
            robot.set_joint_position_target(vel_targets, joint_ids=sdk_ids)
        else:
            orig_apply()

    def write_arm_targets(ids):
        if not mode["vel"]:
            orig_write_arms(ids)

    act_term.apply_actions = apply_actions
    arms._write_targets = write_arm_targets

    def hold_toggle(target: float):
        sq.target[:] = target
        sq.time_left[:] = 1.0e9  # the test drives the toggle, not the training timer

    obs, _ = env.reset()
    hold_toggle(0.0)
    zero_act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=dev)
    dt = env.step_dt

    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    fell_phase = ["-"] * env.num_envs
    stats = {"min_pelvis": torch.full((env.num_envs,), 9.0, device=dev),
             "min_hand_z": torch.full((env.num_envs,), 9.0, device=dev),
             "max_tilt_deg": torch.zeros(env.num_envs, device=dev)}
    switch_jumps = []
    feet_start = None

    def step(action, phase_name, cmd=None):
        nonlocal obs, vel_targets
        start = time.time()
        if mode["vel"]:
            vel_targets = vel.act(cmd if cmd is not None else torch.zeros(env.num_envs, 3, device=dev))
            robot.set_joint_position_target(arms.hand_default.expand(env.num_envs, -1), joint_ids=hand_ids)
        obs, _, term, _, _ = env.step(action)
        new = term & ~fell
        for i in new.nonzero().flatten().tolist():
            fell_phase[i] = phase_name
        fell[:] |= term
        g = robot.data.projected_gravity_b
        tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
        stats["max_tilt_deg"] = torch.maximum(stats["max_tilt_deg"], torch.where(fell, 0.0, tilt))
        if args.real_time:
            time.sleep(max(0.0, dt - (time.time() - start)))

    def run_vel(seconds, vx, name):
        cmd = torch.zeros(env.num_envs, 3, device=dev)
        cmd[:, 0] = vx
        for _ in range(int(seconds / dt)):
            step(zero_act, name, cmd)

    def squat_action():
        with torch.inference_mode():
            return squat_actor(obs["policy"])

    sdk_order_ids, _ = robot.find_joints(rc.SDK_JOINTS, preserve_order=True)

    def switch(to_vel: bool):
        # compare targets in one joint order: velocity policy order for "-> velocity", SDK order for "-> squat"
        before = robot.data.joint_pos_target[:, sdk_ids if to_vel else sdk_order_ids].clone()
        mode["vel"] = to_vel
        if to_vel:
            vel.reset()
            target = vel.act(torch.zeros(env.num_envs, 3, device=dev))
        else:
            # squat policy takes over at s = 0, arms ramp from where they are to the rest pose
            sq.phase[:] = 0.0
            arms.current[:] = robot.data.joint_pos[:, arms.arm_ids]
            arms._start_segment(torch.arange(env.num_envs, device=dev), arms.q_default.expand(env.num_envs, -1),
                                torch.full((env.num_envs,), arms.cfg.return_time, device=dev))
            act_term.process_actions(squat_action())
            target = torch.cat([act_term.processed_actions, arms.q_default.expand(env.num_envs, -1)], dim=1)
        jump = (target - before).abs().max(dim=1)[0]
        switch_jumps.append(("-> velocity" if to_vel else "-> squat", jump.mean().item(), jump.max().item()))

    # ---- timeline ---------------------------------------------------------------------
    run_vel(2.0, 0.0, "vel stand")
    run_vel(3.0, args.walk_speed, "vel walk")
    run_vel(2.0, 0.0, "vel stop")

    if args.vel_only:
        run_vel(3.0, -0.3, "vel walk back")
        run_vel(2.0, 0.0, "vel stop 2")
        g = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
        for i in range(env.num_envs):
            print(f"[vel_only] env {i} fell in: {fell_phase[i]}")
        print(f"\n[vel_only] falls {int(fell.sum())}/{env.num_envs}, peak tilt {stats['max_tilt_deg'].max():.1f} deg, "
              f"final xy offset mean {g.mean(dim=0).tolist()}")
        env.close()
        return

    switch(to_vel=False)
    hold_toggle(0.0)
    for _ in range(int(1.0 / dt)):
        step(squat_action(), "squat-policy stand")
    feet_start = robot.data.body_link_pos_w[:, feet_ids, :2].clone()
    hold_toggle(1.0)
    for _ in range(int((sq.cfg.transition_time + args.hold) / dt)):
        step(squat_action(), "squat down/hold")
        alive = ~fell
        stats["min_pelvis"] = torch.where(alive, torch.minimum(stats["min_pelvis"], sq.pelvis_height()), stats["min_pelvis"])
        hand_z = robot.data.body_link_pos_w[:, wrist_ids, 2].min(dim=1)[0]
        stats["min_hand_z"] = torch.where(alive, torch.minimum(stats["min_hand_z"], hand_z), stats["min_hand_z"])
    slip = (robot.data.body_link_pos_w[:, feet_ids, :2] - feet_start).norm(dim=-1).max(dim=1)[0]
    hold_toggle(0.0)
    for _ in range(int((sq.cfg.transition_time + args.settle) / dt)):
        step(squat_action(), "stand up/settle")
    settle_err = (robot.data.joint_pos[:, sq.joint_ids] - sq.q_stand).abs().max(dim=1)[0]

    switch(to_vel=True)
    run_vel(1.0, 0.0, "vel stand 2")
    run_vel(3.0, args.walk_speed, "vel walk 2")
    run_vel(2.0, 0.0, "vel stop 2")

    # ---- report -------------------------------------------------------------------------
    n = env.num_envs
    print("\n================ hand-over test ================")
    print(f"envs {n}, squat hold {args.hold} s, arms {'off' if args.no_arms else 'random + loads'}")
    print(f"falls: {int(fell.sum())}/{n}")
    for i in range(n):
        if fell[i]:
            print(f"  env {i}: fell during '{fell_phase[i]}'")
    ok = ~fell
    if ok.any():
        print(f"min pelvis height in squat : {stats['min_pelvis'][ok].mean():.3f} m (mean)")
        print(f"min wrist height in squat  : {stats['min_hand_z'][ok].mean():.3f} m mean, "
              f"{stats['min_hand_z'][ok].min():.3f} m lowest (wrist_yaw link origin; fingertips ~0.13 m further)")
        print(f"foot slip during squat     : {slip[ok].mean() * 100:.1f} cm mean, {slip[ok].max() * 100:.1f} cm max")
        print(f"stand pose error at hand-back: {settle_err[ok].mean():.3f} rad mean max-joint")
        print(f"peak pelvis tilt           : {stats['max_tilt_deg'][ok].max():.1f} deg")
    for name, mean, mx in switch_jumps:
        print(f"target jump at switch {name:12s}: {mean:.3f} rad mean, {mx:.3f} rad max")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    app.close()
