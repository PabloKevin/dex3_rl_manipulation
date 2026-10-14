# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Chain the two-stage reach in ONE simulated episode and record it:

  PREGRASP  AppleReachPregrasp policy, hand open, until its own success
            criterion (palm 6 cm in front of the apple, held) fires
  HAND-OFF  hand targets -> pregrasp.FINAL_HAND_PRESHAPE (the wider cup),
            palm tactile input switched on (it is fed zero in part 1)
  FINAL     AppleReachFinal policy, until its success criterion fires
            (gentle palm touch, held) or --final_s elapses

Both tasks share scene, 40-input observation and 8-output action, so one env
(the Pregrasp play env) serves both policies; only the palm_touch_force slot
and the hand targets differ between stages. Nothing is trained or changed.

  conda activate env_isaaclab
  python scripts/play_two_stage.py --headless --enable_cameras \
      --video_dir outputs/two_stage
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play AppleReachPregrasp then AppleReachFinal in one episode.")
parser.add_argument("--pregrasp_ckpt", type=str, default=None, help="default: newest logs/rsl_rl/apple_reach_pregrasp/<run>/model_*.pt")
parser.add_argument("--final_ckpt", type=str, default=None, help="default: newest logs/rsl_rl/apple_reach_final/<run>/model_*.pt")
parser.add_argument("--pregrasp_s", type=float, default=6.0, help="max seconds in part 1")
parser.add_argument("--final_s", type=float, default=4.0, help="max seconds in part 2")
parser.add_argument("--tail_s", type=float, default=1.0, help="seconds kept after the end")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--video_dir", type=str, default=None,
                    help="default: <final run dir>/videos/play, as scripts/play.py")
parser.add_argument("--ramp_s", type=float, default=0.0,
                    help="close the hand to the cup shape over this many seconds at the hand-off (0 = instantly)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import functools  # noqa: E402
import glob  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import envs  # noqa: E402,F401
from envs.agents.rsl_rl_ppo_cfg import AppleReachFinalPPORunnerCfg, AppleReachPregraspPPORunnerCfg  # noqa: E402
from envs.apple_reach_twostage_env_cfg import AppleReachPregraspEnvCfg_PLAY  # noqa: E402
from envs.mdp import pregrasp as pg  # noqa: E402
from envs.mdp import terminations_and_events as te  # noqa: E402

try:
    from envs.apple_grasp_env_cfg import task_mdp  # real palm tactile obs term
except Exception:  # pragma: no cover
    import envs.mdp as task_mdp


def _newest(exp: str) -> str:
    runs = sorted(d for d in glob.glob(os.path.join(REPO, "logs", "rsl_rl", exp, "*")) if "warmstart" not in d)
    for run in reversed(runs):
        ck = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
        if ck:
            return ck[-1]
    raise FileNotFoundError(f"no checkpoint under logs/rsl_rl/{exp}")


def _touch_obs(env, sensor_cfg):
    """Zero in part 1 (as trained), the real palm tactile force in part 2."""
    if getattr(env, "_stage", 1) == 1:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return task_mdp.palm_contact_force(env, sensor_cfg)


@functools.wraps(te.reach_success)  # Isaac Lab checks the term's signature
def _observe_success(env, *args, **params):
    """Run the real reach_success term (keeps its hold timer and the
    env._reach_success cache the reward reads) but never end the episode."""
    done = te.reach_success(env, *args, **params)
    env._chain_success = done.clone()
    return torch.zeros_like(done)


def main():
    pre_ckpt = args_cli.pregrasp_ckpt or _newest("apple_reach_pregrasp")
    fin_ckpt = args_cli.final_ckpt or _newest("apple_reach_final")
    print(f"[two-stage] pregrasp: {pre_ckpt}\n[two-stage] final   : {fin_ckpt}")

    cfg = AppleReachPregraspEnvCfg_PLAY()
    cfg.seed = args_cli.seed
    cfg.sim.device = args_cli.device if args_cli.device is not None else cfg.sim.device
    cfg.episode_length_s = args_cli.pregrasp_s + args_cli.final_s + args_cli.tail_s + 5.0
    # camera: the task cfg's own viewer, exactly as scripts/play.py records it
    # part 2's touch input, switchable
    sensor_cfg = cfg.observations.policy.palm_touch_force.params.get("sensor_cfg")
    from isaaclab.managers import SceneEntityCfg
    cfg.observations.policy.palm_touch_force = ObsTerm(
        func=_touch_obs, params={"sensor_cfg": SceneEntityCfg("contact_forces")}
    )
    # observe, don't act on, the success criterion
    rs = cfg.terminations.reach_success
    rs.params = dict(rs.params)
    rs.func = _observe_success

    env = gym.make("AppleReachPregrasp-Play-v1", cfg=cfg, render_mode="rgb_array")
    # same recording mechanism as scripts/play.py --video
    dt = cfg.sim.dt * cfg.decimation
    n_steps = int((args_cli.pregrasp_s + args_cli.final_s + args_cli.tail_s) / dt)
    video_dir = os.path.abspath(args_cli.video_dir or os.path.join(os.path.dirname(fin_ckpt), "videos", "play"))
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, step_trigger=lambda step: step == 0,
        video_length=n_steps, name_prefix=f"two_stage_seed{args_cli.seed}_ramp{args_cli.ramp_s:g}", disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=AppleReachPregraspPPORunnerCfg().clip_actions)
    uenv = env.unwrapped
    uenv._stage = 1
    uenv._chain_success = torch.zeros(1, dtype=torch.bool, device=uenv.device)

    def load(cfg_cls, ckpt):
        r = OnPolicyRunner(env, cfg_cls().to_dict(), log_dir=None, device=uenv.device)
        r.load(ckpt)
        return r.get_inference_policy(device=uenv.device)

    pol_pre = load(AppleReachPregraspPPORunnerCfg, pre_ckpt)
    pol_fin = load(AppleReachFinalPPORunnerCfg, fin_ckpt)

    robot = uenv.scene["robot"]
    ids = torch.tensor([robot.joint_names.index(j) for j in pg.FINAL_HAND_PRESHAPE], device=uenv.device)
    vals = torch.tensor(list(pg.FINAL_HAND_PRESHAPE.values()), device=uenv.device)
    palm_bid = robot.data.body_names.index("right_hand_palm_link")
    apple = uenv.scene["apple"]

    obs = env.get_observations()
    stage, t_stage, early, p2_ok, d = 1, 0.0, None, False, float("nan")
    tail_left, ramp_left, q_from = None, 0.0, None
    tm = uenv.termination_manager
    for _ in range(n_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = (pol_pre if stage == 1 else pol_fin)(obs)
            obs, _, dones, _ = env.step(actions)
            ok = bool(uenv._chain_success[0])
            d = float(torch.norm(apple.data.root_pos_w[0] - robot.data.body_pos_w[0, palm_bid]))
            fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
            if ramp_left > 0:  # gradual hand closing
                ramp_left = max(0.0, ramp_left - dt)
                a_ = 1.0 - ramp_left / args_cli.ramp_s
                tgt = robot.data.joint_pos_target.clone()
                tgt[:, ids] = q_from + a_ * (vals - q_from)
                robot.set_joint_position_target(tgt)
        t_stage += dt
        if stage == 2 and ok:
            p2_ok = True
        if fired and early is None:
            early = f"terminated in stage {stage} at t={t_stage:.2f}s by {fired}, palm-apple dist {d*100:.1f} cm"
            print(f"[two-stage] {early}")
            tail_left = args_cli.tail_s  # keep recording so the event is visible
        if tail_left is not None:
            tail_left -= dt
            if tail_left <= 0:
                break
            continue
        if stage == 1 and (ok or t_stage >= args_cli.pregrasp_s):
            print(f"[two-stage] HAND-OFF: part-1 success={ok} after {t_stage:.2f}s, palm-apple dist={d*100:.1f} cm")
            stage, t_stage, uenv._stage = 2, 0.0, 2
            uenv._chain_success = torch.zeros(1, dtype=torch.bool, device=uenv.device)
            with torch.inference_mode():
                q_from = robot.data.joint_pos_target[:, ids].clone()
                if args_cli.ramp_s > 0:
                    ramp_left = args_cli.ramp_s
                else:
                    tgt = robot.data.joint_pos_target.clone()
                    tgt[:, ids] = vals
                    robot.set_joint_position_target(tgt)

    print(f"[two-stage] part-2 success reached: {p2_ok}; final palm-apple dist={d*100:.1f} cm; {early or 'no early end'}")
    env.close()  # finalises the video
    print(f"[two-stage] video folder: {video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
