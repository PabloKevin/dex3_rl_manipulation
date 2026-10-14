# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Record the chained two-stage reach (AppleReachFinalChained-Play-v1 / -v2, --chained): frozen
part 1 (newest apple_reach_pregrasp_v2 checkpoint) then the part-2 policy
(newest apple_reach_final_chained checkpoint), in the env part 2 trained in.
Video: <final run>/videos/play/chained_seed<N>-step-0.mp4 (same camera as play.py)."""
import argparse, glob, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pregrasp_ckpt", default=None)
parser.add_argument("--final_ckpt", default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--chained", choices=["v1", "v2"], default="v2",
                    help="v1: pregrasp_v2 + final_chained; v2: pregrasp_v3 + final_chained_v2")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import envs  # noqa: F401
from envs.chained_reach_env import FrozenActor, newest_checkpoint, PREGRASP_LOG_ROOT
from envs.apple_reach_twostage_env_cfg import AppleReachFinalChainedEnvCfg_PLAY, AppleReachFinalChainedV2EnvCfg_PLAY

V = {"v1": (AppleReachFinalChainedEnvCfg_PLAY, "AppleReachFinalChained-Play-v1", "apple_reach_pregrasp_v2", "apple_reach_final_chained"),
     "v2": (AppleReachFinalChainedV2EnvCfg_PLAY, "AppleReachFinalChained-Play-v2", "apple_reach_pregrasp_v3", "apple_reach_final_chained_v2")}
CFG, TASK, P1_EXP, P2_EXP = V[args.chained]

pre = args.pregrasp_ckpt or newest_checkpoint(os.path.join(REPO, "logs", "rsl_rl", P1_EXP))
fin = args.final_ckpt or newest_checkpoint(os.path.join(REPO, "logs", "rsl_rl", P2_EXP))
print(f"[play] part 1: {pre}\n[play] part 2: {fin}")
cfg = CFG()
cfg.seed = args.seed
cfg.pregrasp_ckpt = pre
env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
n = int(args.seconds / u.step_dt)
env = gym.wrappers.RecordVideo(env, video_folder=os.path.join(os.path.dirname(fin), "videos", "play"),
                               step_trigger=lambda s: s == 0, video_length=n,
                               name_prefix=f"chained_seed{args.seed}", disable_logger=True)
learner = FrozenActor(fin, u.device)
robot, apple = u.scene["robot"], u.scene["apple"]
palm = robot.data.body_names.index("right_hand_palm_link")
obs, _ = env.reset()
tm = u.termination_manager
handoff = None; success = False; ended = None; dmin = 9.0
with torch.inference_mode():
    for i in range(n):
        obs, r, te_, tr, ex = env.step(learner(obs["policy"]))
        d = float(torch.norm(apple.data.root_pos_w[0] - robot.data.body_pos_w[0, palm]))
        if bool(u._stage2[0]) and handoff is None:
            handoff = (i * u.step_dt, d, bool(u._p1_success[0]))
        if handoff is not None:
            dmin = min(dmin, d)
        if bool(u._reach_success[0]):
            success = True
        fired = [t for t in tm.active_terms if bool(tm.get_term(t)[0])]
        if fired and ended is None:
            ended = (i * u.step_dt, fired, d)
print(f"[play] seed {args.seed}: hand-off {handoff and (round(handoff[0],2), 'palm-apple %.1f cm' % (handoff[1]*100), 'part1_success=%s' % handoff[2])}")
print(f"[play] seed {args.seed}: part-2 touch success={success}  min palm-apple dist after hand-off={dmin*100:.1f} cm  ended={ended}")
env.close()
simulation_app.close()
