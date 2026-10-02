#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Phase 2 — View the grasp policy (reach -> grasp, chained), interactively.

Single self-contained script, deliberately NOT split across two
terminals/processes like the reach task's run_env.py + reach_apple_hybrid.py
pair: that split existed because reach->classical-control sequencing lived
in an EXTERNAL script (reach_apple_hybrid.py's own phase FSM). For the grasp
task, the reach -> grasp hand-off already happens automatically inside every
episode (the frozen reach policy drives first, then the grasp policy -- see
envs/mdp/grasp_autoplay.py), so a single process loading Isaac Sim ONCE and
supporting 'r'+Enter to reset (no reload) already gives the full picture
with far less complexity. If real-hardware DDS deployment is needed later
(mirroring reach_apple_hybrid.py), that's a separate, later script — this
one is for interactive sim visualization/debugging only.

Usage:
  ~/IsaacLab/isaaclab.sh -p scripts/run_grasp_env.py
  ~/IsaacLab/isaaclab.sh -p scripts/run_grasp_env.py --policy /path/to/policy.pt
  ~/IsaacLab/isaaclab.sh -p scripts/run_grasp_env.py --random_actions   (no policy, sanity-check the env itself)

Press 'r' + Enter in this terminal at any time to reset WITHOUT relaunching
Isaac Sim -- this is the "don't reload isaaclab every trial" part. A manual
reset first RETRACES the arm: the episode's recorded arm trajectory is
played back in reverse (the hand keeps its grip), the hand opens, and only
then do the apple and tree move -- so the arm goes home the way it came,
without sweeping through the tree. The reset itself then puts every joint
exactly at the home pose the reach policy was trained from (also on the
automatic resets after a termination, which do not retrace).

Every reset prints which termination fired last episode (grasp_success /
fruit_damaged / apple_fell_from_grip / apple_dropped / time_out) and a
periodic diagnostic line (per-finger-group force, commanded target, whether
force closure has ever been achieved this episode) so you can follow what's
happening even where the viewport itself doesn't make it obvious.
"""
import argparse
import os
import sys
import threading

from isaaclab.app import AppLauncher

BASE_DIR = "/home/pablo_kevin/dex3_rl_manipulation"

parser = argparse.ArgumentParser(description="Apple Grasp (phase 2) Viewer")
parser.add_argument(
    "--policy", type=str, default="last",
    help="Path to a JIT-exported grasp policy.pt, or 'last' (default) to auto-resolve "
         "the latest logs/rsl_rl/apple_grasp_grasp/*/exported/policy.pt.",
)
parser.add_argument("--random_actions", action="store_true",
                     help="Send random actions instead of loading a policy (env sanity check).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab / torch ─────────────────────────────────────
import torch
import gymnasium as gym
from isaaclab.managers import SceneEntityCfg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: registers gym tasks

from envs.apple_grasp_grasp_env_cfg import AppleGraspGraspEnvCfg_PLAY, GRASP_CENTER_LOCAL, RIGHT_ARM_JOINTS
from isaaclab.utils.math import quat_apply
from envs.mdp.grasp_rewards import _per_finger_group_force

# ── Terminal Reset Listener (identical pattern to scripts/run_env.py) ───────
_reset_requested = False
_reset_lock = threading.Lock()


def _listen_terminal_reset():
    global _reset_requested
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            if line.strip().lower() == "r":
                with _reset_lock:
                    _reset_requested = True
        except Exception:
            break


RETRACE_MAX_S = 3.0      # retrace takes at most this long (replay sped up to fit)
OPEN_HAND_S = 0.5        # then open the hand for this long before resetting


def _drive_targets(env, targets: torch.Tensor, n_control_steps: int) -> None:
    """Hold the robot's joint position targets for n control steps, outside
    env.step (no policy, no action manager), rendering as it goes."""
    robot = env.scene["robot"]
    for _ in range(n_control_steps):
        robot.set_joint_position_target(targets)
        for _ in range(env.cfg.decimation):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        env.sim.render()


def _retrace_home(env, arm_traj: list, arm_ids: list) -> None:
    """Play the episode's recorded arm trajectory backwards (hand keeps its
    current targets, so a held apple comes along), then open the hand."""
    robot = env.scene["robot"]
    targets = robot.data.joint_pos_target.clone()
    if arm_traj:
        stride = max(1, -(-len(arm_traj) // int(RETRACE_MAX_S / env.step_dt)))
        for q in reversed(arm_traj[::stride]):
            targets[0, arm_ids] = q
            _drive_targets(env, targets, 1)
        targets[0, arm_ids] = robot.data.default_joint_pos[0, arm_ids]
        _drive_targets(env, targets, 5)
    hand_ids = [i for i, n in enumerate(robot.joint_names) if n.startswith("right_hand_")]
    targets[0, hand_ids] = robot.data.default_joint_pos[0, hand_ids]
    _drive_targets(env, targets, int(OPEN_HAND_S / env.step_dt))


def _find_last_exported_grasp_policy() -> str:
    """Newest apple_grasp_grasp run that HAS an exported policy.pt. Runs
    are only exported when play.py is run against them, so the newest run
    often has none yet -- say so, instead of crashing on it."""
    runs_dir = os.path.join(BASE_DIR, "logs", "rsl_rl", "apple_grasp_grasp")
    runs = sorted(os.listdir(runs_dir), reverse=True) if os.path.isdir(runs_dir) else []
    for i, run in enumerate(runs):
        path = os.path.join(runs_dir, run, "exported", "policy.pt")
        if os.path.isfile(path):
            if i > 0:
                print(f"[Policy] WARNING: newer run(s) {runs[:i]} have no export -- using {run}. "
                      f"To view the newest, first run:\n  scripts/play.py --task AppleGrasp-Play-v1 "
                      f"--load_run {runs[0]}")
            return path
    raise FileNotFoundError(
        f"No exported grasp policy under {runs_dir}. Run scripts/play.py --task "
        f"AppleGrasp-Play-v1 once (it exports policy.pt), or pass --policy <path>."
    )


def main():
    reset_thread = threading.Thread(target=_listen_terminal_reset, daemon=True)
    reset_thread.start()

    cfg = AppleGraspGraspEnvCfg_PLAY()
    env = gym.make("AppleGrasp-Play-v1", cfg=cfg).unwrapped

    print("\n" + "=" * 60)
    print("APPLE GRASP VIEWER (phase 2: reach -> grasp, chained)")
    print(f"  Action dim : {env.action_space.shape}")
    print(f"  Obs dim    : {env.observation_space['policy'].shape}")
    print("  Press 'r' + Enter in this terminal at any time to RESET")
    print("  (runs the real reach hand-off again, no Isaac Sim reload).")
    print("=" * 60 + "\n")

    policy_module = None
    if not args_cli.random_actions:
        if args_cli.policy == "last":
            policy_path = _find_last_exported_grasp_policy()
        else:
            policy_path = args_cli.policy
        policy_module = torch.jit.load(policy_path, map_location=env.device).eval()
        print(f"[Policy] Loaded {policy_path} — driving the arm+hand every step.")
    else:
        print("[Policy] --random_actions set — sending random actions (no policy loaded).")

    sensor_cfg = SceneEntityCfg("contact_forces")
    sensor_cfg.resolve(env.scene)

    def _diag_line():
        try:
            per_finger = _per_finger_group_force(env, sensor_cfg)
            level = per_finger.mean(dim=-1)[0].item()
        except Exception:
            level = float("nan")
        target = env._target_grip_level[0].item() if hasattr(env, "_target_grip_level") else float("nan")
        ever_closed = bool(env._grasp_ever_closed[0]) if hasattr(env, "_grasp_ever_closed") else False
        # Stem let go (grip latched OR the per-episode release deadline
        # passed): from here on the hand alone holds the fruit.
        on_stem = not bool(env._stem_released[0]) if hasattr(env, "_stem_released") else not ever_closed
        apple = env.scene["apple"]
        apple_z = apple.data.root_pos_w[0, 2].item()
        handoff_z = env._grasp_handoff_apple_pos[0, 2].item() if hasattr(env, "_grasp_handoff_apple_pos") else float("nan")
        phase = "grasp" if getattr(env, "_grasp_phase_active", torch.ones(1, dtype=torch.bool))[0] else "reach"
        # Fruit centre to the palm's grasp point -- what apple_fell_from_grip
        # checks (> 6 cm = left the hand). apple_drop is only the height
        # change since hand-off, which the ARM can cause with the fruit held.
        robot = env.scene["robot"]
        palm = robot.data.body_names.index("right_hand_palm_link")
        grasp_point = robot.data.body_pos_w[0, palm] + quat_apply(
            robot.data.body_quat_w[0, palm], torch.tensor(GRASP_CENTER_LOCAL, device=env.device)
        )
        in_hand = torch.norm(apple.data.root_pos_w[0] - grasp_point).item()
        print(f"  phase={phase:5s}  force={level:5.2f} N  target={target:5.2f} N  "
              f"ever_closed={ever_closed}  on_stem={on_stem}  apple_to_hand={in_hand*100:4.1f} cm  "
              f"apple_drop={handoff_z - apple_z:+.3f} m")

    global _reset_requested
    step = 0
    obs, _ = env.reset()
    robot = env.scene["robot"]
    arm_ids = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    arm_traj = []   # arm joint positions since the last reset, for the retrace

    while simulation_app.is_running():
        with torch.inference_mode(), _reset_lock:
            if _reset_requested:
                _reset_requested = False
                print("\n>>> MANUAL RESET: retracing the arm home, then resetting <<<")
                _retrace_home(env, arm_traj, arm_ids)
                obs, _ = env.reset()
                arm_traj = []
                step = 0
                print(">>> reset done (reach hand-off re-runs from the home pose) <<<\n")

        with torch.inference_mode():
            if args_cli.random_actions:
                raw = env.action_space.sample()
                actions = torch.tensor(raw, device=env.device, dtype=torch.float32)
            else:
                actions = policy_module(obs["policy"])

            obs, reward, terminated, truncated, info = env.step(actions)
            step += 1
            arm_traj.append(robot.data.joint_pos[0, arm_ids].clone())

            if terminated.any() or truncated.any():
                term_log = {k.replace("Episode_Termination/", ""): v
                             for k, v in info.get("log", {}).items() if "Termination" in k}
                fired = [k for k, v in term_log.items() if v]
                print(f"\n>>> EPISODE ENDED at step {step} — {fired or 'unknown'} — auto-reset <<<\n")
                step = 0
                arm_traj = []   # env.step already reset the robot to the home pose

            if step % 25 == 0 or step == 1:
                _diag_line()

    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f"\n>>> CRASH: {e}")
    finally:
        simulation_app.close()
