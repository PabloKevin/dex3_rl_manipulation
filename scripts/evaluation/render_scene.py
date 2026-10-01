# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Render third-person stills of the environment for the paper's Figure 1.

Adds a fixed world-frame camera to the scene (the env's own head_camera is
first-person and disabled in this phase), steps the sim a few frames so the
robot settles into its initial pose and the per-env randomization has been
applied, then writes one PNG per environment plus a stitched contact sheet.

    PY=/home/pablo_kevin/miniconda3/envs/isaaclab_curobo/bin/python
    $PY scripts/evaluation/render_scene.py --headless --enable_cameras --num_envs 3

Needs --enable_cameras; without it Isaac Sim does not create the render product
and the camera returns empty tensors.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render scene stills for the paper.")
parser.add_argument("--task", type=str, default="AppleReach-v1")
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument("--settle_steps", type=int, default=12,
                    help="Sim steps before capture, so the arm settles into its initial pose.")
parser.add_argument("--width", type=int, default=900)
parser.add_argument("--height", type=int, default=760)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--eye", type=float, nargs=3, default=[1.35, -1.25, 1.55],
                    help="Camera position relative to each env origin.")
parser.add_argument("--look", type=float, nargs=3, default=[0.30, -0.30, 1.00],
                    help="Camera target relative to each env origin.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True     # required for any render product
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
import envs  # noqa: E402,F401

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    outdir = args_cli.outdir or os.path.join(_REPO, "papers", "reach_apple", "figures")
    os.makedirs(outdir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # A world-frame camera per env. The env's head_camera is mounted on the
    # robot's head and is None in this phase; this one is a static observer.
    env_cfg.scene.paper_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/PaperCam",
        update_period=0.0,
        width=args_cli.width,
        height=args_cli.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0, focus_distance=2.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=tuple(args_cli.eye), convention="world"),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    u.reset()

    cam = u.scene["paper_camera"]
    origins = u.scene.env_origins
    eye = origins + torch.tensor(args_cli.eye, device=u.device)
    target = origins + torch.tensor(args_cli.look, device=u.device)
    cam.set_world_poses_from_view(eye, target)

    zero = torch.zeros(u.num_envs, u.action_space.shape[1], device=u.device)
    for _ in range(args_cli.settle_steps):
        env.step(zero)

    rgb = cam.data.output["rgb"][..., :3].detach().cpu().numpy().astype(np.uint8)
    print(f"[INFO] captured {rgb.shape[0]} frames of {rgb.shape[1]}x{rgb.shape[2]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for i in range(rgb.shape[0]):
        p = os.path.join(outdir, f"scene_env{i}.png")
        plt.imsave(p, rgb[i])
        paths.append(p)
        print(f"[INFO] wrote {p}")

    # Contact sheet: the three panels (a-c) of the paper's Figure 1. Panel (d)
    # is the geometry schematic from plot_results.py, composed in LaTeX.
    n = rgb.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.4 * args_cli.height / args_cli.width))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.imshow(rgb[i])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.02, 0.97, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="white", va="top",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5))
    fig.subplots_adjust(wspace=0.015, left=0, right=1, top=1, bottom=0)
    sheet = os.path.join(outdir, "scene_strip.png")
    fig.savefig(sheet, dpi=200, bbox_inches="tight", pad_inches=0.01)
    print(f"[INFO] wrote {sheet}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
