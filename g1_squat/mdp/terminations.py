"""Fall detection for the squat (the pelvis legitimately goes down to ~0.35 m)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bodies_below_height(env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Any of the given bodies' origins below a height (e.g. a knee touching the ground)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.any(robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2] < minimum_height, dim=1)


def body_tilt_exceeds(
    env: ManagerBasedRLEnv, max_pitch: float, max_roll: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Body leaning beyond max_pitch (forward/back) or max_roll (sideways), rad."""
    robot: Articulation = env.scene[asset_cfg.name]
    g = quat_apply_inverse(robot.data.body_link_quat_w[:, asset_cfg.body_ids[0]], robot.data.GRAVITY_VEC_W)
    # gravity in the body frame: (0, 0, -1) when upright; x component <- pitch, y <- roll
    pitch = torch.atan2(g[:, 0], -g[:, 2]).abs()
    roll = torch.asin(g[:, 1].clamp(-1.0, 1.0)).abs()
    return (pitch > max_pitch) | (roll > max_roll)
