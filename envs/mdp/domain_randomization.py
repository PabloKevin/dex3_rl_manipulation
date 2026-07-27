# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Cheap per-episode domain randomization: apple mass, apple scale (~radius),
and stem break force/torque. All ops are either the built-in tensor mass API
or plain-Python values — no per-step USD writes, no static-collider re-cook risk.
"""
from __future__ import annotations
import random
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

RAND_PCT = 0.20


def _range(nominal: float, pct: float = RAND_PCT):
    return nominal * (1 - pct), nominal * (1 + pct)


def randomize_apple_scale(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    nominal_radius: float,
) -> None:
    """Uniform scale on the apple RigidObject -> effective radius change.
    Same mechanism as fruit_utils.randomize_fruit_scale_and_mass, radius-only.
    """
    try:
        from pxr import UsdGeom, Gf
        import omni.usd
    except ImportError:
        return

    asset: RigidObject = env.scene[object_cfg.name]
    stage = omni.usd.get_context().get_stage()
    lo, hi = _range(1.0)  # scale factor range, not absolute radius

    for idx in env_ids.tolist():
        s = random.uniform(lo, hi)
        prim_path = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{idx}")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim)
        scale_op = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        if scale_op is None:
            scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(s, s, s))
        # keep env-side bookkeeping in sync, in case other code needs the
        # effective radius (e.g. reward shaping, stem anchor offset)
        if not hasattr(env, "_apple_radius"):
            env._apple_radius = torch.full((env.num_envs,), nominal_radius, device=env.device)
        env._apple_radius[idx] = nominal_radius * s


def randomize_stem_break_params(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    nominal_force: float = 2.5,
    nominal_torque: float = 0.5,
) -> None:
    """Pure-Python per-env break thresholds, read by StemManager._check_break."""
    if not hasattr(env, "_stem_break_force"):
        env._stem_break_force = torch.full((env.num_envs,), nominal_force, device=env.device)
    if not hasattr(env, "_stem_break_torque"):
        env._stem_break_torque = torch.full((env.num_envs,), nominal_torque, device=env.device)

    lo_f, hi_f = _range(nominal_force)
    lo_t, hi_t = _range(nominal_torque)
    for idx in env_ids.tolist():
        env._stem_break_force[idx] = random.uniform(lo_f, hi_f)
        env._stem_break_torque[idx] = random.uniform(lo_t, hi_t)