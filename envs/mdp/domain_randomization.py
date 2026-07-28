# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Cheap per-episode domain randomization: apple mass, apple scale (~radius),
and stem break force/torque. All ops are either the built-in tensor mass API
or plain-Python values — no per-step USD writes, no static-collider re-cook risk.
"""
from __future__ import annotations
import random
import math
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
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    nominal_radius: float,
) -> None:
    """Uniform scale on the apple RigidObject -> effective radius change.
    Same mechanism as fruit_utils.randomize_fruit_scale_and_mass, radius-only.

    Runs at mode="prestartup" — i.e. before sim.reset() creates the GPU tensor
    views — since this is a raw USD write and would otherwise crash/invalidate
    the shared PhysX tensor view if done at "reset" time. Because of that,
    env_ids arrives as None (see EventManager.apply's generic-mode branch);
    treat that as "all envs".
    """
    try:
        from pxr import UsdGeom, Gf
        import omni.usd
    except ImportError:
        return

    asset: RigidObject = env.scene[object_cfg.name]
    stage = omni.usd.get_context().get_stage()
    lo, hi = _range(1.0)  # scale factor range, not absolute radius

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    for idx in env_ids.tolist():
        s = random.uniform(lo, hi)
        prim_path = f"/World/envs/env_{idx}/Apple"   # ← hardcode, don't use asset.cfg.prim_path
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

def _get_or_add_xform_op(prim, op_type):
    from pxr import UsdGeom
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return xform.AddXformOp(op_type)


def _set_translate(prim, x, y, z):
    from pxr import UsdGeom, Gf
    _get_or_add_xform_op(prim, UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(x, y, z))


def _set_rotate_z(prim, angle_rad):
    from pxr import UsdGeom, Gf
    op = _get_or_add_xform_op(prim, UsdGeom.XformOp.TypeOrient)
    half = angle_rad / 2.0
    op.Set(Gf.Quatf(math.cos(half), 0.0, 0.0, math.sin(half)))


def _find_cylinder(stage, base_path):
    """Cylinder geom may live on the prim itself or under a 'geometry' child,
    depending on your Isaac Lab spawner version — try both."""
    from pxr import UsdGeom
    for path in (base_path, f"{base_path}/geometry"):
        prim = stage.GetPrimAtPath(path)
        cyl = UsdGeom.Cylinder(prim) if prim.IsValid() else None
        if cyl:
            return prim, cyl
    return None, None


def randomize_tree_and_apple(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    tree_pos_xy: tuple,          # fixed (x, y) — keep the trunk base put
    tree_height_nominal: float,
    branch_length_nominal: float,
    branch_radius_nominal: float,
    branch_rotation_deg_nominal: float,
    apple_radius_nominal: float,
) -> None:
    try:
        import omni.usd
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    apple: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "_stem_anchor_pos"):
        env._stem_anchor_pos = torch.zeros(env.num_envs, 3, device=env.device)

    h_lo, h_hi = _range(tree_height_nominal)
    l_lo, l_hi = _range(branch_length_nominal)
    r_lo, r_hi = _range(branch_rotation_deg_nominal)

    apple_pos_local = torch.zeros(len(env_ids), 3, device=env.device)

    for k, idx in enumerate(env_ids.tolist()):
        tree_h = random.uniform(h_lo, h_hi)
        branch_len = random.uniform(l_lo, l_hi)
        branch_rot_deg = random.uniform(r_lo, r_hi)
        branch_rot_rad = math.radians(branch_rot_deg)

        tree_x, tree_y = tree_pos_xy
        branch_mid_x = tree_x + (branch_len / 2.0) * math.cos(branch_rot_rad)
        branch_mid_y = tree_y + (branch_len / 2.0) * math.sin(branch_rot_rad)
        branch_tip_x = tree_x + branch_len * math.cos(branch_rot_rad)
        branch_tip_y = tree_y + branch_len * math.sin(branch_rot_rad)
        apple_z = tree_h - branch_radius_nominal - apple_radius_nominal

        env_ns = f"/World/envs/env_{idx}"

        trunk_prim, trunk_cyl = _find_cylinder(stage, f"{env_ns}/TreeTrunk")
        if trunk_cyl:
            trunk_cyl.GetHeightAttr().Set(tree_h)
            _set_translate(trunk_prim, tree_x, tree_y, tree_h / 2.0)

        branch_prim, branch_cyl = _find_cylinder(stage, f"{env_ns}/TreeBranch")
        if branch_cyl:
            branch_cyl.GetHeightAttr().Set(branch_len)
            _set_translate(branch_prim, branch_mid_x, branch_mid_y, tree_h)
            _set_rotate_z(branch_prim, branch_rot_rad)

        apple_pos_local[k] = torch.tensor(
            [branch_tip_x, branch_tip_y, apple_z], device=env.device
        )

    env_origins = env.scene.env_origins[env_ids]
    root_state = apple.data.default_root_state[env_ids].clone()
    root_state[:, :3] = env_origins + apple_pos_local
    root_state[:, 7:] = 0.0
    apple.write_root_state_to_sim(root_state, env_ids=env_ids)

    env._stem_anchor_pos[env_ids] = env_origins + apple_pos_local