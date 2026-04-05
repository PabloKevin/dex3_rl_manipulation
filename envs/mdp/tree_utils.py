# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Tree utilities: create the breakable stem joint between the branch tip and apple.

The stem joint is a PhysX D6 joint with:
  - All translational DOFs locked  (apple position fixed relative to branch tip)
  - All rotational DOFs locked
  - Linear break force:  APPLE_STEM_BREAK_FORCE  N
  - Angular break torque: APPLE_STEM_BREAK_TORQUE N·m

When the robot applies enough downward pull on the apple the joint breaks
and the apple becomes a free rigid body subject to gravity.

Usage in scripts/run_env.py:
    from envs.mdp.tree_utils import create_stem_joints, reset_stem_joints

    # After env.reset() — call once per episode reset:
    stem_joints = create_stem_joints(env)

    # To check if joint broke this step:
    broken = check_stem_broken(env, stem_joints)
"""

from __future__ import annotations
import torch
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def create_stem_joints(env: ManagerBasedRLEnv) -> list:
    """Create a breakable fixed joint between the branch tip and each apple.

    Must be called AFTER env.reset() so all prims exist on the stage.
    Returns a list of joint prim paths (one per env) for later queries.

    How it works:
      - Uses omni.physx USD APIs to create a PhysicsJoint prim
      - Joint type: Fixed (all DOF locked)
      - Break force and torque set to APPLE_STEM_BREAK_FORCE/TORQUE
      - body0 = world frame (None) so branch tip position is fixed in world
      - body1 = Apple rigid body prim

    Note: In Isaac Lab 4.5 / Isaac Sim 4.5, joint break is supported via
    PhysicsJoint.breakForce and breakTorque USD attributes.
    """
    try:
        from pxr import UsdPhysics, Gf, UsdGeom, Sdf
        import omni.usd
    except ImportError:
        print("[tree_utils] WARNING: pxr not available — stem joints skipped.")
        print("  This is OK for pure Python testing; run via isaaclab.sh for physics.")
        return []

    from envs.apple_grasp_env_cfg import (
        APPLE_INIT_POS,
        APPLE_STEM_BREAK_FORCE,
        APPLE_STEM_BREAK_TORQUE,
    )

    stage = omni.usd.get_context().get_stage()
    joint_paths = []

    for env_idx in range(env.num_envs):
        env_ns    = f"/World/envs/env_{env_idx}"
        apple_path = f"{env_ns}/Apple"
        joint_path = f"{env_ns}/AppleStemJoint"

        # Remove old joint if it exists (e.g. after episode reset)
        if stage.GetPrimAtPath(joint_path):
            stage.RemovePrim(joint_path)

        # Create a fixed joint prim
        joint_prim = UsdPhysics.FixedJoint.Define(stage, joint_path)

        # body0: world (None = fixed in world frame)
        # body1: the apple rigid body
        joint_prim.GetBody0Rel().SetTargets([])
        joint_prim.GetBody1Rel().SetTargets([Sdf.Path(apple_path)])

        # Local pose on body1 (apple): joint anchor at apple centre
        joint_prim.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        anchor_pos = Gf.Vec3f(*APPLE_INIT_POS)
        joint_prim.GetLocalPos0Attr().Set(anchor_pos)

        # ── Break force and torque ──────────────────────────────────────
        # PhysX exposes these as custom attributes on the joint prim.
        joint_api = joint_prim.GetPrim()
        if not joint_api.HasAttribute("physics:breakForce"):
            joint_api.CreateAttribute(
                "physics:breakForce", type=Sdf.ValueTypeNames.Float
            ).Set(APPLE_STEM_BREAK_FORCE)
        else:
            joint_api.GetAttribute("physics:breakForce").Set(APPLE_STEM_BREAK_FORCE)

        if not joint_api.HasAttribute("physics:breakTorque"):
            joint_api.CreateAttribute(
                "physics:breakTorque", type=Sdf.ValueTypeNames.Float
            ).Set(APPLE_STEM_BREAK_TORQUE)
        else:
            joint_api.GetAttribute("physics:breakTorque").Set(APPLE_STEM_BREAK_TORQUE)

        joint_paths.append(joint_path)
        print(f"[tree_utils] Created stem joint at {joint_path} "
              f"(break force: {APPLE_STEM_BREAK_FORCE} N)")

    return joint_paths


def reset_stem_joints(env: ManagerBasedRLEnv, joint_paths: list) -> list:
    """Recreate stem joints for all envs (call on episode reset).

    Returns a fresh list of joint paths.
    """
    return create_stem_joints(env)


def check_stem_broken(env: ManagerBasedRLEnv, joint_paths: list) -> torch.Tensor:
    """Check which envs have a broken stem joint.

    Returns bool tensor [num_envs] — True if stem is broken in that env.

    A joint is considered broken if the prim no longer exists on the stage
    (PhysX removes it automatically when break force is exceeded).
    """
    if not joint_paths:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    except ImportError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    broken = []
    for path in joint_paths:
        prim = stage.GetPrimAtPath(path)
        # Joint is broken if prim is gone or has been deactivated
        broken.append(not prim.IsValid() or not prim.IsActive())

    return torch.tensor(broken, dtype=torch.bool, device=env.device)