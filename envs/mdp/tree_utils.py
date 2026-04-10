# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Tree utilities: breakable stem joint between branch tip and apple.

Behaviour (three phases per env):
  HELD       — Apple is locked to branch tip via a FixedJoint.
               Contact force on apple is monitored each step.
               When net downward force > BREAK_FORCE the joint is deleted
               and the stem enters STRETCHING phase.

  STRETCHING — Joint is gone. A spring force pulls the apple back toward
               the anchor point. Spring constant weakens as stretch grows,
               simulating a twig bending then snapping. When stretch
               exceeds SNAP_DISTANCE the spring is set to zero -> RELEASED.

  RELEASED   — Apple is a free rigid body under gravity.

Usage in run_env.py:

    from envs.mdp.tree_utils import StemManager
    stem = StemManager(env)          # call once after env.reset()

    # inside the step loop:
    stem.update(env)
    broken = stem.is_broken()        # list of bools [num_envs]

    # on episode reset:
    stem.reset(env)
"""

from __future__ import annotations
import torch
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── tuneable constants ────────────────────────────────────────────────────────
BREAK_FORCE    = 5.0    # N   — contact force needed to start stretching
SPRING_K       = 40.0   # N/m — spring stiffness during stretch phase
SPRING_DAMPING = 2.0    # N*s/m
SNAP_DISTANCE  = 0.12   # m   — stretch at which spring snaps (apple released)


class _StemState:
    def __init__(self):
        self.phase: str = "HELD"          # HELD | STRETCHING | RELEASED
        self.joint_path: str = ""
        self.anchor_pos: Optional[torch.Tensor] = None


class StemManager:
    """Manages breakable elastic stem joints for all envs."""

    def __init__(self, env: "ManagerBasedRLEnv"):
        self._states: List[_StemState] = []
        self.reset(env)

    # ── public API ───────────────────────────────────────────────────────

    def reset(self, env: "ManagerBasedRLEnv") -> None:
        """(Re)create all stem joints. Call after env.reset()."""
        self._states = []
        for env_idx in range(env.num_envs):
            state = _StemState()
            state.anchor_pos = self._get_anchor(env, env_idx)
            state.joint_path = self._create_joint(env, env_idx, state.anchor_pos)
            state.phase = "HELD"
            self._states.append(state)

    def update(self, env: "ManagerBasedRLEnv") -> None:
        """Call once per simulation step."""
        apple = env.scene["apple"]
        for idx, state in enumerate(self._states):
            if state.phase == "HELD":
                self._check_break(env, apple, idx, state)
            elif state.phase == "STRETCHING":
                self._apply_spring(env, apple, idx, state)

    def is_broken(self) -> List[bool]:
        return [s.phase == "RELEASED" for s in self._states]

    # ── internals ────────────────────────────────────────────────────────

    def _get_anchor(self, env, env_idx: int) -> torch.Tensor:
        from envs.apple_grasp_env_cfg import APPLE_INIT_POS
        origin = env.scene.env_origins[env_idx]
        local  = torch.tensor(list(APPLE_INIT_POS),
                               device=env.device, dtype=torch.float32)
        return origin + local

    def _create_joint(self, env, env_idx: int,
                      anchor_pos: torch.Tensor) -> str:
        try:
            from pxr import UsdPhysics, Gf, Sdf
            import omni.usd
        except ImportError:
            return ""

        stage      = omni.usd.get_context().get_stage()
        apple_path = f"/World/envs/env_{env_idx}/Apple"
        joint_path = f"/World/envs/env_{env_idx}/AppleStemJoint"

        if stage.GetPrimAtPath(joint_path):
            stage.RemovePrim(joint_path)

        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.GetBody0Rel().SetTargets([])
        joint.GetBody1Rel().SetTargets([Sdf.Path(apple_path)])
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.GetLocalPos0Attr().Set(
            Gf.Vec3f(float(anchor_pos[0]),
                     float(anchor_pos[1]),
                     float(anchor_pos[2]))
        )
        # We do NOT use physics:breakForce — unreliable in Isaac Sim 4.5.
        # Force monitoring is done manually in _check_break().
        print(f"[tree_utils] env_{env_idx}: stem joint created")
        return joint_path

    def _delete_joint(self, joint_path: str) -> None:
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath(joint_path):
                stage.RemovePrim(joint_path)
        except Exception:
            pass

    def _check_break(self, env, apple, idx: int,
                     state: _StemState) -> None:
        """Break joint when contact force exceeds threshold."""
        try:
            sensor     = env.scene["contact_forces"]
            forces     = sensor.data.net_forces_w[idx]   # [num_bodies, 3]
            net_force  = forces.norm(dim=-1).sum().item()
        except Exception:
            net_force = 0.0

        if net_force >= BREAK_FORCE:
            print(f"[tree_utils] env_{idx}: BREAKING "
                  f"(force={net_force:.1f} N)")
            self._delete_joint(state.joint_path)
            state.phase = "STRETCHING"

    def _apply_spring(self, env, apple, idx: int,
                      state: _StemState) -> None:
        """Weakening spring during elastic stretch phase."""
        apple_pos  = apple.data.root_pos_w[idx]      # [3]
        stretch    = apple_pos - state.anchor_pos     # [3]
        dist       = stretch.norm().item()

        if dist >= SNAP_DISTANCE:
            print(f"[tree_utils] env_{idx}: SNAPPED — apple free")
            state.phase = "RELEASED"
            return

        # Spring weakens linearly toward snap distance
        k_eff     = SPRING_K * max(0.0, 1.0 - dist / SNAP_DISTANCE)
        spring_f  = -k_eff * stretch

        # Velocity damping
        apple_vel = apple.data.root_lin_vel_w[idx]   # [3]
        damping_f = -SPRING_DAMPING * apple_vel

        total_f   = (spring_f + damping_f).unsqueeze(0).unsqueeze(0)  # [1,1,3]
        zero_t    = torch.zeros_like(total_f)

        env_ids   = torch.tensor([idx], device=env.device, dtype=torch.long)

        try:
            apple.set_external_force_and_torque(
                forces=total_f,
                torques=zero_t,
                body_ids=[0],
                env_ids=env_ids,
            )
        except Exception as e:
            # Isaac Lab 4.5 alternate signature
            try:
                forces_full  = torch.zeros(
                    env.num_envs, apple.num_bodies, 3, device=env.device)
                torques_full = torch.zeros_like(forces_full)
                forces_full[idx, 0] = spring_f + damping_f
                apple.set_external_force_and_torque(
                    forces=forces_full, torques=torques_full)
            except Exception as e2:
                pass   # spring silently disabled if API unavailable