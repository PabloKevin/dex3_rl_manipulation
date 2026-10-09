"""Residual joint-position action around the squat reference.

target = q_ref(s) + scale * a, for the 15 leg + waist joints, where q_ref(s) is the
stand->squat interpolation driven by the squat command's phase. At s = 0 this is
exactly Unitree's velocity-policy action convention (default pose + 0.25 * a).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SquatResidualJointAction(JointPositionAction):
    cfg: SquatResidualJointActionCfg

    def __init__(self, cfg: SquatResidualJointActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._squat = None

    def process_actions(self, actions: torch.Tensor):
        if self._squat is None:
            self._squat = self._env.command_manager.get_term(self.cfg.command_name)
            if list(self._squat.joint_names) != list(self._joint_names):
                raise ValueError(
                    f"action joints {self._joint_names} must match the squat reference joints "
                    f"{self._squat.joint_names} (use preserve_order=True)"
                )
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._squat.ref_joint_pos
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class SquatResidualJointActionCfg(JointPositionActionCfg):
    class_type: type = SquatResidualJointAction
    command_name: str = "squat"
