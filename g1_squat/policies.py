"""Policy loaders for the hand-over test (no onnxruntime needed).

UnitreeVelocityPolicy  — unitree_rl_lab's deployed G1-29dof velocity policy (v0 ONNX +
                         params/deploy.yaml), observation built exactly as deploy.yaml says.
load_squat_actor       — the squat actor MLP from an rsl_rl checkpoint (model_*.pt).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import yaml

from g1_squat import robot_cfg as rc

UNITREE_VELOCITY_DIR = os.path.expanduser("~/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0")


def _mlp(weights: list[tuple[torch.Tensor, torch.Tensor]]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (w, b) in enumerate(weights):
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(weights) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def load_onnx_mlp(path: str) -> nn.Sequential:
    """Gemm/Elu chain exported by Isaac Lab (actor only) -> torch."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(path).graph
    init = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in graph.initializer}
    weights = []
    for node in graph.node:
        if node.op_type == "Gemm":
            weights.append((init[node.input[1]].float(), init[node.input[2]].float()))
        elif node.op_type not in ("Elu",):
            raise ValueError(f"unexpected ONNX op {node.op_type}")
    return _mlp(weights)


def load_squat_actor(checkpoint: str) -> nn.Sequential:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight")})
    if any(k.startswith("actor_obs_normalizer") for k in state):
        raise NotImplementedError("squat actor trained with obs normalization; export it with play.py instead")
    return _mlp([(state[f"actor.{i}.weight"], state[f"actor.{i}.bias"]) for i in idx])


class UnitreeVelocityPolicy:
    """Runs the deployed velocity policy on an Isaac Lab Articulation (G1 with or without hands)."""

    def __init__(self, robot, device: str, policy_dir: str = UNITREE_VELOCITY_DIR):
        with open(os.path.join(policy_dir, "params", "deploy.yaml")) as f:
            p = yaml.safe_load(f)
        self.net = load_onnx_mlp(os.path.join(policy_dir, "exported", "policy.onnx")).to(device).eval()
        self.robot = robot
        self.device = device
        self.policy_joints = [rc.SDK_JOINTS[i] for i in p["joint_ids_map"]]
        self.joint_ids, _ = robot.find_joints(self.policy_joints, preserve_order=True)
        self.default = torch.tensor(p["default_joint_pos"], device=device)
        self.scale = torch.tensor(p["actions"]["JointPositionAction"]["scale"], device=device)
        self.offset = torch.tensor(p["actions"]["JointPositionAction"]["offset"], device=device)
        obs_cfg = p["observations"]
        self.terms = list(obs_cfg.keys())
        self.term_scale = {k: torch.tensor(v["scale"], device=device) for k, v in obs_cfg.items()}
        self.history = {k: v.get("history_length", 1) for k, v in obs_cfg.items()}
        self.hist: dict[str, torch.Tensor] | None = None
        self.last_action = torch.zeros(robot.num_instances, len(self.joint_ids), device=device)

    def reset(self):
        """Call when the policy takes over: history is refilled with the first frame (Isaac Lab/deploy behaviour)."""
        self.hist = None
        self.last_action.zero_()

    def _frame(self, cmd: torch.Tensor) -> dict[str, torch.Tensor]:
        d = self.robot.data
        raw = {
            "base_ang_vel": d.root_ang_vel_b,
            "projected_gravity": d.projected_gravity_b,
            "velocity_commands": cmd,
            "joint_pos_rel": d.joint_pos[:, self.joint_ids] - self.default,
            "joint_vel_rel": d.joint_vel[:, self.joint_ids],
            "last_action": self.last_action,
        }
        return {k: raw[k] * self.term_scale[k] for k in self.terms}

    @torch.no_grad()
    def act(self, cmd: torch.Tensor) -> torch.Tensor:
        """cmd (N, 3) = vx, vy, wz. Returns joint position targets (N, 29) for self.joint_ids."""
        frame = self._frame(cmd)
        if self.hist is None:
            self.hist = {k: v.unsqueeze(1).repeat(1, self.history[k], 1) for k, v in frame.items()}
        else:
            for k, v in frame.items():
                self.hist[k] = torch.cat([self.hist[k][:, 1:], v.unsqueeze(1)], dim=1)
        obs = torch.cat([self.hist[k].flatten(1) for k in self.terms], dim=1)
        action = self.net(obs)
        self.last_action = action
        return action * self.scale + self.offset
