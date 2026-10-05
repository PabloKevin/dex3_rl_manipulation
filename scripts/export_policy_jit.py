#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Export an rsl_rl checkpoint (model_*.pt) to a TorchScript policy without
starting Isaac Sim -- what play.py writes to <run>/exported/policy.pt, for
the benches / grasp hand-off to load: obs -> (obs - mean) / (std + 1e-2) ->
actor MLP (ELU) -> deterministic action.

  python scripts/export_policy_jit.py logs/rsl_rl/apple_reach_tree/<run>/model_700.pt
      -> logs/rsl_rl/apple_reach_tree/<run>/exported/policy_700.pt
  --check <policy.pt>   compare against an existing export on random inputs
"""
import argparse
import os

import torch


class _Policy(torch.nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
        mods = []
        for j, i in enumerate(layers):
            w = sd[f"actor.{i}.weight"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(sd[f"actor.{i}.bias"])
            mods.append(lin)
            if j < len(layers) - 1:
                mods.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*mods)
        n_in = sd[f"actor.{layers[0]}.weight"].shape[1]
        mean = sd.get("actor_obs_normalizer._mean", torch.zeros(1, n_in))
        std = sd.get("actor_obs_normalizer._std", torch.ones(1, n_in))
        self.register_buffer("mean", mean.clone().float())
        self.register_buffer("std", std.clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor((x - self.mean) / (self.std + 1e-2))


def export(ckpt: str, out: str | None = None) -> str:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    pol = _Policy(sd).eval()
    if out is None:
        it = os.path.basename(ckpt).split("_")[-1].split(".")[0]
        out = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "exported", f"policy_{it}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.jit.script(pol).save(out)
    return out


def input_dim(policy_path: str) -> int:
    """Observation size of an exported policy (first Linear layer)."""
    m = torch.jit.load(policy_path, map_location="cpu")
    for name, p in m.named_parameters():
        if p.dim() == 2:
            return int(p.shape[1])
    raise ValueError(f"no linear layer in {policy_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", default=None, help="existing exported policy to compare against")
    a = ap.parse_args()
    path = export(a.checkpoint, a.out)
    print(f"[export] {a.checkpoint} -> {path} (obs {input_dim(path)})")
    if a.check:
        mine, ref = torch.jit.load(path), torch.jit.load(a.check, map_location="cpu")
        x = torch.randn(4096, input_dim(path)) * 2.0
        with torch.no_grad():
            d = (mine(x) - ref(x)).abs().max().item()
        print(f"[export] max |difference| vs {a.check}: {d:.2e}")
