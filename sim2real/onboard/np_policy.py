"""np_policy.py -- an exported rsl_rl reach policy in numpy, for the Jetson (no torch).

The TorchScript policy.pt (scripts/export_policy_jit.py / play.py) is
    x -> (x - mean) / (std + eps) -> [Linear, ELU] x3 -> Linear
sim2real/export_policy_npz.py writes its weights to a .npz and checks this forward pass
against the TorchScript one. Batch 1, float32 like the export, ~25k multiply-adds: well
under a millisecond on the Jetson's CPU. Python 3.8.
"""
from __future__ import annotations

import numpy as np


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class NumpyPolicy:
    """Same interface as reach_io.ReachPolicy: n_in, n_out, layout, __call__(obs) -> raw."""

    def __init__(self, path: str, layout: str = ""):
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.mean = d["mean"].astype(np.float32).reshape(-1)
        self.std = d["std"].astype(np.float32).reshape(-1)
        self.eps = np.float32(d["eps"])
        n = int(d["n_layers"])
        self.Ws = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.bs = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.n_in, self.n_out = int(self.Ws[0].shape[1]), int(self.Ws[-1].shape[0])
        self.layout = layout or (str(d["layout"]) if "layout" in d.files else "")
        self.source = str(d["source"]) if "source" in d.files else ""

    def __call__(self, obs) -> np.ndarray:
        h = (np.asarray(obs, np.float32).reshape(-1) - self.mean) / (self.std + self.eps)
        last = len(self.Ws) - 1
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            h = W @ h + b
            if i < last:
                h = _elu(h)
        return h.astype(float)
