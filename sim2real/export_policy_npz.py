#!/usr/bin/env python3
"""export_policy_npz.py -- a TorchScript reach policy (policy.pt) -> .npz for the Jetson runner
(sim2real/onboard/np_policy.py: numpy only, no torch on the robot).

Reads the normalizer (mean, std, eps) and the Linear layers, writes them with the layout name
and the source path, then CHECKS the numpy forward pass against the TorchScript one on random
observations (max |difference| must stay under 1e-4) and on zeros.

  python3 sim2real/export_policy_npz.py --policy <policy.pt> --layout single [--out <file.npz>]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "onboard"))
from np_policy import NumpyPolicy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="TorchScript policy.pt")
    ap.add_argument("--layout", required=True, help="reach_io layout name (single, single_tactile, ...)")
    ap.add_argument("--out", default=None,
                    help="default: sim2real/onboard/policies/<experiment>_<run>_<file>.npz (the name run_toy "
                         "--onboard asks the robot for)")
    ap.add_argument("--n_check", type=int, default=2000)
    a = ap.parse_args()
    import torch

    net = torch.jit.load(a.policy, map_location="cpu").eval()
    sd = {k: v.detach().cpu().numpy() for k, v in list(net.named_parameters()) + list(net.named_buffers())}
    lin = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("actor.") and k.endswith(".weight")},
                 key=lambda s: int(s.split(".")[1]))
    acts = [getattr(net.actor, str(i)).original_name for i in range(len(list(net.actor.children())))]
    if any(n not in ("Linear", "ELU") for n in acts):
        sys.exit(f"unexpected actor modules {acts}: np_policy.py does Linear + ELU only")
    eps = float(getattr(net.normalizer, "eps", 1e-2))
    out = {"mean": sd["normalizer._mean"], "std": sd["normalizer._std"], "eps": np.float32(eps),
           "n_layers": np.int64(len(lin)), "layout": np.str_(a.layout),
           "source": np.str_(os.path.abspath(a.policy))}
    for i, k in enumerate(lin):
        out[f"W{i}"] = sd[f"{k}.weight"]
        out[f"b{i}"] = sd[f"{k}.bias"]
    if a.out is None:
        sys.path.insert(0, os.path.join(HERE, "dual"))
        from onboard_link import npz_name
        os.makedirs(os.path.join(HERE, "onboard", "policies"), exist_ok=True)
        a.out = os.path.join(HERE, "onboard", "policies", npz_name(a.policy))
    path = a.out
    np.savez(path, **out)

    pol = NumpyPolicy(path)
    rng = np.random.default_rng(0)
    obs = rng.normal(0.0, 1.0, (a.n_check, pol.n_in)).astype(np.float32) * pol.std + pol.mean
    obs[0] = 0.0
    with torch.no_grad():
        ref = net(torch.as_tensor(obs)).numpy()
    got = np.stack([pol(o) for o in obs])
    err = float(np.max(np.abs(got - ref)))
    print(f"{path}: {pol.n_in} in / {pol.n_out} out, {len(lin)} layers, eps {eps:g}; "
          f"numpy vs TorchScript on {a.n_check} observations: max |diff| {err:.2e}")
    if err > 1e-4:
        print("MISMATCH: do not deploy this file")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
