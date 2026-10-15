"""reach_v1.py -- the harvest test bench's reach policy on the robot (run_toy.py v1).

scripts/harvest_test_bench.py runs AppleGrasp-v1, whose reach phase is the FROZEN v1 reach
(envs/mdp/grasp_autoplay.PINNED_REACH_POLICY = envs/agents/best_weights/
grasp_handoff_reach_apple.pt; apple_grasp_reach 2026-09-18, commit 5f6276a): right arm only,
29 inputs / 7 outputs. grasp_autoplay._reach_obs layout "v1" (= apple_grasp_env_cfg
ObservationsCfg.PolicyObs):
  arm q - default (7), arm dq (7), apple centre - palm point (3), palm quat in the torso
  frame (4), previous raw action (7), palm contact force from the apple (1) = 29
That is reach_io's "tree" vector (37) without its last 8 entries (the clearance ring).
The action is the same as "tree": target = default + 0.5 x raw, rate-limited at 2.0 rad/s,
8.0 rad/s^2 (AppleGrasp-v1 right_arm: reach_scale 0.5, max_velocity 2.0, max_acceleration 8.0;
reach_io ARM_SCALE / ARM_MAX_VELOCITY / ARM_MAX_ACCELERATION), palm point (0.08, 0, 0).
One difference kept: the sim clips the target to +-3.0 rad, the toy to the joint limits
(tighter; the robot cannot go further anyway).

install() adds the layout to reach_io IN THIS PROCESS ONLY: reach_io.py and toy_reach.py are
not changed, and every other toy (single, reach, apple, ...) runs exactly as before.
"""
from __future__ import annotations

import os

V1_DIMS = (29, 7)
RING = 8


def default_policy(repo: str) -> str:
    return os.path.join(repo, "envs", "agents", "best_weights", "grasp_handoff_reach_apple.pt")


def install(R) -> None:
    if getattr(R, "_v1_installed", False):
        return
    R._v1_installed = True
    R.LAYOUTS = {**R.LAYOUTS, V1_DIMS[0]: "v1"}
    R.LAYOUT_DIMS = dict(R.LAYOUT_DIMS, v1=V1_DIMS)
    build = R.build_obs

    def build_obs(layout, kin, st, prev_raw, fr=None):
        if layout != "v1":
            return build(layout, kin, st, prev_raw, fr)
        obs, fr = build("tree", kin, st, prev_raw, fr)
        return obs[:-RING].copy(), fr
    R.build_obs = build_obs
