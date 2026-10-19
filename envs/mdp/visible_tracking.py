# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
AppleReachVisible-v2: the head camera keeps tracking the apple during the
reach, and the reach may start from any clear arm pose.

Camera tracking (step_camera_tracking, every policy step). v1 perceives the
apple ONCE at the start and holds it (the robot's toys detect at home). Here
env._apple_meas -- what apple_pos_palm_measured_torso feeds the policy -- is
refreshed like the robot's detector would refresh it:
  - a detection every DETECT period (per env, ~10 Hz, the throttled colour
    stream), of where the apple was LATENCY ago (per episode: transport +
    inference + depth);
  - only when the camera sees the apple whole (visible_reach.sees_t) and no
    arm / hand point sits on the camera -> apple segment in front of it
    (occluded: the measurement is held, as a tracker would);
  - a hand ALMOST in the way (within one apple radius more) shifts the
    centre along the viewing ray (a box cut by the hand, the depth of the
    hand in it), and a random share of frames is missed;
  - per-detection noise on top of the episode's bias
    (single_reach.reset_apple_measurement: calibration, radius), which
    stays: tracking follows the apple's MOTION (a bump, the stem swinging),
    it does not undo the calibration error;
  - a share of episodes never update (frozen_prob): the v1 behaviour, so the
    policy still works with today's toys that keep one point.

Wide starts (reset_arm_wide, after the tree is placed): with probability
`prob` the right arm starts at a pose of the arm pose library
(scripts/build_arm_pose_library.py: full-body sphere check at the waist
yaws), clear of the body at this episode's yaw, inside the action clip (the
robot's joint limits), the grasp pocket at least `apple_clear` from every
apple and nearer the waist axis than the target by `nearer` (out of the
canopy and the trunk behind it). Otherwise the start stays home + jitter.
The policy then learns to come in from wherever it is -- the robot can
restart it after a stop instead of driving home first.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg

from .single_reach import fingertip_points
from .tree_reach import _neighbour_state
from .visible_reach import _unturn, camera_pose_w, sees_t

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#: arm / hand points that can hide the apple from the head camera, with a radius (m)
OCCLUDERS = (
    ("right_elbow_link", 0.045),
    ("right_wrist_roll_link", 0.040),
    ("right_wrist_yaw_link", 0.040),
    ("right_hand_palm_link", 0.045),
)
TIP_RADIUS = 0.015


class _Track:
    def __init__(self, env: "ManagerBasedRLEnv", hist_len: int):
        N, dev = env.num_envs, env.device
        true = env.scene["apple"].data.root_pos_w
        self.hist = true.unsqueeze(0).repeat(hist_len, 1, 1)       # [H, N, 3] true apple, newest first
        self.bias = torch.zeros(N, 3, device=dev)
        self.lat = torch.zeros(N, dtype=torch.long, device=dev)
        self.timer = torch.zeros(N, device=dev)
        self.frozen = torch.zeros(N, dtype=torch.bool, device=dev)
        self.n_frames = torch.zeros(N, device=dev)
        self.n_det = torch.zeros(N, device=dev)
        self.n_occl = torch.zeros(N, device=dev)
        self.n_out = torch.zeros(N, device=dev)
        robot = env.scene["robot"]
        names = robot.data.body_names
        occ = [(names.index(n), r) for n, r in OCCLUDERS if n in names]
        missing = [n for n, _ in OCCLUDERS if n not in names]
        if missing:
            print(f"[visible_tracking] occluder bodies not in the robot, skipped: {missing}")
        self.occ_ids = [i for i, _ in occ]
        self.occ_r = torch.tensor([r for _, r in occ], device=dev)


def reset_camera_tracking(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    latency_s: tuple[float, float],
    detect_period_s: tuple[float, float],
    frozen_prob: float,
) -> None:
    """mode="reset", after single_reach.reset_apple_measurement: the episode's
    bias (kept), latency, first detection time and frozen flag."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if len(ids) == 0:
        return
    dt = env.step_dt
    hist_len = int(np.ceil(latency_s[1] / dt)) + 2
    T = getattr(env, "_vt", None)
    if T is None:
        T = env._vt = _Track(env, hist_len)
    n, dev = len(ids), env.device
    true = env.scene["apple"].data.root_pos_w[ids]
    meas = getattr(env, "_apple_meas", None)
    T.bias[ids] = (meas[ids] - true) if meas is not None else 0.0
    T.hist[:, ids] = true.unsqueeze(0)
    lat = latency_s[0] + (latency_s[1] - latency_s[0]) * torch.rand(n, device=dev)
    T.lat[ids] = (lat / dt).round().long().clamp(0, T.hist.shape[0] - 1)
    p0, p1 = detect_period_s
    T.timer[ids] = (p0 + (p1 - p0) * torch.rand(n, device=dev)) * torch.rand(n, device=dev)
    T.frozen[ids] = torch.rand(n, device=dev) < frozen_prob
    for t in (T.n_frames, T.n_det, T.n_occl, T.n_out):
        t[ids] = 0.0


def step_camera_tracking(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    camera: dict,
    detect_period_s: tuple[float, float],
    noise_std: float,
    miss_prob: float,
    partial_shift: float,
    apple_radius: float,
) -> None:
    """mode="interval" every step (interval 0, global): push the true apple
    into the history, then for the envs whose frame is due update
    env._apple_meas from the delayed true apple when the camera sees it."""
    T = getattr(env, "_vt", None)
    if T is None or getattr(env, "_apple_meas", None) is None:
        return
    N, dev = env.num_envs, env.device
    ar = torch.arange(N, device=dev)
    true = env.scene["apple"].data.root_pos_w
    T.hist = torch.roll(T.hist, 1, dims=0)
    T.hist[0] = true
    T.timer -= env.step_dt
    due = (T.timer <= 0.0) & ~T.frozen & (env.episode_length_buf > 0)
    if not bool(due.any()):
        return
    p0, p1 = detect_period_s
    T.timer = torch.where(due, T.timer + p0 + (p1 - p0) * torch.rand(N, device=dev), T.timer)
    seen_at = T.hist[T.lat, ar]                                              # [N, 3] where it was
    cam_p, cam_R = camera_pose_w(env, camera)
    in_view = sees_t(seen_at.unsqueeze(1), cam_p, cam_R, camera)[:, 0]
    # the arm and hand on the camera -> apple segment, in front of the apple
    robot = env.scene["robot"]
    pts = torch.cat([robot.data.body_pos_w[:, T.occ_ids], fingertip_points(env)], dim=1)   # [N, B, 3]
    rad = torch.cat([T.occ_r, torch.full((pts.shape[1] - len(T.occ_ids),), TIP_RADIUS, device=dev)])
    d = seen_at - cam_p                                                      # [N, 3]
    L = d.norm(dim=-1).clamp(min=1e-6)
    u = d / L.unsqueeze(-1)
    s = ((pts - cam_p.unsqueeze(1)) * u.unsqueeze(1)).sum(-1)                # along the ray [N, B]
    perp = (pts - cam_p.unsqueeze(1) - s.unsqueeze(-1) * u.unsqueeze(1)).norm(dim=-1)
    front = (s > 0.0) & (s < L.unsqueeze(-1) - 0.5 * apple_radius)
    gap = perp - rad.unsqueeze(0)                                            # [N, B] to the ray
    occluded = (front & (gap < 0.5 * apple_radius)).any(-1)
    partial = (front & (gap < 1.5 * apple_radius)).any(-1) & ~occluded
    missed = torch.rand(N, device=dev) < miss_prob
    det = due & in_view & ~occluded & ~missed
    meas = seen_at + T.bias + noise_std * torch.randn(N, 3, device=dev)
    shift = partial_shift * torch.rand(N, device=dev) * partial.float()       # the hand's depth in the box
    meas = meas + shift.unsqueeze(-1) * u
    env._apple_meas = torch.where(det.unsqueeze(-1), meas, env._apple_meas)
    T.n_frames += due.float()
    T.n_det += det.float()
    T.n_occl += (due & in_view & occluded).float()
    T.n_out += (due & ~in_view).float()


def camera_tracking_metrics(env: "ManagerBasedRLEnv", env_ids: Sequence[int]) -> dict:
    """Curriculum term (logs only): shares of the camera frames of the finished episodes."""
    T = getattr(env, "_vt", None)
    if T is None:
        return {}
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    live = ~T.frozen[ids]
    fr = T.n_frames[ids][live].sum().clamp(min=1.0)
    meas = getattr(env, "_apple_meas", None)
    out = {
        "detected_frac": float(T.n_det[ids][live].sum() / fr),
        "occluded_frac": float(T.n_occl[ids][live].sum() / fr),
        "out_of_view_frac": float(T.n_out[ids][live].sum() / fr),
        "frozen_frac": float(T.frozen[ids].float().mean()),
    }
    if meas is not None:
        err = (meas[ids] - env.scene["apple"].data.root_pos_w[ids]).norm(dim=-1)
        out["meas_err_cm"] = float(100.0 * err.mean())
    if hasattr(env, "_wide_start"):
        out["wide_start_frac"] = float(env._wide_start[ids].float().mean())
    return out


# ─────────────────────────────────────────────
#  Wide starts
# ─────────────────────────────────────────────

class _StartLib:
    def __init__(self, env: "ManagerBasedRLEnv", path: str, start_clearance: float, q_lo, q_hi):
        robot = env.scene["robot"]
        dev = env.device
        lib = np.load(path)
        names = [str(n) for n in lib["joint_names"]]
        self.jids = [robot.joint_names.index(n) for n in names]
        q = torch.tensor(lib["q"], device=dev, dtype=torch.float32)
        clear = torch.tensor(lib["clearance"], device=dev, dtype=torch.float32)
        ok = (q >= torch.tensor(q_lo, device=dev)).all(-1) & (q <= torch.tensor(q_hi, device=dev)).all(-1)
        self.q = q[ok]
        self.pocket = torch.tensor(lib["pocket_torso"], device=dev, dtype=torch.float32)[ok]
        self.bclear = torch.minimum(clear[ok][:, :-1], clear[ok][:, 1:]) >= start_clearance    # [M, B]
        self.yaw_grid = torch.tensor(lib["yaw_grid"], device=dev, dtype=torch.float32)
        # the torso at waist yaw 0 (env-local), from the camera pose v1 stored there
        v = env._camera_view0
        cam = env.cfg.head_camera
        A = torch.tensor(cam["A"], device=dev, dtype=torch.float32)
        t = torch.tensor(cam["t"], device=dev, dtype=torch.float32)
        R0 = torch.tensor(v["R"], device=dev, dtype=torch.float32) @ torch.linalg.inv(A)
        self.torso_p0 = torch.tensor(v["pos"], device=dev, dtype=torch.float32) - R0 @ t
        self.torso_R0 = R0
        print(f"[visible_tracking] wide starts: {int(ok.sum())} of {len(ok)} library poses inside the "
              f"robot's joint limits")


def reset_arm_wide(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    library_file: str,
    prob: float,
    q_lo: Sequence[float],
    q_hi: Sequence[float],
    start_clearance: float = 0.02,
    apple_clear: float = 0.12,
    nearer: float = 0.06,
    n_candidates: int = 64,
) -> None:
    """mode="reset", after visible_reach.place_tree_fixed_waist and reset_arm_jitter (declaration
    order): a clear library start for a share `prob` of the episodes (module docstring)."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if len(ids) == 0 or not hasattr(env, "_fixed_waist_yaw"):
        return
    if not hasattr(env, "_wide_lib"):
        env._wide_lib = _StartLib(env, library_file, start_clearance, q_lo, q_hi)
        env._wide_start = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    Lb = env._wide_lib
    dev = env.device
    env._wide_start[ids] = False
    pick = ids[torch.rand(len(ids), device=dev) < prob]
    n = len(pick)
    if n == 0:
        return
    yaw = env._fixed_waist_yaw[pick]
    b = (torch.searchsorted(Lb.yaw_grid, yaw.contiguous()) - 1).clamp(0, Lb.bclear.shape[1] - 1)
    cand = torch.randint(0, len(Lb.q), (n, n_candidates), device=dev)
    ok = Lb.bclear[cand, b.unsqueeze(-1)]                                     # [n, C] body clear at this yaw
    pocket = Lb.torso_p0 + Lb.pocket[cand] @ Lb.torso_R0.T                    # [n, C, 3] yaw-0 env-local
    target = _unturn(env, pick, env.scene["apple"].data.root_pos_w[pick].unsqueeze(1))   # [n, 1, 3]
    apples = target
    if hasattr(env, "_tree"):
        nq, hang = _neighbour_state(env)
        nb = _unturn(env, pick, nq[pick])
        far = torch.full_like(nb, 1.0e3)
        apples = torch.cat([target, torch.where(hang[pick].unsqueeze(-1), nb, far)], dim=1)
    dmin = (pocket.unsqueeze(2) - apples.unsqueeze(1)).norm(dim=-1).amin(-1)  # [n, C]
    ok &= dmin >= apple_clear
    pv = torch.tensor(env._waist_pivot, device=dev, dtype=torch.float32)
    r_pocket = (pocket[..., :2] - pv).norm(dim=-1)
    r_target = (target[:, 0, :2] - pv).norm(dim=-1)
    ok &= r_pocket <= (r_target - nearer).unsqueeze(-1)
    has = ok.any(-1)
    if not bool(has.any()):
        return
    first = torch.argmax(ok.float(), dim=-1)
    sel = pick[has]
    q = Lb.q[cand[has, first[has]]]
    robot = env.scene["robot"]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=Lb.jids, env_ids=sel)
    robot.set_joint_position_target(q, joint_ids=Lb.jids, env_ids=sel)
    env._wide_start[sel] = True
