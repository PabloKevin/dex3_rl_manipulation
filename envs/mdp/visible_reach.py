# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""AppleReachVisible-v1 pieces (envs/apple_reach_visible_env_cfg.py).

  - head camera model: the real D435i (g1_fruits_picker camera_config.yaml
    intrinsics + calibrated camera_extrinsics.yaml pose, both relative to
    torso_link). Which apples it sees whole -- the reach only gets targets
    the robot can detect;
  - waist held at a fixed yaw per episode (the robot's scan script turns the
    torso, then the reach runs with it held): the tree is sampled for the
    torso at yaw 0 and turned with it, so the arm / camera / tree geometry
    is the same at every yaw and only the legs move relative to the arm;
  - inputs in the torso's yaw frame (the policy then works at any yaw);
  - staging-point approach rewards: rise / lower next to the body to the
    apple's height, then go in level (path length along that L, the
    near-apple rewards gated on being level inside the cluster, a keep-out
    penalty for being off-level there);
  - play.py visuals: the camera's frustum, which apples it sees, and its
    image as a picture-in-picture in the recorded video.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, yaw_quat

from . import reach_rewards as rr
from .apple_tree import place_apple_tree

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────
#  Head camera model
# ─────────────────────────────────────────────

def _cam_rt(camera: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(A [3, 3] optical (x right, y UP, z forward) -> torso_link, t [3])."""
    return (torch.tensor(camera["A"], dtype=torch.float32, device=device),
            torch.tensor(camera["t"], dtype=torch.float32, device=device))


def camera_pose_w(env: "ManagerBasedRLEnv", camera: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Live camera position [N, 3] and rotation [N, 3, 3] (world; columns =
    optical x right, y up, z forward), from torso_link."""
    robot = env.scene["robot"]
    b = robot.data.body_names.index(camera.get("torso_body", "torso_link"))
    A, t = _cam_rt(camera, env.device)
    Rt = matrix_from_quat(robot.data.body_quat_w[:, b])
    return robot.data.body_pos_w[:, b] + Rt @ t, Rt @ A


def sees_t(points: torch.Tensor, cam_pos: torch.Tensor, cam_R: torch.Tensor, camera: dict) -> torch.Tensor:
    """[..., K] bool: tree_sampler.camera_sees in torch. points [..., K, 3],
    cam_pos [..., 3], cam_R [..., 3, 3] (same frame)."""
    pc = ((points - cam_pos.unsqueeze(-2)).unsqueeze(-2) @ cam_R.unsqueeze(-3)).squeeze(-2)
    x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
    zz = z.clamp(min=1e-6)
    u = camera["cx"] + camera["fx"] * x / zz
    v = camera["cy"] - camera["fy"] * y / zz
    rp = camera["fx"] * camera["radius"] / zz + camera["margin_px"]
    lo, hi = camera["z_range"]
    return (z >= lo) & (z <= hi) & (u - rp >= 0) & (u + rp <= camera["width"]) & (v - rp >= 0) \
        & (v + rp <= camera["height"])


def camera_view_yaw0(env: "ManagerBasedRLEnv", camera: dict) -> dict:
    """tree_sampler.camera_sees' view dict in env 0's env-local frame at the
    CURRENT pose (call it at the default pose, waist yaw 0)."""
    pos, R = camera_pose_w(env, camera)
    view = {k: camera[k] for k in ("fx", "fy", "cx", "cy", "width", "height", "radius", "margin_px")}
    view["z_range"] = tuple(float(z) for z in camera["z_range"])
    view["pos"] = (pos[0] - env.scene.env_origins[0]).cpu().double().numpy().tolist()
    view["R"] = R[0].cpu().double().numpy().tolist()
    return view


def _unturn(env: "ManagerBasedRLEnv", ids: torch.Tensor, p_w: torch.Tensor) -> torch.Tensor:
    """World points [n, K, 3] of envs `ids` -> env-local, turned back by
    each env's fixed waist yaw about the waist axis: the yaw-0 frame the tree
    was sampled in (where env._camera_view0 holds)."""
    p = p_w - env.scene.env_origins[ids].unsqueeze(1)
    pv = torch.tensor(env._waist_pivot, dtype=p.dtype, device=p.device)
    yaw = -env._fixed_waist_yaw[ids]
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    x, y = p[..., 0] - pv[0], p[..., 1] - pv[1]
    return torch.stack([c * x - s * y + pv[0], s * x + c * y + pv[1], p[..., 2]], dim=-1)


def _sees_yaw0(env: "ManagerBasedRLEnv", ids: torch.Tensor, p_w: torch.Tensor) -> torch.Tensor:
    """[n, K] bool: the camera sees these world points at reset (robot data
    still stale then: the fixed yaw is undone instead of reading the torso)."""
    v = env._camera_view0
    pos = torch.tensor(v["pos"], dtype=torch.float32, device=env.device)
    R = torch.tensor(v["R"], dtype=torch.float32, device=env.device)
    return sees_t(_unturn(env, ids, p_w), pos, R, v)


# ─────────────────────────────────────────────
#  Events
# ─────────────────────────────────────────────

def place_tree_fixed_waist(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    yaw_range_deg: tuple[float, float],
    camera: dict,
    n_apples: int,
    seed_envelope: dict,
    member_envelope: dict,
    workspace_map: str | None,
    standoff: float,
    stem_heading_max_deg: float,
    stem_length_range: tuple[float, float],
    trunk_dist: float,
    trunk_jitter: float,
    cluster_sizes: tuple[int, ...],
    cluster_weights: tuple[float, ...],
    cluster_gap: tuple[float, float],
    sep_min: float,
    order: str = "top_down",
    stem_params: dict | None = None,
    new_tree_each_reset: bool = False,
    stem_radius: float = 0.003,
    branch_radius: float = 0.015,
    trunk_radius: float = 0.06,
    spindle: dict | None = None,
    sampler_workers: int = 0,
    waist_joint: str = "waist_yaw_joint",
) -> None:
    """mode="reset", in place of apple_tree.place_apple_tree: this episode's
    waist yaw, uniform in yaw_range_deg, written as the joint position AND
    its PD target (no action term drives the waist: it stays there), then a
    new tree for the torso at yaw 0 -- its first apple seen whole by the
    head camera (Envelope view) and reachable (the map), the target an
    apple the camera sees -- turned with the torso about the waist axis."""
    robot = env.scene["robot"]
    N, dev = env.num_envs, env.device
    if not hasattr(env, "_fixed_waist_yaw"):
        # first reset: the robot still at its default pose (waist 0)
        env._fixed_waist_yaw = torch.zeros(N, device=dev)
        env._fw_joint = robot.joint_names.index(waist_joint)
        pel = robot.data.body_names.index("pelvis")
        env._waist_pivot = (robot.data.body_pos_w[0, pel] - env.scene.env_origins[0])[:2].cpu().numpy().tolist()
        env._camera_view0 = camera_view_yaw0(env, camera)
    ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
    lo, hi = (math.radians(a) for a in yaw_range_deg)
    yaw = lo + (hi - lo) * torch.rand(len(ids), device=dev)
    env._fixed_waist_yaw[ids] = yaw
    q = yaw.unsqueeze(-1)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=[env._fw_joint], env_ids=ids)
    robot.set_joint_position_target(q, joint_ids=[env._fw_joint], env_ids=ids)
    view = env._camera_view0
    place_apple_tree(
        env, env_ids, n_apples, {**seed_envelope, "view": view}, member_envelope, workspace_map, standoff,
        stem_heading_max_deg, stem_length_range, trunk_dist, trunk_jitter, cluster_sizes, cluster_weights,
        cluster_gap, sep_min, order=order, stem_params=stem_params, new_tree_each_reset=new_tree_each_reset,
        stem_radius=stem_radius, branch_radius=branch_radius, trunk_radius=trunk_radius, spindle=spindle,
        sampler_workers=sampler_workers, fixed_waist=True, target_view=view)


def mask_unseen_neighbours(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """mode="reset", after single_reach.reset_apple_measurement: neighbours
    the head camera does not see whole read as absent in the perceived
    clearance ring (the robot cannot know about them). Also records, for
    the metrics, whether the target and how many neighbours were seen."""
    if getattr(env, "_neigh_meas", None) is None or not hasattr(env, "_camera_view0"):
        return
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    seen = _sees_yaw0(env, ids, env._neigh_meas[ids])
    if not hasattr(env, "_vis_neigh_frac"):
        env._vis_neigh_frac = torch.zeros(env.num_envs, device=env.device)
        env._vis_target = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    hang = env._neigh_hanging_meas[ids]
    env._vis_neigh_frac[ids] = (seen & hang).float().sum(-1) / hang.float().sum(-1).clamp(min=1.0)
    env._neigh_hanging_meas[ids] = hang & seen
    apple = env.scene["apple"].data.root_pos_w[ids].unsqueeze(1)
    env._vis_target[ids] = _sees_yaw0(env, ids, apple)[:, 0]


# ─────────────────────────────────────────────
#  Observations
# ─────────────────────────────────────────────

def apple_pos_palm_measured_torso(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                                  local_offset: Sequence[float] = (0.0, 0.0, 0.0),
                                  torso_body: str = "torso_link") -> torch.Tensor:
    """[N, 3] single_reach.apple_pos_palm_measured (perceived apple - palm
    reference point) in the torso's YAW frame instead of the world: the same
    reach reads the same at every waist yaw."""
    from .single_reach import apple_pos_palm_measured

    robot = env.scene["robot"]
    if not hasattr(env, "_vr_torso_id"):
        env._vr_torso_id = robot.data.body_names.index(torso_body)
    v = apple_pos_palm_measured(env, object_cfg, robot_cfg, local_offset)
    return quat_apply_inverse(yaw_quat(robot.data.body_quat_w[:, env._vr_torso_id]), v)


# ─────────────────────────────────────────────
#  Staging-point approach
# ─────────────────────────────────────────────
# G: the goal (the palm reference point at the grasp pose, true apple).
# S: the staging point -- at G's height, on the horizontal line from the
#    waist axis through the apple, `back` m short of G horizontally but never
#    nearer the waist axis than `body_reach` (the hand's reach at rest), nor
#    than where the palm started this episode (along that line): a point
#    next to the body for near fruit, in front of the cluster for far, and
#    straight up / down from the start is never inside the zone.
# The L path goes palm -> S -> G. The cluster ZONE is ahead of S (or of the
# palm's start, if that is further out) along the approach line: nothing up
# to `zone_start` past it, full from zone_start + zone_ramp (measured along the line, so a hand rising next to the body for
# a near apple, where S sits at body_reach, is never in it).

def _staging(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, standoff: float,
             palm_local_offset: Sequence[float], back: float, body_reach: float):
    p = rr._palm_ref_pos(env, robot_cfg, palm_local_offset)
    G = rr._grasp_target_pos(env, object_cfg, standoff)
    u = rr._approach_dir(env, object_cfg)
    O = env.scene["robot"].data.body_pos_w[:, env._approach_from_body]
    hG = ((G - O)[:, :2] * u[:, :2]).sum(-1)
    sp = ((p - O)[:, :2] * u[:, :2]).sum(-1)
    home = getattr(env, "_stage_home", None)
    if home is None or home.shape != sp.shape:
        home = sp
    home = env._stage_home = torch.where(env.episode_length_buf <= 1, sp, home)   # the palm where the episode starts
    hS = torch.minimum(torch.maximum((hG - back).clamp(min=body_reach), home), hG)
    S = G - (hG - hS).unsqueeze(-1) * u
    # the zone's origin: S, or the palm's start when that is already further
    # out (fruit low beside the hip, the hand starting past the goal)
    return p, G, S, u, O, torch.maximum(hS, home)


def _zone(p: torch.Tensor, u: torch.Tensor, O: torch.Tensor, hS: torch.Tensor, start: float,
          ramp: float) -> torch.Tensor:
    """[N] 0 up to `start` past S along the approach line, 1 from start + ramp."""
    sp = ((p - O)[:, :2] * u[:, :2]).sum(-1)
    return ((sp - hS - start) / ramp).clamp(0.0, 1.0)


def _level_gate(p: torch.Tensor, G: torch.Tensor, tol: float, band: float) -> torch.Tensor:
    """[N] 1 within `tol` of G's height, 0 from tol + band."""
    return (1.0 - ((p[:, 2] - G[:, 2]).abs() - tol) / band).clamp(0.0, 1.0)


def staging_path_length(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg,
                        standoff: float, palm_local_offset: Sequence[float], back: float = 0.15,
                        body_reach: float = 0.28, level_tol: float = 0.04, level_band: float = 0.05) -> torch.Tensor:
    """[N] what is left of the L path: straight to G when level with it,
    via S otherwise (|p - S| + |S - G|), blended by the level gate. Rising
    or lowering next to the body shortens it 1:1; going in off-level does
    not, so the diagonal through the cluster is never the short way."""
    p, G, S, _, _, _ = _staging(env, robot_cfg, object_cfg, standoff, palm_local_offset, back, body_reach)
    g = _level_gate(p, G, level_tol, level_band)
    via = (p - S).norm(dim=-1) + (S - G).norm(dim=-1)
    return g * (p - G).norm(dim=-1) + (1.0 - g) * via


def reward_path_progress(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg,
                         standoff: float, palm_local_offset: Sequence[float], back: float = 0.15,
                         body_reach: float = 0.28, level_tol: float = 0.04, level_band: float = 0.05) -> torch.Tensor:
    """[N] potential-based shaping on staging_path_length: metres of the L
    path closed this step (replaces reach_progress, which paid the straight
    line). 0 on an episode's first step."""
    L = staging_path_length(env, robot_cfg, object_cfg, standoff, palm_local_offset, back, body_reach, level_tol,
                            level_band)
    prev = getattr(env, "_prev_path_len", None)
    if prev is None or prev.shape != L.shape:
        prev = L.clone()
    r = torch.where(env.episode_length_buf <= 1, torch.zeros_like(L), prev - L)
    env._prev_path_len = L.clone()
    return r


def reward_path_gaussian(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg,
                         standoff: float, palm_local_offset: Sequence[float], std: float = 0.25,
                         back: float = 0.15, body_reach: float = 0.28, level_tol: float = 0.04,
                         level_band: float = 0.05) -> torch.Tensor:
    """[N] exp(-L^2 / 2 std^2) on staging_path_length (replaces reach_coarse)."""
    L = staging_path_length(env, robot_cfg, object_cfg, standoff, palm_local_offset, back, body_reach, level_tol,
                            level_band)
    return torch.exp(-(L ** 2) / (2.0 * std ** 2))


def penalty_cluster_keepout(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg,
                            standoff: float, palm_local_offset: Sequence[float], back: float = 0.15,
                            body_reach: float = 0.28, zone_start: float = 0.03, zone_ramp: float = 0.05,
                            tol: float = 0.05, ref: float = 0.04, cap: float = 3.0) -> torch.Tensor:
    """[N] the palm inside the cluster zone (past the staging point toward
    the apple) while off G's height by more than `tol`: zone x min((|dz| -
    tol) / ref, cap)^2 -- rising or dropping through the fruit under / over
    the target. Also keeps each episode's worst |dz| in the full zone for
    the metrics (env._keepout_peak, m)."""
    p, G, _, u, O, hS = _staging(env, robot_cfg, object_cfg, standoff, palm_local_offset, back, body_reach)
    z = _zone(p, u, O, hS, zone_start, zone_ramp)
    dz = (p[:, 2] - G[:, 2]).abs()
    peak = getattr(env, "_keepout_peak", None)
    if peak is None or peak.shape != dz.shape:
        peak = torch.zeros_like(dz)
    peak = torch.where(env.episode_length_buf <= 1, torch.zeros_like(dz), peak)
    env._keepout_peak = torch.where(z >= 1.0, torch.maximum(peak, dz), peak)
    return z * (((dz - tol) / ref).clamp(min=0.0, max=cap) ** 2)


def level_gated(env: "ManagerBasedRLEnv", term: Callable, term_params: dict, robot_cfg: SceneEntityCfg,
                object_cfg: SceneEntityCfg, standoff: float, palm_local_offset: Sequence[float],
                back: float = 0.15, body_reach: float = 0.28, zone_start: float = 0.03, zone_ramp: float = 0.05,
                level_tol: float = 0.04, level_band: float = 0.05) -> torch.Tensor:
    """[N] term(env, **term_params) x gate: the gate is 1 outside the cluster
    zone and, inside it, the level gate -- a near-apple reward paid while
    coming in off-level from below / above is not paid. (The manager does
    not resolve SceneEntityCfg nested in term_params: done here, once.)"""
    done = getattr(env, "_lg_resolved", None)
    if done is None:
        done = env._lg_resolved = set()
    if id(term_params) not in done:
        for v in term_params.values():
            if isinstance(v, SceneEntityCfg):
                v.resolve(env.scene)
        done.add(id(term_params))
    p, G, _, u, O, hS = _staging(env, robot_cfg, object_cfg, standoff, palm_local_offset, back, body_reach)
    gate = 1.0 - _zone(p, u, O, hS, zone_start, zone_ramp) * (1.0 - _level_gate(p, G, level_tol, level_band))
    return term(env, **term_params) * gate


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def visible_reach_metrics(env: "ManagerBasedRLEnv", env_ids: Sequence[int], ema: float = 0.05,
                          offlevel_cm: float = 4.0) -> dict:
    """Curriculum-term logger (changes nothing), run before the reset, on
    the episodes ending: target seen at the start (should be 1.0), share of
    hanging neighbours seen, worst off-level depth inside the cluster zone
    (cm) and the share of episodes that went in more than `offlevel_cm`
    off-level, the fixed waist yaw (deg, |mean|). Curriculum/<term>/<key>."""
    st = getattr(env, "_vr_metrics", None)
    if st is None:
        st = env._vr_metrics = {}
    if not hasattr(env, "_vis_target") or env.common_step_counter < 1:
        return dict(st)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return dict(st)
    peak = getattr(env, "_keepout_peak", torch.zeros(env.num_envs, device=env.device))[ids] * 100.0
    vals = {
        "target_seen": float(env._vis_target[ids].float().mean()),
        "neighbours_seen": float(env._vis_neigh_frac[ids].mean()),
        "offlevel_peak_cm": float(peak.mean()),
        "offlevel_share": float((peak > offlevel_cm).float().mean()),
        "waist_yaw_deg": float(torch.rad2deg(env._fixed_waist_yaw[ids]).mean()),
    }
    for k, m in vals.items():
        st[k] = m if k not in st else (1.0 - ema) * st[k] + ema * m
    return dict(st)


# ─────────────────────────────────────────────
#  play.py visuals
# ─────────────────────────────────────────────

def camera_quat_ros(camera: dict) -> tuple[float, float, float, float]:
    """(w, x, y, z) of the camera in torso_link, ROS optical convention (x
    right, y DOWN, z forward) -- CameraCfg.OffsetCfg(convention="ros")."""
    R = np.asarray(camera["A"], float) @ np.diag([1.0, -1.0, 1.0])
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def head_camera_sensor_cfg(camera: dict, robot_prim: str = "{ENV_REGEX_NS}/Robot", downscale: int = 1):
    """CameraCfg for the head camera's own image (play.py --video)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    f = camera
    return CameraCfg(
        prim_path=f"{robot_prim}/{camera.get('torso_body', 'torso_link')}/head_d435i",
        update_period=0.0,
        width=int(f["width"]) // downscale,
        height=int(f["height"]) // downscale,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[f["fx"] / downscale, 0.0, f["cx"] / downscale, 0.0, f["fy"] / downscale,
                              f["cy"] / downscale, 0.0, 0.0, 1.0],
            width=int(f["width"]) // downscale, height=int(f["height"]) // downscale, clipping_range=(0.05, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in f["t"]), rot=camera_quat_ros(camera),
                                   convention="ros"),
    )


class HeadCameraViz:
    """What the head camera sees, drawn in the scene every step (env 0):
    the view frustum as yellow edges out to `far` m along the optical axis,
    and a marker 6 cm over every apple -- green if the camera sees it
    whole, magenta if not; blue at the perceived target (the policy's apple
    input)."""

    def __init__(self, env: "ManagerBasedRLEnv", camera: dict, far: float = 0.9):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.env, self.camera, self.far = env, camera, far
        mat = sim_utils.PreviewSurfaceCfg
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/HeadCameraView",
            markers={
                "edge": sim_utils.CylinderCfg(radius=0.003, height=1.0, axis="Z",
                                              visual_material=mat(diffuse_color=(1.0, 0.8, 0.0))),
                "seen": sim_utils.SphereCfg(radius=0.012, visual_material=mat(diffuse_color=(0.1, 0.9, 0.1))),
                "unseen": sim_utils.SphereCfg(radius=0.012, visual_material=mat(diffuse_color=(0.85, 0.1, 0.9))),
                "perceived": sim_utils.SphereCfg(radius=0.010, visual_material=mat(diffuse_color=(0.1, 0.3, 1.0))),
            },
        )
        self.markers = VisualizationMarkers(cfg)
        W, H = float(camera["width"]), float(camera["height"])
        corners = [(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)]
        # optical-frame directions with z = 1 (y up)
        self.rays = torch.tensor([[(u - camera["cx"]) / camera["fx"], -(v - camera["cy"]) / camera["fy"], 1.0]
                                  for u, v in corners], device=env.device)

    @staticmethod
    def _quat_z_to(d: torch.Tensor) -> torch.Tensor:
        """(w, x, y, z) [K, 4] turning +Z onto the unit vectors d [K, 3]."""
        z = torch.tensor([0.0, 0.0, 1.0], device=d.device).expand_as(d)
        axis = torch.cross(z, d, dim=-1)
        w = 1.0 + (z * d).sum(-1, keepdim=True)
        q = torch.cat([w, axis], dim=-1)
        flip = w.squeeze(-1) < 1e-6
        q[flip] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=d.device)
        return q / q.norm(dim=-1, keepdim=True)

    def update(self) -> None:
        env, i = self.env, 0
        pos, R = camera_pose_w(env, self.camera)
        c, Rc = pos[i], R[i]
        far = c + (self.rays * self.far) @ Rc.T                   # [4, 3]
        a = torch.cat([c.expand(4, 3), far])                         # 4 rays, then the far rectangle
        b = torch.cat([far, far.roll(-1, 0)])
        d = b - a
        L = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        trans, quats = [(a + b) / 2], [self._quat_z_to(d / L)]
        scales = [torch.cat([torch.ones_like(L), torch.ones_like(L), L], dim=-1)]
        idx = [torch.zeros(8, dtype=torch.long, device=env.device)]
        if hasattr(env, "_tree"):
            apples = torch.stack([o.data.root_pos_w[i] for o in env._tree.objs])      # [n, 3]
            seen = sees_t(apples.unsqueeze(0), pos[i:i + 1], R[i:i + 1], self.camera)[0]
            trans.append(apples + torch.tensor([0.0, 0.0, 0.06], device=env.device))
            idx.append(torch.where(seen, 1, 2).long())
            quats.append(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).expand(len(apples), 4))
            scales.append(torch.ones(len(apples), 3, device=env.device))
        meas = getattr(env, "_apple_meas", None)
        if meas is not None:
            trans.append(meas[i:i + 1])
            idx.append(torch.tensor([3], device=env.device))
            quats.append(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device))
            scales.append(torch.ones(1, 3, device=env.device))
        self.markers.visualize(translations=torch.cat(trans), orientations=torch.cat(quats),
                               scales=torch.cat(scales), marker_indices=torch.cat(idx))


def camera_view_wrapper(env, camera: dict, sensor_name: str | None = None, scale: int = 1):
    """gymnasium wrapper for play.py: HeadCameraViz updated after every step
    (before RecordVideo grabs the frame), and, when `sensor_name` names the
    head camera sensor in the scene, its own image pasted into the top-right
    corner of every rendered frame (yellow border)."""
    import gymnasium as gym

    class _CameraView(gym.Wrapper):
        def __init__(self, inner):
            super().__init__(inner)
            self.viz = HeadCameraViz(inner.unwrapped, camera)

        def reset(self, **kw):
            out = self.env.reset(**kw)
            self.viz.update()
            return out

        def step(self, action):
            out = self.env.step(action)
            self.viz.update()
            return out

        def render(self):
            frame = self.env.render()
            if frame is None or sensor_name is None:
                return frame
            img = self.env.unwrapped.scene[sensor_name].data.output.get("rgb")
            if img is None:
                return frame
            img = img[0, ..., :3].cpu().numpy().astype(np.uint8)
            if scale > 1:
                img = img.repeat(scale, axis=0).repeat(scale, axis=1)
            frame = np.array(frame, copy=True)
            h, w = img.shape[:2]
            b, m = 4, 10
            if h + 2 * b + m > frame.shape[0] or w + 2 * b + m > frame.shape[1]:
                return frame
            x0 = frame.shape[1] - w - b - m
            frame[m:m + h + 2 * b, x0 - b:x0 + w + b] = (255, 204, 0)
            frame[m + b:m + b + h, x0:x0 + w] = img
            return frame

    return _CameraView(env)


# ─────────────────────────────────────────────
#  Grasp (AppleGraspVisible-v2)
# ─────────────────────────────────────────────

def penalty_self_collision_grasp(env: "ManagerBasedRLEnv", margin: float = 0.015, ref: float = 0.05) -> torch.Tensor:
    """[N] self_collision.penalty_self_collision during the GRASP phase only
    (the frozen reach is not the trainee's): the right arm / hand inside the
    rest of the body. The robot has no physical self-collision in the sim, so
    without this the grasp's arm passed through the torso."""
    from .grasp_autoplay import grasp_phase_active
    from .self_collision import penalty_self_collision

    return penalty_self_collision(env, margin=margin, ref=ref) * grasp_phase_active(env).float()
