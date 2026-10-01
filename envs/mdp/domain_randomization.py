# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Domain randomization for the apple-grasp scene.

Fixed once per env at "prestartup"/"startup" (before/just-after the GPU tensor
views exist): apple scale, apple mass, trunk height, branch/stem length — anything
that requires a raw USD collider-geometry write, since that's illegal against
a live PhysX GPU-direct-API tensor view.

Re-sampled every episode at "reset": apple position and height (via the
tensor API), branch yaw, the tree translation that follows from both, and
stem break thresholds (plain Python values).
"""
from __future__ import annotations
import random
import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

RAND_PCT = 0.20


def _range(nominal: float, pct: float = RAND_PCT):
    return nominal * (1 - pct), nominal * (1 + pct)

def randomize_apple_scale(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    nominal_radius: float,
) -> None:
    """Uniform scale on the apple RigidObject -> effective radius change.
    Same mechanism as fruit_utils.randomize_fruit_scale_and_mass, radius-only.

    Runs at mode="prestartup" — i.e. before sim.reset() creates the GPU tensor
    views — since this is a raw USD write and would otherwise crash/invalidate
    the shared PhysX tensor view if done at "reset" time. Because of that,
    env_ids arrives as None (see EventManager.apply's generic-mode branch);
    treat that as "all envs".
    """
    try:
        from pxr import UsdGeom, Gf
        import omni.usd
    except ImportError:
        return

    asset: RigidObject = env.scene[object_cfg.name]
    stage = omni.usd.get_context().get_stage()
    lo, hi = _range(1.0)  # scale factor range, not absolute radius

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    for idx in env_ids.tolist():
        s = random.uniform(lo, hi)
        prim_path = f"/World/envs/env_{idx}/Apple"   # ← hardcode, don't use asset.cfg.prim_path
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim)
        scale_op = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        if scale_op is None:
            scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(s, s, s))
        # keep env-side bookkeeping in sync, in case other code needs the
        # effective radius (e.g. reward shaping, stem anchor offset)
        if not hasattr(env, "_apple_radius"):
            env._apple_radius = torch.full((env.num_envs,), nominal_radius, device=env.device)
        env._apple_radius[idx] = nominal_radius * s


def randomize_stem_break_params(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    nominal_force: float = 2.5,
    nominal_torque: float = 0.5,
) -> None:
    """Pure-Python per-env break thresholds, read by StemManager._check_break."""
    if not hasattr(env, "_stem_break_force"):
        env._stem_break_force = torch.full((env.num_envs,), nominal_force, device=env.device)
    if not hasattr(env, "_stem_break_torque"):
        env._stem_break_torque = torch.full((env.num_envs,), nominal_torque, device=env.device)

    lo_f, hi_f = _range(nominal_force)
    lo_t, hi_t = _range(nominal_torque)
    for idx in env_ids.tolist():
        env._stem_break_force[idx] = random.uniform(lo_f, hi_f)
        env._stem_break_torque[idx] = random.uniform(lo_t, hi_t)

def randomize_tactile_gain(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    gain_range: tuple[float, float] = (0.7, 1.3),
) -> None:
    """Independent per-taxel sensor-calibration gain, resampled every episode.

    Grasp-phase-only randomizer (the reach task never calls this): tactile
    observations (envs/mdp/observations.py::tactile_slip/_finger_balance/
    _force_level) multiply the ground-truth apple-filtered contact force by
    env._tactile_gain before exposing it to the policy, so training can't
    lean on an absolute-force threshold that real hardware can't reproduce
    (tactile_layout.SIM_FORCE_TO_RAW_GAIN is an admitted, unverified
    placeholder — see that module's docstring). Independent PER TAXEL (not
    one shared scalar) models realistic sensor-to-sensor miscalibration and
    forces reliance on relative/gradient signals rather than a shared,
    cancel-able scale factor.

    Deliberately separate from tactile_layout.SIM_FORCE_TO_RAW_GAIN/
    force_to_raw, which stay untouched — those are the sim-viewer/real-
    hardware wire-format default scripts/run_env.py's DDS tactile publisher
    depends on for scripts/tactile_monitor.py. This is a training-time-only
    mechanism living purely on `env`.
    """
    import scripts.tactile_layout as tactile_layout

    n_taxels = len(tactile_layout.TAXELS)
    if not hasattr(env, "_tactile_gain"):
        env._tactile_gain = torch.ones(env.num_envs, n_taxels, device=env.device)

    lo, hi = gain_range
    n = len(env_ids)
    env._tactile_gain[env_ids] = torch.empty(n, n_taxels, device=env.device).uniform_(lo, hi)


def randomize_tactile_footprint(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    full_deg_range: tuple[float, float] = (15.0, 40.0),
    falloff_deg_range: tuple[float, float] = (15.0, 35.0),
) -> None:
    """Per-episode, per-taxel sensing footprint of the pads: how far off the
    pad normal a contact can land and still be read (observations.
    _sensed_tactile_forces via contact_geometry.pad_face_weight).

    Sets env._tactile_full_deg (read in full up to this angle) and
    env._tactile_cut_deg (= full + falloff; read nothing beyond), [N, 9].

    Why randomized: in sim the per-link sensor feels every face, the real
    Dex3-1 pads only their pad face, and how far toward the edge they still
    respond is unmeasured. The tripod grasp lands index/middle on their
    pads' INNER EDGES (~20-26 deg off the normal, see contact_geometry's
    TACTILE_*_DEFAULT note), right where that uncertainty is -- so a policy
    that relied on edge contacts reading in full would be relying on a sim
    artifact. The low end of full_deg_range (15 deg) makes those contacts
    read only partly in some episodes; the high end of full+falloff (75 deg)
    still never reads a pure side-face contact (~90 deg) or the back.

    Grasp-phase-only, like randomize_tactile_gain; independent per taxel
    for the same reason (pad-to-pad variation, not one shared factor).
    """
    import scripts.tactile_layout as tactile_layout

    n_taxels = len(tactile_layout.TAXELS)
    if not hasattr(env, "_tactile_full_deg"):
        env._tactile_full_deg = torch.full((env.num_envs, n_taxels), 30.0, device=env.device)
        env._tactile_cut_deg = torch.full((env.num_envs, n_taxels), 60.0, device=env.device)

    n = len(env_ids)
    full = torch.empty(n, n_taxels, device=env.device).uniform_(*full_deg_range)
    falloff = torch.empty(n, n_taxels, device=env.device).uniform_(*falloff_deg_range)
    env._tactile_full_deg[env_ids] = full
    env._tactile_cut_deg[env_ids] = full + falloff


def randomize_fruit_damage_force(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    low: float = 20.0,
    high: float = 40.0,
    apple_low: float = 22.0,
    apple_high: float = 28.0,
    apple_weight_pct: float = 0.6,
    apple_mass_nominal: float = 0.15,
    mu: float = 0.5,
) -> None:
    """Per-episode bio-yield/bruising-force threshold, AND the derived
    command-conditioned grip setpoint that must sit strictly under it.

    Damage force: the thesis's fruit-mechanics literature review (Li et al.,
    2026, texture-analyzer compression tests on Red Fuji apples: 24.33 N
    upper region, 26.37-26.38 N middle/lower; broader pome-fruit literature —
    citrus, grapes, pears — spanning roughly 20-40 N) is compiled BECAUSE the
    thesis hasn't fixed a final fruit species, so the threshold is randomized
    per episode rather than hardcoded to one number. Sampled as a two-tier
    mixture: with probability `apple_weight_pct`, sample tightly around the
    best-documented Red Fuji apple range [apple_low, apple_high]; otherwise
    sample the full reported range [low, high], so training generalizes
    rather than overfitting to one species.

    Caveat kept visible here (not just in the design doc): the literature
    number is a local texture-analyzer PROBE force, not a perfect physical
    analog to a simulated per-link contact-force magnitude — it's the best
    available literature anchor, not a lab-measured constant for this setup.

    Target grip setpoint (env._target_grip_level): sampled uniformly between
    a rough friction-based minimum-to-not-slip estimate,
    F_min ~= (apple_mass * g) / (3 * mu) (mu here is an ASSUMED constant —
    no PhysX material is explicitly configured on the apple in this scene to
    read a real friction coefficient from — and this is an order-of-magnitude
    estimate, not a rigorous static-equilibrium solve), and THIS SAME
    episode's just-sampled damage force. Computed in the same function
    (rather than a separate event) because the target's upper bound depends
    on this call's own damage-force sample — EventsCfg ordering alone can't
    guarantee that dependency across two separate event terms as cleanly.
    """
    if not hasattr(env, "_fruit_damage_force"):
        env._fruit_damage_force = torch.full(
            (env.num_envs,), (low + high) / 2.0, device=env.device
        )
    if not hasattr(env, "_target_grip_level"):
        env._target_grip_level = torch.zeros(env.num_envs, device=env.device)

    n = len(env_ids)
    use_apple = torch.rand(n, device=env.device) < apple_weight_pct
    wide = torch.empty(n, device=env.device).uniform_(low, high)
    tight = torch.empty(n, device=env.device).uniform_(apple_low, apple_high)
    damage = torch.where(use_apple, tight, wide)
    env._fruit_damage_force[env_ids] = damage

    g = 9.81
    f_min = (apple_mass_nominal * g) / (3.0 * mu)
    lo_target = torch.full((n,), f_min, device=env.device)
    hi_target = torch.maximum(damage, lo_target + 1e-3)  # guard a degenerate/inverted band
    env._target_grip_level[env_ids] = lo_target + torch.rand(n, device=env.device) * (
        hi_target - lo_target
    )


def _get_or_add_xform_op(prim, op_type):
    from pxr import UsdGeom
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return xform.AddXformOp(op_type)


def _set_translate(prim, x, y, z):
    from pxr import UsdGeom, Gf
    _get_or_add_xform_op(prim, UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(x, y, z))


def _set_rotate_z(prim, angle_rad):
    from pxr import UsdGeom, Gf
    op = _get_or_add_xform_op(prim, UsdGeom.XformOp.TypeOrient)
    half = angle_rad / 2.0
    w, x, y, z = math.cos(half), 0.0, 0.0, math.sin(half)
    # existing xformOp:orient attrs authored by Isaac Lab's create_prim are
    # GfQuatd (double precision) — match whatever precision is already there
    # instead of hardcoding Quatf, or USD raises a type-mismatch error.
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        op.Set(Gf.Quatf(w, x, y, z))
    else:
        op.Set(Gf.Quatd(w, x, y, z))


def _get_cylinder_mesh(stage, base_path):
    """The actual Cylinder geometry for an Isaac Lab CylinderCfg spawn lives at
    '{base_path}/geometry/mesh' — '{base_path}' itself is just the Xform root
    (see isaaclab.sim.spawners.shapes._spawn_geom_from_prim_type). Also check
    prim.IsA(UsdGeom.Cylinder) explicitly: UsdGeom.Cylinder(prim) on a non-
    Cylinder prim still constructs a "valid" (truthy) schema wrapper, it just
    has no height/radius attributes — a classic silent-no-op USD footgun.
    """
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(f"{base_path}/geometry/mesh")
    if prim.IsValid() and prim.IsA(UsdGeom.Cylinder):
        return prim, UsdGeom.Cylinder(prim)
    return None, None


def randomize_tree_geometry(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    trunk_height_range: tuple[float, float],
    branch_length_nominal: float,
    stem_length_range: tuple[float, float],
) -> None:
    """One-time (mode="prestartup") per-env trunk height / branch length /
    stem length pick.

    Resizing cylinder geometry is a raw USD write, same constraint as
    randomize_apple_scale — must happen before sim.reset() creates the GPU
    tensor views, so it can only run once per env, not every episode. The
    sampled values are cached on `env` so randomize_tree_and_apple (mode=
    "reset") can reuse them every episode; it only translates/rotates the
    (now fixed-shape) trunk, branch and stem.
    """
    try:
        import omni.usd
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    l_lo, l_hi = _range(branch_length_nominal)

    env._trunk_height = torch.full((env.num_envs,), sum(trunk_height_range) / 2.0, device=env.device)
    env._branch_length = torch.full((env.num_envs,), branch_length_nominal, device=env.device)
    env._stem_length = torch.full((env.num_envs,), sum(stem_length_range) / 2.0, device=env.device)

    for idx in env_ids.tolist():
        trunk_h = random.uniform(*trunk_height_range)
        branch_len = random.uniform(l_lo, l_hi)
        stem_len = random.uniform(*stem_length_range)
        env._trunk_height[idx] = trunk_h
        env._branch_length[idx] = branch_len
        env._stem_length[idx] = stem_len

        env_ns = f"/World/envs/env_{idx}"
        for name, h in (("TreeTrunk", trunk_h), ("TreeBranch", branch_len), ("AppleStem", stem_len)):
            _, cyl = _get_cylinder_mesh(stage, f"{env_ns}/{name}")
            if cyl:
                cyl.GetHeightAttr().Set(h)


def randomize_tree_and_apple(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    apple_radius_nominal: float,
    reach_min: float,
    reach_max: float,
    reach_azimuth_center_deg: float,
    reach_azimuth_span_deg: float,
    target_dz_range: tuple[float, float],
    branch_yaw_spread_deg: float,
    standoff: float,
    reach_min_forward: float = 0.0,
    reach_min_horizontal: float = 0.0,
    workspace_map: str | None = None,
    shoulder_body_name: str = "right_shoulder_pitch_link",
) -> None:
    """Per-episode (mode="reset") apple placement, reachability-aware, on a
    commercial fruiting-wall tree: a tall vertical trunk with a short,
    horizontal branch whose tip holds the apple by a horizontal stem.

    The GRASP TARGET (the point the palm's reference point is scored
    against, reach_rewards._grasp_target_pos: `standoff` beyond the apple
    surface, on the side opposite the stem) is sampled in the right
    shoulder's own spherical coordinates:
      - height dz above the shoulder, uniform in target_dz_range;
      - 3D distance in [reach_min, reach_max] (solved for the horizontal
        radius given dz);
      - azimuth in reach_azimuth_center_deg +- reach_azimuth_span_deg/2;
      - and, by resampling, at least reach_min_forward in front of the
        shoulder (x) and reach_min_horizontal away from it horizontally, so
        a high target cannot sit almost on top of the shoulder (the 3D
        distance alone allows that, and the arm then has to contort);
      - and, if `workspace_map` is given, inside a FEASIBLE cell of that map
        (scripts/probe_side_reach.py: reachable with the palm along the stem
        and no self-collision).
    Measured at right_shoulder_pitch_link (env-local (0.0, -0.10, 1.085)).

    The branch points from the trunk back toward the robot (toward the
    shoulder), yawed by up to +-branch_yaw_spread_deg (imperfect real
    trees). The apple then sits beyond the target along the stem, so the
    trunk is BEHIND the apple: tree position is derived, not sampled.

    Caches env._stem_anchor_pos (apple centre, for the lift reward /
    tree_utils) and env._stem_dir (unit, apple -> branch tip).

    Trunk/branch/stem are visual-only AssetBaseCfg prims moved by USD
    xform writes; the apple is kinematic in the reach task and moved
    through the tensor API.
    """
    try:
        import omni.usd
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    apple: RigidObject = env.scene[object_cfg.name]
    robot = env.scene["robot"]

    n = env.num_envs
    if not hasattr(env, "_stem_anchor_pos"):
        env._stem_anchor_pos = torch.zeros(n, 3, device=env.device)
    if not hasattr(env, "_stem_dir"):
        env._stem_dir = torch.zeros(n, 3, device=env.device)
    # randomize_tree_geometry (prestartup) should always run first; fall back
    # to nominal only to avoid a hard crash if events are ever reordered.
    for attr, val in (("_trunk_height", 2.25), ("_branch_length", 0.12), ("_stem_length", 0.03)):
        if not hasattr(env, attr):
            setattr(env, attr, torch.full((n,), val, device=env.device))

    if not hasattr(env, "_shoulder_local_xyz"):
        shoulder_id = robot.data.body_names.index(shoulder_body_name)
        shoulder_world = robot.data.body_pos_w[0, shoulder_id, :]
        origin0 = env.scene.env_origins[0, :]
        env._shoulder_local_xyz = (shoulder_world - origin0).clone()
    shoulder_x, shoulder_y, shoulder_z = env._shoulder_local_xyz.tolist()

    feasible_cell = None
    if workspace_map is not None:
        if getattr(env, "_workspace_map_path", None) != workspace_map:
            import numpy as np
            m = np.load(workspace_map)
            env._workspace_map = (m["x_edges"], m["y_edges"], m["z_edges"], m["feasible"])
            env._workspace_map_path = workspace_map
        xe, ye, ze, feas = env._workspace_map

        def feasible_cell(x, y, z):
            i = int(math.floor((x - xe[0]) / (xe[1] - xe[0])))
            j = int(math.floor((y - ye[0]) / (ye[1] - ye[0])))
            k = int(math.floor((z - ze[0]) / (ze[1] - ze[0])))
            n = feas.shape
            return 0 <= i < n[0] and 0 <= j < n[1] and 0 <= k < n[2] and bool(feas[i, j, k])

    radius = getattr(env, "_apple_radius", None)
    az_lo = reach_azimuth_center_deg - reach_azimuth_span_deg / 2.0
    az_hi = reach_azimuth_center_deg + reach_azimuth_span_deg / 2.0

    apple_pos_local = torch.zeros(len(env_ids), 3, device=env.device)
    stem_dir = torch.zeros(len(env_ids), 3, device=env.device)

    for k, idx in enumerate(env_ids.tolist()):
        r_apple = float(radius[idx]) if isinstance(radius, torch.Tensor) else apple_radius_nominal
        branch_len = env._branch_length[idx].item()
        stem_len = env._stem_length[idx].item()
        trunk_h = env._trunk_height[idx].item()

        for _ in range(2000):
            dz = random.uniform(*target_dz_range)
            r_xy_max = math.sqrt(max(reach_max ** 2 - dz ** 2, 1e-4))
            r_xy_min = math.sqrt(max(reach_min ** 2 - dz ** 2, 0.0))
            r_xy_min = max(r_xy_min, reach_min_horizontal)
            if r_xy_min >= r_xy_max:
                continue
            r_xy = random.uniform(r_xy_min, r_xy_max)
            az = math.radians(random.uniform(az_lo, az_hi))
            if r_xy * math.cos(az) < reach_min_forward:
                continue
            if feasible_cell is None or feasible_cell(r_xy * math.cos(az), r_xy * math.sin(az), dz):
                break
        else:
            raise RuntimeError("randomize_tree_and_apple: reach envelope constraints are unsatisfiable")
        target_x = shoulder_x + r_xy * math.cos(az)
        target_y = shoulder_y + r_xy * math.sin(az)

        # Branch direction (trunk -> tip): back toward the robot, yawed. The
        # stem continues it, so the apple sits beyond the target along -b.
        b = az + math.pi + math.radians(random.uniform(-branch_yaw_spread_deg, branch_yaw_spread_deg))
        bx, by = math.cos(b), math.sin(b)
        apple_x = target_x - bx * (r_apple + standoff)
        apple_y = target_y - by * (r_apple + standoff)
        apple_z = shoulder_z + dz
        tip_x = apple_x - bx * (r_apple + stem_len)
        tip_y = apple_y - by * (r_apple + stem_len)
        stem_mid = r_apple + stem_len / 2.0
        trunk_x, trunk_y = tip_x - bx * branch_len, tip_y - by * branch_len

        env_ns = f"/World/envs/env_{idx}"
        placements = (
            ("TreeBase", (trunk_x, trunk_y, 0.03), None),
            ("TreeTrunk", (trunk_x, trunk_y, trunk_h / 2.0), None),
            ("TreeBranch", (tip_x - bx * branch_len / 2.0, tip_y - by * branch_len / 2.0, apple_z), b),
            ("AppleStem", (apple_x - bx * stem_mid, apple_y - by * stem_mid, apple_z), b),
        )
        for name, pos, yaw in placements:
            prim = stage.GetPrimAtPath(f"{env_ns}/{name}")
            if prim.IsValid():
                _set_translate(prim, *pos)
                if yaw is not None:
                    _set_rotate_z(prim, yaw)

        apple_pos_local[k] = torch.tensor([apple_x, apple_y, apple_z], device=env.device)
        stem_dir[k] = torch.tensor([-bx, -by, 0.0], device=env.device)

    env_origins = env.scene.env_origins[env_ids]
    root_state = apple.data.default_root_state[env_ids].clone()
    root_state[:, :3] = env_origins + apple_pos_local
    root_state[:, 7:] = 0.0
    apple.write_root_state_to_sim(root_state, env_ids=env_ids)

    env._stem_anchor_pos[env_ids] = env_origins + apple_pos_local
    env._stem_dir[env_ids] = stem_dir

def randomize_closure_threshold(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    thresh_scale_range: tuple[float, float] = (0.6, 1.6),
) -> None:
    """Per-episode scale on the grip detector's contact threshold
    (grasp_rewards._closure_state).

    Models the one real uncertainty left in that detector: the
    pressure-to-newton calibration of the Dex3-1 pads, which
    scripts/tactile_layout.py openly flags as an unverified placeholder
    (SIM_FORCE_TO_RAW_GAIN). Scaling it per episode stops the policy
    depending on a grip being declared at one exact force, the way it would
    if the threshold were a constant it could memorise.

    Episode-level on purpose: a calibration is unknown across robots but
    stable within a grasp -- a detector that flickers step to step is a
    different (and worse) problem, not this one.
    """
    if not hasattr(env, "_closure_thresh_scale"):
        env._closure_thresh_scale = torch.ones(env.num_envs, device=env.device)
    lo, hi = thresh_scale_range
    env._closure_thresh_scale[env_ids] = torch.empty(len(env_ids), device=env.device).uniform_(lo, hi)
