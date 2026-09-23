# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Domain randomization for the apple-grasp scene.

Fixed once per env at "prestartup"/"startup" (before/just-after the GPU tensor
views exist): apple scale, apple mass, trunk height, branch length — anything
that requires a raw USD collider-geometry write, since that's illegal against
a live PhysX GPU-direct-API tensor view.

Re-sampled every episode at "reset": whole-tree (x, y) translation, branch
yaw rotation, apple position (via the tensor API, tracking the branch tip),
and stem break thresholds (plain Python values).
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
    tree_height_nominal: float,
    branch_length_nominal: float,
) -> None:
    """One-time (mode="prestartup") per-env trunk-height / branch-length pick.

    Resizing collider geometry is a raw USD write, same constraint as
    randomize_apple_scale — must happen before sim.reset() creates the GPU
    tensor views, so it can only run once per env, not every episode. The
    sampled values are cached on `env` so randomize_tree_and_apple (mode=
    "reset") can reuse them every episode without touching cylinder geometry
    again — it only translates/rotates the (now-fixed-shape) trunk & branch.
    """
    try:
        import omni.usd
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    h_lo, h_hi = _range(tree_height_nominal)
    l_lo, l_hi = _range(branch_length_nominal)

    env._tree_height = torch.full((env.num_envs,), tree_height_nominal, device=env.device)
    env._branch_length = torch.full((env.num_envs,), branch_length_nominal, device=env.device)

    for idx in env_ids.tolist():
        tree_h = random.uniform(h_lo, h_hi)
        branch_len = random.uniform(l_lo, l_hi)
        env._tree_height[idx] = tree_h
        env._branch_length[idx] = branch_len

        env_ns = f"/World/envs/env_{idx}"

        _, trunk_cyl = _get_cylinder_mesh(stage, f"{env_ns}/TreeTrunk")
        if trunk_cyl:
            trunk_cyl.GetHeightAttr().Set(tree_h)

        _, branch_cyl = _get_cylinder_mesh(stage, f"{env_ns}/TreeBranch")
        if branch_cyl:
            branch_cyl.GetHeightAttr().Set(branch_len)


def randomize_tree_and_apple(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    branch_radius_nominal: float,
    branch_rotation_deg_nominal: float,
    apple_radius_nominal: float,
    tree_height_nominal: float,
    branch_length_nominal: float,
    reach_min: float = 0.25,
    reach_max: float = 0.42,
    reach_azimuth_center_deg: float = -45.0,
    reach_azimuth_span_deg: float = 70.0,
    offset_z: float = 0.08,
    shoulder_body_name: str = "right_shoulder_pitch_link",
) -> None:
    """Per-episode (mode="reset") apple placement, reachability-aware.

    The apple's (x, y) is now sampled directly in the right shoulder's own
    polar coordinates — radius in [reach_min, reach_max], azimuth in
    [reach_azimuth_center_deg +- reach_azimuth_span_deg/2] — instead of being
    an incidental byproduct of independently-jittered tree position + branch
    rotation. The old scheme (TREE_POS +-15cm, BRANCH_ROTATION +-20%,
    BRANCH_LENGTH +-20% ~ 0.4-0.6m) had no notion of the arm's actual
    workspace, so it routinely spawned the apple out of reach (or even
    behind the shoulder) — confirmed empirically via probe_arm_reach.py:
    right_shoulder_pitch_link sits at env-local (0.0, -0.10, 1.085), and the
    rigid chain shoulder->elbow->wrist->palm sums to only ~0.42 m, a hard
    upper bound on how far the hand can possibly reach (before even
    accounting for reach_rewards.PALM_LOCAL_OFFSET pushing the true grasp
    reference point another ~0.14 m out). reach_max=0.45 below already
    accounts for that; if you push PALM_LOCAL_OFFSET further out, tighten
    reach_max to match, and re-run probe_arm_reach.py if you ever change the
    USD/arm.

    dz (shoulder height vs. grasp-point height) comes from the already-fixed
    per-env tree_height, so the azimuth/radius sampling below solves for the
    horizontal radius that keeps the FULL 3D distance within
    [reach_min, reach_max], not just the horizontal component.

    branch_rotation_deg_nominal / its +-20% range is kept only for visual
    branch yaw variety — the trunk (tree_x, tree_y) is now solved backward
    from the chosen branch tip (= apple xy) and that sampled rotation, rather
    than being sampled first.

    Trunk height and branch length are fixed per env (set once by
    randomize_tree_geometry at "prestartup") — the branch's world yaw is
    still re-sampled every reset for visual variety; the resulting tree
    translation is derived, not sampled. Trunk/branch are moved by
    translating/rotating their root Xform prim (a cheap, legal USD write for
    these *static* AssetBaseCfg colliders — unlike the Apple, they are not
    dynamic RigidObjects tracked by the GPU tensor view, so this doesn't hit
    the eENABLE_DIRECT_GPU_API restriction). The apple's pose is written
    through the tensor API. The apple itself is kinematic (see
    AppleGraspSceneCfg.apple for why), so this teleport is all that's needed
    to hold it there for the whole episode — no joint required.
    """
    try:
        import omni.usd
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    apple: RigidObject = env.scene[object_cfg.name]
    robot = env.scene["robot"]

    if not hasattr(env, "_stem_anchor_pos"):
        env._stem_anchor_pos = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "_tree_height"):
        # randomize_tree_geometry (prestartup) should always run first; fall back
        # to nominal only to avoid a hard crash if events are ever reordered.
        env._tree_height = torch.full((env.num_envs,), tree_height_nominal, device=env.device)
    if not hasattr(env, "_branch_length"):
        env._branch_length = torch.full((env.num_envs,), branch_length_nominal, device=env.device)

    if not hasattr(env, "_shoulder_local_xyz"):
        shoulder_id = robot.data.body_names.index(shoulder_body_name)
        shoulder_world = robot.data.body_pos_w[0, shoulder_id, :]
        origin0 = env.scene.env_origins[0, :]
        env._shoulder_local_xyz = (shoulder_world - origin0).clone()
    shoulder_x, shoulder_y, shoulder_z = env._shoulder_local_xyz.tolist()

    r_lo, r_hi = _range(branch_rotation_deg_nominal)
    az_lo = reach_azimuth_center_deg - reach_azimuth_span_deg / 2.0
    az_hi = reach_azimuth_center_deg + reach_azimuth_span_deg / 2.0

    apple_pos_local = torch.zeros(len(env_ids), 3, device=env.device)

    for k, idx in enumerate(env_ids.tolist()):
        tree_h = env._tree_height[idx].item()
        branch_len = env._branch_length[idx].item()
        apple_z = tree_h - branch_radius_nominal - apple_radius_nominal
        dz = (apple_z - offset_z) - shoulder_z

        r_xy_max = math.sqrt(max(reach_max ** 2 - dz ** 2, 1e-4))
        r_xy_min = math.sqrt(max(reach_min ** 2 - dz ** 2, 0.0))
        if r_xy_min >= r_xy_max:
            # dz alone already eats (close to) the whole reach budget — this
            # env's randomized tree_height put the branch almost directly
            # above/below the shoulder. Fall back to the smallest feasible
            # horizontal radius rather than sampling an empty range.
            r_xy_min, r_xy_max = 0.0, max(r_xy_max, 0.02)

        r_xy = random.uniform(r_xy_min, r_xy_max)
        az_deg = random.uniform(az_lo, az_hi)
        az_rad = math.radians(az_deg)

        apple_x = shoulder_x + r_xy * math.cos(az_rad)
        apple_y = shoulder_y + r_xy * math.sin(az_rad)

        branch_rot_deg = random.uniform(r_lo, r_hi)
        branch_rot_rad = math.radians(branch_rot_deg)

        # Trunk position solved backward so the branch tip lands exactly at
        # (apple_x, apple_y): branch_tip = tree_xy + branch_len * dir(rot).
        tree_x = apple_x - branch_len * math.cos(branch_rot_rad)
        tree_y = apple_y - branch_len * math.sin(branch_rot_rad)
        branch_mid_x = tree_x + (branch_len / 2.0) * math.cos(branch_rot_rad)
        branch_mid_y = tree_y + (branch_len / 2.0) * math.sin(branch_rot_rad)

        env_ns = f"/World/envs/env_{idx}"

        base_prim = stage.GetPrimAtPath(f"{env_ns}/TreeBase")
        if base_prim.IsValid():
            _set_translate(base_prim, tree_x, tree_y, 0.03)

        trunk_prim = stage.GetPrimAtPath(f"{env_ns}/TreeTrunk")
        if trunk_prim.IsValid():
            _set_translate(trunk_prim, tree_x, tree_y, tree_h / 2.0)

        branch_prim = stage.GetPrimAtPath(f"{env_ns}/TreeBranch")
        if branch_prim.IsValid():
            _set_translate(branch_prim, branch_mid_x, branch_mid_y, tree_h)
            _set_rotate_z(branch_prim, branch_rot_rad)

        apple_pos_local[k] = torch.tensor(
            [apple_x, apple_y, apple_z], device=env.device
        )

    env_origins = env.scene.env_origins[env_ids]
    root_state = apple.data.default_root_state[env_ids].clone()
    root_state[:, :3] = env_origins + apple_pos_local
    root_state[:, 7:] = 0.0
    apple.write_root_state_to_sim(root_state, env_ids=env_ids)

    env._stem_anchor_pos[env_ids] = env_origins + apple_pos_local

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
