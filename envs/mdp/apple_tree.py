# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Multi-apple procedural tree: the same commercial fruiting wall as the
single-apple tasks (tall trunk, short horizontal branches, horizontal stems
pointing into the wall), carrying N apples -- all inside the envelope the
frozen reach / grasp policies were trained on, so the robot can harvest the
whole tree from where it stands, one apple at a time
(scripts/tree_harvest_bench.py).

Apples come in CLUSTERS of 1-4 (cluster_sizes / cluster_weights): the
apples of one cluster touch, or press a few mm into each other
(cluster_gap); different clusters keep sep_min of air between them. Each
apple hangs on its own branch from the one trunk behind them all.

Every apple is a dynamic sphere held by its own BreakableStem (envs/mdp/
harvest.py), so the hand and fingers really collide with the neighbours of
the apple they are picking: they swing on their stems, and break off and
fall when pushed hard enough.

Entities: "apple" is ALWAYS the current target -- the frozen policies, the
apple-filtered contact sensors, the reach->grasp pin and every reward read
it by that name -- and "apple_1".."apple_{N-1}" are the others. Moving on
to the next apple (an env reset with the tree kept, see place_apple_tree)
SWAPS poses: "apple" jumps to the next spot and the neighbour that hung
there takes the harvested fruit's place (in the basket, on the ground, or
back on the tree if its stem never broke). Hence one radius per tree: the
fruit that takes a spot must be the size of the one that hung there.

Every fruit, the target included, hangs on its stem from the start (the
grasp task's rigid reach-phase pin is turned off in this env): pinned, the
target was teleported back into the hand every physics step, so the hand
sank into it.

Per env, env._tree holds the spots (apple centre, stem direction, cluster),
the harvest order, who sits where and each spot's status; the bench reads
it for its report.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#: Spot status
WAITING, TARGET, DONE, KNOCKED = 0, 1, 2, 3

#: Prims that carry each spot's branch and stem (slot 0 keeps the single-
#: apple scene's names).
def branch_prim(k: int) -> str:
    return "TreeBranch" if k == 0 else f"TreeBranch_{k}"


def stem_prim(k: int) -> str:
    return "AppleStem" if k == 0 else f"AppleStem_{k}"


def apple_entity(k: int) -> str:
    return "apple" if k == 0 else f"apple_{k}"


def apple_prim(k: int) -> str:
    return "Apple" if k == 0 else f"Apple_{k}"

from .tree_sampler import (  # noqa: E402,F401
    Envelope, SamplerPool, harvest_order, make_spindle_job, sample_spindle_tree, sample_tree, trunk_axis,
)




# ─────────────────────────────────────────────
#  Events
# ─────────────────────────────────────────────

def setup_tree_prims(env: "ManagerBasedRLEnv", env_ids, n_apples: int, trunk_height_range: tuple[float, float],
                     radius_range: tuple[float, float], spawn_radius: float) -> None:
    """prestartup: trunk height per env; every branch / stem cylinder made 1 m
    long (place_apple_tree sets its length with a scale op, which can change
    at run time); ONE apple radius per env, for all its apples."""
    from pxr import Gf, UsdGeom
    import omni.usd
    from .domain_randomization import _get_cylinder_mesh

    stage = omni.usd.get_context().get_stage()
    env._trunk_height = torch.zeros(env.num_envs, device=env.device)
    env._apple_radius = torch.zeros(env.num_envs, device=env.device)
    for i in range(env.num_envs):
        ns = f"/World/envs/env_{i}"
        h = random.uniform(*trunk_height_range)
        env._trunk_height[i] = h
        _, cyl = _get_cylinder_mesh(stage, f"{ns}/TreeTrunk")
        if cyl:
            cyl.GetHeightAttr().Set(h)
        for k in range(n_apples):
            for name in (branch_prim(k), stem_prim(k)):
                _, cyl = _get_cylinder_mesh(stage, f"{ns}/{name}")
                if cyl:
                    cyl.GetHeightAttr().Set(1.0)
        r = random.uniform(*radius_range)
        env._apple_radius[i] = r
        for k in range(n_apples):
            prim = stage.GetPrimAtPath(f"{ns}/{apple_prim(k)}")
            if not prim.IsValid():
                continue
            xf = UsdGeom.Xformable(prim)
            op = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeScale), None) \
                or xf.AddScaleOp()
            s = r / spawn_radius
            op.Set(Gf.Vec3f(s, s, s) if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d(s, s, s))


def reset_joints_on_new_tree(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg) -> None:
    """mode="reset", in place of the exact joint reset: teleport the robot to
    its default pose only when a NEW tree starts. Moving on to the next
    apple of the same tree leaves the joints alone -- the arm is already
    home (the bench drove it there), and teleporting it there anyway snapped
    it by its gravity sag (the jerk at every apple change)."""
    from isaaclab.envs import mdp

    ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    T = getattr(env, "_tree", None)
    new = [i for i in ids if T is None or bool(T.new_tree[i]) or T.trees[i] is None]
    if new:
        mdp.reset_joints_by_offset(env, torch.tensor(new, device=env.device), position_range=(0.0, 0.0),
                                   velocity_range=(0.0, 0.0), asset_cfg=asset_cfg)


def hold_targets_on_new_tree(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg) -> None:
    """mode="reset": PD targets to the default pose, new trees only (see
    reset_joints_on_new_tree); on the next apple every target stays where
    the bench left it."""
    from .terminations_and_events import hold_default_joint_targets

    ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    T = getattr(env, "_tree", None)
    new = [i for i in ids if T is None or bool(T.new_tree[i]) or T.trees[i] is None]
    if new:
        hold_default_joint_targets(env, torch.tensor(new, device=env.device), asset_cfg)


def _seg_dist(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance [P] from points p [P, 3] to the segment a-b."""
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum().clamp(min=1e-12)).clamp(0.0, 1.0)
    return (p - (a + t.unsqueeze(-1) * ab)).norm(dim=-1)


def access_cost(pos: torch.Tensor, hanging: torch.Tensor, stem: torch.Tensor, s: int, r: float,
                palm_home: torch.Tensor) -> float:
    """How much the hand would run into the OTHER hanging fruit picking spot s
    (m of overlap, summed): the fruit within the volumes the hand sweeps --
      - the fingers closing around it: a ring r + 4 cm around its centre,
      - palm + wrist behind it (the robot side, the palm normal runs along
        the stem): 22 cm back from it, 5.5 cm wide,
      - the way in: from where the palm waits at home to that approach point.
    pos [M, 3] current fruit centres, hanging [M] bool, stem [M, 3]."""
    a = pos[s]
    back = a - stem[s] * (r + 0.22)
    others = [k for k in range(len(pos)) if k != s and bool(hanging[k])]
    if not others:
        return 0.0
    q = pos[others]
    over = (2 * r + 0.04 - (q - a).norm(dim=-1)).clamp(min=0.0)
    over += (r + 0.055 - _seg_dist(q, a - stem[s] * r * 0.5, back)).clamp(min=0.0)
    over += (r + 0.05 - _seg_dist(q, palm_home, back)).clamp(min=0.0)
    return float(over.sum())


def _set_cylinder(stage, path: str, center, yaw: float, length: float) -> None:
    """Place an X-axis, 1 m cylinder: centre, yaw about z, length by x-scale."""
    from pxr import Gf, UsdGeom
    from .domain_randomization import _get_or_add_xform_op, _set_rotate_z, _set_translate

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    _set_translate(prim, *map(float, center))
    _set_rotate_z(prim, yaw)
    op = _get_or_add_xform_op(prim, UsdGeom.XformOp.TypeScale)
    v = (max(length, 1e-3), 1.0, 1.0)
    op.Set(Gf.Vec3f(*v) if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d(*v))


def _set_z_segment(stage, path: str, a, b, spawn_height: float) -> None:
    """Place a Z-axis cylinder prim (spawned spawn_height long) between the
    points a and b: centre, tilt and length (z scale)."""
    from pxr import Gf, UsdGeom
    from .domain_randomization import _get_or_add_xform_op, _set_translate

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = b - a
    length = float(np.linalg.norm(v))
    _set_translate(prim, *map(float, (a + b) / 2))
    u = v / max(length, 1e-9)
    axis = np.cross([0.0, 0.0, 1.0], u)                 # rotation taking +z onto u
    sa, ca = float(np.linalg.norm(axis)), float(u[2])
    ang = math.atan2(sa, ca)
    axis = axis / sa if sa > 1e-9 else np.array([1.0, 0.0, 0.0])
    w, x, y, z = math.cos(ang / 2), *(axis * math.sin(ang / 2))
    op = _get_or_add_xform_op(prim, UsdGeom.XformOp.TypeOrient)
    op.Set(Gf.Quatf(w, x, y, z) if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd(w, x, y, z))
    sc = _get_or_add_xform_op(prim, UsdGeom.XformOp.TypeScale)
    vs = (1.0, 1.0, max(length, 1e-3) / spawn_height)
    sc.Set(Gf.Vec3f(*vs) if sc.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d(*vs))


class _TreeState:
    def __init__(self, env: "ManagerBasedRLEnv", n: int, stem_params: dict | None = None):
        from .harvest import BreakableStem, StemParams

        p = StemParams(**(stem_params or {}))

        N, dev = env.num_envs, env.device
        self.n = n
        self.spot_pos = torch.zeros(N, n, 3, device=dev)        # env-local apple centre
        self.spot_stem = torch.zeros(N, n, 3, device=dev)       # unit, apple -> branch tip
        self.spot_tip = torch.zeros(N, n, 3, device=dev)
        self.cluster = torch.zeros(N, n, dtype=torch.long, device=dev)
        self.trunk = torch.zeros(N, 2, device=dev)             # base x, y
        self.trunk_kink = torch.full((N,), 1.0e3, device=dev)  # spindle: height where the lean starts
        self.trunk_lean = torch.zeros(N, 2, device=dev)        # spindle: dx/dz, dy/dz above the kink
        self.order = torch.zeros(N, n, dtype=torch.long, device=dev)
        self.cursor = torch.zeros(N, dtype=torch.long, device=dev)   # index into order of the current target
        self.status = torch.zeros(N, n, dtype=torch.long, device=dev)
        self.occupant = torch.zeros(N, n, dtype=torch.long, device=dev)   # entity sitting at each spot
        self.knocked_by = torch.full((N, n), -1, dtype=torch.long, device=dev)
        self.new_tree = torch.ones(N, dtype=torch.bool, device=dev)
        self.next_request = torch.zeros(N, dtype=torch.bool, device=dev)
        self.retry_request = torch.zeros(N, dtype=torch.bool, device=dev)   # same target again
        self.target_stem = BreakableStem(env, "apple", params=p)
        self.stems = [None] + [BreakableStem(env, apple_entity(k), params=p) for k in range(1, n)]
        self.objs = [env.scene[apple_entity(k)] for k in range(n)]
        self.trees: list[dict | None] = [None] * N

    def current_spot(self, i: int) -> int:
        return int(self.order[i, min(int(self.cursor[i]), self.n - 1)])

    def remaining(self, i: int) -> int:
        """Spots still to harvest after the current one (fruit still on its stem)."""
        self.refresh_knocked(i)
        return int((self.status[i] == WAITING).sum())

    def refresh_knocked(self, i: int, by: int = -1) -> None:
        for s in range(self.n):
            e = int(self.occupant[i, s])
            if self.status[i, s] == WAITING and e > 0 and bool(self.stems[e].broken[i]):
                self.status[i, s] = KNOCKED
                self.knocked_by[i, s] = by

    def apply(self) -> None:
        self.target_stem.apply()
        for s in self.stems[1:]:
            s.apply()


def _pick_next(T: _TreeState, i: int, cursor: int, pos: torch.Tensor, hanging: torch.Tensor, r: float,
               palm_home: torch.Tensor) -> None:
    """"clearance" order: put the hanging fruit the hand can reach with the
    least contact with the others (access_cost) at order[cursor]; ties (within
    5 mm) go to the lowest. Re-decided at every apple, from where the fruit
    hangs NOW (pushed, knocked off)."""
    cands = [int(T.order[i, k]) for k in range(cursor, T.n) if int(T.status[i, int(T.order[i, k])]) == WAITING]
    if not cands:
        return
    stem = T.spot_stem[i]
    costs = {s: access_cost(pos, hanging, stem, s, r, palm_home) for s in cands}
    best = min(cands, key=lambda s: (round(costs[s] / 0.005), float(pos[s, 2])))
    k = [int(x) for x in T.order[i]].index(best)
    T.order[i, k], T.order[i, cursor] = T.order[i, cursor].clone(), torch.tensor(best, device=T.order.device)


def tree_state(env: "ManagerBasedRLEnv") -> _TreeState:
    return env._tree


def next_apple_requested(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Termination term: the bench asks for the next apple (an env reset that
    keeps the tree)."""
    if not hasattr(env, "_tree"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._tree.next_request.clone()


def _draw(fn, tries: int = 25):
    """fn() until it returns: sample_spindle_tree / sample_tree raise
    RuntimeError when none of their 300 candidate trees fits, which for an
    unconstrained tree happens about once in several thousand draws -- rare
    enough to be invisible in a short run and fatal in a 512-env one (a
    reset every few seconds). Each draw is independent: just draw again."""
    err = None
    for _ in range(tries):
        try:
            return fn()
        except RuntimeError as e:
            err = e
    raise err


def get_sampler_pool(env: "ManagerBasedRLEnv", workers: int, seed_envelope: dict, member_envelope: dict,
                     workspace_map: str | None, shoulder_body_name: str = "right_shoulder_pitch_link") -> SamplerPool:
    """The environment's tree-sampling worker pool (tree_sampler.SamplerPool),
    started on first use with the same envelopes place_apple_tree builds."""
    pool = getattr(env, "_sampler_pool", None)
    if pool is None:
        robot = env.scene["robot"]
        sh = (robot.data.body_pos_w[0, robot.data.body_names.index(shoulder_body_name)]
              - env.scene.env_origins[0]).cpu().numpy()
        pool = env._sampler_pool = SamplerPool(workers, sh, seed_envelope, member_envelope, workspace_map)
        env._tree_prefetch = {}
    return pool


def place_apple_tree(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
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
    shoulder_body_name: str = "right_shoulder_pitch_link",
    spindle: dict | None = None,
    sampler_workers: int = 0,
    trees: dict | None = None,
    target_spot0: bool = False,
) -> None:
    """mode="reset", in place of randomize_tree_and_apple. Per env: a NEW
    tree when env._tree.new_tree is set (the first reset, and whenever the
    bench asks), otherwise the NEXT apple of the same tree (swap, see the
    module docstring). stem_params overrides harvest.StemParams fields for
    every stem of the tree (target and neighbours alike).
    new_tree_each_reset: a fresh tree at every reset (reach training: one
    apple per episode; with order="random" the target is any of them,
    cluster members included). Leaves the same bookkeeping as randomize_tree_and_
    apple for the target ("apple" pose, _stem_anchor_pos, _stem_dir,
    _tree_segments / _tree_radii: target stem first, then the other stems,
    the branches and the trunk).
    spindle: a tall-spindle tree instead (sample_spindle_tree: one thin
    trunk, vertical then leaning, short branches); keys kink_range,
    lean_deg, branch_range, trunk_radius, and top_prim (the upper trunk
    segment's prim, a 1 m Z cylinder). The capsule list then ends with the
    two trunk segments (other trees: the trunk, then a dummy far away).
    sampler_workers > 0 (spindle trees): the trees are sampled by that many
    worker PROCESSES (tree_sampler.SamplerPool), each environment's NEXT tree
    already in flight while the current one is in use, so a reset does not
    wait for the sampler. 0: sampled here, as before.
    trees: {env index: tree dict} already sampled (pregrasp.place_tree_at_
    pregrasp); target_spot0: with those, apple 0 of the tree is the target."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    robot = env.scene["robot"]
    dev = env.device
    N = env.num_envs
    if not hasattr(env, "_tree"):
        env._tree = _TreeState(env, n_apples, stem_params)
        shoulder = robot.data.body_pos_w[0, robot.data.body_names.index(shoulder_body_name)] - env.scene.env_origins[0]
        env._shoulder_local_xyz = shoulder.clone()
        ws = None
        if workspace_map is not None:
            m = np.load(workspace_map)
            ws = (m["x_edges"], m["y_edges"], m["z_edges"], m["feasible"])
        sh = shoulder.cpu().numpy()
        env._tree_envs = (Envelope(sh, workspace=ws, **seed_envelope), Envelope(sh, workspace=ws, **member_envelope))
        env._tree_palm_home = (robot.data.body_pos_w[0, robot.data.body_names.index("right_hand_palm_link")]
                               - env.scene.env_origins[0]).clone()
        env._stem_anchor_pos = torch.zeros(N, 3, device=dev)
        env._stem_dir = torch.zeros(N, 3, device=dev)
        env._tree_segments = torch.full((N, 2 * n_apples + 2, 2, 3), 1.0e3, device=dev)
        env._tree_radii = torch.tensor([stem_radius] * n_apples + [branch_radius] * n_apples + [trunk_radius] * 2,
                                       device=dev)
        orig = env.scene.write_data_to_sim

        def write_with_stems():                       # every PHYSICS substep, before the wrenches go out
            env._tree.apply()
            orig()

        env.scene.write_data_to_sim = write_with_stems
    T = env._tree
    seed_env, member_env = env._tree_envs
    n = T.n
    ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    rng = random.Random(random.getrandbits(32))
    up = torch.tensor([0.0, 0.0, 1.0], device=dev)
    g = 9.81

    for i in ids:
        origin = env.scene.env_origins[i]
        idt = torch.tensor([i], device=dev)
        r = float(env._apple_radius[i])
        if bool(T.retry_request[i]) and not new_tree_each_reset and T.trees[i] is not None:
            # another attempt at the same apple (it is still on its stem):
            # nothing moves, the bookkeeping below is refreshed
            T.retry_request[i] = False
        elif new_tree_each_reset or bool(T.new_tree[i]) or T.trees[i] is None:
            # a target forced in front of a given hand pose (pregrasp.place_tree_at_pregrasp)
            forced = getattr(env, "_forced_seed", {}).pop(i, None)
            pre = None if trees is None else trees.pop(i, None)
            if pre is not None:
                tree = pre
                forced = True if target_spot0 else None
            elif spindle is not None and sampler_workers > 0 and forced is None:
                pool = get_sampler_pool(env, sampler_workers, seed_envelope, member_envelope, workspace_map,
                                        shoulder_body_name)

                def _job():
                    return make_spindle_job(rng, n, r, standoff, stem_heading_max_deg, stem_length_range,
                                            cluster_sizes, cluster_weights, cluster_gap, sep_min, spindle)

                fut = env._tree_prefetch.pop(i, None) or pool.submit(_job())
                try:
                    tree = fut.result()
                except RuntimeError:             # no tree fits, or the pool is gone: here, again
                    tree = _draw(lambda: sample_spindle_tree(
                        rng, n, r, seed_env, member_env, standoff, stem_heading_max_deg, stem_length_range,
                        cluster_sizes, cluster_weights, cluster_gap, sep_min, spindle))
                env._tree_prefetch[i] = pool.submit(_job())      # the NEXT reset's tree, meanwhile
            elif spindle is not None:
                # a forced target that fits no tree is the CALLER's to re-pick (pregrasp.place_tree_at_pregrasp);
                # an unconstrained draw that fails is just drawn again
                tree = _draw(lambda: sample_spindle_tree(
                    rng, n, r, seed_env, member_env, standoff, stem_heading_max_deg, stem_length_range,
                    cluster_sizes, cluster_weights, cluster_gap, sep_min, spindle,
                    seed=None if forced is None else (forced["apple"], forced["heading"]),
                    keep_out=None if forced is None else (forced["pts"], forced["rad"])),
                    tries=25 if forced is None else 1)
            elif forced is not None:
                raise ValueError("place_apple_tree: a forced target needs the spindle tree")
            else:
                tree = _draw(lambda: sample_tree(
                    rng, n, r, seed_env, member_env, standoff, trunk_dist, trunk_jitter, stem_heading_max_deg,
                    stem_length_range, cluster_sizes, cluster_weights, cluster_gap, sep_min))
            T.trees[i] = tree
            T.trunk_kink[i] = float(tree.get("kink", 1.0e3))
            T.trunk_lean[i] = torch.tensor(tree.get("lean", np.zeros(2)), device=dev, dtype=torch.float32)
            T.new_tree[i] = False
            T.spot_pos[i] = torch.tensor(tree["apple"], device=dev, dtype=torch.float32)
            T.spot_stem[i] = torch.tensor(tree["stem_dir"], device=dev, dtype=torch.float32)
            T.spot_tip[i] = torch.tensor(tree["tip"], device=dev, dtype=torch.float32)
            T.cluster[i] = torch.tensor(tree["cluster"], device=dev)
            T.trunk[i] = torch.tensor(tree["trunk"], device=dev, dtype=torch.float32)
            o = harvest_order(tree, order, rng)
            if forced is not None:                       # the forced apple (spot 0) is the target
                o = np.concatenate([[0], o[o != 0]])
            T.order[i] = torch.tensor(o, device=dev)
            T.cursor[i] = 0
            T.status[i] = WAITING
            T.knocked_by[i] = -1
            # visuals: trunk, and per spot a branch (trunk -> tip) and a stem (apple -> tip)
            ns = f"/World/envs/env_{i}"
            from .domain_randomization import _set_translate
            H = float(env._trunk_height[i])
            if spindle is not None:
                kink = float(tree["kink"])
                _set_z_segment(stage, f"{ns}/TreeTrunk", trunk_axis(tree, 0.0), trunk_axis(tree, kink), H)
                _set_z_segment(stage, f"{ns}/{spindle.get('top_prim', 'TreeTrunkTop')}",
                               trunk_axis(tree, kink), trunk_axis(tree, H), 1.0)
                p = stage.GetPrimAtPath(f"{ns}/TreeBase")
                if p.IsValid():
                    _set_translate(p, float(tree["trunk"][0]), float(tree["trunk"][1]), 0.03)
            else:
                for name, z in (("TreeBase", 0.03), ("TreeTrunk", H / 2)):
                    p = stage.GetPrimAtPath(f"{ns}/{name}")
                    if p.IsValid():
                        _set_translate(p, float(tree["trunk"][0]), float(tree["trunk"][1]), z)
            for k in range(n):
                a, d, tip, L = tree["apple"][k], tree["stem_dir"][k], tree["tip"][k], tree["stem_len"][k]
                _set_cylinder(stage, f"{ns}/{stem_prim(k)}", a + d * (r + L / 2), math.atan2(d[1], d[0]), L)
                tr = trunk_axis(tree, float(tip[2]))
                if spindle is not None:                  # branch ends at the bark, not the axis
                    tr = tr - d * spindle["trunk_radius"]
                b = tr - tip
                _set_cylinder(stage, f"{ns}/{branch_prim(k)}", (tip + tr) / 2, math.atan2(b[1], b[0]),
                              float(np.linalg.norm(b)))
            if order == "clearance":
                _pick_next(T, i, 0, T.spot_pos[i], torch.ones(n, dtype=torch.bool, device=dev), r, env._tree_palm_home)
            # apples: the first in harvest order is the target ("apple"), the others in spot order
            first = int(T.order[i, 0])
            others = [s for s in range(n) if s != first]
            T.occupant[i, first] = 0
            for e, s in enumerate(others, start=1):
                T.occupant[i, s] = e
            T.status[i, first] = TARGET
            for s in range(n):
                e = int(T.occupant[i, s])
                obj = T.objs[e]
                st = obj.data.default_root_state[idt].clone()
                st[:, :3] = origin + T.spot_pos[i, s]
                st[:, 7:] = 0.0
                obj.write_root_state_to_sim(st, env_ids=idt)
                # every fruit hangs on its own stem, the target included (no pin)
                m = float(obj.cfg.spawn.mass_props.mass)
                (T.stems[e] if e > 0 else T.target_stem).engage(
                    idt, axis_w=T.spot_stem[i, s:s + 1], pos=st[:, :3], quat=st[:, 3:7],
                    preload=(m * g * up).unsqueeze(0))
        else:
            # next apple of the same tree: the fruit just handled stays where it is,
            # as the neighbour entity that hung at the next spot
            prev = T.current_spot(i)
            prev_attached = bool(T.target_stem.active[i]) and not bool(T.target_stem.broken[i])
            T.status[i, prev] = DONE                     # its turn is over, whatever happened
            T.refresh_knocked(i, by=prev)
            T.cursor[i] += 1
            if order == "clearance" and int(T.cursor[i]) < n:
                pos = torch.stack([o.data.root_pos_w[i] for o in T.objs]) - origin           # by entity
                pos_spot = pos[T.occupant[i]]
                hang = T.status[i] == WAITING
                _pick_next(T, i, int(T.cursor[i]), pos_spot, hang, r, env._tree_palm_home)
            while int(T.cursor[i]) < n and int(T.status[i, int(T.order[i, int(T.cursor[i])])]) != WAITING:
                T.cursor[i] += 1
            if int(T.cursor[i]) >= n:
                # nothing left: keep "apple" where it is (the bench stops asking)
                T.cursor[i] = n
                T.next_request[i] = False
                continue
            nxt = T.current_spot(i)
            e = int(T.occupant[i, nxt])
            target, other = T.objs[0], T.objs[e]
            s_t = target.data.root_state_w[idt].clone()
            s_o = other.data.root_state_w[idt].clone()
            s_o[:, 7:] = 0.0
            T.stems[e].release(idt)
            target.write_root_state_to_sim(s_o, env_ids=idt)
            other.write_root_state_to_sim(s_t, env_ids=idt)
            T.occupant[i, nxt], T.occupant[i, prev] = 0, e
            if prev_attached:
                # the last target never came off: it hangs on as entity e
                m = float(other.cfg.spawn.mass_props.mass)
                T.stems[e].engage(idt, axis_w=T.spot_stem[i, prev:prev + 1], pos=s_t[:, :3], quat=s_t[:, 3:7],
                                  preload=(m * g * up).unsqueeze(0))
            m = float(target.cfg.spawn.mass_props.mass)
            T.target_stem.engage(idt, axis_w=T.spot_stem[i, nxt:nxt + 1], pos=s_o[:, :3], quat=s_o[:, 3:7],
                                 preload=(m * g * up).unsqueeze(0))
            T.status[i, nxt] = TARGET

        # bookkeeping for the target, as randomize_tree_and_apple leaves it
        s = T.current_spot(i)
        env._stem_anchor_pos[i] = env.scene[apple_entity(0)].data.root_pos_w[i]
        env._stem_dir[i] = T.spot_stem[i, s]
        stems = [s] + [k for k in range(n) if k != s]
        def axis(z):
            xy = T.trunk[i] + T.trunk_lean[i] * (z - T.trunk_kink[i]).clamp(min=0.0)
            return torch.cat([xy, z.view(1)])

        seg = torch.full((2 * n + 2, 2, 3), 1.0e3, device=dev)
        for j, k in enumerate(stems):
            a = T.spot_pos[i, k] + T.spot_stem[i, k] * r
            seg[j] = torch.stack([a, T.spot_tip[i, k]])
            tip = T.spot_tip[i, k]
            seg[n + j] = torch.stack([tip, axis(tip[2])])
        zero, top = torch.zeros((), device=dev), env._trunk_height[i]
        kink = torch.minimum(T.trunk_kink[i], top)
        seg[2 * n] = torch.stack([axis(zero), axis(kink)])
        if bool(T.trunk_kink[i] < top):
            seg[2 * n + 1] = torch.stack([axis(kink), axis(top)])
        env._tree_segments[i] = seg + origin
        T.next_request[i] = False
