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


# ─────────────────────────────────────────────
#  Tree sampling (pure numpy, env-local frame)
# ─────────────────────────────────────────────

class Envelope:
    """Shoulder-relative reach envelope + safe-workspace map, as
    domain_randomization.randomize_tree_and_apple samples it."""

    def __init__(self, shoulder, reach_min, reach_max, az_center_deg, az_span_deg, dz_range,
                 min_forward=0.0, min_horizontal=0.0, workspace=None):
        self.sh = np.asarray(shoulder, float)
        self.rmin, self.rmax = reach_min, reach_max
        self.az = (math.radians(az_center_deg - az_span_deg / 2), math.radians(az_center_deg + az_span_deg / 2))
        self.dz, self.fwd, self.hor = dz_range, min_forward, min_horizontal
        self.ws = workspace

    def feasible(self, rel) -> bool:
        if self.ws is None:
            return True
        xe, ye, ze, feas = self.ws
        idx = [int(math.floor((v - e[0]) / (e[1] - e[0]))) for v, e in zip(rel, (xe, ye, ze))]
        return all(0 <= i < n for i, n in zip(idx, feas.shape)) and bool(feas[tuple(idx)])

    def contains(self, target, slack=0.0) -> bool:
        rel = target - self.sh
        r, rxy = np.linalg.norm(rel), np.linalg.norm(rel[:2])
        az = math.atan2(rel[1], rel[0])
        return (self.rmin - slack <= r <= self.rmax + slack and self.dz[0] - slack <= rel[2] <= self.dz[1] + slack
                and self.az[0] - slack / 0.3 <= az <= self.az[1] + slack / 0.3
                and rel[0] >= self.fwd - slack and rxy >= self.hor - slack and self.feasible(rel))

    def sample_target(self, rng: random.Random):
        for _ in range(500):
            dz = rng.uniform(*self.dz)
            rxy_max = math.sqrt(max(self.rmax ** 2 - dz ** 2, 1e-4))
            rxy_min = max(math.sqrt(max(self.rmin ** 2 - dz ** 2, 0.0)), self.hor)
            if rxy_min >= rxy_max:
                continue
            rxy, az = rng.uniform(rxy_min, rxy_max), rng.uniform(*self.az)
            rel = np.array([rxy * math.cos(az), rxy * math.sin(az), dz])
            if rel[0] >= self.fwd and self.feasible(rel):
                return self.sh + rel
        raise RuntimeError("Envelope.sample_target: envelope unsatisfiable")


def sample_tree(rng: random.Random, n: int, radius: float, seed_env: Envelope, member_env: Envelope,
                standoff: float, trunk_dist: float, trunk_jitter: float, heading_max_deg: float,
                stem_len_range: tuple[float, float], cluster_sizes, cluster_weights,
                cluster_gap: tuple[float, float], sep_min: float, tries: int = 200) -> dict:
    """One tree: n apple centres in clusters, every one reachable (its grasp
    target -- standoff beyond the apple surface, opposite its stem -- inside
    the envelope), stems pointing at the trunk within heading_max_deg of +x.

    Cluster seeds come from seed_env (the grasp task's placement envelope),
    the other members of a cluster from member_env (the reach policy's
    wider training envelope): they sit wherever their seed's neighbourhood
    puts them."""
    sh = seed_env.sh
    az_c = 0.5 * sum(seed_env.az)
    for _ in range(tries):
        trunk = sh[:2] + trunk_dist * np.array([math.cos(az_c), math.sin(az_c)]) \
            + np.array([rng.uniform(-trunk_jitter, trunk_jitter), rng.uniform(-trunk_jitter, trunk_jitter)])
        lim = math.radians(heading_max_deg)

        def heading_from(p):
            h = math.atan2(trunk[1] - p[1], trunk[0] - p[0])
            return max(-lim, min(lim, h))

        def apple_from_target(t):
            h = heading_from(t)
            d = np.array([math.cos(h), math.sin(h), 0.0])
            return t + d * (radius + standoff), d

        apples, dirs, cluster = [], [], []
        ok = True
        c = 0
        while len(apples) < n and ok:
            size = min(rng.choices(cluster_sizes, weights=cluster_weights)[0], n - len(apples))
            placed = False
            for _ in range(60):                                   # the seed
                t = seed_env.sample_target(rng)
                a, d = apple_from_target(t)
                if all(np.linalg.norm(a - b) >= 2 * radius + sep_min for b in apples):
                    placed = True
                    break
            if not placed:
                ok = False
                break
            members, mdirs = [a], [d]
            for _ in range(size - 1):                             # the rest of the cluster, around it
                placed = False
                for _ in range(80):
                    anchor = members[rng.randrange(len(members))]
                    u = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])
                    u -= 0.7 * np.dot(u, mdirs[0]) * mdirs[0]     # mostly beside, not in front/behind
                    if np.linalg.norm(u) < 1e-3:
                        continue
                    u /= np.linalg.norm(u)
                    p = anchor + u * (2 * radius + rng.uniform(*cluster_gap))
                    h = heading_from(p)
                    dp = np.array([math.cos(h), math.sin(h), 0.0])
                    t = p - dp * (radius + standoff)
                    if not member_env.contains(t, slack=0.01):
                        continue
                    if any(np.linalg.norm(p - m) < 2 * radius + cluster_gap[0] - 1e-6 for m in members):
                        continue
                    if any(np.linalg.norm(p - b) < 2 * radius + sep_min for b in apples):
                        continue
                    members.append(p); mdirs.append(dp)
                    placed = True
                    break
                if not placed:
                    break                                         # a smaller cluster than drawn
            apples += members; dirs += mdirs; cluster += [c] * len(members)
            c += 1
        if not ok or len(apples) < n:
            continue
        apples, dirs = np.array(apples), np.array(dirs)
        stem_len = np.array([rng.uniform(*stem_len_range) for _ in range(n)])
        tips = apples + dirs * (radius + stem_len)[:, None]
        return {"apple": apples, "stem_dir": dirs, "stem_len": stem_len, "tip": tips,
                "cluster": np.array(cluster), "trunk": trunk}
    raise RuntimeError(f"sample_tree: could not place {n} apples in {tries} trees -- too many for the envelope?")


def trunk_axis(tree: dict, z: float) -> np.ndarray:
    """Trunk axis point (env-local) at height z: vertical from tree["trunk"]
    (base x, y) up to tree["kink"], straight along tree["lean"] (dx/dz,
    dy/dz) above it. Trees without a kink are plain vertical trunks."""
    k = tree.get("kink", float("inf"))
    lean = tree.get("lean", np.zeros(2))
    xy = np.asarray(tree["trunk"], float) + lean * max(z - k, 0.0)
    return np.array([xy[0], xy[1], z])


def sample_spindle_tree(rng: random.Random, n: int, radius: float, seed_env: Envelope, member_env: Envelope,
                        standoff: float, heading_max_deg: float, stem_len_range: tuple[float, float],
                        cluster_sizes, cluster_weights, cluster_gap: tuple[float, float], sep_min: float,
                        spindle: dict, tries: int = 300) -> dict:
    """Tall-spindle tree: ONE thin trunk, vertical up to a kink (spindle
    ["kink_range"], m), then straight, leaning spindle["lean_deg"] in a
    random direction. Every apple hangs off it on a short branch
    (spindle["branch_range"]) plus its stem, on the side facing the aisle:
    the stem (apple -> trunk) points into the row, within heading_max_deg of
    +x, as in the fruiting-wall tree. Clusters (sizes / gap / sep_min as in
    sample_tree) stack along the trunk. Every apple's grasp target lies in
    the envelope, as in sample_tree.

    Same dict as sample_tree, plus "kink" (z) and "lean" (dx/dz, dy/dz)."""
    lim = math.radians(heading_max_deg)
    tr = spindle["trunk_radius"]

    def hang(axis_pt, h, stem_len, branch):
        """apple centre, stem direction, tip for an apple on the trunk."""
        d = np.array([math.cos(h), math.sin(h), 0.0])
        a = axis_pt - d * (tr + branch + stem_len + radius)
        return a, d

    for _ in range(tries):
        t0 = seed_env.sample_target(rng)
        h0 = rng.uniform(-lim, lim)
        L0, b0 = rng.uniform(*stem_len_range), rng.uniform(*spindle["branch_range"])
        d0 = np.array([math.cos(h0), math.sin(h0), 0.0])
        a0 = t0 + d0 * (radius + standoff)
        p0 = a0 + d0 * (radius + L0 + b0 + tr)                 # trunk axis at the seed's height
        kink = rng.uniform(*spindle["kink_range"])
        alpha, psi = math.radians(rng.uniform(*spindle["lean_deg"])), rng.uniform(0.0, 2 * math.pi)
        lean = math.tan(alpha) * np.array([math.cos(psi), math.sin(psi)])
        tree = {"trunk": p0[:2] - lean * max(p0[2] - kink, 0.0), "kink": kink, "lean": lean}
        z_lo = seed_env.sh[2] + seed_env.dz[0] - 0.02
        z_hi = seed_env.sh[2] + seed_env.dz[1] + 0.02

        apples, dirs, stems, cluster = [a0], [d0], [L0], [0]
        size = min(rng.choices(cluster_sizes, weights=cluster_weights)[0], n)
        c, ok = 0, True
        while len(apples) < n and ok:
            if c > 0:                                          # a new cluster's seed, anywhere on the trunk
                size = min(rng.choices(cluster_sizes, weights=cluster_weights)[0], n - len(apples))
                for _ in range(80):
                    h, L, b = rng.uniform(-lim, lim), rng.uniform(*stem_len_range), rng.uniform(*spindle["branch_range"])
                    a, d = hang(trunk_axis(tree, rng.uniform(z_lo, z_hi)), h, L, b)
                    if seed_env.contains(a - d * (radius + standoff)) and \
                            all(np.linalg.norm(a - q) >= 2 * radius + sep_min for q in apples):
                        apples.append(a); dirs.append(d); stems.append(L); cluster.append(c)
                        break
                else:
                    ok = False
                    break
            members = [len(apples) - 1]
            for _ in range(size - 1):                          # the rest of the cluster, touching it
                for _ in range(80):
                    anchor = apples[members[rng.randrange(len(members))]]
                    z = anchor[2] + rng.uniform(-1.0, 1.0) * (2 * radius + cluster_gap[1])
                    h, L, b = rng.uniform(-lim, lim), rng.uniform(*stem_len_range), rng.uniform(*spindle["branch_range"])
                    a, d = hang(trunk_axis(tree, z), h, L, b)
                    gaps_in = [np.linalg.norm(a - apples[m]) - 2 * radius for m in members]
                    if not (cluster_gap[0] - 1e-6 <= min(gaps_in) <= cluster_gap[1] + 0.01):
                        continue
                    if any(np.linalg.norm(a - q) < 2 * radius + sep_min
                           for j, q in enumerate(apples) if cluster[j] != c):
                        continue
                    if not member_env.contains(a - d * (radius + standoff), slack=0.01):
                        continue
                    apples.append(a); dirs.append(d); stems.append(L); cluster.append(c); members.append(len(apples) - 1)
                    break
                else:
                    break                                      # a smaller cluster than drawn
            c += 1
        if not ok or len(apples) < n:
            continue
        apples, dirs, stems = np.array(apples), np.array(dirs), np.array(stems)
        tree.update({"apple": apples, "stem_dir": dirs, "stem_len": stems,
                     "tip": apples + dirs * (radius + stems)[:, None], "cluster": np.array(cluster)})
        return tree
    raise RuntimeError(f"sample_spindle_tree: could not place {n} apples on one trunk in {tries} trees")


def harvest_order(tree: dict, mode: str, rng: random.Random) -> np.ndarray:
    n = len(tree["apple"])
    if mode == "random":
        o = list(range(n)); rng.shuffle(o)
        return np.array(o)
    z = tree["apple"][:, 2]
    return np.argsort(-z if mode == "top_down" else z, kind="stable")      # ("clearance" starts bottom-up)


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
    two trunk segments (other trees: the trunk, then a dummy far away)."""
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
            if spindle is not None:
                tree = sample_spindle_tree(rng, n, r, seed_env, member_env, standoff, stem_heading_max_deg,
                                           stem_length_range, cluster_sizes, cluster_weights, cluster_gap,
                                           sep_min, spindle)
            else:
                tree = sample_tree(rng, n, r, seed_env, member_env, standoff, trunk_dist, trunk_jitter,
                                   stem_heading_max_deg, stem_length_range, cluster_sizes, cluster_weights,
                                   cluster_gap, sep_min)
            T.trees[i] = tree
            T.trunk_kink[i] = float(tree.get("kink", 1.0e3))
            T.trunk_lean[i] = torch.tensor(tree.get("lean", np.zeros(2)), device=dev, dtype=torch.float32)
            T.new_tree[i] = False
            T.spot_pos[i] = torch.tensor(tree["apple"], device=dev, dtype=torch.float32)
            T.spot_stem[i] = torch.tensor(tree["stem_dir"], device=dev, dtype=torch.float32)
            T.spot_tip[i] = torch.tensor(tree["tip"], device=dev, dtype=torch.float32)
            T.cluster[i] = torch.tensor(tree["cluster"], device=dev)
            T.trunk[i] = torch.tensor(tree["trunk"], device=dev, dtype=torch.float32)
            T.order[i] = torch.tensor(harvest_order(tree, order, rng), device=dev)
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
