# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Tree sampling (pure numpy / math / random, env-local frame) and a pool of
worker PROCESSES that run it ahead of the simulation.

Split out of apple_tree.py so that a worker can load it without Isaac Lab:
`python tree_sampler.py --worker` imports nothing but numpy, and speaks
pickles over stdin / stdout (see SamplerPool). Why processes: sampling one
spindle tree costs ~100 ms of pure-Python numpy-on-3-vectors, it does not
depend on the simulator at all, and resetting hundreds of environments
runs it serially in the training process (a thread cannot help: the GIL).
So place_apple_tree asks a pool for each environment's NEXT tree right
after taking its current one, and a reset only picks up a finished tree.
"""

from __future__ import annotations

import math
import os
import queue
import random
import subprocess
import sys
import threading
from concurrent.futures import Future

import numpy as np


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
                        spindle: dict, tries: int = 300, seed: tuple | None = None,
                        keep_out: tuple | None = None) -> dict:
    """Tall-spindle tree: ONE thin trunk, vertical up to a kink (spindle
    ["kink_range"], m), then straight, leaning spindle["lean_deg"] in a
    random direction. Every apple hangs off it on a short branch
    (spindle["branch_range"]) plus its stem, on the side facing the aisle:
    the stem (apple -> trunk) points into the row, within heading_max_deg of
    +x, as in the fruiting-wall tree. Clusters (sizes / gap / sep_min as in
    sample_tree) stack along the trunk. Every apple's grasp target lies in
    the envelope, as in sample_tree.

    seed: (apple centre [3], stem heading [rad]) -- apple 0 hangs exactly
    there instead of being sampled from seed_env (the final-approach task
    builds the tree around a fruit placed in front of a given hand pose).
    keep_out: (points [P, 3], radii [P]) -- no OTHER apple within its radius
    + radii of these points (the hand and forearm at that start pose).

    Same dict as sample_tree, plus "kink" (z) and "lean" (dx/dz, dy/dz)."""
    lim = math.radians(heading_max_deg)
    tr = spindle["trunk_radius"]

    def clear(a):
        if keep_out is None:
            return True
        pts, rad = keep_out
        return bool(np.all(np.linalg.norm(pts - a, axis=1) >= radius + rad))

    def hang(axis_pt, h, stem_len, branch):
        """apple centre, stem direction, tip for an apple on the trunk."""
        d = np.array([math.cos(h), math.sin(h), 0.0])
        a = axis_pt - d * (tr + branch + stem_len + radius)
        return a, d

    for _ in range(tries):
        if seed is None:
            t0 = seed_env.sample_target(rng)
            h0 = rng.uniform(-lim, lim)
        else:
            h0 = float(seed[1])
        L0, b0 = rng.uniform(*stem_len_range), rng.uniform(*spindle["branch_range"])
        d0 = np.array([math.cos(h0), math.sin(h0), 0.0])
        a0 = t0 + d0 * (radius + standoff) if seed is None else np.asarray(seed[0], float)
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
                    if seed_env.contains(a - d * (radius + standoff)) and clear(a) and \
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
                    if not member_env.contains(a - d * (radius + standoff), slack=0.01) or not clear(a):
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
#  Worker processes
# ─────────────────────────────────────────────

def make_spindle_job(rng: random.Random, n, radius, standoff, heading_max_deg, stem_len_range, cluster_sizes,
                     cluster_weights, cluster_gap, sep_min, spindle, seed=None, keep_out=None) -> dict:
    """One sample_spindle_tree call, as a picklable dict (the worker supplies
    the envelopes). Its random stream is seeded from `rng`."""
    return dict(rng_seed=rng.getrandbits(32), n=n, radius=radius, standoff=standoff,
                heading_max_deg=heading_max_deg, stem_len_range=tuple(stem_len_range),
                cluster_sizes=tuple(cluster_sizes), cluster_weights=tuple(cluster_weights),
                cluster_gap=tuple(cluster_gap), sep_min=sep_min, spindle=dict(spindle), seed=seed, keep_out=keep_out)


# Plain subprocesses over pickled pipes, NOT multiprocessing: spawn / forkserver
# re-import the parent's __main__ in every child, and scripts/train.py launches
# Isaac Sim at module level.

def _spindle_job(job: dict, seed_env: "Envelope", member_env: "Envelope") -> dict:
    """The kwargs of one sample_spindle_tree call (SamplerPool.submit's dict
    minus the two envelopes, which the worker holds)."""
    job = dict(job)
    rng = random.Random(job.pop("rng_seed"))
    return sample_spindle_tree(
        rng, job["n"], job["radius"], seed_env, member_env, job["standoff"], job["heading_max_deg"],
        job["stem_len_range"], job["cluster_sizes"], job["cluster_weights"], job["cluster_gap"], job["sep_min"],
        job["spindle"], seed=job.get("seed"), keep_out=job.get("keep_out"))


def _worker_main() -> None:
    import pickle
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                    # a stray print must not corrupt the pickle stream
    inp = sys.stdin.buffer
    try:
        init = pickle.load(inp)
    except EOFError:
        return
    ws = None
    if init.get("workspace_map"):
        m = np.load(init["workspace_map"])
        ws = (m["x_edges"], m["y_edges"], m["z_edges"], m["feasible"])
    sh = np.asarray(init["shoulder"], float)
    seed_env = Envelope(sh, workspace=ws, **init["seed_envelope"])
    member_env = Envelope(sh, workspace=ws, **init["member_envelope"])
    while True:
        try:
            job = pickle.load(inp)
        except EOFError:
            return
        try:
            res = ("ok", _spindle_job(job, seed_env, member_env))
        except RuntimeError as e:              # no tree fits (sample_spindle_tree's own failure)
            res = ("fail", str(e))
        except Exception as e:                 # a bug: report it, keep serving
            res = ("error", f"{type(e).__name__}: {e}")
        try:
            pickle.dump(res, out, protocol=4)
            out.flush()
        except (BrokenPipeError, OSError):     # the training process is gone: nothing to answer
            return


class SamplerPool:
    """`workers` subprocesses running sample_spindle_tree. submit(job) returns
    a Future: .result() is the tree dict, or raises RuntimeError exactly where
    the inline call would have. Each worker is served by one thread that only
    waits on a pipe (no GIL contention). Died workers fail their job with
    RuntimeError("worker ..."); the caller then falls back to sampling inline."""

    def __init__(self, workers: int, shoulder, seed_envelope: dict, member_envelope: dict,
                 workspace_map: str | None):
        import pickle
        self._pickle = pickle
        self._q: "queue.Queue" = queue.Queue()
        self._procs = []
        self._closed = False
        self._alive = 0
        self._lock = threading.Lock()
        init = {"shoulder": np.asarray(shoulder, float).tolist(), "seed_envelope": dict(seed_envelope),
                "member_envelope": dict(member_envelope), "workspace_map": workspace_map}
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1")
        for _ in range(max(1, int(workers))):
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, env=env, close_fds=True)
            pickle.dump(init, p.stdin, protocol=4)
            p.stdin.flush()
            self._procs.append(p)
            self._alive += 1
            threading.Thread(target=self._serve, args=(p,), daemon=True).start()
        import atexit
        atexit.register(self.close)

    def submit(self, job: dict) -> Future:
        f: Future = Future()
        if self._closed or self._alive <= 0:
            f.set_exception(RuntimeError("tree worker lost: no worker left"))
        else:
            self._q.put((job, f))
        return f

    def _worker_gone(self) -> None:
        """One serving thread ended for good; with the last one, fail what is still queued."""
        with self._lock:
            self._alive -= 1
            last = self._alive <= 0
        if last:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    return
                if item is not None:
                    item[1].set_exception(RuntimeError("tree worker lost: no worker left"))

    def _serve(self, p) -> None:
        while True:
            item = self._q.get()
            if item is None:                   # close()
                return
            job, f = item
            try:
                self._pickle.dump(job, p.stdin, protocol=4)
                p.stdin.flush()
                status, val = self._pickle.load(p.stdout)
            except Exception as e:             # the worker died or the pipe broke
                f.set_exception(RuntimeError(f"tree worker lost: {type(e).__name__}: {e}"))
                self._worker_gone()
                return
            if status == "ok":
                f.set_result(val)
            elif status == "fail":
                f.set_exception(RuntimeError(val))
            else:
                f.set_exception(RuntimeError(f"tree worker error: {val}"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._procs:
            self._q.put(None)
        for p in self._procs:
            try:
                p.stdin.close()
            except Exception:
                pass
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
