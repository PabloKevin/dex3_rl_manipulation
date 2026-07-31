# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Numpy-only evaluation math and .npz IO shared by evaluate_policy.py and plot_results.py.

Deliberately imports NOTHING from torch or isaaclab. evaluate_policy.py has to
boot Isaac Sim; plot_results.py must not. Keeping every derived metric here means
the numbers in the paper's figures and the numbers in its tables come from the
same code path, and plot_results.py stays a plain ~1 s python process.

npz schema (version 1) -- N completed episodes, T = max_episode_length (250),
J = 7 actuated arm joints:

  per-episode, shape (N,) unless noted
    ep_index          int32    global completion order
    ep_env_id         int16    which parallel env produced it
    ep_global_step    int32    rollout loop step at completion (warm-up filtering)
    ep_length         int16    control steps, 1-based
    ep_success        bool     reach_success fired
    ep_timeout        bool     time_out fired
    ep_dropped        bool     apple_dropped fired
    d_final           float32  m, palm-to-grasp-point distance at the terminal step
    d_min             float32  m, best distance reached during the episode
    theta_final       float32  rad, palm-normal-vs-+Z angle at the terminal step
    theta_min         float32  rad
    first_reach_step  int16    first step with d<pos_tol AND theta<ori_tol; -1 if never
    t_first_reach     float32  s; NaN if never reached
    t_success         float32  s; NaN if not a success
    frac_in_tol_after float32  fraction of steps in tolerance after first entry; NaN if never
    n_tol_exits       int16    times it left the tolerance ball after entering; -1 if never
    apple_pos_w       float32  (N,3) world
    shoulder_pos_w    float32  (N,3) world
    r_xy              float32  m, shoulder-centred horizontal radius to the apple
    r3d_target        float32  m, shoulder-centred 3-D radius to the GRASP POINT
    azimuth_deg       float32  deg, atan2(dy, dx) about the shoulder
    dz_target         float32  m, signed height of the grasp point above the shoulder
    q_final           float32  (N,J) rad

  per-timestep (absent when the rollout ran with --no_traces)
    traj_mask   bool    (N,T)   True where the episode was still running
    traj_q      float32 (N,T,J) measured joint positions
    traj_dq     float32 (N,T,J) measured joint velocities
    traj_a_raw  float32 (N,T,J) raw policy output
    traj_u      float32 (N,T,J) rate-limited commanded joint target
    traj_d      float32 (N,T)   position error
    traj_theta  float32 (N,T)   orientation error

  metadata
    meta_json   <U      ()      0-d JSON string; npz cannot store dicts
    dt          float32 ()      control timestep, 0.02 s
    pos_tol     float32 ()      0.04 m
    ori_tol_rad float32 ()      0.30 rad
    hold_steps  int32   ()      25
    offset_z    float32 ()      GRASP_OFFSET_Z used for the grasp point
    joint_names <U32    (J,)    RIGHT_ARM_JOINTS, order preserved
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Sequence

import numpy as np

NPZ_VERSION = 1

# Number of leading steps to discard from every trace before differentiating.
# RateLimitedJointPositionAction re-anchors _prev_target/_prev_vel to the actual
# joint state when episode_length_buf <= 1, and reset_arm_jitter perturbs the
# arm by +-0.20 rad, so the first couple of steps contain a reset transient that
# is not policy behaviour.
SKIP_FIRST = 3


# ═════════════════════════════════════════════════════════════════════════
#  IO
# ═════════════════════════════════════════════════════════════════════════

def save_results(path: str, episodes: dict, traces: dict | None, meta: dict) -> None:
    """Write one compressed npz holding episodes + optional traces + metadata."""
    meta = dict(meta)
    meta["npz_version"] = NPZ_VERSION
    payload = dict(episodes)
    if traces:
        payload.update(traces)
    payload["meta_json"] = np.array(json.dumps(meta, default=str))
    np.savez_compressed(path, **payload)


def load_results(path: str) -> SimpleNamespace:
    """Load an npz into a namespace with `.meta` already parsed from JSON.

    Every array is materialised (npz members are lazy file handles, which go
    stale once the archive is closed).
    """
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data.pop("meta_json")))
    ns = SimpleNamespace(**data)
    ns.meta = meta
    ns.path = path
    ns.has_traces = "traj_q" in data
    ns.dt = float(data["dt"]) if "dt" in data else 0.02
    ns.n_episodes = int(data["ep_success"].shape[0])
    return ns


# ═════════════════════════════════════════════════════════════════════════
#  Finite differences
# ═════════════════════════════════════════════════════════════════════════

def joint_accel(dq: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """First central difference of joint velocity -> rad/s^2. Shape preserved.

    Edge samples (t=0 and t=L-1) are set to NaN rather than one-sided, so a
    caller that forgets to mask cannot silently mix a low-order edge estimate
    into an RMS.
    """
    dq = np.asarray(dq, dtype=np.float64)
    out = np.full_like(dq, np.nan)
    out[..., 1:-1, :] = (dq[..., 2:, :] - dq[..., :-2, :]) / (2.0 * dt)
    return out


def joint_jerk(dq: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """Second CENTRAL difference of joint velocity -> rad/s^3. Shape preserved.

        j[t] = (dq[t+1] - 2*dq[t] + dq[t-1]) / dt^2

    Differentiating the reported joint VELOCITY twice divides by dt^2 = 4e-4
    (a 2 500x noise gain). Taking a third difference of joint POSITION instead
    would divide by dt^3 = 8e-6 (125 000x) and drown the signal in PhysX
    quantisation noise -- which is why this uses dq, not q.

    Central rather than backward: zero phase lag, so the jerk trace lines up in
    time with the velocity trace plotted above it, and the truncation error is
    O(dt^2) instead of O(dt).

    Edge samples are NaN; use `valid_window` / `per_episode_jerk_stats` to mask.
    """
    dq = np.asarray(dq, dtype=np.float64)
    out = np.full_like(dq, np.nan)
    out[..., 1:-1, :] = (dq[..., 2:, :] - 2.0 * dq[..., 1:-1, :] + dq[..., :-2, :]) / (dt * dt)
    return out


def command_jerk(u: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """Third backward difference of the COMMANDED joint target -> rad/s^3.

        j_cmd[t] = (u[t] - 3*u[t-1] + 3*u[t-2] - u[t-3]) / dt^3

    This is the quantity the rate limiter bounds one derivative below: it caps
    the commanded acceleration at `max_acceleration`, so the commanded jerk is
    analytically bounded by

        |j_cmd| <= 2 * max_acceleration / dt

    (the largest possible step change in a bounded acceleration, divided by dt).
    At max_acceleration = 15 rad/s^2 and dt = 0.02 s that ceiling is
    1500 rad/s^3 -- see `command_jerk_ceiling`. With the limiter disabled there
    is no such bound, which is exactly the ablation figure.

    Backward rather than central: u is a command sequence, and the limiter's own
    recursion is causal, so the causal difference is what corresponds to it.
    The first three samples are NaN.
    """
    u = np.asarray(u, dtype=np.float64)
    out = np.full_like(u, np.nan)
    out[..., 3:, :] = (u[..., 3:, :] - 3.0 * u[..., 2:-1, :]
                       + 3.0 * u[..., 1:-2, :] - u[..., :-3, :]) / (dt ** 3)
    return out


def command_jerk_ceiling(max_acceleration: float, dt: float = 0.02) -> float:
    """Analytic upper bound on |j_cmd| under the rate limiter, rad/s^3."""
    return 2.0 * max_acceleration / dt


def command_velocity(u: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """Backward difference of the commanded target -> rad/s. First sample NaN.

    Under the limiter this must never exceed cfg.max_velocity; asserting that
    is how evaluate_policy.py proves the ablation actually took effect.
    """
    u = np.asarray(u, dtype=np.float64)
    out = np.full_like(u, np.nan)
    out[..., 1:, :] = (u[..., 1:, :] - u[..., :-1, :]) / dt
    return out


def command_accel(u: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """Second backward difference of the commanded target -> rad/s^2. First 2 NaN."""
    u = np.asarray(u, dtype=np.float64)
    out = np.full_like(u, np.nan)
    out[..., 2:, :] = (u[..., 2:, :] - 2.0 * u[..., 1:-1, :] + u[..., :-2, :]) / (dt * dt)
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Masking
# ═════════════════════════════════════════════════════════════════════════

def valid_window(mask: np.ndarray, skip_first: int = SKIP_FIRST,
                 skip_last: int = 2) -> np.ndarray:
    """Shrink a (N,T) validity mask so finite-difference stencils never cross an edge.

    Drops `skip_first` leading steps (reset transient) and `skip_last` trailing
    steps (the stencil needs t+-1, and the trace is truncated at termination, so
    the last samples would differentiate against zero padding).
    """
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    lengths = mask.sum(axis=-1)
    T = mask.shape[-1]
    idx = np.arange(T)[None, :]
    out &= idx >= skip_first
    out &= idx < np.maximum(lengths - skip_last, 0)[:, None]
    return out


def masked_percentile(x: np.ndarray, mask: np.ndarray, q: Sequence[float],
                      min_count: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestep percentiles across episodes, ignoring the zero padding.

    Without this, every trace-aggregate curve collapses toward zero after ~2 s:
    successful episodes terminate early, and their padded tail would otherwise
    be averaged in as real data.

    Args:
        x: (N, T) values.
        mask: (N, T) True where the episode was running.
        q: percentiles in [0, 100].
        min_count: timesteps with fewer than this many live episodes give NaN.

    Returns:
        (percentiles (len(q), T), n_valid (T,))
    """
    x = np.asarray(x, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    xm = np.where(mask, x, np.nan)
    n_valid = mask.sum(axis=0)
    with np.errstate(all="ignore"):
        out = np.nanpercentile(xm, list(q), axis=0)
    out = np.atleast_2d(out)
    out[:, n_valid < min_count] = np.nan
    return out, n_valid


# ═════════════════════════════════════════════════════════════════════════
#  Derived per-episode statistics
# ═════════════════════════════════════════════════════════════════════════

def per_episode_jerk_stats(traj_dq: np.ndarray, traj_mask: np.ndarray,
                           dt: float = 0.02, skip_first: int = SKIP_FIRST) -> dict:
    """RMS / p95 / max of ||j(t)||_2 over the 7 joints, one number per episode.

    The 2-norm across joints collapses each timestep to a single magnitude, so
    an episode gets one jerk number rather than seven -- otherwise the summary
    table needs a joint dimension nobody reads.
    """
    j = joint_jerk(traj_dq, dt)                      # (N,T,J)
    jn = np.linalg.norm(np.nan_to_num(j), axis=-1)   # (N,T)
    good = valid_window(traj_mask, skip_first=skip_first, skip_last=2)
    good &= np.isfinite(j).all(axis=-1)

    n = jn.shape[0]
    rms = np.full(n, np.nan)
    p95 = np.full(n, np.nan)
    mx = np.full(n, np.nan)
    for i in range(n):
        v = jn[i][good[i]]
        if v.size:
            rms[i] = np.sqrt(np.mean(v ** 2))
            p95[i] = np.percentile(v, 95)
            mx[i] = v.max()
    return {"jerk_rms": rms, "jerk_p95": p95, "jerk_max": mx}


def per_episode_vel_stats(traj_dq: np.ndarray, traj_mask: np.ndarray,
                          skip_first: int = SKIP_FIRST) -> dict:
    """Peak and RMS of ||dq(t)||_2, one number per episode."""
    vn = np.linalg.norm(np.nan_to_num(traj_dq), axis=-1)
    good = valid_window(traj_mask, skip_first=skip_first, skip_last=0)
    n = vn.shape[0]
    peak = np.full(n, np.nan)
    rms = np.full(n, np.nan)
    for i in range(n):
        v = vn[i][good[i]]
        if v.size:
            peak[i] = v.max()
            rms[i] = np.sqrt(np.mean(v ** 2))
    return {"vel_peak": peak, "vel_rms": rms}


def check_rate_limits(traj_u: np.ndarray, traj_mask: np.ndarray, dt: float,
                      max_velocity: float, max_acceleration: float,
                      tol: float = 1e-3) -> dict:
    """Measure the realised bounds on the commanded trajectory.

    Returns the observed maxima plus pass/fail against the configured ceilings.
    With the limiter ON both must hold; with it OFF (max_* set to 1e9) the
    observed velocity is expected to blow past the nominal 2.0 rad/s, which is
    what proves the ablation actually took effect.
    """
    v = np.abs(command_velocity(traj_u, dt))
    a = np.abs(command_accel(traj_u, dt))
    good = valid_window(traj_mask, skip_first=SKIP_FIRST, skip_last=0)[..., None]
    vg = v[np.broadcast_to(good, v.shape) & np.isfinite(v)]
    ag = a[np.broadcast_to(good, a.shape) & np.isfinite(a)]
    v_max = float(vg.max()) if vg.size else float("nan")
    a_max = float(ag.max()) if ag.size else float("nan")
    return {
        "cmd_vel_max": v_max,
        "cmd_acc_max": a_max,
        "vel_within": bool(v_max <= max_velocity + tol),
        "acc_within": bool(a_max <= max_acceleration + tol),
    }


# ═════════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion.

    Wilson rather than normal-approximation: the success rate here may sit near
    0 or 1, where the normal interval runs outside [0, 1] and understates the
    uncertainty.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("mean", "median", "std", "p95", "n")} | {"n": 0}
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "p95": float(np.percentile(x, 95)),
        "n": int(x.size),
    }


def summarize(res: SimpleNamespace) -> dict:
    """Every number the paper's results table quotes, from one place.

    Position/orientation error rows are reported BOTH pooled and
    success-conditioned. A pooled mean that mixes settled successes with
    timed-out failures is not a meaningful headline number, and quoting only
    the success-conditioned one would flatter the policy.
    """
    succ = res.ep_success.astype(bool)
    n = succ.size
    k = int(succ.sum())
    lo, hi = wilson_ci(k, n)

    reached = res.first_reach_step >= 0
    out: dict = {
        "n_episodes": n,
        "n_success": k,
        "success_rate": k / n if n else float("nan"),
        "success_ci95": (lo, hi),
        "n_reached": int(reached.sum()),
        "reach_rate": float(reached.mean()) if n else float("nan"),
        "n_timeout": int(res.ep_timeout.astype(bool).sum()),
        "meta": getattr(res, "meta", {}),
    }

    out["d_final_mm_all"] = _stats(res.d_final * 1e3)
    out["d_final_mm_succ"] = _stats(res.d_final[succ] * 1e3)
    out["d_min_mm_all"] = _stats(res.d_min * 1e3)
    out["theta_final_deg_all"] = _stats(np.degrees(res.theta_final))
    out["theta_final_deg_succ"] = _stats(np.degrees(res.theta_final[succ]))
    out["theta_min_deg_all"] = _stats(np.degrees(res.theta_min))
    out["t_first_reach_s"] = _stats(res.t_first_reach)
    out["t_success_s"] = _stats(res.t_success)
    out["ep_len_s"] = _stats(res.ep_length * res.dt)
    out["frac_in_tol_after"] = _stats(res.frac_in_tol_after)
    out["n_tol_exits"] = _stats(np.where(res.n_tol_exits >= 0, res.n_tol_exits, np.nan))

    # Position criterion alone, ignoring orientation -- decomposes the joint
    # success criterion and shows how much of the failure is orientation.
    pos_tol = float(res.pos_tol) if hasattr(res, "pos_tol") else 0.04
    ori_tol = float(res.ori_tol_rad) if hasattr(res, "ori_tol_rad") else 0.30
    out["pos_only_rate"] = float((res.d_min < pos_tol).mean())
    out["ori_only_rate"] = float((res.theta_min < ori_tol).mean())

    if getattr(res, "has_traces", False):
        out.update({k2: _stats(v) for k2, v in
                    per_episode_jerk_stats(res.traj_dq, res.traj_mask, res.dt).items()})
        out.update({k2: _stats(v) for k2, v in
                    per_episode_vel_stats(res.traj_dq, res.traj_mask).items()})
        meta = getattr(res, "meta", {})
        out["rate_limit_check"] = check_rate_limits(
            res.traj_u, res.traj_mask, res.dt,
            float(meta.get("max_velocity", 2.0)),
            float(meta.get("max_acceleration", 15.0)),
        )
    return out


_ROWS = [
    ("final position error [mm], all", "d_final_mm_all"),
    ("final position error [mm], success", "d_final_mm_succ"),
    ("min position error [mm], all", "d_min_mm_all"),
    ("final orientation error [deg], all", "theta_final_deg_all"),
    ("final orientation error [deg], success", "theta_final_deg_succ"),
    ("min orientation error [deg], all", "theta_min_deg_all"),
    ("time to first reach [s]", "t_first_reach_s"),
    ("time to settled success [s]", "t_success_s"),
    ("episode duration [s]", "ep_len_s"),
    ("fraction in tolerance after entry", "frac_in_tol_after"),
    ("tolerance exits after entry", "n_tol_exits"),
    ("joint jerk RMS [rad/s^3]", "jerk_rms"),
    ("joint jerk p95 [rad/s^3]", "jerk_p95"),
    ("joint jerk max [rad/s^3]", "jerk_max"),
    ("peak |dq| [rad/s]", "vel_peak"),
    ("RMS |dq| [rad/s]", "vel_rms"),
]


def format_summary_table(s: dict) -> str:
    """Plain-text summary. numpy + str only -- pandas is not installed here."""
    L = []
    L.append("=" * 78)
    m = s.get("meta", {})
    if m:
        L.append(f"checkpoint     : {m.get('checkpoint', '?')}")
        L.append(f"task / envs    : {m.get('task', '?')}  x{m.get('num_envs', '?')}")
        L.append(f"rate limiter   : {m.get('rate_limit', '?')} "
                 f"(v_max={m.get('max_velocity')}, a_max={m.get('max_acceleration')})")
        L.append(f"seed / scene   : {m.get('seed', '?')} / {m.get('scene_seed', '?')}")
    lo, hi = s["success_ci95"]
    L.append("-" * 78)
    L.append(f"episodes           : {s['n_episodes']}")
    L.append(f"SUCCESS RATE       : {s['success_rate']:.3f}  "
             f"({s['n_success']}/{s['n_episodes']}, 95% CI [{lo:.3f}, {hi:.3f}])")
    L.append(f"  reached tol      : {s['reach_rate']:.3f}   (entered the ball at least once)")
    L.append(f"  position only    : {s['pos_only_rate']:.3f}   (d_min < pos_tol, ignoring orientation)")
    L.append(f"  orientation only : {s['ori_only_rate']:.3f}   (theta_min < ori_tol, ignoring position)")
    L.append(f"  timed out        : {s['n_timeout']}")
    L.append("-" * 78)
    L.append(f"{'metric':<38}{'mean':>9}{'median':>9}{'std':>9}{'p95':>9}{'n':>5}")
    for label, key in _ROWS:
        st = s.get(key)
        if not st or st.get("n", 0) == 0:
            continue
        L.append(f"{label:<38}{st['mean']:>9.3f}{st['median']:>9.3f}"
                 f"{st['std']:>9.3f}{st['p95']:>9.3f}{st['n']:>5d}")
    chk = s.get("rate_limit_check")
    if chk:
        L.append("-" * 78)
        L.append(f"CHECK commanded |dq| max : {chk['cmd_vel_max']:.3f} rad/s   "
                 f"{'OK' if chk['vel_within'] else 'EXCEEDS CONFIGURED LIMIT'}")
        L.append(f"CHECK commanded |ddq| max: {chk['cmd_acc_max']:.3f} rad/s^2 "
                 f"{'OK' if chk['acc_within'] else 'EXCEEDS CONFIGURED LIMIT'}")
    L.append("=" * 78)
    return "\n".join(L)


def latex_summary_table(s: dict, s_ablation: dict | None = None,
                        caption: str = "Evaluation of the learned reach-and-orient policy.",
                        label: str = "tab:results") -> str:
    """Emit the paper's results table so the .tex never hand-copies a number."""
    def cell(st, key, fmt="{:.1f}"):
        v = st.get(key)
        if not v or v.get("n", 0) == 0:
            return "--"
        return fmt.format(v["median"]) + r" {\scriptsize$\pm$" + fmt.format(v["std"]) + "}"

    lo, hi = s["success_ci95"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\begin{tabular}{lr}", r"\toprule",
         r"Metric & Value \\", r"\midrule",
         rf"Episodes & {s['n_episodes']} \\",
         rf"Success rate & {s['success_rate']*100:.1f}\% "
         rf"{{\scriptsize[{lo*100:.1f}, {hi*100:.1f}]}} \\",
         rf"Reached tolerance at least once & {s['reach_rate']*100:.1f}\% \\",
         rf"Position criterion only & {s['pos_only_rate']*100:.1f}\% \\",
         rf"Orientation criterion only & {s['ori_only_rate']*100:.1f}\% \\",
         r"\midrule",
         r"Final position error, success [mm] & " + cell(s, "d_final_mm_succ") + r" \\",
         r"Final orientation error, success [deg] & " + cell(s, "theta_final_deg_succ") + r" \\",
         r"Time to first reach [s] & " + cell(s, "t_first_reach_s", "{:.2f}") + r" \\",
         r"Time to settled success [s] & " + cell(s, "t_success_s", "{:.2f}") + r" \\"]
    if s_ablation is not None:
        L += [r"\midrule",
              r"\multicolumn{2}{l}{\emph{Commanded-trajectory smoothness}} \\",
              r"Joint jerk RMS, limiter on [rad/s$^3$] & " + cell(s, "jerk_rms", "{:.0f}") + r" \\",
              r"Joint jerk RMS, limiter off [rad/s$^3$] & " + cell(s_ablation, "jerk_rms", "{:.0f}") + r" \\",
              r"Peak $\|\dot q\|$, limiter on [rad/s] & " + cell(s, "vel_peak", "{:.2f}") + r" \\",
              r"Peak $\|\dot q\|$, limiter off [rad/s] & " + cell(s_ablation, "vel_peak", "{:.2f}") + r" \\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ═════════════════════════════════════════════════════════════════════════
#  Representative-episode selection (for the flagship trajectory figure)
# ═════════════════════════════════════════════════════════════════════════

def select_representative(res: SimpleNamespace) -> int:
    """Index of a typical successful episode: median settle time, median reach distance.

    Deterministic and reportable -- the caller prints the chosen ep_index so the
    figure can be regenerated exactly. Falls back to the closest-approach
    episode if nothing succeeded, so the figure still renders while the policy
    is being tuned.
    """
    succ = res.ep_success.astype(bool)
    if not succ.any():
        return int(np.argmin(res.d_min))
    idx = np.flatnonzero(succ)
    ts = res.t_success[idx]
    rr = res.r3d_target[idx]
    score = (np.abs(ts - np.nanmedian(ts)) / (np.nanstd(ts) + 1e-9)
             + np.abs(rr - np.nanmedian(rr)) / (np.nanstd(rr) + 1e-9))
    return int(idx[np.argmin(score)])
