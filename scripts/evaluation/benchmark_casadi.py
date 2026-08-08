#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Benchmark the CasADi/IPOPT IK reacher against the learned RL policy.

Runs the *same* targets the RL policy was evaluated on -- reconstructed from
results/eval_on.npz, which stores each episode's sampled grasp point in
shoulder-centred polar coordinates -- through scripts/reach_apple_casadi.py's
solver, and reports solve time and kinematic residual.

    PY=/home/pablo_kevin/miniconda3/envs/isaaclab_curobo/bin/python
    $PY scripts/benchmark_casadi.py --results results/eval_on.npz \\
        --out results/casadi_bench.npz

No Isaac Sim and no DDS: this measures the solver, not the closed loop.

WHAT IS AND IS NOT COMPARABLE
-----------------------------
The two systems are measured at different points in their pipelines, and
conflating them would flatter the optimizer:

  * CasADi residual is OPEN-LOOP KINEMATIC error -- "given this target, does a
    joint configuration exist that puts the grasp reference point there, and
    does IPOPT find it". The arm is never simulated. A 0 mm residual means the
    NLP converged, not that the robot got there.

  * RL error is CLOSED-LOOP REALIZED error after 5 s of physics, including
    actuator lag, the rate limiter, and the 0.5 s hold requirement.

So CasADi's residual is a lower bound on what its deployment would achieve,
while the RL number is the achieved result. The honest shared quantities are
(a) per-decision compute time and (b) how often each can satisfy the task
tolerances at all. Both are reported; the summary labels which is which.

A second caveat, reported rather than silently corrected: reach_apple_casadi.py
declares PALM_LOCAL_OFFSET = (0.08, 0, 0) and grasp_offset_z = 0.0, whereas the
trained environment uses (0.14, 0, 0) and 0.0677. This benchmark defaults to the
ENVIRONMENT's values so both systems are scored against one geometry; pass
--use-script-offsets to measure the deployed script's own convention instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import eval_utils as eu  # noqa: E402

# Shoulder (right_shoulder_pitch joint) origin expressed in torso_link, taken
# from reach_apple_casadi._ARM_CHAIN[0] -- the same URDF numbers the FK is built
# from, so the frame conversion below cannot drift from the kinematic model.
SHOULDER_IN_TORSO = np.array([0.0039563, -0.10021, 0.24778])


def _load_casadi_module():
    """Import reach_apple_casadi, whose DDS setup runs at import time.

    That module initialises a DDS publisher inside a try/except at module
    scope. Importing it here is intentional -- reusing its FK chain and solver
    is the whole point, and a re-implementation could silently diverge from the
    thing being benchmarked -- but stdout is noisy and, if a DDS daemon is up, a
    publisher gets created. Nothing is ever published: this script only calls
    CasadiIK.solve() and the FK function.
    """
    import reach_apple_casadi as rac
    return rac


def targets_from_results(res, n: int | None = None, seed: int = 0):
    """Reconstruct grasp-point targets in the torso frame from an eval npz.

    The probe recorded, per episode, the grasp point relative to the shoulder as
    (r_xy, azimuth_deg, dz_target) -- literally the coordinates the environment's
    randomiser samples in. Rebuilding the Cartesian point from those and shifting
    by the fixed shoulder-in-torso offset gives the CasADi solver exactly the
    targets the policy saw.

    Assumes the torso is world-aligned. The environment holds waist_pitch at
    0.1 rad, so this introduces a small constant tilt; it affects both the goal
    and the "up" vector identically for every sample and does not change the
    ranking between methods.
    """
    az = np.radians(res.azimuth_deg)
    xyz_shoulder = np.stack([res.r_xy * np.cos(az),
                             res.r_xy * np.sin(az),
                             res.dz_target], axis=1)
    goals = xyz_shoulder + SHOULDER_IN_TORSO[None, :]
    if n is not None and n < goals.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(goals.shape[0], size=n, replace=False)
        return goals[idx], idx
    return goals, np.arange(goals.shape[0])


def rl_inference_time(checkpoint_dir: str, obs_dim: int = 28, n: int = 2000) -> dict:
    """Time a forward pass of the exported TorchScript policy on CPU and GPU.

    The RL side's per-decision cost is one forward pass; this is the quantity
    comparable to an IPOPT solve. Batch size 1, matching deployment.
    """
    import torch
    path = os.path.join(checkpoint_dir, "exported", "policy.pt")
    if not os.path.exists(path):
        return {"error": f"no exported policy at {path}"}
    out = {}
    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        m = torch.jit.load(path, map_location=dev).eval()
        x = torch.zeros(1, obs_dim, device=dev)
        with torch.inference_mode():
            for _ in range(50):
                m(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                m(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / n
        out[f"{dev}_ms"] = dt * 1e3
    return out


def replay_trajectory(rac, ik, goal, q_start, steps, dt, v_max, a_max, up=(0.0, 0.0, 1.0),
                      stop_cfg=None, offset_z: float = 0.0):
    """Offline replay of the deployed CasADi control loop for one target.

    Reproduces reach_apple_casadi.py's architecture: every tick re-solves the
    IK NLP warm-started from the previous *solution*, and the resulting target
    is slewed by the environment-side rate limiter. The IK converges in a tick
    or two, so what the arm actually follows is the rate-limited ramp toward a
    near-constant solution -- which is precisely the contrast with the policy,
    whose command keeps changing every tick.

    The joint state is taken to equal the rate-limited command, i.e. perfect
    tracking. That is a kinematic replay, not physics: it yields the COMMANDED
    trajectory, which is the quantity comparable to the policy's own commanded
    trajectory (traj_u in the evaluation npz).
    """
    lo, hi = rac.FALLBACK_LOWER, rac.FALLBACK_UPPER
    q_cmd = np.asarray(q_start, dtype=float).copy()
    v = np.zeros(7)
    q_ik = list(q_start)
    q_hist = np.zeros((steps, 7))
    t_hist = np.zeros(steps)

    stop = _make_stop(stop_cfg)
    fruit = np.asarray(goal, dtype=float) + np.array([0.0, 0.0, offset_z])

    for k in range(steps):
        t0 = time.perf_counter()
        q_ik = ik.solve(list(goal), list(up), q_ik, lo, hi)
        t_hist[k] = (time.perf_counter() - t0) * 1e3
        q_des = q_ik
        if stop is not None:
            import torch
            t_now = torch.tensor([k * dt], dtype=torch.float32)
            palm, _ = eu.forward_kinematics(q_cmd, (0.0, 0.0, 0.0))
            d = torch.tensor([float(np.linalg.norm(palm - fruit))], dtype=torch.float32)
            stop.update(t_now, d,
                        torch.tensor(q_cmd, dtype=torch.float32).unsqueeze(0),
                        torch.tensor(v, dtype=torch.float32).unsqueeze(0))
            ov, qc = stop.command(t_now, torch.tensor(q_des, dtype=torch.float32).unsqueeze(0))
            if bool(ov[0]):
                q_des = qc[0].numpy().tolist()
        q_cmd, v = eu.rate_limit_step(q_des, q_cmd, v, dt, v_max, a_max)
        q_hist[k] = q_cmd
    return q_hist, t_hist


def _make_stop(cfg):
    """One-arm BatchQuinticStop, or None when the supervisor is disabled."""
    if cfg is None:
        return None
    from proximity_stop_batch import BatchQuinticStop
    return BatchQuinticStop(1, 7, cfg=cfg, device="cpu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(_REPO, "results", "eval_on.npz"))
    p.add_argument("--out", default=os.path.join(_REPO, "results", "casadi_bench.npz"))
    p.add_argument("--traj-out", default=os.path.join(_REPO, "results", "casadi_traj.npz"),
                   help="Where to write the replayed commanded trajectory.")
    p.add_argument("--traj-episode", type=int, default=None,
                   help="Episode index to replay (default: the representative successful "
                        "episode the RL trajectory figure uses).")
    p.add_argument("--traj-steps", type=int, default=250)
    p.add_argument("--n", type=int, default=None, help="Subsample targets (default: all).")
    p.add_argument("--w_pos", type=float, default=200.0)
    p.add_argument("--w_ori", type=float, default=20.0)
    p.add_argument("--w_reg", type=float, default=0.01)
    p.add_argument("--proximity_stop", type=str, default="on", choices=["on", "off"],
                   help="Apply the quintic proximity-stop supervisor, as deployed.")
    p.add_argument("--stop_dist", type=float, default=0.09)
    p.add_argument("--stop_dwell", type=float, default=0.6)
    p.add_argument("--stop_duration", type=float, default=0.4)
    # reach_apple_casadi.py deploys a 5 s watchdog (its help text explains why:
    # closed-loop IK convergence legitimately takes longer to settle than a
    # policy forward pass), so mirror that rather than run_env.py's 2.5 s.
    p.add_argument("--stop_watchdog", type=float, default=5.0)
    p.add_argument("--use-script-offsets", action="store_true",
                   help="Score with reach_apple_casadi.py's own palm offset / grasp offset "
                        "instead of the trained environment's.")
    args = p.parse_args()

    res = eu.load_results(args.results)
    pos_tol = float(res.pos_tol)
    ori_tol = float(res.ori_tol_rad)
    env_palm_offset = tuple(res.meta["palm_local_offset"])
    env_offset_z = float(res.meta["grasp_offset_z"])

    print("[INFO] importing reach_apple_casadi (its DDS setup runs at import) ...")
    rac = _load_casadi_module()
    script_palm_offset = tuple(rac.PALM_LOCAL_OFFSET)
    script_offset_z = float(rac.GRASP_OFFSET_Z_DEFAULT)

    if args.use_script_offsets:
        palm_offset, offset_z, which = script_palm_offset, script_offset_z, "script"
    else:
        palm_offset, offset_z, which = env_palm_offset, env_offset_z, "environment"
        # Point the module's module-level constant at the environment's value so
        # the FK/IK graph built below scores the same geometry the RL policy did.
        rac.PALM_LOCAL_OFFSET = palm_offset

    print(f"[INFO] scoring with the {which} geometry: "
          f"palm_local_offset={tuple(round(v,4) for v in palm_offset)}, offset_z={offset_z:.4f}")
    if script_palm_offset != tuple(env_palm_offset) or abs(script_offset_z - env_offset_z) > 1e-6:
        print(f"[WARN] reach_apple_casadi.py and the environment disagree on grasp geometry: "
              f"script palm_offset={script_palm_offset}, offset_z={script_offset_z:.4f} vs "
              f"env palm_offset={tuple(round(v,4) for v in env_palm_offset)}, "
              f"offset_z={env_offset_z:.4f}. The deployed script will aim "
              f"{abs(script_offset_z - env_offset_z)*100:.1f} cm from where the reward scores.")

    goals, idx = targets_from_results(res, args.n)
    n = goals.shape[0]
    print(f"[INFO] {n} targets reconstructed from {os.path.basename(args.results)}")

    ik = rac.CasadiIK(w_pos=args.w_pos, w_ori=args.w_ori, w_reg=args.w_reg)
    q_default = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    up = [0.0, 0.0, 1.0]
    lo, hi = rac.FALLBACK_LOWER, rac.FALLBACK_UPPER

    # Two regimes. Cold = every solve from the default pose (worst case, and
    # what the first tick of an episode actually does). Warm = each solve
    # initialised from the previous solution, which is what the 50 Hz loop does
    # on every subsequent tick.
    out: dict[str, np.ndarray] = {}
    for regime in ("cold", "warm"):
        t_ms = np.full(n, np.nan)
        d_err = np.full(n, np.nan)
        th_err = np.full(n, np.nan)
        ok = np.zeros(n, dtype=bool)
        q_prev = list(q_default)
        for i in range(n):
            start = list(q_default) if regime == "cold" else q_prev
            t0 = time.perf_counter()
            q = ik.solve(goals[i].tolist(), up, start, lo, hi)
            t_ms[i] = (time.perf_counter() - t0) * 1e3
            gp, nvec = ik.fk_fn(q)
            gp = np.asarray(gp).flatten()
            nvec = np.asarray(nvec).flatten()
            d_err[i] = float(np.linalg.norm(gp - goals[i]))
            th_err[i] = float(np.arccos(np.clip(nvec[2], -1.0, 1.0)))
            ok[i] = (d_err[i] < pos_tol) and (th_err[i] < ori_tol)
            q_prev = q
            if (i + 1) % 100 == 0:
                print(f"  [{regime}] {i+1}/{n}  median solve "
                      f"{np.nanmedian(t_ms[:i+1]):.1f} ms  "
                      f"median residual {np.nanmedian(d_err[:i+1])*1e3:.1f} mm")
        out[f"{regime}_solve_ms"] = t_ms
        out[f"{regime}_pos_err"] = d_err
        out[f"{regime}_theta_err"] = th_err
        out[f"{regime}_ok"] = ok

    # ── replay one target through the full deployed loop, for the
    #    trajectory-comparison figure ────────────────────────────────────
    ep = args.traj_episode if args.traj_episode is not None else eu.select_representative(res)
    goals_all, _ = targets_from_results(res)
    q_start = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]   # env's default right-arm pose
    dt = float(res.dt)
    v_max = float(res.meta.get("max_velocity", 2.0))
    a_max = float(res.meta.get("max_acceleration", 15.0))
    print(f"[INFO] replaying episode {ep} through the rate-limited loop "
          f"({args.traj_steps} ticks at {1/dt:.0f} Hz)")
    stop_cfg = None
    if args.proximity_stop == "on":
        from proximity_stop_batch import BatchStopConfig
        stop_cfg = BatchStopConfig(stop_dist=args.stop_dist, dwell_time=args.stop_dwell,
                                   stop_duration=args.stop_duration,
                                   watchdog_time=args.stop_watchdog)
    q_traj, t_traj = replay_trajectory(rac, ik, goals_all[ep], q_start,
                                       args.traj_steps, dt, v_max, a_max,
                                       stop_cfg=stop_cfg, offset_z=offset_z)
    ee_traj, _ = eu.forward_kinematics(q_traj, palm_offset)
    np.savez_compressed(
        args.traj_out,
        q=q_traj.astype(np.float32),
        ee=ee_traj.astype(np.float32),
        solve_ms=t_traj.astype(np.float32),
        goal_torso=np.asarray(goals_all[ep], dtype=np.float32),
        meta_json=np.array(json.dumps({
            "episode_index": int(ep), "dt": dt, "max_velocity": v_max,
            "max_acceleration": a_max, "palm_local_offset": list(palm_offset),
            "grasp_offset_z": offset_z, "source_results": args.results,
        }, default=float)),
    )
    print(f"[INFO] wrote {args.traj_out}")
    eu.check_fk_against_casadi()

    rl_t = rl_inference_time(res.meta["run_dir"])
    summary = _summarize(out, res, rl_t, n, pos_tol, ori_tol)
    text = _format(summary, which, palm_offset, offset_z)
    print(text)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {
        "source_results": args.results,
        "n_targets": int(n),
        "geometry": which,
        "palm_local_offset": list(palm_offset),
        "grasp_offset_z": offset_z,
        "script_palm_local_offset": list(script_palm_offset),
        "script_grasp_offset_z": script_offset_z,
        "weights": {"w_pos": args.w_pos, "w_ori": args.w_ori, "w_reg": args.w_reg},
        "pos_tol": pos_tol,
        "ori_tol_rad": ori_tol,
        "rl_inference": rl_t,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out["episode_index"] = idx.astype(np.int32)
    out["goal_torso"] = goals.astype(np.float32)
    np.savez_compressed(args.out, meta_json=np.array(json.dumps(meta, default=float)), **out)
    print(f"[INFO] wrote {args.out}")
    with open(os.path.splitext(args.out)[0] + "_summary.txt", "w") as f:
        f.write(text + "\n")


def _summarize(out, res, rl_t, n, pos_tol, ori_tol) -> dict:
    s = {"n_targets": int(n), "rl_inference": rl_t}
    for regime in ("cold", "warm"):
        s[regime] = {
            "solve_ms_median": float(np.nanmedian(out[f"{regime}_solve_ms"])),
            "solve_ms_mean": float(np.nanmean(out[f"{regime}_solve_ms"])),
            "solve_ms_p95": float(np.nanpercentile(out[f"{regime}_solve_ms"], 95)),
            "solve_ms_max": float(np.nanmax(out[f"{regime}_solve_ms"])),
            "pos_err_mm_median": float(np.nanmedian(out[f"{regime}_pos_err"]) * 1e3),
            # rms, not median: residuals are strongly bimodal (a converged
            # majority near zero, a badly-off tail from targets outside the
            # workspace), and the median sits inside the converged cluster
            # and reports it as if it were the whole population. Squaring
            # before averaging weights the tail the way it actually matters.
            "pos_err_mm_rms": float(np.sqrt(np.nanmean(out[f"{regime}_pos_err"] ** 2)) * 1e3),
            "pos_err_mm_p95": float(np.nanpercentile(out[f"{regime}_pos_err"], 95) * 1e3),
            "theta_deg_median": float(np.degrees(np.nanmedian(out[f"{regime}_theta_err"]))),
            "theta_deg_rms": float(np.degrees(np.sqrt(np.nanmean(out[f"{regime}_theta_err"] ** 2)))),
            "theta_deg_p95": float(np.degrees(np.nanpercentile(out[f"{regime}_theta_err"], 95))),
            "within_tol_rate": float(out[f"{regime}_ok"].mean()),
            "pos_within_rate": float((out[f"{regime}_pos_err"] < pos_tol).mean()),
            "ori_within_rate": float((out[f"{regime}_theta_err"] < ori_tol).mean()),
        }
    rl = eu.summarize(res)
    s["rl"] = {
        "success_rate": rl["success_rate"],
        "pos_only_rate": rl["pos_only_rate"],
        "ori_only_rate": rl["ori_only_rate"],
        "d_final_mm_median": rl["d_final_mm_succ"]["median"],
        "theta_final_deg_median": rl["theta_final_deg_succ"]["median"],
        "t_success_median": rl["t_success_s"]["median"],
    }
    return s


def _format(s, which, palm_offset, offset_z) -> str:
    L = ["=" * 78,
         f"CasADi IK benchmark  --  {s['n_targets']} targets, {which} geometry",
         f"  palm_local_offset={tuple(round(v,4) for v in palm_offset)}  offset_z={offset_z:.4f}",
         "-" * 78,
         f"{'':22}{'cold start':>16}{'warm start':>16}",
         "-" * 78]
    rows = [("solve time median [ms]", "solve_ms_median", "{:.1f}"),
            ("solve time p95 [ms]", "solve_ms_p95", "{:.1f}"),
            ("solve time max [ms]", "solve_ms_max", "{:.1f}"),
            ("pos residual med [mm]", "pos_err_mm_median", "{:.2f}"),
            ("pos residual rms [mm]", "pos_err_mm_rms", "{:.2f}"),
            ("pos residual p95 [mm]", "pos_err_mm_p95", "{:.2f}"),
            ("ori residual med [deg]", "theta_deg_median", "{:.2f}"),
            ("ori residual rms [deg]", "theta_deg_rms", "{:.2f}"),
            ("ori residual p95 [deg]", "theta_deg_p95", "{:.2f}"),
            ("within both tol", "within_tol_rate", "{:.3f}"),
            ("within pos tol", "pos_within_rate", "{:.3f}"),
            ("within ori tol", "ori_within_rate", "{:.3f}")]
    for label, key, fmt in rows:
        L.append(f"{label:22}{fmt.format(s['cold'][key]):>16}{fmt.format(s['warm'][key]):>16}")
    L += ["-" * 78,
          "RL policy (closed-loop, 5 s of physics, includes the 0.5 s hold):",
          f"  success rate                {s['rl']['success_rate']:.3f}",
          f"  position criterion alone    {s['rl']['pos_only_rate']:.3f}",
          f"  orientation criterion alone {s['rl']['ori_only_rate']:.3f}",
          f"  final pos error (succ) [mm] {s['rl']['d_final_mm_median']:.1f}",
          f"  final ori error (succ) [deg]{s['rl']['theta_final_deg_median']:.1f}",
          f"  time to settled success [s] {s['rl']['t_success_median']:.2f}"]
    rl_t = s.get("rl_inference", {})
    if "error" not in rl_t:
        L.append("  forward-pass time [ms]      " +
                 "  ".join(f"{k}={v:.3f}" for k, v in rl_t.items()))
    L += ["-" * 78,
          "NOTE: CasADi residuals are open-loop kinematic error (the arm is never",
          "      simulated); RL errors are closed-loop realised error after 5 s of",
          "      physics. Solve time vs forward-pass time is the directly",
          "      comparable pair; so is 'can this target be satisfied at all'.",
          "=" * 78]
    return "\n".join(L)


if __name__ == "__main__":
    main()
