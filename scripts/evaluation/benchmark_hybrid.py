#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Benchmark the hybrid RL->CasADi reacher (scripts/reach_apple_hybrid.py).

    PY=/home/pablo_kevin/miniconda3/envs/unitree_g1/bin/python
    $PY scripts/evaluation/benchmark_hybrid.py --out results/hybrid_bench.npz

Method. The hybrid's first phase is the trained policy, which cannot be replayed
open-loop -- its observations depend on the simulated state. So instead of
re-simulating it, this replays the hybrid's *switching logic* over the recorded
RL rollouts in results/eval_on.npz, which are real closed-loop physics:

  1. For each episode, walk the recorded trajectory and find the tick where
     reach_apple_hybrid.py would hand off: raw palm-to-fruit distance below
     --switch_dist, or --rl_max_time elapsed, whichever comes first.
  2. Take the arm's measured joint state at that tick as the CasADi phase's
     initial condition, and run the real IPOPT solve from there, through the
     same rate limiter the environment applies.
  3. Score the resulting pose with the same forward kinematics and the same
     tolerances used everywhere else.

Phase 1 is therefore genuine closed-loop data and phase 2 is a kinematic replay
(perfect tracking of the commanded target), the same caveat that applies to
benchmark_casadi.py. Compute cost is accounted honestly: one policy forward pass
per RL tick plus one measured IPOPT solve per CasADi tick.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _HERE)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _REPO)

import eval_utils as eu  # noqa: E402
from benchmark_casadi import (  # noqa: E402
    SHOULDER_IN_TORSO, _load_casadi_module, rl_inference_time, targets_from_results,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(_REPO, "results", "eval_on.npz"))
    p.add_argument("--out", default=os.path.join(_REPO, "results", "hybrid_bench.npz"))
    p.add_argument("--traj-out", default=os.path.join(_REPO, "results", "hybrid_traj.npz"))
    p.add_argument("--switch_dist", type=float, default=0.10,
                   help="Raw palm-to-fruit distance at which the hybrid hands off (m).")
    p.add_argument("--rl_max_time", type=float, default=2.5,
                   help="Unconditional handoff time (s).")
    p.add_argument("--proximity_stop", type=str, default="on", choices=["on", "off"],
                   help="Apply the quintic supervisor during the CasADi phase, as "
                        "reach_apple_hybrid.py does (it arms the stop only after handoff).")
    # Defaults mirror reach_apple_hybrid.py's own, NOT run_env.py's. The
    # watchdog in particular must exceed --rl_max_time: it is driven by the
    # EPISODE clock, so a 2.5 s watchdog with a 2.5 s handoff fires on the very
    # first CasADi tick and freezes the arm before the solver can move it
    # (measured: within-tolerance collapses from 77% to 15%).
    p.add_argument("--stop_dist", type=float, default=0.03)
    p.add_argument("--stop_dwell", type=float, default=0.3)
    p.add_argument("--stop_duration", type=float, default=0.3)
    p.add_argument("--stop_watchdog", type=float, default=5.0)
    p.add_argument("--casadi_steps", type=int, default=100,
                   help="Ticks to run the CasADi phase for after handoff.")
    p.add_argument("--n", type=int, default=None, help="Subsample episodes.")
    p.add_argument("--traj-episode", type=int, default=None,
                   help="Episode whose trajectory is saved for the comparison figure. "
                        "Defaults to the same representative episode plot_results.py uses "
                        "for the other controllers, so all rows show one target.")
    p.add_argument("--w_pos", type=float, default=200.0)
    p.add_argument("--w_ori", type=float, default=20.0)
    p.add_argument("--w_reg", type=float, default=0.01)
    args = p.parse_args()

    res = eu.load_results(args.results)
    pos_tol, ori_tol = float(res.pos_tol), float(res.ori_tol_rad)
    dt = float(res.dt)
    palm_off = tuple(res.meta["palm_local_offset"])
    offset_z = float(res.meta["grasp_offset_z"])
    v_max = float(res.meta.get("max_velocity", 2.0))
    a_max = float(res.meta.get("max_acceleration", 15.0))

    rac = _load_casadi_module()
    rac.PALM_LOCAL_OFFSET = palm_off
    ik = rac.CasadiIK(w_pos=args.w_pos, w_ori=args.w_ori, w_reg=args.w_reg)
    lo, hi = rac.FALLBACK_LOWER, rac.FALLBACK_UPPER

    goals, _ = targets_from_results(res)
    n_all = goals.shape[0]
    traj_ep = args.traj_episode if args.traj_episode is not None \
        else eu.select_representative(res)
    idx = np.arange(n_all) if args.n is None else \
        np.random.default_rng(0).choice(n_all, size=min(args.n, n_all), replace=False)
    n = len(idx)
    print(f"[INFO] replaying the hybrid switch over {n} recorded episodes")

    max_rl_tick = int(round(args.rl_max_time / dt))
    up = [0.0, 0.0, 1.0]

    stop_cfg = None
    if args.proximity_stop == "on":
        from proximity_stop_batch import BatchQuinticStop, BatchStopConfig
        stop_cfg = BatchStopConfig(stop_dist=args.stop_dist, dwell_time=args.stop_dwell,
                                   stop_duration=args.stop_duration,
                                   watchdog_time=args.stop_watchdog)
    stopped_at = np.full(n, np.nan)

    hand_tick = np.zeros(n, dtype=np.int32)
    hand_reason = np.zeros(n, dtype=np.int8)          # 0 = proximity, 1 = timeout
    d_final = np.full(n, np.nan)
    th_final = np.full(n, np.nan)
    t_total = np.full(n, np.nan)
    n_cas = np.zeros(n, dtype=np.int32)
    solve_ms = np.full(n, np.nan)
    peak_v = np.full(n, np.nan)
    peak_a = np.full(n, np.nan)
    jerk_rms = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    traj_store = None

    for j, e in enumerate(idx):
        L = int(res.traj_mask[e].sum())
        q_meas = res.traj_q[e, :L]
        # Raw palm-link-origin to fruit distance -- the quantity the hybrid's
        # switch and the proximity stop both use (NOT the grasp-point metric the
        # reward scores, which already includes the palm offset).
        palm, _ = eu.forward_kinematics(q_meas, (0.0, 0.0, 0.0))
        fruit = goals[e] + np.array([0.0, 0.0, offset_z])
        raw_d = np.linalg.norm(palm - fruit[None, :], axis=1)

        near = np.flatnonzero(raw_d < args.switch_dist)
        if near.size:
            k = int(near[0]); hand_reason[j] = 0
        else:
            k = min(max_rl_tick, L - 1); hand_reason[j] = 1
        hand_tick[j] = k

        # CasADi phase, continuing from the arm's state at handoff.
        #
        # The rate limiter lives in the ENVIRONMENT's action term, not in the
        # client, so it does not reset when the client switches phase: both its
        # position and velocity buffers carry straight through. Seeding from the
        # last COMMANDED target (traj_u), not the measured joint position, is
        # what reproduces that -- the limiter integrates its own command, and the
        # two differ by the actuator's tracking error. Seeding from the measured
        # position instead injects that tracking error as a step discontinuity
        # and reports an acceleration spike the real system does not have.
        if k >= 1:
            q_cmd = res.traj_u[e, k].astype(float).copy()
            v = np.clip((res.traj_u[e, k].astype(float)
                         - res.traj_u[e, k - 1].astype(float)) / dt, -v_max, v_max)
        else:
            q_cmd = q_meas[k].astype(float).copy()
            v = np.zeros(7)
        q_ik = list(q_cmd)
        hist = np.zeros((args.casadi_steps, 7))
        tsum = 0.0
        stop_fired_t = np.nan
        # reach_apple_hybrid.py arms the supervisor only once the CasADi phase
        # begins, and drives it with the EPISODE clock (t_ep), not a phase-local
        # one -- so the watchdog is measured from episode start.
        stop = BatchQuinticStop(1, 7, cfg=stop_cfg, device="cpu") if stop_cfg else None
        fruit = goals[e] + np.array([0.0, 0.0, offset_z])
        import torch
        for s in range(args.casadi_steps):
            t0 = time.perf_counter()
            q_ik = ik.solve(goals[e].tolist(), up, q_ik, lo, hi)
            tsum += (time.perf_counter() - t0) * 1e3
            q_des = q_ik
            if stop is not None:
                t_now = torch.tensor([(k + s) * dt], dtype=torch.float32)
                palm, _ = eu.forward_kinematics(q_cmd, (0.0, 0.0, 0.0))
                d = torch.tensor([float(np.linalg.norm(palm - fruit))], dtype=torch.float32)
                stop.update(t_now, d,
                            torch.tensor(q_cmd, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(v, dtype=torch.float32).unsqueeze(0))
                ov, qc = stop.command(t_now, torch.tensor(q_des, dtype=torch.float32).unsqueeze(0))
                if bool(ov[0]):
                    q_des = qc[0].numpy().tolist()
                    if np.isnan(stop_fired_t):
                        stop_fired_t = float((k + s) * dt)
            q_cmd, v = eu.rate_limit_step(q_des, q_cmd, v, dt, v_max, a_max)
            hist[s] = q_cmd
        n_cas[j] = args.casadi_steps
        solve_ms[j] = tsum / args.casadi_steps
        stopped_at[j] = stop_fired_t

        gp, R = eu.forward_kinematics(hist[-1], palm_off)
        nvec = R @ np.asarray(rac.PALM_NORMAL_AXIS, dtype=float)
        d_final[j] = float(np.linalg.norm(gp - goals[e]))
        th_final[j] = float(np.arccos(np.clip(nvec[2], -1.0, 1.0)))
        ok[j] = (d_final[j] < pos_tol) and (th_final[j] < ori_tol)

        # Full commanded trajectory: recorded RL command up to handoff, then the
        # CasADi command. Derivatives are taken over the joined sequence so the
        # handoff discontinuity, if any, shows up rather than being hidden.
        full = np.concatenate([res.traj_u[e, :k + 1].astype(float), hist], axis=0)
        vel = np.gradient(full, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jrk = np.gradient(acc, dt, axis=0)
        sl = slice(2, -2)
        peak_v[j] = float(np.max(np.linalg.norm(vel[sl], axis=1)))
        peak_a[j] = float(np.max(np.linalg.norm(acc[sl], axis=1)))
        jerk_rms[j] = float(np.sqrt(np.mean(np.linalg.norm(jrk[sl], axis=1) ** 2)))
        t_total[j] = (k + args.casadi_steps) * dt

        if int(e) == int(traj_ep):
            traj_store = {"q": full.astype(np.float32),
                          "handoff_tick": int(k),
                          "episode_index": int(e)}
        if (j + 1) % 100 == 0:
            print(f"  {j+1}/{n}  handoff median tick {np.median(hand_tick[:j+1]):.0f}  "
                  f"within tol {ok[:j+1].mean():.3f}")

    rl_ms = rl_inference_time(res.meta["run_dir"]).get("cpu_ms", float("nan"))
    compute_ms = hand_tick * rl_ms + n_cas * solve_ms

    summary = {
        "n_episodes": int(n),
        "switch_dist": args.switch_dist,
        "rl_max_time": args.rl_max_time,
        "handoff_time_s_median": float(np.median(hand_tick) * dt),
        "handoff_by_proximity": float((hand_reason == 0).mean()),
        "within_tol_rate": float(ok.mean()),
        "pos_within_rate": float((d_final < pos_tol).mean()),
        "ori_within_rate": float((th_final < ori_tol).mean()),
        "d_final_mm_median": float(np.median(d_final) * 1e3),
        # rms, not median -- see benchmark_casadi.py's _summarize: this
        # population mixes solved and unreachable targets, and the median
        # sits inside the solved cluster and understates the tail.
        "d_final_mm_rms": float(np.sqrt(np.mean(d_final ** 2)) * 1e3),
        "d_final_mm_p95": float(np.percentile(d_final, 95) * 1e3),
        "theta_deg_median": float(np.degrees(np.median(th_final))),
        "theta_deg_rms": float(np.degrees(np.sqrt(np.mean(th_final ** 2)))),
        "t_total_s_median": float(np.median(t_total)),
        "peak_vel_median": float(np.median(peak_v)),
        "peak_acc_median": float(np.median(peak_a)),
        "jerk_rms_median": float(np.median(jerk_rms)),
        "compute_ms_per_episode_median": float(np.median(compute_ms)),
        "rl_forward_ms": rl_ms,
        "casadi_solve_ms_median": float(np.median(solve_ms)),
        "proximity_stop": args.proximity_stop,
        "stop_engaged_rate": float(np.isfinite(stopped_at).mean()),
        "stop_time_s_median": float(np.nanmedian(stopped_at)) if np.isfinite(stopped_at).any() else float("nan"),
    }
    print(_format(summary))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, handoff_tick=hand_tick, handoff_reason=hand_reason,
        d_final=d_final.astype(np.float32), theta_final=th_final.astype(np.float32),
        t_total=t_total.astype(np.float32), ok=ok, peak_vel=peak_v.astype(np.float32),
        peak_acc=peak_a.astype(np.float32), jerk_rms=jerk_rms.astype(np.float32),
        compute_ms=compute_ms.astype(np.float32), episode_index=idx.astype(np.int32),
        meta_json=np.array(json.dumps({"summary": summary, "dt": dt,
                                       "palm_local_offset": list(palm_off),
                                       "grasp_offset_z": offset_z,
                                       "source_results": args.results}, default=float)),
    )
    print(f"[INFO] wrote {args.out}")

    if traj_store is not None:
        ee, _ = eu.forward_kinematics(traj_store["q"], palm_off)
        np.savez_compressed(
            args.traj_out, q=traj_store["q"], ee=ee.astype(np.float32),
            meta_json=np.array(json.dumps({
                "episode_index": traj_store["episode_index"],
                "handoff_tick": traj_store["handoff_tick"], "dt": dt,
                "max_velocity": v_max, "max_acceleration": a_max,
                "palm_local_offset": list(palm_off)}, default=float)))
        print(f"[INFO] wrote {args.traj_out} (episode {traj_store['episode_index']}, "
              f"handoff at tick {traj_store['handoff_tick']})")

    with open(os.path.splitext(args.out)[0] + "_summary.txt", "w") as f:
        f.write(_format(summary) + "\n")


def _format(s: dict) -> str:
    L = ["=" * 70, f"Hybrid RL->CasADi reacher  --  {s['n_episodes']} episodes",
         f"  switch at {s['switch_dist']*100:.0f} cm or {s['rl_max_time']:.1f} s", "-" * 70,
         f"handoff time, median [s]        {s['handoff_time_s_median']:.2f}",
         f"handoff triggered by proximity  {s['handoff_by_proximity']:.3f}",
         f"within both tolerances          {s['within_tol_rate']:.3f}",
         f"  position only                 {s['pos_within_rate']:.3f}",
         f"  orientation only              {s['ori_within_rate']:.3f}",
         f"final pos error, median [mm]    {s['d_final_mm_median']:.2f}",
         f"final pos error, rms [mm]       {s['d_final_mm_rms']:.2f}",
         f"final pos error, p95 [mm]       {s['d_final_mm_p95']:.1f}",
         f"final ori error, median [deg]   {s['theta_deg_median']:.2f}",
         f"final ori error, rms [deg]      {s['theta_deg_rms']:.2f}",
         f"total time, median [s]          {s['t_total_s_median']:.2f}",
         f"peak |dq| median [rad/s]        {s['peak_vel_median']:.2f}",
         f"peak |ddq| median [rad/s^2]     {s['peak_acc_median']:.1f}",
         f"jerk RMS median [rad/s^3]       {s['jerk_rms_median']:.0f}",
         "-" * 70,
         f"compute per episode, median [ms] {s['compute_ms_per_episode_median']:.1f}",
         f"  RL forward pass [ms]           {s['rl_forward_ms']:.3f}",
         f"  CasADi solve [ms]              {s['casadi_solve_ms_median']:.2f}",
         "=" * 70]
    return "\n".join(L)


if __name__ == "__main__":
    main()
