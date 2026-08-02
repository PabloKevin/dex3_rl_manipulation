# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Generate the paper's figures and its results macros from evaluation .npz files.

Plain python + matplotlib: no isaaclab, no Isaac Sim boot, no pandas/seaborn.

    PY=/home/pablo_kevin/miniconda3/envs/isaaclab_curobo/bin/python
    $PY scripts/plot_results.py \\
        --results results/eval_on.npz --results-ablation results/eval_off.npz \\
        --outdir paper/figures --emit-macros paper/results_macros.tex

Figures (names match \\includegraphics in paper/main.tex):
    fig1_training.pdf        training curves, reported run vs. negative control
    fig_eval.pdf             d(t), theta(t), workspace coverage
    fig3_smooth_control.pdf  joint q/dq + rate-limiter jerk ablation
    fig_geometry.pdf         task-geometry schematic (panel d of Fig. 1)

Charting rules enforced here, for print legibility:
  * No dual y-axes anywhere -- differing units always get separate panels.
  * Every two-series comparison pairs colour with linestyle, and every
    two-class scatter pairs colour with marker shape, so identity survives
    greyscale printing.
  * Threshold rules are neutral gray + dashed; gridlines are solid hairlines.
    Dashing therefore always carries meaning.
  * Trace aggregates use masked percentiles so zero padding after an episode
    terminates is never averaged in as data.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_utils as eu
import plot_style as ps

# Measured rigid-chain length shoulder->elbow->wrist->palm (probe_arm_reach.py),
# plus PALM_LOCAL_OFFSET out to the grasp reference point. Drawn on the coverage
# panel because the sampler's r_max exceeds it.
ARM_REACH_M = 0.42 + 0.14

TB_TAGS = [
    "Train/mean_reward",
    "Episode_Termination/reach_success",
    "Episode_Reward/hold",
    "Episode_Reward/palm_up",
    "Episode_Reward/reach_fine",
    "Policy/mean_noise_std",
]


# ═════════════════════════════════════════════════════════════════════════
#  TensorBoard
# ═════════════════════════════════════════════════════════════════════════

def read_tb_scalars(run_dir: str, tags=TB_TAGS) -> dict:
    """Read scalars from a run directory's tfevents file.

    size_guidance={"scalars": 0} is REQUIRED: the default reservoir-samples
    scalars down to 1000 points, and these runs have 1700-1900, so the default
    silently subsamples -- and subsamples two runs onto different step grids,
    which would misalign any overlay.

    Tags ending in "/time" are excluded: their `step` field is wall-clock
    seconds, not the training iteration.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"no tfevents file in {run_dir}")
    if len(files) > 1:
        print(f"[warn] {len(files)} tfevents files in {run_dir}; EventAccumulator will merge them")

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])
    out = {}
    for t in tags:
        if t.endswith("/time"):
            continue
        if t not in available:
            print(f"[warn] tag {t!r} missing from {os.path.basename(run_dir)}")
            continue
        s = ea.Scalars(t)
        out[t] = (np.array([e.step for e in s], dtype=float),
                  np.array([e.value for e in s], dtype=float))
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Fig. 1 (paper Fig. 2) -- training curves
# ═════════════════════════════════════════════════════════════════════════

def fig_training(main_dir: str, control_dir: str | None, outdir: str) -> str:
    main = read_tb_scalars(main_dir)
    ctrl = read_tb_scalars(control_dir) if control_dir else {}

    fig, axes = ps.figure("single", height=3.0, nrows=2, ncols=1, sharex=True)
    ax_r, ax_s = axes

    series = [("reported policy", main, ps.C_BLUE, ps.LS_MAIN)]
    if ctrl:
        series.append(("mis-specified $\\delta_z$", ctrl, ps.C_ORANGE, ps.LS_ALT))

    for label, d, colour, ls in series:
        if "Train/mean_reward" in d:
            x, y = d["Train/mean_reward"]
            ps.plot_smoothed(ax_r, x, y, colour, label=label, linestyle=ls)
            ps.label_end(ax_r, x[-1], ps.rolling_mean(y)[-1], f"{y[-20:].mean():.0f}")
        if "Episode_Termination/reach_success" in d:
            x, y = d["Episode_Termination/reach_success"]
            ps.plot_smoothed(ax_s, x, 100.0 * y, colour, linestyle=ls)
            ps.label_end(ax_s, x[-1], 100.0 * ps.rolling_mean(100.0 * y)[-1] / 100.0,
                         f"{100*y[-20:].mean():.0f}%")

    ax_r.set_ylabel("mean return")
    ax_s.set_ylabel("success rate [%]")
    ax_s.set_xlabel("training iteration")
    ax_s.set_ylim(bottom=-1)
    ax_r.legend(loc="center right", ncol=1)
    for a, tag in zip(axes, ("(a)", "(b)")):
        ps.hairline_grid(a)
        ps.panel_tag(a, tag, x=-0.14)
    return ps.finish(fig, os.path.join(outdir, "fig1_training.pdf"))


# ═════════════════════════════════════════════════════════════════════════
#  Fig. 2 (paper Fig. 3) -- error traces + workspace coverage
# ═════════════════════════════════════════════════════════════════════════

def _band(ax, t, pct, colour, label=None):
    p10, p25, p50, p75, p90 = pct
    ax.fill_between(t, p10, p90, color=colour, alpha=0.07, linewidth=0)
    ax.fill_between(t, p25, p75, color=colour, alpha=0.14, linewidth=0)
    ax.plot(t, p50, color=colour, linewidth=1.1, label=label)


def fig_eval(res, outdir: str) -> str:
    fig, axes = ps.figure("double", height=2.35, nrows=1, ncols=3)
    ax_d, ax_th, ax_w = axes

    mask = res.traj_mask
    t = np.arange(mask.shape[1]) * res.dt
    n_live = mask.sum(axis=0)
    # Truncate where too few episodes remain live: successful episodes
    # terminate early, so the far tail is a handful of failures and the
    # percentile band there says nothing about typical behaviour.
    keep = n_live >= max(1, int(0.2 * res.n_episodes))
    tk = t[keep]

    pct_d, _ = eu.masked_percentile(res.traj_d * 1e3, mask, [10, 25, 50, 75, 90])
    _band(ax_d, tk, pct_d[:, keep], ps.C_BLUE)
    ps.threshold(ax_d, float(res.pos_tol) * 1e3, "h", "tolerance")
    ax_d.set_ylabel("position error [mm]")
    ax_d.set_xlabel("time [s]")

    pct_th, _ = eu.masked_percentile(np.degrees(res.traj_theta), mask, [10, 25, 50, 75, 90])
    _band(ax_th, tk, pct_th[:, keep], ps.C_BLUE)
    # label on the left: the orientation curve rises back toward the rule on the
    # right, so a right-aligned label would sit on the data
    ps.threshold(ax_th, np.degrees(float(res.ori_tol_rad)), "h", "tolerance", label_x=0.36)
    ax_th.set_ylabel("orientation error [deg]")
    ax_th.set_xlabel("time [s]")

    succ = res.ep_success.astype(bool)
    err_mm = res.d_final * 1e3
    vmax = float(np.percentile(err_mm, 95))
    # y is the 3-D shoulder-to-grasp-point radius: that is the quantity the
    # sampler constrains to [reach_min, reach_max], so the r_max rule below is
    # drawn in the same coordinate it actually bounds.
    common = dict(cmap=ps.SEQ_BLUE, vmin=0.0, vmax=vmax, s=9, linewidths=0.4)
    ax_w.scatter(res.azimuth_deg[~succ], res.r3d_target[~succ], c=err_mm[~succ],
                 marker="x", **common)
    sc = ax_w.scatter(res.azimuth_deg[succ], res.r3d_target[succ], c=err_mm[succ],
                      marker="o", edgecolors="white", **common)
    m = res.meta
    ps.threshold(ax_w, m["reach_max"], "h", "sampler $r_{\\max}$", halo=True)
    ps.threshold(ax_w, ARM_REACH_M, "h", "measured arm reach", label_x=0.62, halo=True)
    ax_w.set_xlabel("fruit azimuth [deg]")
    ax_w.set_ylabel("shoulder radius $r_{3D}$ [m]")
    # Neutral legend markers: shape carries the class, not colour (colour is
    # already spent on the sequential error encoding).
    from matplotlib.lines import Line2D
    # opaque background: this legend sits over dense scatter, and legibility
    # beats the frameless default here
    ax_w.legend(handles=[Line2D([], [], ls="", marker="o", mfc="none", mec=ps.INK_2,
                                ms=3.2, mew=0.7, label="success"),
                         Line2D([], [], ls="", marker="x", color=ps.INK_2,
                                ms=3.2, mew=0.7, label="failure")],
                loc="lower left", ncol=2, fontsize=6, frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.88,
                borderpad=0.3).set_zorder(10)
    cb = fig.colorbar(sc, ax=ax_w, orientation="vertical", fraction=0.046, pad=0.03)
    cb.set_label("final pos. error [mm]", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)

    for a, tag in zip(axes, ("(a)", "(b)", "(c)")):
        ps.hairline_grid(a)
        ps.panel_tag(a, tag, x=-0.22)
    return ps.finish(fig, os.path.join(outdir, "fig_eval.pdf"))


# ═════════════════════════════════════════════════════════════════════════
#  Fig. 3 (paper Fig. 4) -- joint trajectories + rate-limiter ablation
# ═════════════════════════════════════════════════════════════════════════

def fig_smooth(res, res_abl, outdir: str) -> tuple[str, int]:
    idx = eu.select_representative(res)
    n = int(res.traj_mask[idx].sum())
    t = np.arange(n) * res.dt
    names = [str(s) for s in res.joint_names]

    fig, axes = ps.figure("double", height=3.5, nrows=2, ncols=2)
    (ax_q, ax_dq), (ax_v, ax_j) = axes

    for k in range(res.traj_q.shape[2]):
        c = ps.JOINT_RAMP[k % len(ps.JOINT_RAMP)]
        ax_q.plot(t, res.traj_q[idx, :n, k], color=c, linewidth=1.0)
        ax_dq.plot(t, res.traj_dq[idx, :n, k], color=c, linewidth=1.0)
    ax_q.set_ylabel("joint position [rad]")
    ax_dq.set_ylabel("joint velocity [rad/s]")
    ps.joint_key(ax_q, names, loc="best", ncol=2)

    v_max = float(res.meta.get("max_velocity", 2.0))
    ps.threshold(ax_dq, v_max, "h", "$v_{\\max}$")
    ps.threshold(ax_dq, -v_max, "h")

    # (c,d) aggregate to the 2-norm over joints: 14 lines would be unreadable,
    # and the claim is about overall motion magnitude, not any single joint.
    def agg(r, fn):
        """Median over episodes, with the finite-difference stencil edges dropped.

        eu.valid_window trims the leading reset transient and the trailing
        samples whose stencil would reach into the zero padding; without it the
        aggregate shows spurious spikes at t=0 and at the horizon.
        """
        m = eu.valid_window(r.traj_mask, skip_first=eu.SKIP_FIRST, skip_last=3)
        out, _ = eu.masked_percentile(fn(r), m, [50])
        return np.arange(m.shape[1]) * r.dt, out[0], m.sum(axis=0)

    pairs = [("limiter on", res, ps.C_BLUE, ps.LS_MAIN)]
    if res_abl is not None:
        pairs.append(("limiter off", res_abl, ps.C_ORANGE, ps.LS_ALT))

    for label, r, colour, ls in pairs:
        tt, vv, nl = agg(r, lambda x: np.linalg.norm(x.traj_dq, axis=-1))
        k = nl >= max(1, int(0.2 * r.n_episodes))
        ax_v.plot(tt[k], vv[k], color=colour, linestyle=ls, label=label, linewidth=1.1)

        tt, jj, nl = agg(r, lambda x: np.linalg.norm(
            np.nan_to_num(eu.joint_jerk(x.traj_dq, x.dt)), axis=-1))
        k = nl >= max(1, int(0.2 * r.n_episodes))
        ax_j.plot(tt[k], np.maximum(jj[k], 1e-1), color=colour, linestyle=ls,
                  label=label, linewidth=1.1)

    ax_v.set_ylabel(r"$\|\dot q\|$ [rad/s]")
    ax_j.set_ylabel(r"$\|j\|$ [rad/s$^3$]")
    ax_j.set_yscale("log")
    ps.threshold(ax_j, eu.command_jerk_ceiling(float(res.meta.get("max_acceleration", 15.0)), res.dt),
                 "h", "commanded ceiling")
    ax_v.legend(loc="upper right", ncol=1)

    for a in (ax_v, ax_j):
        a.set_xlabel("time [s]")
    for a, tag in zip((ax_q, ax_dq, ax_v, ax_j), ("(a)", "(b)", "(c)", "(d)")):
        ps.hairline_grid(a)
        ps.panel_tag(a, tag, x=-0.15)
    return ps.finish(fig, os.path.join(outdir, "fig3_smooth_control.pdf")), idx


# ═════════════════════════════════════════════════════════════════════════
#  Geometry schematic (paper Fig. 1d)
# ═════════════════════════════════════════════════════════════════════════

def fig_geometry(res, outdir: str) -> str:
    """Side-view schematic of the grasp point, palm reference point and theta.

    Schematic, not to scale: the palm is drawn well below and left of the grasp
    point so that d is a legible diagonal and theta has room for an arc. The
    palm is deliberately tilted off vertical, since theta = 0 (the target) would
    make the angle invisible.
    """
    import matplotlib.patches as mpatches

    fig, ax = ps.figure("single", height=2.15)
    dz = float(res.meta.get("grasp_offset_z", 0.09))
    r_apple, tilt = 0.042, np.deg2rad(24.0)

    # fruit + grasp point, with delta_z between them
    ax.add_patch(mpatches.Circle((0, 0), r_apple, facecolor="#e34948",
                                 edgecolor="none", alpha=0.9, zorder=3))
    ax.annotate("fruit", xy=(0, 0), xytext=(0.055, 0.020), fontsize=6.5, color=ps.INK_2)
    ax.plot([0], [-dz], marker="o", color=ps.INK, ms=3.6, zorder=5)
    ax.annotate("grasp point", xy=(0, -dz), xytext=(0.030, -dz + 0.004),
                fontsize=6.5, color=ps.INK_2, ha="left", va="bottom")
    ax.annotate("", xy=(0, 0), xytext=(0, -dz),
                arrowprops=dict(arrowstyle="<->", color=ps.INK_MUTED, lw=0.7,
                                shrinkA=0, shrinkB=0))
    # placed clear of the fruit disc (radius r_apple) so it never sits on red
    ax.annotate(r"$\delta_z$", xy=(r_apple + 0.012, -dz / 2), fontsize=7.5,
                color=ps.INK_2, ha="left", va="center")

    # palm, tilted, with its link origin and the grasp reference point
    px, py, L = -0.195, -dz - 0.085, 0.115
    ux, uy = np.cos(tilt), np.sin(tilt)
    gx, gy = px + L * ux, py + L * uy
    ax.plot([px, gx], [py, gy], color=ps.C_BLUE, lw=2.4,
            solid_capstyle="round", zorder=3)
    ax.plot([px], [py], marker="o", color=ps.C_BLUE, ms=3.0, zorder=5)
    ax.plot([gx], [gy], marker="o", color=ps.C_BLUE, ms=3.6, zorder=5)
    ax.annotate("palm link origin", xy=(px, py), xytext=(px - 0.012, py - 0.030),
                fontsize=6.5, color=ps.INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=ps.INK_MUTED, lw=0.5))
    ax.annotate("grasp reference point", xy=(gx, gy), xytext=(gx + 0.020, gy - 0.036),
                fontsize=6.5, color=ps.INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=ps.INK_MUTED, lw=0.5))

    # d
    ax.annotate("", xy=(0, -dz), xytext=(gx, gy),
                arrowprops=dict(arrowstyle="<->", color=ps.INK, lw=0.9,
                                shrinkA=2, shrinkB=2))
    ax.annotate("$d$", xy=((gx + 0.0) / 2 - 0.004, (gy - dz) / 2 + 0.008),
                fontsize=8, color=ps.INK, ha="right", va="bottom")

    # palm normal (perpendicular to the palm) vs world +z, with the theta arc
    nx, ny = -np.sin(tilt), np.cos(tilt)
    ax.annotate("", xy=(gx + 0.075 * nx, gy + 0.075 * ny), xytext=(gx, gy),
                arrowprops=dict(arrowstyle="->", color=ps.C_AQUA, lw=1.2))
    ax.annotate("palm normal", xy=(gx + 0.075 * nx, gy + 0.075 * ny),
                xytext=(gx + 0.075 * nx - 0.006, gy + 0.075 * ny + 0.008),
                fontsize=6.5, color=ps.INK_2, ha="center")
    ax.annotate("", xy=(gx, gy + 0.078), xytext=(gx, gy),
                arrowprops=dict(arrowstyle="->", color=ps.INK_MUTED, lw=0.8,
                                linestyle="dashed"))
    ax.annotate("world $+z$", xy=(gx, gy + 0.078), xytext=(gx + 0.007, gy + 0.080),
                fontsize=6.5, color=ps.INK_2, ha="left", va="bottom")
    ax.add_patch(mpatches.Arc((gx, gy), 0.085, 0.085, angle=0,
                              theta1=90.0, theta2=90.0 + np.rad2deg(tilt),
                              color=ps.INK, lw=0.9, zorder=4))
    ax.annotate(r"$\theta$", xy=(gx - 0.016, gy + 0.050), fontsize=8.5,
                color=ps.INK, ha="center", va="center")

    ax.set_xlim(-0.20, 0.17)
    ax.set_ylim(-dz - 0.135, 0.075)
    ax.set_aspect("equal")
    ax.axis("off")
    return ps.finish(fig, os.path.join(outdir, "fig_geometry.pdf"))


# ═════════════════════════════════════════════════════════════════════════
#  Macros
# ═════════════════════════════════════════════════════════════════════════

def emit_macros(path: str, s: dict, s_abl: dict | None, res) -> str:
    def md(st, fmt="{:.1f}"):
        return fmt.format(st["median"]) if st and st.get("n", 0) else r"\PLACEHOLDER"

    lo, hi = s["success_ci95"]
    chk = s.get("rate_limit_check", {})
    jerk_on = s.get("jerk_rms", {}).get("median")
    jerk_off = (s_abl or {}).get("jerk_rms", {}).get("median")
    ratio = (f"{jerk_off / jerk_on:.1f}$\\times$"
             if jerk_on and jerk_off and jerk_on > 0 else r"\PLACEHOLDER")

    L = [
        "% AUTO-GENERATED by scripts/plot_results.py --emit-macros. Do not edit by hand.",
        f"% source: {os.path.basename(res.path)}  checkpoint iter "
        f"{res.meta.get('checkpoint_iteration','?')}  ({res.meta.get('timestamp','?')})",
        r"\newcommand{\PLACEHOLDER}{\textcolor{red}{\textbf{??}}}",
        "",
        f"\\newcommand{{\\RESnepisodes}}{{{s['n_episodes']}}}",
        f"\\newcommand{{\\RESnepisodesAbl}}{{{(s_abl or {}).get('n_episodes', '??')}}}",
        f"\\newcommand{{\\RESckpt}}{{{res.meta.get('checkpoint_iteration','?')}}}",
        f"\\newcommand{{\\RESfracintol}}{{{100*s['frac_in_tol_after']['median']:.0f}\\%}}",
        f"\\newcommand{{\\RESablsuccess}}{{{100*(s_abl or {}).get('success_rate', float('nan')):.1f}\\%}}",
        "",
        f"\\newcommand{{\\RESsuccess}}{{{100*s['success_rate']:.1f}\\%}}",
        f"\\newcommand{{\\RESsuccessCI}}{{[{100*lo:.1f}, {100*hi:.1f}]}}",
        f"\\newcommand{{\\RESreachrate}}{{{100*s['reach_rate']:.1f}\\%}}",
        f"\\newcommand{{\\RESposonly}}{{{100*s['pos_only_rate']:.1f}\\%}}",
        f"\\newcommand{{\\RESorionly}}{{{100*s['ori_only_rate']:.1f}\\%}}",
        "",
        f"\\newcommand{{\\RESdfinal}}{{{md(s.get('d_final_mm_succ'))}}}",
        f"\\newcommand{{\\RESthetafinal}}{{{md(s.get('theta_final_deg_succ'))}}}",
        f"\\newcommand{{\\REStreach}}{{{md(s.get('t_first_reach_s'), '{:.2f}')}}}",
        f"\\newcommand{{\\REStsuccess}}{{{md(s.get('t_success_s'), '{:.2f}')}}}",
        "",
        f"\\newcommand{{\\RESjerkOn}}{{{md(s.get('jerk_rms'), '{:.0f}')}}}",
        f"\\newcommand{{\\RESjerkOff}}{{{md((s_abl or {}).get('jerk_rms'), '{:.0f}')}}}",
        f"\\newcommand{{\\RESjerkRatio}}{{{ratio}}}",
        f"\\newcommand{{\\RESvelOn}}{{{md(s.get('vel_peak'), '{:.2f}')}}}",
        f"\\newcommand{{\\RESvelOff}}{{{md((s_abl or {}).get('vel_peak'), '{:.2f}')}}}",
        f"\\newcommand{{\\REScmdvelOn}}{{{chk.get('cmd_vel_max', float('nan')):.2f}\\,rad/s}}",
        f"\\newcommand{{\\REScmdaccOn}}{{{chk.get('cmd_acc_max', float('nan')):.1f}\\,rad/s$^2$}}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(repo, "results", "eval_on.npz"))
    p.add_argument("--results-ablation", default=os.path.join(repo, "results", "eval_off.npz"))
    p.add_argument("--logdir", default=os.path.join(repo, "logs", "rsl_rl", "apple_grasp_reach"))
    p.add_argument("--run-main", default=None, help="Run dir name; default = newest.")
    p.add_argument("--run-control", default="2026-07-31_05-42-13")
    p.add_argument("--outdir", default=os.path.join(repo, "paper", "figures"))
    p.add_argument("--emit-macros", default=os.path.join(repo, "paper", "results_macros.tex"))
    p.add_argument("--only", default=None, help="training | eval | smooth | geometry")
    args = p.parse_args()

    ps.apply_ieee_style()
    os.makedirs(args.outdir, exist_ok=True)

    res = eu.load_results(args.results)
    res_abl = (eu.load_results(args.results_ablation)
               if args.results_ablation and os.path.exists(args.results_ablation) else None)
    if res_abl is None:
        print(f"[warn] no ablation results at {args.results_ablation}; "
              "the limiter comparison will be omitted")

    run_main = args.run_main or os.path.basename(res.meta.get("run_dir", ""))
    main_dir = os.path.join(args.logdir, run_main)
    ctrl_dir = os.path.join(args.logdir, args.run_control)
    if not os.path.isdir(ctrl_dir):
        print(f"[warn] control run {ctrl_dir} not found; plotting the main run alone")
        ctrl_dir = None

    only = args.only
    made = []
    if only in (None, "training"):
        made.append(fig_training(main_dir, ctrl_dir, args.outdir))
    if only in (None, "eval"):
        made.append(fig_eval(res, args.outdir))
    if only in (None, "smooth"):
        path, idx = fig_smooth(res, res_abl, args.outdir)
        made.append(path)
        print(f"[info] representative episode index {idx} "
              f"(success={bool(res.ep_success[idx])}, "
              f"t_success={res.t_success[idx]:.2f}s, r3d={res.r3d_target[idx]:.3f}m)")
    if only in (None, "geometry"):
        made.append(fig_geometry(res, args.outdir))

    for f in made:
        print(f"[info] wrote {f}")

    if args.emit_macros and only is None:
        s = eu.summarize(res)
        s_abl = eu.summarize(res_abl) if res_abl is not None else None
        print(f"[info] wrote {emit_macros(args.emit_macros, s, s_abl, res)}")


if __name__ == "__main__":
    main()
