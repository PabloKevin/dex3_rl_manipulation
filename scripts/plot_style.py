# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Shared matplotlib style for the IEEE double-column paper figures.

Deliberately dependency-light: matplotlib + numpy only, no isaaclab, no torch,
no seaborn, no pandas. That lets plot_results.py run in a plain python process
without paying the ~40 s Isaac Sim startup cost.

Design rules this module encodes (see the paper's figure captions):

  * Vector PDF with TrueType fonts (``pdf.fonttype = 42``). Type-3 fonts are
    rejected by IEEE PDF eXpress, and matplotlib's default IS Type-3.
  * Serif body text at 8 pt to sit next to IEEEtran's Times without clashing.
  * Recessive chrome: hairline solid y-grid, no top/right spines, no legend
    frame, no minor ticks.
  * Colour carries identity ONLY together with a second encoding -- linestyle
    for line pairs, marker shape for scatter classes. IEEE papers get printed
    and photocopied in greyscale.
  * Text never wears a series colour; labels use the neutral ink tokens.

Palette provenance: taken verbatim from the data-viz reference palette, in its
documented fixed slot order. Scatter/small-multiple forms (all-pairs
comparison) are capped at the first three slots, the only ones that clear the
all-pairs CVD/normal-vision gates; line/bar forms use the *adjacent* pairlist,
which the full eight-slot order clears in both light and dark modes, so
JOINT_RAMP below uses all seven non-red slots for the 7-DoF joint legend. Node
is not installed in this environment, so ``validate_palette.js`` could NOT be
re-run here -- do not substitute hexes without installing node and
validating.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # no display in a headless conda env
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# ═════════════════════════════════════════════════════════════════════════
#  Canvas sizes -- IEEEtran conference, US Letter, two columns
# ═════════════════════════════════════════════════════════════════════════

COL_1 = 3.5    # inches, \columnwidth
COL_2 = 7.16   # inches, \textwidth

# ═════════════════════════════════════════════════════════════════════════
#  Palette
# ═════════════════════════════════════════════════════════════════════════

# Categorical slots 1-3. Capped at three on purpose: the reference palette
# documents these three as the set that clears the all-pairs gates in light
# mode; slot 4 (yellow) fails against slot 2 (orange).
C_BLUE = "#2a78d6"     # slot 1 -- main run / rate limiter ON / primary series
C_ORANGE = "#eb6834"   # slot 2 -- negative control / rate limiter OFF
C_AQUA = "#1baf7a"     # slot 3 -- third series, used sparingly
CATEGORICAL = (C_BLUE, C_ORANGE, C_AQUA)

# Categorical hues for the 7 arm joints -- one per joint, maximally
# distinguishable rather than an ordinal ramp, since with 7 overlapping line
# traces in one small panel adjacent shades of blue read as noise. Slots 1-7
# of the reference palette's fixed 8-slot categorical order (slot 8, red, is
# reserved for threshold/error accents elsewhere and left unused here).
JOINT_RAMP = (
    "#2a78d6",  # slot 1 blue     shoulder_pitch
    "#eb6834",  # slot 2 orange   shoulder_roll
    "#1baf7a",  # slot 3 aqua     shoulder_yaw
    "#eda100",  # slot 4 yellow   elbow
    "#e87ba4",  # slot 5 magenta  wrist_roll
    "#008300",  # slot 6 green    wrist_pitch
    "#4a3aa7",  # slot 7 violet   wrist_yaw
)

# Sequential single hue, light->dark, for continuous magnitude (final position
# error on the workspace scatter). Full 100->700 range is legal for sequential.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

# Ink / chrome tokens. Text and threshold rules use these, never a series hue.
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
SURFACE = "#ffffff"
GRID = "#E3E2DE"
AXIS = "#B8B7B2"
TOL_LINE = "#8a8984"   # tolerance/threshold rules: neutral gray, dashed

# Linestyles pairing with CATEGORICAL, so identity survives greyscale print.
LS_MAIN = "-"
LS_ALT = (0, (4, 1.6))

# ═════════════════════════════════════════════════════════════════════════
#  rcParams
# ═════════════════════════════════════════════════════════════════════════

RC = {
    # --- output: vector PDF, TrueType (Type-3 is rejected by PDF eXpress) ---
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "savefig.transparent": False,
    "figure.dpi": 200,          # only affects the PNG previews

    # --- type: match IEEEtran's Times as closely as the installed fonts allow ---
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 8,

    # --- marks: thin, print-legible ---
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "patch.linewidth": 0.6,

    # --- chrome: recessive ---
    "axes.linewidth": 0.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",      # SOLID hairline. Dashing is reserved for meaning.
    "grid.alpha": 1.0,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "text.color": INK,

    # --- legend: present but frameless ---
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.3,

    "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "errorbar.capsize": 0,
}


def apply_ieee_style() -> None:
    """Install the rcParams globally. Call once at the top of a plotting script."""
    plt.rcParams.update(RC)


# ═════════════════════════════════════════════════════════════════════════
#  Figure helpers
# ═════════════════════════════════════════════════════════════════════════

def figure(width: str = "single", height: float | None = None, aspect: float = 0.62,
           nrows: int = 1, ncols: int = 1, **kw):
    """Create a correctly-sized figure for the paper.

    Args:
        width: "single" (3.5 in, \\columnwidth) or "double" (7.16 in, \\textwidth).
        height: explicit height in inches; overrides `aspect`.
        aspect: height/width ratio used when `height` is None.
        nrows, ncols: passed to subplots.

    Returns:
        (fig, axes) exactly as plt.subplots, so a 1x1 call yields a bare Axes.
    """
    w = COL_2 if width == "double" else COL_1
    h = height if height is not None else w * aspect
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return fig, axes


def despine(ax) -> None:
    """Drop the top/right spines (redundant with rcParams, but explicit for twins)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def hairline_grid(ax, axis: str = "y") -> None:
    """Force the recessive hairline grid on one axis only."""
    ax.grid(False)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.4, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def threshold(ax, value: float, orientation: str = "h", label: str | None = None,
              label_x: float = 0.985, label_y_pad: float = 0.0,
              halo: bool = False) -> None:
    """Draw a tolerance/threshold rule: neutral gray, dashed, annotated.

    Dashed is reserved for thresholds and for the second series in a paired
    comparison -- never for gridlines -- so a dashed line in these figures
    always means "this carries information".

    `halo` puts a semi-opaque background behind the label, for rules that cross
    dense scatter where the text would otherwise be unreadable.
    """
    bbox = (dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8)
            if halo else None)
    if orientation == "h":
        ax.axhline(value, color=TOL_LINE, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        if label:
            ax.annotate(label, xy=(label_x, value), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=6.5, color=INK_MUTED,
                        xytext=(0, 1.5 + label_y_pad), textcoords="offset points",
                        bbox=bbox, zorder=6)
    else:
        ax.axvline(value, color=TOL_LINE, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        if label:
            ax.annotate(label, xy=(value, 0.98), xycoords=("data", "axes fraction"),
                        ha="left", va="top", fontsize=6.5, color=INK_MUTED,
                        xytext=(2, 0), textcoords="offset points",
                        bbox=bbox, zorder=6)


def label_end(ax, x, y, text: str, color: str = INK_2, dx: float = 3.0,
              dy: float = 0.0, fontsize: float = 6.5, **kw) -> None:
    """Direct-label a series at its right endpoint.

    `color` defaults to secondary ink: the label is TEXT, so it wears a text
    token, not the series colour. The line beside it carries the identity.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha="left", va="center", fontsize=fontsize, color=color, **kw)


def panel_tag(ax, tag: str, x: float = -0.16, y: float = 1.04) -> None:
    """Put an (a)/(b)/(c) tag on a subplot, in primary ink.

    Axes3D.text() takes (x, y, z, s), so 3-D panels go through text2D().
    """
    if hasattr(ax, "text2D"):
        ax.text2D(x, y, tag, transform=ax.transAxes, fontsize=8,
                  fontweight="bold", color=INK, ha="left", va="bottom")
    else:
        ax.text(x, y, tag, transform=ax.transAxes, fontsize=8, fontweight="bold",
                color=INK, ha="left", va="bottom")


def joint_key(ax, names: Sequence[str], loc: str = "upper right",
              ncol: int = 2, short: bool = True) -> None:
    """Legend for the 7-joint ordinal ramp, in proximal->distal order."""
    from matplotlib.lines import Line2D
    labels = [_short_joint(n) for n in names] if short else list(names)
    handles = [Line2D([0], [0], color=JOINT_RAMP[i % len(JOINT_RAMP)], lw=1.1)
               for i in range(len(labels))]
    # Opaque background: this key sits over trajectory curves in the comparison
    # figure, where a frameless legend is unreadable.
    ax.legend(handles, labels, loc=loc, ncol=ncol, fontsize=6,
              handlelength=1.2, columnspacing=0.8, labelspacing=0.22,
              borderaxespad=0.2, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9, borderpad=0.25).set_zorder(10)


def _short_joint(name: str) -> str:
    """right_shoulder_pitch_joint -> sh. pitch"""
    s = name.replace("right_", "").replace("_joint", "")
    s = s.replace("shoulder_", "sh. ").replace("wrist_", "wr. ")
    return s


def finish(fig, path: str, png_preview: bool = True, tight_pad: float = 0.25) -> str:
    """Lay out, save vector PDF (+ a PNG preview), close, return the PDF path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        fig.tight_layout(pad=tight_pad)
    except Exception:
        pass  # constrained_layout figures raise; harmless
    fig.savefig(path, format="pdf")
    if png_preview:
        fig.savefig(os.path.splitext(path)[0] + ".png", format="png", dpi=200)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════════
#  Small numeric helpers used by more than one figure
# ═════════════════════════════════════════════════════════════════════════

def rolling_mean(y: np.ndarray, w: int = 21) -> np.ndarray:
    """Centered moving average that keeps the curve spanning the full x-range.

    The denominator is convolved the same way as the numerator, so the window
    shrinks at the edges instead of the curve fading out or being padded with
    zeros. numpy only -- pandas is not installed in this environment.
    """
    y = np.asarray(y, dtype=float)
    if w <= 1 or y.size < 3:
        return y
    w = min(w, y.size if y.size % 2 else y.size - 1)
    if w % 2 == 0:
        w -= 1
    k = np.ones(w)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")
    return num / den


def plot_smoothed(ax, x, y, color: str, label: str | None = None,
                  linestyle=LS_MAIN, w: int = 21, raw_alpha: float = 0.22,
                  lw: float = 1.1, zorder: int = 3):
    """Raw series faint behind a smoothed series at full weight.

    Showing both is the honest option: the reader sees the real noise level and
    the trend, instead of only a smoothed curve that hides the variance.
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    ax.plot(x, y, color=color, alpha=raw_alpha, linewidth=0.5,
            linestyle="-", zorder=zorder - 1)
    ys = rolling_mean(y, w)
    (line,) = ax.plot(x, ys, color=color, linewidth=lw, linestyle=linestyle,
                      label=label, zorder=zorder)
    return line, ys
