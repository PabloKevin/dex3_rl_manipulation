"""apple_depth.py -- the apple's own depth in its box, COPIED VERBATIM from g1_fruits_picker/common/object_locator.py
(lines 40-56 and 115-315: RADIUS_*, APPLE_RADIUS_M, _DEPTH_GAP_M, _robust_depth_at, _robust_bbox_depth, _apple_depth),
so the Jetson's detections get exactly the laptop's depth. object_locator itself imports ultralytics (not on the
Jetson). Keep in step if those functions change there."""
import math
from typing import Optional, Sequence, Tuple, List, Dict, Any

import cv2
import numpy as np



#: plausible fruit radii [m]. Outside this band the apparent-size estimate is rejected
#: (merged detections, truncation at the image edge, a bad depth sample) and consumers
#: fall back to their class default rather than acting on a wrong number.
RADIUS_MIN_M = 0.018
RADIUS_MAX_M = 0.060
#: THE APPLE RADIUS ASSUMED BY THE DEPTH SELECTION [m]. Fixed, because today's apples
#: are identical artificial ones and the planner uses 3.5 cm for all of them (the
#: apparent-size estimate below was tried on the robot and was not reliable). ON A REAL
#: TREE THIS IS AN OPEN ITEM: apples vary 2.5-4.5 cm in radius, which moves the
#: expected depth from the box size by +-30 % and the surface->centre offset by +-1 cm.
#: A per-apple radius that can be trusted is needed before then (operator, 2026-10-05).
APPLE_RADIUS_M = 0.035
#: depth values further apart than this are different surfaces [m]
_DEPTH_GAP_M = 0.04



def _robust_depth_at(
    depth: np.ndarray,
    cx: int,
    cy: int,
    window: int = 5,
    invalid_values: Sequence[int] = (0,),
) -> Optional[float]:
    """
    Returns robust depth value from a window around (cx, cy).
    Works for uint16 depth (mm) or float depth (m).
    """
    if depth is None:
        return None

    h, w = depth.shape[:2]
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None

    r = max(0, window // 2)
    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)

    patch = depth[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    # Flatten and filter invalids
    flat = patch.reshape(-1)

    # If integer depth, invalid often equals 0
    if np.issubdtype(flat.dtype, np.integer):
        mask = np.ones_like(flat, dtype=bool)
        for v in invalid_values:
            mask &= (flat != v)
        flat = flat[mask]
    else:
        # float depth: filter NaN/inf and non-positive
        flat = flat[np.isfinite(flat)]
        flat = flat[flat > 0.0]

    if flat.size == 0:
        return None

    return float(np.median(flat))


def _robust_bbox_depth(
    depth: np.ndarray,
    bbox: Sequence[float],
    shrink: float = 0.5,
    band_m: float = 0.03,
    depth_scale_for_int: float = 0.001,
) -> Optional[float]:
    """
    Foreground-surface depth for a detection bbox.

    Samples the central `shrink` fraction of the bbox (not a tiny window at
    the center pixel, which flips to the background when the bbox center
    jitters near the object's edge), then returns the median of the nearest
    depth mode: values within `band_m` of the 25th percentile.  Background
    pixels bleeding into the box are farther than the fruit surface and get
    rejected by the band.

    Returns depth in the *native* units of the depth image (mm for uint16),
    like _robust_depth_at, or None if no valid samples.
    """
    if depth is None:
        return None

    h, w = depth.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = max(1.0, (x2 - x1) * shrink / 2.0)
    hh = max(1.0, (y2 - y1) * shrink / 2.0)
    u1, u2 = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    v1, v2 = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    if u2 <= u1 or v2 <= v1:
        return None

    flat = depth[v1:v2, u1:u2].reshape(-1)
    if np.issubdtype(flat.dtype, np.integer):
        flat = flat[flat != 0]
        band = band_m / depth_scale_for_int   # metres → native (mm)
    else:
        flat = flat[np.isfinite(flat)]
        flat = flat[flat > 0.0]
        band = band_m

    if flat.size == 0:
        return None

    z0 = np.percentile(flat, 25)
    near = flat[np.abs(flat.astype(np.float64) - z0) <= band]
    if near.size == 0:
        return float(z0)
    return float(np.median(near))


def _apple_depth(
    depth: np.ndarray,
    bbox: Sequence[float],
    f_px: float,
    rgb_bgr: Optional[np.ndarray] = None,
    red: bool = False,
    radius_m: float = APPLE_RADIUS_M,
    depth_scale_for_int: float = 0.001,
    shrink: float = 0.7,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Depth of the APPLE's front surface in a detection box [m], and how it was chosen.

    WHY NOT THE NEAREST SURFACE. _robust_bbox_depth takes the nearest depth mode in the
    box, which rejects background but hands the box to a LEAF in front of the apple: on
    2026-10-05 two attempts placed an apple 10 cm nearer and 7 cm higher than it was,
    along the camera ray, because the leaf's depth was taken for the apple's.

    HOW. The depths in the central `shrink` of the box are split into clusters at gaps
    of _DEPTH_GAP_M. A leaf is thin and nearer, the background is farther and spread,
    the apple face spans at most one radius. The apples are identical, so the box's
    size gives an expected distance on its own: z_box = r f / (half the smaller side).
    The cluster taken is the one, among those big enough (>= 10 % of the pixels) and
    thin enough (span <= 1.3 r), whose near edge is closest to z_box. For a red apple
    the colour votes too: if the red pixels' median depth is not inside that cluster,
    the cluster that holds the red pixels wins. The surface is the chosen cluster's
    25th percentile, the front of the face rather than its limb.

    Returns (surface depth in metres or None, quality). quality carries the pixel
    count, the chosen cluster's span, z_box and the method, so a caller can refuse
    thin evidence and a recording can show what was decided."""
    q: Dict[str, Any] = {"method": "none", "pixels": 0}
    if depth is None:
        return None, q
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = max(1.0, (x2 - x1) * shrink / 2.0)
    hh = max(1.0, (y2 - y1) * shrink / 2.0)
    u1, u2 = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    v1, v2 = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    if u2 <= u1 or v2 <= v1:
        return None, q
    patch = depth[v1:v2, u1:u2]
    if np.issubdtype(patch.dtype, np.integer):
        zm = patch.astype(np.float64) * float(depth_scale_for_int)
    else:
        zm = patch.astype(np.float64)
    valid = np.isfinite(zm) & (zm > 0.0)
    flat = zm[valid]
    q["pixels"] = int(flat.size)
    if flat.size == 0:
        return None, q
    # expected distance from the box size alone
    half_px = 0.5 * max(1.0, min(x2 - x1, y2 - y1))
    z_box = float(radius_m * f_px / half_px)
    q["z_box_m"] = round(z_box, 4)
    # clusters at gaps
    srt = np.sort(flat)
    cuts = np.where(np.diff(srt) > _DEPTH_GAP_M)[0] + 1
    clusters = np.split(srt, cuts)
    cands = []
    for c in clusters:
        p10, p25, p90 = (float(np.percentile(c, k)) for k in (10, 25, 90))
        cands.append({"n": int(c.size), "p10": p10, "p25": p25, "p90": p90, "span": p90 - p10})
    q["clusters"] = [[round(c["p10"], 3), round(c["p90"], 3), c["n"]] for c in cands]
    ok = [c for c in cands if c["n"] >= max(10, 0.10 * flat.size) and c["span"] <= 1.3 * radius_m]
    if not ok:
        q["method"] = "fallback_nearest"
        d = _robust_bbox_depth(depth, bbox, depth_scale_for_int=depth_scale_for_int)
        if d is None:
            return None, q
        return (float(d) * depth_scale_for_int if np.issubdtype(depth.dtype, np.integer) else float(d)), q
    # z_box is the distance of the apple's CENTRE (the silhouette is the equator); the
    # cluster's near edge is its front surface, one radius nearer
    best = min(ok, key=lambda c: abs(c["p10"] - (z_box - radius_m)))
    q["method"] = "size"
    # the colour's vote, for a red apple with a colour frame
    if red and rgb_bgr is not None and rgb_bgr.ndim == 3:
        try:
            crop = rgb_bgr[v1:v2, u1:u2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hh_, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            redm = ((hh_ <= 10) | (hh_ >= 170)) & (ss >= 90) & (vv >= 50) & valid
            q["red_pixels"] = int(redm.sum())
            if redm.sum() >= 20:
                zr = float(np.median(zm[redm]))
                q["red_depth_m"] = round(zr, 4)
                if not (best["p10"] - 0.02 <= zr <= best["p90"] + 0.02):
                    holder = min(cands, key=lambda c: 0.0 if c["p10"] - 0.02 <= zr <= c["p90"] + 0.02
                                 else min(abs(zr - c["p10"]), abs(zr - c["p90"])))
                    if holder["n"] >= 10:
                        best = holder
                        q["method"] = "size+colour"
                else:
                    q["method"] = "size+colour"
        except Exception:
            pass
    q["span_m"] = round(best["span"], 4)
    q["cluster_pixels"] = best["n"]
    q["size_vs_depth_m"] = round((z_box - radius_m) - best["p25"], 4)    # ~0 when the box agrees
    return float(best["p25"]), q


