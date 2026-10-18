"""Hot-pixel selection relative to the thermal background (R5)."""

# Shared by the thermal candidate detector (which regions are heat sources)
# and the candidate localizer (which depth points lie on them), so both use
# the same definition of "hot".
#
# A pixel is hot when it is at least delta_k above the scene background,
# the 75th percentile temperature of the frame (BACKGROUND_PERCENTILE). A
# region is kept when it is also delta_k above its local background: the
# median of a ring of non-hot pixels around its bounding box.
#
# In large_demo the ground is ~293 K and trees/rocks ~300.7 K (7.7 K warmer),
# so a 3-5 K threshold would mark every tree; 10 K keeps vegetation out and
# still finds animals far below the old fixed 315 K threshold. The 75th
# percentile, not the median: frames with much cold sky had a median of
# ~290.8 K, so the threshold fell to ~300.8 K, a hair above the trees, and
# tree pixels behind an animal pulled its depth estimate (mean error 0.35 ->
# 0.71 m in a run). The percentile follows the vegetation when it fills the
# view and the ground otherwise.


import numpy as np

BACKGROUND_PERCENTILE = 75.0


def scene_background_k(kelvin, percentile=BACKGROUND_PERCENTILE):
    """Background temperature of a frame: a percentile of its finite pixels [K]."""
    values = np.asarray(kelvin, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    return float(np.percentile(values, percentile))


def relative_hot_mask(kelvin, delta_k):
    """Boolean mask of pixels at least delta_k above the scene background."""
    kelvin = np.asarray(kelvin, dtype=float)
    background = scene_background_k(kelvin)
    if not np.isfinite(background):
        return np.zeros(kelvin.shape, dtype=bool)
    with np.errstate(invalid='ignore'):
        return kelvin >= background + float(delta_k)


def hot_mask(kelvin, mode, absolute_k, delta_k):
    """Hot mask for mode 'relative' (default) or 'absolute' (old fixed K)."""
    if mode == 'absolute':
        with np.errstate(invalid='ignore'):
            return np.asarray(kelvin, dtype=float) >= float(absolute_k)
    if mode != 'relative':
        raise ValueError("threshold mode must be 'relative' or 'absolute'")
    return relative_hot_mask(kelvin, delta_k)


def local_background_k(kelvin, hot, box, margin_px):
    """Median of the non-hot pixels in a ring around box = (x, y, w, h)."""
    kelvin = np.asarray(kelvin, dtype=float)
    height, width = kelvin.shape[:2]
    x, y, w, h = (int(v) for v in box)
    x1, y1 = max(0, x - margin_px), max(0, y - margin_px)
    x2, y2 = min(width, x + w + margin_px), min(height, y + h + margin_px)
    ring = np.ones((y2 - y1, x2 - x1), dtype=bool)
    ring[max(0, y - y1):min(y2, y + h) - y1, max(0, x - x1):min(x2, x + w) - x1] = False
    window = kelvin[y1:y2, x1:x2]
    candidates = ring & ~np.asarray(hot, dtype=bool)[y1:y2, x1:x2] & np.isfinite(window)
    if not candidates.any():
        return float('nan')
    return float(np.median(window[candidates]))
