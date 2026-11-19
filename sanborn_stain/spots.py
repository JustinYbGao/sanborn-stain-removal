"""Continue the existing color correction through small gaps on warm paper."""

import math

import cv2
import numpy as np

from .lines import restore_rgb as restore_previous
from .paper import lab_of, paper_corridor

VERSION = 'paper-correction-with-isolated-spot-recovery-v5'
WORK_SIDE = 2400
MIN_PAPER_GREEN = 125
COLOR_TOLERANCE = np.array([13, 13, 21], np.float32)
RING_OUTER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
RING_INNER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLOSING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
SEPARATE_SPOTS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
INK_MARGIN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def spot_components(raw, corrected, paper_rgb):
    """Find small warm depressions in the current correction."""
    raw_lab = lab_of(raw)
    paper = lab_of(np.asarray(paper_rgb, dtype=np.uint8).reshape(1, 1, 3))[0, 0]
    blue_gain = corrected[..., 2].astype(np.int16) - raw[..., 2]
    donor = paper_corridor(raw_lab, paper)
    donor &= (raw[..., 1] > MIN_PAPER_GREEN) & (blue_gain >= 8)
    if not donor.any():
        return None
    smooth = cv2.GaussianBlur(corrected[..., 1:], (3, 3), .65)
    deficit = cv2.subtract(cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, CLOSING_KERNEL), smooth)
    paper_like = paper_corridor(raw_lab, paper) & (raw[..., 1] > MIN_PAPER_GREEN)
    warm = raw_lab[..., 2] > paper[2] + 4
    residual = paper_like & warm & (deficit[..., 0] >= 8) & (deficit[..., 1] >= 12)
    # Thin bridges in the residual can join neighboring round spots.
    cores = cv2.morphologyEx(residual.astype(np.uint8), cv2.MORPH_OPEN, SEPARATE_SPOTS)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
    return labels, stats, range(1, count), donor & ~residual


def apply_spot(raw, corrected, changed, spot, window, work_shape, original_color, paper_color):
    """Apply the local correction within a feathered component at native scale."""
    height, width = changed.shape
    work_h, work_w = work_shape
    x0, y0, x1, y1 = window
    left, top = math.floor(x0 * width / work_w), math.floor(y0 * height / work_h)
    right, bottom = math.ceil(x1 * width / work_w), math.ceil(y1 * height / work_h)
    rows = np.s_[top:bottom, left:right]
    patch = raw[rows]
    x = (np.arange(left, right, dtype=np.float32) + .5) * (work_w / width) - .5 - x0
    y = (np.arange(top, bottom, dtype=np.float32) + .5) * (work_h / height) - .5 - y0
    grid_x, grid_y = np.meshgrid(x, y)
    hard_mask = cv2.remap(spot, grid_x, grid_y, cv2.INTER_NEAREST)
    # Reject marks at actual native ink, including thin strokes missed at work scale.
    ink = (np.max(patch, axis=2) <= 110).astype(np.uint8)
    near_ink = cv2.dilate(ink, INK_MARGIN)
    if np.any(hard_mask & near_ink):
        return False
    expanded = cv2.dilate(spot, SEPARATE_SPOTS)
    feather = cv2.GaussianBlur(expanded.astype(np.float32), (0, 0), .85)
    blend = cv2.remap(feather, grid_x, grid_y, cv2.INTER_LINEAR)
    blend[near_ink != 0] = 0
    error = (np.abs(patch - original_color) / COLOR_TOLERANCE).max(axis=2)
    blend *= np.clip((1 - error) * 2, 0, 1)
    blend *= np.clip((patch[..., 1].astype(np.float32) - MIN_PAPER_GREEN) / 9, 0, 1)
    gain = np.maximum(paper_color - original_color, 0)
    target = np.minimum(patch + gain, paper_color + 3)
    current = corrected[rows]
    result = np.rint(current + blend[..., None] * np.maximum(target - current, 0)).clip(0, 255).astype(np.uint8)
    edited = np.any(result != current, axis=2)
    if not edited.any():
        return False
    current[:] = result
    changed[rows] |= edited
    return True


def correct_spots(raw, corrected, changed, paper_rgb):
    if paper_rgb is None or not changed.any():
        return 0
    h, w = changed.shape
    scale = min(1, WORK_SIDE / max(w, h))
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    raw_small = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
    old_small = cv2.resize(corrected, size, interpolation=cv2.INTER_AREA)
    found = spot_components(raw_small, old_small, paper_rgb)
    if found is None:
        return 0
    labels, stats, ids, donors = found
    count = 0
    margin = 9
    for index in ids:
        x, y, width, height, area = stats[index].tolist()
        if not 5 <= area <= 560 or max(width, height) > 36 or max(width, height) > 2.8 * min(width, height):
            continue
        if area < .30 * width * height:
            continue
        x0, y0, x1, y1 = x - margin, y - margin, x + width + margin, y + height + margin
        if x0 < 0 or y0 < 0 or x1 > size[0] or y1 > size[1]:
            continue
        box = np.s_[y0:y1, x0:x1]
        spot = (labels[box] == index).astype(np.uint8)
        ring = cv2.dilate(spot, RING_OUTER) > cv2.dilate(spot, RING_INNER)
        support = ring & donors[box]
        if support.sum() < .50 * ring.sum():
            continue
        original_color = np.median(raw_small[box][support], axis=0).astype(np.float32)
        center = np.median(raw_small[box][spot > 0], axis=0)
        if np.any(np.abs(center - original_color) > .85 * COLOR_TOLERANCE):
            continue
        paper_color = np.median(old_small[box][support], axis=0).astype(np.float32)
        count += apply_spot(raw, corrected, changed, spot, (x0, y0, x1, y1),
                            raw_small.shape[:2], original_color, paper_color)
    return count


def restore_rgb(rgb):
    out, changed, review, diagnostics = restore_previous(rgb)
    count = correct_spots(rgb, out, changed, diagnostics['paper_rgb'])
    review &= ~changed
    return out, changed, review, {**diagnostics, 'algorithm': VERSION, 'supported_spots_corrected': count}
