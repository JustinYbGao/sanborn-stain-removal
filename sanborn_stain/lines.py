"""Carry the measured correction into warm paper near dark lines."""

import cv2
import numpy as np

from .chroma import restore_rgb as restore_previous
from .paper import ROW_CHUNK, lab_of, paper_corridor, quiet_pixels

VERSION = 'paper-chroma-with-ink-neighborhood-v4'
MIN_PAPER_GREEN = 125


def donor_field(raw, previous, paper_rgb, work_size):
    """Average source colors and corrections from already corrected warm paper."""
    source = cv2.resize(raw, work_size, interpolation=cv2.INTER_AREA)
    target = cv2.resize(previous, work_size, interpolation=cv2.INTER_AREA)
    lab = lab_of(source)
    paper = lab_of(np.asarray(paper_rgb, np.uint8).reshape(1, 1, 3))[0, 0]
    shift = target.astype(np.float32) - source
    donors = quiet_pixels(lab) & paper_corridor(lab, paper)
    donors &= (lab[..., 2] >= paper[2] + 4) & (shift[..., 2] - shift[..., 1] >= 5)
    if not donors.any():
        return None
    weights = donors.astype(np.float32)
    packed = np.dstack((weights, source * weights[..., None], shift * weights[..., None]))
    return cv2.GaussianBlur(packed, (0, 0), sigmaX=5)


def correct_near_ink(raw, previous, changed, paper_rgb, work_size):
    """Modify paper matching the donor colors, with smooth spatial and color gates."""
    if paper_rgb is None or not changed.any():
        return 0
    field = donor_field(raw, previous, paper_rgb, tuple(work_size))
    if field is None:
        return 0
    height, width = changed.shape
    work_width, work_height = work_size
    radius = min(38, max(5, round(6 * height / work_height)))
    x_coords = (np.arange(width, dtype=np.float32) + .5) * (work_width / width) - .5
    ink_ceiling = min(105, round(.6 * paper_rgb[1]))
    paper_gap = paper_rgb[1] - paper_rgb[2]
    changed_count = 0
    for top in range(0, height, ROW_CHUNK):
        end = min(top + ROW_CHUNK, height)
        lo, hi = max(0, top - radius - 1), min(height, end + radius + 1)
        expanded = raw[lo:hi]
        ink = np.max(expanded, axis=2) <= ink_ceiling
        if not ink.any():
            continue
        distance = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)[top-lo:end-lo]
        rows = slice(top, end)
        source = raw[rows]
        eligible = (distance > 0) & (distance < radius)
        eligible &= source[..., 1] > MIN_PAPER_GREEN
        eligible &= source[..., 1].astype(np.int16) - source[..., 2] >= paper_gap + 4
        if not eligible.any():
            continue
        y_coords = (np.arange(top, end, dtype=np.float32) + .5) * (work_height / height) - .5
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        local = cv2.remap(field, grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        eligible &= local[..., 0] > .08
        local = local[eligible]
        weight = local[:, 0]
        donors = local[:, 1:] / weight[:, None]
        pixels = source[eligible].astype(np.float32)
        # The bright paper must match the measured raw paper in all three
        # channels. Dark strokes and their gray antialiasing fail this test.
        error = (np.abs(donors[:, :3] - pixels) / np.array([9, 9, 15], np.float32)).max(axis=1)
        strength = np.clip((1 - error) * 2, 0, 1)
        strength *= np.clip((weight - .08) / .16, 0, 1)
        strength *= np.clip((radius - distance[eligible]) / (radius * .35), 0, 1)
        strength *= np.clip((pixels[:, 1] - MIN_PAPER_GREEN) / 10, 0, 1)
        existing = previous[rows][eligible]
        proposal = np.rint(pixels + strength[:, None] * donors[:, 3:]).clip(0, 255).astype(np.uint8)
        result = np.maximum(existing, proposal)
        extra = np.any(result != existing, axis=1)
        if extra.any():
            previous[rows][eligible] = result
            changed[rows][eligible] |= extra
            changed_count += int(extra.sum())
    return changed_count


def restore_rgb(rgb):
    previous, changed, review, diagnostics = restore_previous(rgb)
    count = correct_near_ink(rgb, previous, changed, diagnostics['paper_rgb'], diagnostics.get('work_dimensions'))
    review &= ~changed
    return previous, changed, review, {**diagnostics, 'algorithm': VERSION, 'near_ink_extra_pixels': count}
