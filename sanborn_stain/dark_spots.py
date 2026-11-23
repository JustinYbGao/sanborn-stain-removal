"""Extend the measured RGB correction across enclosed, under-corrected paper."""

import cv2
import numpy as np

from . import spots
from .local_guard import screen
from .paper import ROW_CHUNK

VERSION = 'paper-correction-with-dark-spot-repair-v6'


def _target(raw, clean):
    h, w = raw.shape[:2]
    size = (max(1, w // 5), max(1, h // 5))
    def small(im):
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        return cv2.GaussianBlur(im, (0, 0), 1)
    src = small(raw)
    gain = small(clean) - src
    value = cv2.cvtColor(gain, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # Start at paper with very little correction inside a more strongly corrected neighborhood.
    seeds = (value <= 6) & (cv2.GaussianBlur(value, (0, 0), 6) - value >= 5)
    seeds &= (gray >= 122) & (np.ptp(src, axis=2) < 41)
    seeds[:12] = seeds[-12:] = seeds[:, :12] = seeds[:, -12:] = False
    yy, xx = np.where(seeds)
    if not len(xx):
        return None
    # A one-sided transition at a wall or print boundary is insufficient evidence.
    votes = np.zeros(len(xx), np.int16)
    total = np.zeros((len(xx), 3), np.float32)
    for dy, dx in ((0, 11), (8, 8), (11, 0), (8, -8), (0, -11), (-8, -8), (-11, 0), (-8, 8)):
        idx = (yy + dy, xx + dx)
        eligible = value[idx] >= np.maximum(12, value[yy, xx] + 10)
        eligible &= np.max(np.abs(src[idx] - src[yy, xx]), axis=1) <= 15
        votes += eligible
        total += gain[idx] * eligible[:, None]
    keep = votes >= 6
    if not keep.any():
        return None
    idx = (yy[keep], xx[keep])
    seeds[:] = False
    seeds[idx] = True
    sums = np.zeros_like(src)
    sums[idx] = total[keep] / votes[keep, None]
    sums = cv2.GaussianBlur(sums, (0, 0), 5)
    weights = cv2.GaussianBlur(seeds.astype(np.float32), (0, 0), 5)
    target = sums / np.maximum(weights, 1e-08)[:, :, None]
    distance = cv2.distanceTransform((~seeds).astype('uint8'), cv2.DIST_L2, 5)
    blend = np.clip((10 - distance) / 6, 0, 1)
    return (src, target, blend)


def repair(raw, clean):
    """Modify v5 RGB in place, within local basins of its correction field."""
    grid = _target(raw, clean)
    if grid is None:
        return clean
    guide, gain, weight = grid
    h, w = raw.shape[:2]
    gh, gw = weight.shape
    # Restrict native-resolution work to boxes around actual proposals.
    support = cv2.dilate((weight > 0).astype('uint8'), np.ones((3, 3), np.uint8))
    n, labels, boxes, _ = cv2.connectedComponentsWithStats(support)
    fields = np.dstack((guide, gain, weight))
    for k in range(1, n):
        left, top, width, height, _ = boxes[k]
        x0, x1 = (int(left * w / gw), min(w, int(np.ceil((left + width) * w / gw))))
        y0, y1 = (int(top * h / gh), min(h, int(np.ceil((top + height) * h / gh))))
        xs = (np.arange(x0, x1, dtype=np.float32) + 0.5) * (gw / w) - 0.5
        # Two component bounding boxes can overlap; only apply each support once.
        bounds = np.s_[top:top + height, left:left + width]
        component = fields[bounds].copy()
        component[:, :, -1] *= labels[bounds] == k
        for y in range(y0, y1, 192):
            end = min(y + 192, y1)
            ys = (np.arange(y, end, dtype=np.float32) + 0.5) * (gh / h) - 0.5
            mapx, mapy = np.meshgrid(xs - int(left), ys - int(top))
            field = cv2.remap(component, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            region = np.s_[y:end, x0:x1]
            src = raw[region].astype(np.float32)
            old = clean[region]
            # Fade out on native ink, strong printing colors and high-contrast strokes.
            error = abs(src - field[:, :, :3]).max(2)
            alpha = np.clip((20 - error) / 8, 0, 1) * np.clip((48 - np.ptp(src, axis=2)) / 8, 0, 1)
            alpha *= field[:, :, 6] * (src.max(2) > 100)
            delta = src + field[:, :, 3:6] - old
            # Only raise an under-corrected pixel; preserve the donor's channel ratios.
            alpha *= cv2.cvtColor(delta, cv2.COLOR_RGB2GRAY) > 0
            old[:] = np.rint(old + alpha[:, :, None] * delta).clip(0, 255).astype(np.uint8)
    return clean

def restore_rgb(rgb):
    """Apply the frozen dark-spot repair and report final RGB changes."""
    previous, changed, review, diagnostics = spots.restore_rgb(rgb)
    result = screen(rgb, previous, repair(rgb, previous.copy()))
    for top in range(0, len(rgb), ROW_CHUNK):
        rows = slice(top, top + ROW_CHUNK)
        changed[rows] = np.any(result[rows] != rgb[rows], axis=2)
    review &= ~changed
    return (result, changed, review, {**diagnostics, 'algorithm': VERSION})
