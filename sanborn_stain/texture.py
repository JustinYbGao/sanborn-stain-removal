"""Reduce grain introduced by cleanup while retaining weak printed detail."""

import cv2
import numpy as np

from .paper import ROW_CHUNK

_PAPER_NEIGHBORHOOD = np.ones((5, 5), np.uint8)


def _smooth_gain(raw, cleaned):
    output = cleaned.copy()
    for top in range(0, len(raw), ROW_CHUNK):
        end = min(len(raw), top + ROW_CHUNK)
        lo, hi = max(0, top - 3), min(len(raw), end + 3)
        rows = slice(top - lo, end - lo)
        original = raw[lo:hi]
        gain = cleaned[lo:hi].astype(np.float32) - original
        adjustment = cv2.GaussianBlur(gain, (7, 7), 1.4) - gain
        span = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, _PAPER_NEIGHBORHOOD).max(2)
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        chroma = np.ptp(original, axis=2)
        weight = np.clip((24 - span.astype(np.float32)) / 10, 0, 1)
        weight *= np.clip((45 - chroma.astype(np.float32)) / 10, 0, 1)
        weight *= np.clip((gray.astype(np.float32) - 115) / 12, 0, 1)
        weight *= np.clip((np.abs(adjustment).max(2) - 1) / 3, 0, 1)
        step = np.clip(adjustment[rows], -24, 24) * weight[rows, :, None]
        output[top:end] = np.rint(cleaned[top:end] + step).clip(0, 255).astype(np.uint8)
    return output


def _fine_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.medianBlur(gray, 7).astype(np.int16) - gray


def repair_grain(raw, cleaned):
    """Smooth the RGB correction and compensate where it weakens source detail."""
    proposed = _smooth_gain(raw, cleaned)
    output = proposed.copy()
    for top in range(0, len(raw), ROW_CHUNK):
        end = min(len(raw), top + ROW_CHUNK)
        lo, hi = max(0, top - 6), min(len(raw), end + 6)
        if cv2.norm(proposed[lo:hi], cleaned[lo:hi], cv2.NORM_INF) == 0:
            continue
        source = _fine_contrast(raw[lo:hi])
        before = _fine_contrast(cleaned[lo:hi])
        after = _fine_contrast(proposed[lo:hi])
        sign = np.sign(source)
        weak = (np.abs(source) >= 5) & (np.abs(source) <= 24)
        loss = weak & (sign * (source - after) >= 3) & (sign * (before - after) >= 3)
        if not loss.any():
            continue
        rows = slice(top - lo, end - lo)
        span = cv2.morphologyEx(raw[lo:hi], cv2.MORPH_GRADIENT, _PAPER_NEIGHBORHOOD).max(2)[rows]
        centers = raw[top:end]
        light_paper = (span < 24) & (np.ptp(centers, axis=2) < 45)
        light_paper &= cv2.cvtColor(centers, cv2.COLOR_RGB2GRAY) > 115
        target = sign * np.minimum(np.abs(source), sign * before)
        # Change all channels equally to correct contrast without adding color.
        adjustment = np.where(loss[rows] & light_paper, np.clip((after - target)[rows], -12, 12), 0)
        output[top:end] = np.clip(proposed[top:end].astype(np.int16) + adjustment[:, :, None],
                                  0, 255).astype(np.uint8)
    return output
