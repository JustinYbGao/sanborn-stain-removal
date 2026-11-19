"""Reduce the yellow cast left in paper pixels already corrected by V2."""

import numpy as np

from .paper import ROW_CHUNK, restore_rgb as restore_v2

VERSION = 'paper-residual-chroma-v3'


def correct_residual_color(raw, corrected, changed, paper_rgb):
    """Add only the remaining blue deficit, in place, within the V2 edit mask."""
    if paper_rgb is None or not changed.any():
        return
    _, paper_green, paper_blue = paper_rgb
    paper_gap = paper_green - paper_blue
    for top in range(0, raw.shape[0], ROW_CHUNK):
        rows = slice(top, top + ROW_CHUNK)
        mask = changed[rows]
        if not mask.any():
            continue
        original = raw[rows][mask].astype(np.int16)
        result = corrected[rows][mask].astype(np.int16)
        # Scale the extra correction down for slight warmth and the faded
        # boundary of a V2 edit. Strong paper stains have a larger G–B gap.
        excess = original[:, 1] - original[:, 2] - paper_gap
        warm = np.clip((excess - 7) / 13, 0, 1)
        progress = (result[:, 1] - original[:, 1]) / np.maximum(paper_green - original[:, 1], 8)
        support = np.clip((progress - .15) / .55, 0, 1)
        # Match the paper's color at the brightness already chosen by V2.
        # Keep a small share of the residual to retain paper texture.
        goal_blue = np.minimum(result[:, 1] * (paper_blue / max(paper_green, 1)), 255)
        extra_blue = np.rint(.9 * warm * support * np.maximum(goal_blue - result[:, 2], 0))
        corrected[rows, :, 2][mask] = result[:, 2] + extra_blue


def restore_rgb(rgb):
    corrected, changed, review, diagnostics = restore_v2(rgb)
    correct_residual_color(rgb, corrected, changed, diagnostics['paper_rgb'])
    return corrected, changed, review, {**diagnostics, 'algorithm': VERSION}
