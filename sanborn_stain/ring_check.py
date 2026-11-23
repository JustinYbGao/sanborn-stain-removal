"""Native-scale center/ring check sampled on the five-pixel paper grid."""
import cv2
import numpy as np


def increases(src, old, new, quiet, near_edit):
    bad = np.zeros(src.shape, bool)
    eligible = quiet & near_edit
    eligible[:11] = eligible[-11:] = eligible[:, :11] = eligible[:, -11:] = False
    y, x = np.where(eligible)
    if not len(x):
        return bad
    centers = [cv2.medianBlur(im.astype('uint8'), 3).astype(np.float32) for im in (src, old, new)]
    radial = np.arange(-10, 11)
    rr = radial[:, None] ** 2 + radial[None, :] ** 2
    offsets = [np.where((rr >= a * a) & (rr <= b * b)) for a, b in ((5, 7), (6, 8), (7, 10))]
    for start in range(0, len(x), 4096):
        ys = y[start:start + 4096]
        xs = x[start:start + 4096]
        for dy, dx in offsets:
            iy = ys[:, None] + dy - 10
            ix = xs[:, None] + dx - 10
            contrasts = [np.median(im[iy, ix], axis=1) - center[ys, xs] for center, im in zip(centers, (src, old, new))]
            prev = np.abs(contrasts[1] - contrasts[0])
            next_ = np.abs(contrasts[2] - contrasts[0])
            hit = (next_ >= 6) & (next_ - prev >= 4)
            bad[ys[hit], xs[hit]] = True
    return bad
