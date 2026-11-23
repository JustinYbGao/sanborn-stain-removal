"""Roll back connected edits that introduce dark or bright contrast on smooth paper."""
import cv2
import numpy as np
from PIL import Image
from .ring_check import increases


def screen(raw, v5, proposed):
    h, w = raw.shape[:2]
    size = (max(1, w // 5), max(1, h // 5))
    def reduce(im):
        return np.asarray(Image.fromarray(im).resize(size, Image.Resampling.BOX))
    def gray(im):
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    def band(im, sigma):
        return cv2.GaussianBlur(im, (0, 0), sigma) - cv2.GaussianBlur(im, (0, 0), 1.4)
    src = reduce(raw)
    ref = gray(src)
    old = gray(reduce(v5))
    kernel = np.ones((17, 17), 'uint8')
    lo = cv2.erode(ref, kernel)
    span = cv2.dilate(ref, kernel) - lo
    quiet = (lo > 106) & (span <= 32) & (np.ptp(src, axis=2) < 48)
    paper = (lo > 119) & (span <= 25) & (np.ptp(src, axis=2) < 40) & (np.abs(band(ref, 6)) <= 9)
    paper[:10] = paper[-10:] = paper[:, :10] = paper[:, -10:] = False
    any_edit = np.zeros((h, w), 'uint8')
    for y in range(0, h, 192):
        sl = np.s_[y:y + 192]
        edge_slice = raw[max(0, y - 1):min(h, y + 193)]
        edge = cv2.morphologyEx(cv2.cvtColor(edge_slice, cv2.COLOR_RGB2GRAY), cv2.MORPH_GRADIENT, np.ones((3, 3), 'uint8'))
        preserve = edge[int(y > 0):int(y > 0) + len(raw[sl])] >= 35
        np.copyto(proposed[sl], v5[sl], where=preserve[:, :, None])
        any_edit[sl] = (proposed[sl] != v5[sl]).any(2).astype('uint8') * 255
    support = cv2.resize(any_edit, size, interpolation=cv2.INTER_AREA) > 0
    del any_edit
    if not support.any():
        return proposed
    _, labels = cv2.connectedComponents(cv2.dilate(support.astype('uint8'), np.ones((3, 3), 'uint8')))
    active = np.ones(labels.max() + 1, bool)
    active[0] = False
    # Include a second scale to catch broad halos extending outside the repaired hole.
    ref_bands = [band(ref, sigma) for sigma in (6, 9)]
    old_bands = [band(old, sigma) - ref_band for sigma, ref_band in zip((6, 9), ref_bands)]
    gx = np.minimum(np.arange(w) * size[0] // w, size[0] - 1)
    passes = 0
    while active.any():
        new = gray(reduce(proposed))
        near_edit = cv2.dilate(active[labels].astype('uint8'), np.ones((23, 23), 'uint8')).astype(bool)
        bad = increases(ref, old, new, quiet, near_edit)
        for before, ref_band, sigma in zip(old_bands, ref_bands, (6, 9)):
            after = band(new, sigma) - ref_band
            for sign in (1, -1):
                a, b = (before * sign, after * sign)
                if sigma == 6 and np.count_nonzero(paper & (b >= 6)) > np.count_nonzero(paper & (a >= 6)):
                    bad |= paper & (a < 6) & (b >= 6)
                bad |= quiet & (b >= 8) & (b - np.maximum(a, 0) > 4)
        if not bad.any():
            break
        if passes < 3:
            # Restore v5 only around the conflicting neighborhood; a continuous
            # weight avoids a rectangular seam through an unrelated good repair.
            distance = cv2.distanceTransform((~bad).astype('uint8'), cv2.DIST_L2, 5)
            weight = np.clip((distance - (8 + 2 * passes)) / 6, 0, 1)
            native = cv2.resize(np.rint(255 * weight).astype('uint8'), (w, h), interpolation=cv2.INTER_LINEAR)
            for y in range(0, h, 192):
                sl = np.s_[y:y + 192]
                take = native[sl] < 255
                if take.any():
                    factor = native[sl][take, None].astype(np.float32) / 255
                    old_rgb = v5[sl][take].astype(np.float32)
                    proposed[sl][take] = np.rint(old_rgb + (proposed[sl][take] - old_rgb) * factor).astype('uint8')
            passes += 1
            continue
        near = cv2.dilate(bad.astype('uint8'), np.ones((43, 43), 'uint8')).astype(bool)
        reject = np.zeros_like(active)
        reject[np.unique(labels[near])] = True
        reject &= active
        if not reject.any():
            return v5
        active[reject] = False
        remove = reject[labels]
        for y in range(0, h, 192):
            end = min(y + 192, h)
            gy = np.arange(y, end) * size[1] // h
            mask = remove[gy[:, None], gx]
            np.copyto(proposed[y:end], v5[y:end], where=mask[:, :, None])
    return proposed
