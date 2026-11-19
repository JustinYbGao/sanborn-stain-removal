"""Conservative paper correction before Sanborn building over-segmentation."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WORK_SIDE = 1600
ROW_CHUNK = 192
EDGE_LIMIT = 3.0
VERSION = 'paper-connected-gain-v2'

_ENCODED = np.arange(256, dtype=np.float32) / 255
_LINEAR = np.where(_ENCODED <= .04045, _ENCODED / 12.92,
                   ((_ENCODED + .055) / 1.055) ** 2.4)
_LOG = np.log(np.maximum(_LINEAR, 1e-5))
_KERNEL = np.ones((3, 3), np.uint8)


def lab_of(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def paper_corridor(lab, paper):
    """Admit warm or dim shifts of this paper while excluding strong ink colors."""
    luma, red, yellow = np.moveaxis(lab - paper, -1, 0)
    return ((luma >= -25) & (luma <= 7) & (yellow >= -5) & (yellow <= 27)
            & (red >= -5) & (red <= 4 + .45 * np.maximum(yellow, 0)))


def quiet_pixels(lab):
    local_range = cv2.morphologyEx(lab, cv2.MORPH_GRADIENT, _KERNEL)
    return ((local_range[..., 0] <= EDGE_LIMIT) & (local_range[..., 1] <= 6)
            & (local_range[..., 2] <= 7))


def normalized_blur(values, support, sigma):
    weights = support.astype(np.float32)
    divisor = cv2.GaussianBlur(weights, (0, 0), sigma)
    numerator = cv2.GaussianBlur(values * weights[..., None], (0, 0), sigma)
    return numerator / np.maximum(divisor[..., None], 1e-5)


def gain_field(work):
    """Estimate gains only on paper connected to a clean-looking local anchor."""
    lab = lab_of(work)
    h, w = lab.shape[:2]
    border_y, border_x = max(1, h // 25), max(1, w // 25)
    quiet = quiet_pixels(lab)
    samples = lab[border_y:-border_y, border_x:-border_x][quiet[border_y:-border_y, border_x:-border_x]]
    if not len(samples):
        return None, None, None
    bright = samples[samples[:, 0] >= np.quantile(samples[:, 0], .65)]
    chroma = np.linalg.norm(bright[:, 1:], axis=1)
    paper = np.median(bright[chroma <= np.quantile(chroma, .45)], axis=0)
    corridor = paper_corridor(lab, paper)
    corridor &= quiet
    anchor = corridor & (np.abs(lab[..., 0] - paper[0]) <= 3)
    anchor &= np.linalg.norm(lab[..., 1:] - paper[1:], axis=2) <= 3
    if anchor.mean() < .01:
        return None, None, None
    n, objects = cv2.connectedComponents(corridor.astype(np.uint8), connectivity=4)
    supported = np.zeros(n, bool)
    supported[np.unique(objects[anchor])] = True
    supported[0] = False
    reachable = supported[objects]

    # Estimate smooth brightness separately from more localized color shifts.
    clean_rgb = np.median(work[anchor], axis=0).round().astype(np.uint8)
    log_rgb = _LOG[work]
    wide = normalized_blur(log_rgb, reachable, max(2.0, max(h, w) / 105))
    small_sigma = max(1.1, max(h, w) / 780)
    local = normalized_blur(np.concatenate((log_rgb, lab[..., 2:3]), axis=2),
                            reachable, small_sigma)
    ref = _LOG[clean_rgb]
    mean_gain = (ref - wide).mean(axis=2)
    shade = np.clip(mean_gain - .035, 0, .39)
    warm = np.clip((local[..., 3] - paper[2] - 2) / 5, 0, 1)
    local_gain = np.clip(ref - local[..., :3] - .022, 0, .52)
    gains = (1 - warm[..., None]) * shade[..., None] + warm[..., None] * local_gain
    # Fade the effect as it approaches an ink, color or canvas boundary.
    interior = cv2.distanceTransform(np.pad(reachable, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    gains *= np.clip((interior - 1) / 4, 0, 1)[..., None]
    return paper, gains.astype(np.float32), clean_rgb


def restore_rgb(rgb):
    """Return corrected uint8 RGB, exact change mask, review mask and diagnostics."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected an H×W×3 uint8 RGB array.')
    height, width = rgb.shape[:2]
    if min(height, width) < 8 or max(height, width) >= 32767:
        raise ValueError('Image dimensions must be between 8 and 32766 pixels.')
    scale = min(1., WORK_SIDE / max(height, width))
    work_size = (max(8, round(width * scale)), max(8, round(height * scale)))
    work = cv2.resize(rgb, work_size, interpolation=cv2.INTER_AREA)
    paper, gains, clean_rgb = gain_field(work)
    out = rgb.copy()
    changed = np.zeros((height, width), bool)
    review = np.zeros_like(changed)
    if gains is None:
        return out, changed, review, {'paper_rgb': None, 'reason': 'insufficient_paper_anchors'}

    sample_x = (np.arange(width, dtype=np.float32) + .5) * (work_size[0] / width) - .5
    for top in range(0, height, ROW_CHUNK):
        bottom = min(height, top + ROW_CHUNK)
        # One extra row on each side keeps the native edge check continuous.
        lo, hi = max(0, top - 1), min(height, bottom + 1)
        source = rgb[top:bottom]
        padded_lab = lab_of(rgb[lo:hi])
        rows = slice(top - lo, bottom - lo)
        lab = padded_lab[rows]
        quiet = quiet_pixels(padded_lab)[rows]
        sample_y = (np.arange(top, bottom, dtype=np.float32) + .5) * (work_size[1] / height) - .5
        map_x, map_y = np.meshgrid(sample_x, sample_y)
        gain = cv2.remap(gains, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Native checks cover lines that can disappear on the estimation grid.
        dim = np.clip((paper[0] - lab[..., 0] - 1.3) / 4, 0, 1)
        eligible = paper_corridor(lab, paper) & quiet & (dim > 0) & (gain.max(axis=2) > .001)
        if eligible.any():
            values = _LINEAR[source[eligible]] * np.exp(gain[eligible] * dim[eligible, None])
            values = np.clip(values, 0, 1)
            encoded = np.where(values <= .0031308, 12.92 * values,
                               1.055 * values ** (1 / 2.4) - .055)
            out[top:bottom][eligible] = np.rint(encoded * 255).astype(np.uint8)
        changed[top:bottom] = np.any(out[top:bottom] != source, axis=2)
        plausible = paper_corridor(lab, paper) & (lab[..., 2] > paper[2] + 4) & (dim > .5)
        review[top:bottom] = plausible & ~changed[top:bottom]
    return out, changed, review, {'paper_rgb': clean_rgb.tolist(), 'work_dimensions': list(work_size)}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rgb8(source):
    """Validate source precision before Pillow can coerce it into RGB uint8."""
    with Image.open(source) as image:
        if image.mode != 'RGB':
            raise ValueError('Use an original 8-bit-per-channel RGB image.')
        if image.format == 'TIFF':
            bits = image.tag_v2.get(258, (1,))
            if isinstance(bits, int):
                bits = (bits,)
            if any(value != 8 for value in bits):
                raise ValueError('Only 8-bit-per-channel RGB TIFF is supported.')
        elif image.format == 'PNG':
            with source.open('rb') as stream:
                if stream.read(25)[24] != 8:
                    raise ValueError('Only 8-bit-per-channel RGB PNG is supported.')
        elif image.format == 'JPEG2000':
            with source.open('rb') as stream:
                header = stream.read(65536)
            position = header.find(b'ihdr')
            if (position < 4 or int.from_bytes(header[position-4:position], 'big') != 22
                    or len(header) < position + 18 or header[position+14] != 7):
                raise ValueError('Use a JP2 container with uniform unsigned 8-bit RGB components.')
        elif image.format != 'JPEG' or getattr(image, 'bits', 8) != 8:
            raise ValueError('Supported inputs are 8-bit RGB TIFF, JP2, PNG, and JPEG.')
        metadata = {'source_format': image.format, 'source_mode': image.mode,
                    'source_bits_per_channel': 8, 'selected_frame': 0,
                    'stored_orientation': int(image.getexif().get(274, 1))}
        return np.asarray(image), metadata


def restore_file(source, destination):
    """Write a same-stem RGB PNG and diagnostics to a new directory."""
    if destination.exists():
        raise FileExistsError(f'Choose a new output directory: {destination}')
    source_hash = sha256(source)
    rgb, source_metadata = read_rgb8(source)
    out, changed, review, diagnostics = restore_rgb(rgb)
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / f'{source.stem}.png'
    Image.fromarray(out).save(target)
    audit = destination / 'diagnostics'
    audit.mkdir()
    for name, mask in (('changed', changed), ('review', review)):
        Image.fromarray(mask).save(audit / f'{name}.png')
    result = {
        'algorithm': VERSION,
        'implementation_sha256': sha256(Path(__file__)),
        'source': str(source.resolve()),
        'source_sha256': source_hash,
        **source_metadata,
        'output': target.name,
        'output_sha256': sha256(target),
        'dimensions': list(rgb.shape[1::-1]),
        'encoding': 'RGB uint8, decoded raster coordinates',
        'changed_pixels': int(changed.sum()),
        'review_pixels': int(review.sum()),
        **diagnostics,
    }
    (audit / 'metadata.json').write_text(json.dumps(result, indent=2) + '\n')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original RGB TIFF, JP2, PNG, or JPEG')
    parser.add_argument('output_directory', type=Path, help='New directory for the RGB PNG and diagnostics')
    args = parser.parse_args()
    print(restore_file(args.source, args.output_directory))


if __name__ == '__main__':
    main()
