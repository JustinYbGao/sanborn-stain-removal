"""File output for the current stain correction pipeline."""

import json
from pathlib import Path
import shutil

from PIL import Image

from . import chroma, lines, paper, spots


def restore_file(source, destination):
    """Write the corrected PNG, two diagnostic masks, and hashes to a new folder."""
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Choose a new output directory: {destination}')
    source_hash = paper.sha256(source)
    rgb, source_metadata = paper.read_rgb8(source)
    corrected, changed, review, diagnostics = spots.restore_rgb(rgb)
    implementation_hashes = {
        name: paper.sha256(Path(module.__file__))
        for name, module in (('paper', paper), ('chroma', chroma), ('lines', lines), ('spots', spots))
    }
    destination.mkdir(parents=True, exist_ok=False)
    try:
        target = destination / f'{source.stem}-cleaned.png'
        Image.fromarray(corrected).save(target)
        audit = destination / 'diagnostics'
        audit.mkdir()
        for name, mask in (('changed', changed), ('review', review)):
            Image.fromarray(mask).save(audit / f'{name}.png')
        metadata = {
            'algorithm': spots.VERSION,
            'implementation_sha256': implementation_hashes,
            'source': source.name,
            'source_sha256': source_hash,
            **source_metadata,
            'output': target.name,
            'output_sha256': paper.sha256(target),
            'dimensions': list(rgb.shape[1::-1]),
            'encoding': 'RGB uint8, decoded raster coordinates',
            'changed_pixels': int(changed.sum()),
            'review_pixels': int(review.sum()),
            **diagnostics,
        }
        (audit / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException:
        shutil.rmtree(destination)
        raise
    return target
