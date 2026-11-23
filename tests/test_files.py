import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from sanborn_stain import restore_file, restore_rgb
from sanborn_stain.paper import sha256
from sanborn_stain.dark_spots import VERSION
from tests.test_paper import drawing


class FileOutputTest(unittest.TestCase):
    def test_cli_writes_current_result_and_traceable_masks(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'sheet.tif'
            destination = root / 'result'
            clean, x, y = drawing()
            stain = np.exp(-((x - 86)**2 + (y - 104)**2) / (2 * 21**2))
            raw = np.rint(clean - stain[..., None] * [21, 32, 66]).astype(np.uint8)
            Image.fromarray(raw).save(source)
            response = subprocess.run(
                [sys.executable, '-m', 'sanborn_stain', str(source), str(destination)],
                text=True, capture_output=True,
            )
            self.assertEqual(response.returncode, 0, response.stderr)
            result = destination / 'sheet-cleaned.png'
            expected, changed, review, _ = restore_rgb(raw)
            with Image.open(result) as im:
                self.assertEqual((im.mode, im.size), ('RGB', (360, 260)))
                np.testing.assert_array_equal(np.asarray(im), expected)
            for name, mask in [('changed', changed), ('review', review)]:
                with Image.open(destination / 'diagnostics' / f'{name}.png') as im:
                    np.testing.assert_array_equal(np.asarray(im), mask)
            metadata = json.loads((destination / 'diagnostics' / 'metadata.json').read_text())
            self.assertEqual(metadata['algorithm'], VERSION)
            self.assertEqual(metadata['source_sha256'], sha256(source))
            self.assertEqual(metadata['output_sha256'], sha256(result))
            self.assertEqual(metadata['dimensions'], [360, 260])
            self.assertEqual(set(metadata['implementation_sha256']), {'paper', 'chroma', 'lines', 'spots', 'dark_spots', 'local_guard', 'ring_check'})

    def test_existing_output_directory_is_preserved(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / 'result'
            destination.mkdir()
            sentinel = destination / 'keep.txt'
            sentinel.write_text('previous result')
            with self.assertRaises(FileExistsError):
                restore_file(root / 'missing.png', destination)
            self.assertEqual(sentinel.read_text(), 'previous result')

    def test_rejected_input_leaves_no_output_directory(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'gray.png'
            Image.fromarray(np.zeros((30, 40), np.uint8)).save(source)
            destination = root / 'result'
            with self.assertRaises(ValueError):
                restore_file(source, destination)
            self.assertFalse(destination.exists())
