import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import cv2
from PIL import Image

from sanborn_stain.paper import restore_file, restore_rgb


def drawing():
    """Flat cream paper, a shallow yellow fill and a high-contrast dark wall."""
    clean = np.full((260, 360, 3), [225, 215, 191], np.uint8)
    clean[42:103, 163:250] = [202, 190, 159]
    clean[42:103, 163:166] = [50, 47, 41]
    clean[42:45, 163:250] = [50, 47, 41]
    clean[151:215, 224:306] = [215, 155, 160]
    clean[145:218, 215:219] = [53, 49, 47]
    y, x = np.ogrid[:260, :360]
    return clean, x, y


class RestoreTests(unittest.TestCase):
    def test_clean_low_contrast_fill_is_byte_identical(self):
        raw = np.full((260, 360, 3), [225, 215, 191], np.uint8)
        raw[60:180, 160:280] = [210, 200, 175]
        restored, _, _, _ = restore_rgb(raw)
        np.testing.assert_array_equal(restored, raw)

    def test_high_depth_tiff_is_rejected_before_writing(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'sheet.tif'
            destination = Path(folder) / 'result'
            raw = np.linspace(10000, 50000, 2400, dtype=np.uint16).reshape(40, 60)
            Image.fromarray(raw).save(source)
            with self.assertRaises(ValueError):
                restore_file(source, destination)
            self.assertFalse(destination.exists())
            rgb16 = np.full((40, 60, 3), [10000, 30000, 50000], np.uint16)
            cv2.imwrite(str(source), rgb16)
            with self.assertRaises(ValueError):
                restore_file(source, destination)
            self.assertFalse(destination.exists())

    def test_diffuse_warm_spot_is_attenuated(self):
        clean, x, y = drawing()
        spot = np.exp(-((x-86)**2+(y-104)**2)/(2*21**2))
        raw = np.rint(clean.astype(np.float32) - spot[..., None]*[15, 23, 40]).astype(np.uint8)
        restored, changed, _, _ = restore_rgb(raw)
        inside = spot >= .75
        error = lambda rgb: np.abs(rgb[inside].astype(int) - clean[inside]).mean()
        self.assertLess(error(restored), .7 * error(raw))
        np.testing.assert_array_equal(changed, (restored != raw).any(axis=-1))
        np.testing.assert_array_equal(restored[42:103, 163:250], raw[42:103, 163:250])
        np.testing.assert_array_equal(restored[145:218, 215:306], raw[145:218, 215:306])

    def test_broad_shadow_is_attenuated_without_whitening_clean_paper(self):
        clean, x, y = drawing()
        shadow = np.exp(-((x-71)**2+(y-183)**2)/(2*35**2))
        raw = np.rint(clean.astype(np.float32) * (1 - .15*shadow[..., None])).astype(np.uint8)
        restored, _, _, _ = restore_rgb(raw)
        inside = shadow >= .75
        error = lambda rgb: np.abs(rgb[inside].astype(int) - clean[inside]).mean()
        self.assertLess(error(restored), .7 * error(raw))
        np.testing.assert_array_equal(restored[:32, 320:], raw[:32, 320:])

    def test_clean_map_is_byte_identical(self):
        raw, _, _ = drawing()
        restored, changed, _, _ = restore_rgb(raw)
        self.assertEqual(restored.shape, raw.shape)
        self.assertEqual(restored.dtype, np.uint8)
        np.testing.assert_array_equal(restored, raw)
        self.assertFalse(changed.any())

    def test_uncertain_dark_and_saturated_marks_pass_through(self):
        raw, _, _ = drawing()
        raw[72:103, 45:91] = [112, 72, 38]
        raw[132:180, 70:130] = [125, 175, 195]
        restored, changed, _, _ = restore_rgb(raw)
        self.assertFalse(changed[72:103, 45:91].any())
        self.assertFalse(changed[132:180, 70:130].any())
        np.testing.assert_array_equal(restored, raw)


if __name__ == '__main__':
    unittest.main()
