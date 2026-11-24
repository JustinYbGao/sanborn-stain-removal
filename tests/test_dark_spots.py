from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from sanborn_stain import restore_rgb
from sanborn_stain import web


class DarkSpotDefaultTest(TestCase):
    def test_default_repairs_dark_hole_and_updates_masks(self):
        raw = np.full((600, 800, 3), 150, np.uint8)
        previous = np.full_like(raw, 175)
        cv2.circle(previous, (300, 300), 25, (150, 150, 150), -1)
        changed = np.any(previous != raw, axis=2)
        review = ~changed
        with patch('sanborn_stain.spots.restore_rgb', return_value=(previous, changed, review, {})):
            result, mask, remaining, diagnostics = restore_rgb(raw)
        self.assertGreater(int(result[300, 300, 0]), 168)
        np.testing.assert_array_equal(mask, np.any(result != raw, axis=2))
        self.assertFalse(remaining[300, 300])
        self.assertFalse((mask & remaining).any())
        self.assertEqual(diagnostics['algorithm'], 'paper-correction-with-guarded-texture-v7')
        self.assertIs(web.restore_rgb, restore_rgb)
