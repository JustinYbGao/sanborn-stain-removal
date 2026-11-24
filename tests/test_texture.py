import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sanborn_stain import dark_spots, texture


class PaperDetailTest(unittest.TestCase):
    def test_paper_gain_texture_reduces_dark_specks_and_preserves_ink(self):
        raw = np.full((240, 260, 3), 157, np.uint8)
        cv2.line(raw, (15, 50), (235, 175), (64, 65, 66), 1)
        raw[25:43, 35:66] = (206, 159, 110)
        corrected = raw + 14
        rng = np.random.default_rng(374)
        lost_gain = rng.random(raw.shape[:2]) < 0.18
        corrected[lost_gain] = raw[lost_gain]
        result = texture.repair_grain(raw, corrected)
        paper = raw.min(2) == 157
        def extra_dark(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return (cv2.medianBlur(gray, 7).astype(np.int16) - gray) >= 8

        self.assertLess(np.count_nonzero(paper & extra_dark(result)),
                        np.count_nonzero(paper & extra_dark(corrected)) // 2)
        np.testing.assert_array_equal(result[~paper], corrected[~paper])

    def test_even_paper_does_not_trigger_a_blotch_proposal(self):
        raw = np.full((192, 193, 3), 161, np.uint8)
        with patch.object(dark_spots, 'screen', side_effect=AssertionError('Unneeded protection pass')):
            output, changed, review, _ = dark_spots.restore_rgb(raw)
        np.testing.assert_array_equal(output, raw)
        self.assertFalse(changed.any())
        self.assertFalse((changed & review).any())
