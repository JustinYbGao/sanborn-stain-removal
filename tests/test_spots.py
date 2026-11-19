import unittest

import numpy as np

from sanborn_stain.spots import correct_spots


class SpotCorrectionTest(unittest.TestCase):
    def test_fills_compact_gap_in_paper_correction(self):
        raw = np.full((180, 220, 3), [155, 148, 108], np.uint8)
        expected = np.array([180, 176, 151])
        old = np.full_like(raw, expected)
        y, x = np.ogrid[:180, :220]
        center = (y - 75)**2 + (x - 100)**2
        halo = np.exp(-center / (2 * 3.8**2))
        old[:] = np.rint(expected - halo[..., None] * [24, 27, 41]).astype(np.uint8)
        mask = np.any(old != raw, axis=-1)
        before = old.copy()
        count = correct_spots(raw, old, mask, expected.tolist())
        core = center < 9
        self.assertGreater(count, 0)
        self.assertLess(np.abs(old[core].astype(int) - expected).mean(),
                        .3 * np.abs(before[core].astype(int) - expected).mean())
        np.testing.assert_array_equal(mask, np.any(old != raw, axis=2))

    def test_preserves_ink_brown_print_and_an_unsupported_dark_fleck(self):
        raw = np.full((180, 220, 3), [155, 148, 108], np.uint8)
        old = np.full_like(raw, [180, 176, 151])
        raw[50:100, 43:45] = [38, 37, 34]
        raw[48:56, 95:103] = [112, 82, 57]
        raw[115:123, 125:133] = [153, 126, 68]
        old[48:56, 95:103] = raw[48:56, 95:103]
        old[115:123, 125:133] = raw[115:123, 125:133]
        old[50:100, 43:45] = raw[50:100, 43:45]
        mask = np.any(old != raw, axis=-1)
        before = old.copy()
        correct_spots(raw, old, mask, [180, 176, 151])
        np.testing.assert_array_equal(old, before)

    def test_preserves_uncorrected_original_paper(self):
        raw = np.full((120, 150, 3), [180, 176, 151], np.uint8)
        raw[50:56, 60:66] = [161, 150, 104]
        old = raw.copy()
        mask = np.zeros(raw.shape[:2], bool)
        self.assertEqual(correct_spots(raw, old, mask, [180, 176, 151]), 0)
        np.testing.assert_array_equal(old, raw)


if __name__ == '__main__':
    unittest.main()
