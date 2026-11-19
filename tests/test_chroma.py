import unittest

import numpy as np

from sanborn_stain.paper import lab_of, restore_rgb as previous_restore
from sanborn_stain.chroma import restore_rgb


class ResidualColorTest(unittest.TestCase):
    def test_brown_stain_is_neutralized_toward_paper_without_expanding_the_edit(self):
        paper = np.array([184, 181, 157], dtype=np.uint8)
        y, x = np.ogrid[:260, :360]
        stain = np.exp(-((x - 86)**2 + (y - 104)**2) / (2 * 21**2))
        raw = np.rint(paper - stain[..., None] * [21, 32, 66]).astype(np.uint8)
        raw[42:103, 163:250] = [202, 190, 159]
        raw[42:103, 163:166] = [50, 47, 41]
        raw[42:45, 163:250] = [50, 47, 41]
        old, old_changed, _, _ = previous_restore(raw)
        cleaned, changed, _, _ = restore_rgb(raw)
        inside = stain >= .9
        reference_b = lab_of(paper.reshape(1, 1, 3))[0, 0, 2]
        old_error = np.median(np.abs(lab_of(old)[inside, 2] - reference_b))
        error = np.median(np.abs(lab_of(cleaned)[inside, 2] - reference_b))
        self.assertLess(error, 0.35 * old_error)
        np.testing.assert_array_equal(cleaned[~old_changed], raw[~old_changed])
        np.testing.assert_array_equal(cleaned[42:103, 163:250], raw[42:103, 163:250])
        np.testing.assert_array_equal(changed, np.any(cleaned != raw, axis=2))

    def test_neutral_shading_uses_the_previous_correction(self):
        y, x = np.ogrid[:220, :300]
        shadow = np.exp(-((x - 80)**2 + (y - 100)**2) / (2 * 35**2))
        raw = np.rint((1 - .19 * shadow[..., None]) * [225, 215, 191]).astype(np.uint8)
        old, _, _, _ = previous_restore(raw)
        cleaned, _, _, _ = restore_rgb(raw)
        np.testing.assert_array_equal(cleaned, old)


if __name__ == '__main__':
    unittest.main()
