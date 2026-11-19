import unittest

import numpy as np

from sanborn_stain.chroma import restore_rgb as previous
from sanborn_stain.lines import correct_near_ink, restore_rgb


class LinePaperTest(unittest.TestCase):
    def test_warm_paper_near_ink_matches_clean_paper_more_closely(self):
        y, x = np.ogrid[:260, :360]
        paper = np.array([184, 181, 157], np.uint8)
        spot = np.exp(-((x - 136)**2 + (y - 117)**2) / (2 * 28**2))
        raw = np.rint(paper - spot[..., None] * [19, 31, 61]).astype(np.uint8)
        raw[87:151, 134:136] = [43, 42, 35]
        raw[87:151, 133] = [119, 112, 78]
        raw[87:151, 136] = [122, 115, 81]
        raw[113:116, 125:127] = [91, 74, 56]
        old, _, _, _ = previous(raw)
        new, changed, _, _ = restore_rgb(raw)

        paper_strip = np.s_[101:131, 130:133]
        old_error = np.abs(old[paper_strip].astype(int) - paper).mean()
        new_error = np.abs(new[paper_strip].astype(int) - paper).mean()
        self.assertLess(new_error, .55 * old_error)
        np.testing.assert_array_equal(new[87:151, 133:137], raw[87:151, 133:137])
        np.testing.assert_array_equal(new[113:116, 125:127], raw[113:116, 125:127])
        np.testing.assert_array_equal(changed, np.any(new != raw, axis=-1))

    def test_nearby_yellow_and_pink_print_survive(self):
        y, x = np.ogrid[:260, :360]
        stain = np.exp(-((x - 132)**2 + (y - 110)**2) / (2 * 27**2))
        raw = np.rint([184, 181, 157] - stain[..., None] * [20, 30, 61]).astype(np.uint8)
        raw[85:111, 135:174] = [192, 154, 76]
        raw[111:140, 135:174] = [203, 144, 143]
        raw[85:140, 133:135] = [44, 39, 35]
        new, _, _, _ = restore_rgb(raw)
        np.testing.assert_array_equal(new[85:140, 133:174], raw[85:140, 133:174])

    def test_neutral_shading_around_ink_uses_existing_correction(self):
        y, x = np.ogrid[:220, :300]
        shadow = np.exp(-((x - 80)**2 + (y - 100)**2) / (2 * 35**2))
        raw = np.rint((1 - .19 * shadow[..., None]) * [225, 215, 191]).astype(np.uint8)
        raw[70:125, 79:81] = [54, 52, 45]
        old, _, _, _ = previous(raw)
        new, _, _, _ = restore_rgb(raw)
        np.testing.assert_array_equal(new, old)

    def test_dark_gray_edge_pixels_are_left_alone_in_dark_paper(self):
        raw = np.full((100, 100, 3), [125, 123, 89], dtype=np.uint8)
        raw[:, 49:51] = [52, 49, 45]
        old = np.full_like(raw, [170, 167, 141])
        old[:, 47:53] = raw[:, 47:53]
        changed = np.any(old != raw, axis=-1)
        correct_near_ink(raw, old, changed, [173, 170, 144], (100, 100))
        np.testing.assert_array_equal(old[:, 47:53], raw[:, 47:53])


if __name__ == '__main__':
    unittest.main()
