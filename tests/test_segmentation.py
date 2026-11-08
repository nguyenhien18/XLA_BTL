"""Behavioral checks: Otsu reference agreement and separation of touching objects."""
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segmentation import otsu_threshold, segment_otsu, segment_watershed


class SegmentationTests(unittest.TestCase):
    def test_otsu_threshold_agrees_with_reference_on_actual_demo(self):
        for name in (ROOT / "data/splits/demo.txt").read_text().split():
            with self.subTest(image=name), Image.open(ROOT / "data/prepared/images" / f"{name}.png") as image:
                array = np.array(image)
                self.assertEqual(otsu_threshold(array), int(threshold_otsu(array)))

    def test_constant_images_produce_no_false_instances(self):
        for value in [0, 71, 255]:
            array = np.full((16, 16), value, dtype=np.uint8)
            self.assertEqual(otsu_threshold(array), value)
            self.assertEqual(int(segment_otsu(array)[0].max()), 0)
            self.assertEqual(int(segment_watershed(array)[0].max()), 0)

    def test_watershed_separates_touching_disks(self):
        y, x = np.indices((64, 64))
        mask = ((x - 24) ** 2 + (y - 32) ** 2 <= 12 ** 2) | ((x - 40) ** 2 + (y - 32) ** 2 <= 12 ** 2)
        array = (mask * 255).astype(np.uint8)
        self.assertEqual(int(segment_otsu(array)[0].max()), 1)
        result = segment_watershed(array, min_distance=5)[0]
        self.assertEqual(len(np.unique(result[result > 0])), 2)
        np.testing.assert_array_equal(result > 0, mask)


if __name__ == "__main__":
    unittest.main(verbosity=2)
