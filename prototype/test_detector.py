import unittest
from pathlib import Path
import cv2 as cv
import numpy as np
from robust_detect.detector import Classifier, UNIT, plane_groups

ROOT = Path(__file__).resolve().parents[1]


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.ref = cv.imread(str(ROOT/'web/public/grass_block_top.png'), 0)
        self.c = Classifier(self.ref)
        self.quad = UNIT * 128 + 8

    def tile(self, k):
        green = cv.resize(np.rot90(self.ref, -k), (128, 128), interpolation=cv.INTER_NEAREST)
        im = np.zeros((146, 146, 3), np.uint8)
        # Per-channel biome tint and exposure.
        im[8:136, 8:136] = np.stack([green*.5+10, green*.8+15, green*.35+5], axis=2)
        return im

    def test_all_rotations_tint(self):
        for k in range(4):
            result = self.c.classify(self.tile(k), self.quad)
            self.assertIsNotNone(result)
            self.assertEqual(result['rotation'], k)

    def test_small_occluder(self):
        im = self.tile(2)
        im[38:52, 60:74] = [255, 0, 255]
        result = self.c.classify(im, self.quad)
        self.assertIsNotNone(result)
        self.assertEqual(result['rotation'], 2)

    def test_large_occluder_rejected(self):
        im = self.tile(2)
        im[10:130, 10:90] = [255, 0, 255]
        self.assertIsNone(self.c.classify(im, self.quad))

    def test_constant_and_outside_rejected(self):
        self.assertIsNone(self.c.classify(np.full((146, 146, 3), 100, np.uint8), self.quad))
        self.assertIsNone(self.c.classify(self.tile(1), self.quad - 100))

    def test_foreshortened_rejected(self):
        self.assertIsNone(self.c.classify(self.tile(0), UNIT * [128, 6] + 8))

    def test_height_groups_not_merged(self):
        # Three coplanar tiles plus another plane displaced by noninteger image offset.
        seeds = [{'quad': (UNIT * 80 + [10+x*80, 10]).tolist()} for x in range(3)]
        seeds += [{'quad': (UNIT * 80 + [25, 34]).tolist()}]
        groups = plane_groups(seeds)
        self.assertEqual(sorted(map(lambda g: len(g['seed_ids']), groups)), [1, 3])


if __name__ == '__main__':
    unittest.main()
