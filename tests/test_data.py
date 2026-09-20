from __future__ import annotations

import unittest

import numpy as np
import torch

from data import rasterize_calibrated_tof


class CalibratedToFRasterizationTests(unittest.TestCase):
    def test_rectangles_are_clipped_and_invalid_zones_stay_empty(self) -> None:
        hist_data = np.zeros((64, 2), dtype=np.float32)
        rect_data = np.zeros((64, 4), dtype=np.int32)
        mask = np.zeros(64, dtype=np.bool_)
        hist_data[0] = (2.5, 0.2)
        rect_data[0] = (-2, 3, 5, 9)
        mask[0] = True
        hist_data[1] = (8.0, 0.4)
        rect_data[1] = (5, 0, 9, 4)

        features = rasterize_calibrated_tof(
            hist_data,
            rect_data,
            mask,
            image_size=(8, 10),
        )

        self.assertEqual(features.shape, (3, 8, 10))
        torch.testing.assert_close(features[0, 0:5, 3:9], torch.full((5, 6), 2.5))
        torch.testing.assert_close(features[1, 0:5, 3:9], torch.full((5, 6), 0.2))
        torch.testing.assert_close(features[2, 0:5, 3:9], torch.ones(5, 6))
        self.assertEqual(features[2, 5:, :].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
