from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch

from data import ZJUL5Dataset, build_calibrated_tof_tokens, rasterize_calibrated_tof


class CalibratedToFRasterizationTests(unittest.TestCase):
    def test_out_of_range_ground_truth_is_excluded_not_clamped(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample_path = root / "sample.h5"
            rgb = np.zeros((8, 10, 3), dtype=np.uint8)
            depth = np.ones((8, 10), dtype=np.float32)
            depth[0, :5] = np.array([0.05, 0.1, 10.0, 12.0, np.nan])
            hist_data = np.zeros((64, 2), dtype=np.float32)
            rect_data = np.zeros((64, 4), dtype=np.int32)
            mask = np.zeros(64, dtype=np.bool_)
            with h5py.File(sample_path, "w") as sample_file:
                sample_file["rgb"] = rgb
                sample_file["depth"] = depth
                sample_file["hist_data"] = hist_data
                sample_file["fr"] = rect_data
                sample_file["mask"] = mask
            (root / "data.json").write_text(
                json.dumps({"train": [{"filename": sample_path.name}]}),
                encoding="utf-8",
            )
            sample = ZJUL5Dataset(
                root,
                split="train",
                normalize_image=False,
                min_depth=0.1,
                max_depth=10.0,
            )[0]

        self.assertEqual(
            sample["target_valid_mask"][0, 0, :5].tolist(),
            [False, True, True, False, False],
        )
        torch.testing.assert_close(
            sample["target_depth"][0, 0, :5],
            torch.tensor([0.0, 0.1, 10.0, 0.0, 0.0]),
        )

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

    def test_tokens_keep_metric_values_and_normalized_clipped_geometry(self) -> None:
        hist_data = np.zeros((64, 2), dtype=np.float32)
        rect_data = np.zeros((64, 4), dtype=np.int32)
        mask = np.zeros(64, dtype=np.bool_)
        hist_data[0] = (2.5, 0.2)
        rect_data[0] = (-2, 3, 5, 9)
        mask[0] = True
        hist_data[1] = (4.0, 0.3)
        rect_data[1] = (5, 0, 9, 4)

        tokens = build_calibrated_tof_tokens(
            hist_data, rect_data, mask, image_size=(8, 10)
        )

        self.assertEqual(tokens.shape, (64, 7))
        torch.testing.assert_close(
            tokens[0],
            torch.tensor([2.5, 0.2, 1.0, 0.3125, 0.6, 0.625, 0.6]),
        )
        self.assertEqual(tokens[1, 0].item(), 0.0)
        self.assertEqual(tokens[1, 1].item(), 0.0)
        self.assertEqual(tokens[1, 2].item(), 0.0)

    def test_horizontal_flip_updates_token_geometry(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample_path = root / "sample.h5"
            manifest_path = root / "data.json"
            rgb = np.zeros((8, 10, 3), dtype=np.uint8)
            depth = np.ones((8, 10), dtype=np.float32)
            hist_data = np.zeros((64, 2), dtype=np.float32)
            rect_data = np.zeros((64, 4), dtype=np.int32)
            mask = np.zeros(64, dtype=np.bool_)
            hist_data[0] = (2.0, 0.1)
            rect_data[0] = (0, 1, 8, 3)
            mask[0] = True
            with h5py.File(sample_path, "w") as sample_file:
                sample_file["rgb"] = rgb
                sample_file["depth"] = depth
                sample_file["hist_data"] = hist_data
                sample_file["fr"] = rect_data
                sample_file["mask"] = mask
            manifest_path.write_text(
                json.dumps({"train": [{"filename": sample_path.name}]}),
                encoding="utf-8",
            )
            dataset = ZJUL5Dataset(
                root,
                split="train",
                normalize_image=False,
                augment=True,
                horizontal_flip_probability=1.0,
                photometric_probability=0.0,
            )

            sample = dataset[0]

        self.assertAlmostEqual(sample["tof_tokens"][0, 4].item(), 0.8)
        torch.testing.assert_close(
            sample["tof_features"][2, :, 7:9], torch.ones(8, 2)
        )


if __name__ == "__main__":
    unittest.main()
