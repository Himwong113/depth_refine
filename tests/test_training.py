from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from training import evaluate


class FixedPredictionModel(nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(
        self,
        image: torch.Tensor,
        tof_features: torch.Tensor,
        tof_tokens: torch.Tensor,
    ) -> torch.Tensor:
        del tof_features, tof_tokens
        return self.prediction.expand(image.shape[0], -1, -1, -1)


class EvaluationMetricTests(unittest.TestCase):
    def test_reports_full_corrected_metrics_by_region_and_depth_bin(self) -> None:
        target = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        prediction = torch.tensor([[[[1.0, 1.0, 3.0, 3.0]]]])
        tokens = torch.zeros(1, 64, 7)
        tokens[0, 0] = torch.tensor([1.0, 0.1, 1.0, 0.5, 0.25, 1.0, 0.5])
        batch = {
            "image": torch.zeros(1, 3, 1, 4),
            "sparse_depth": torch.zeros(1, 1, 1, 4),
            "tof_tokens": tokens,
            "target_depth": target,
            "target_valid_mask": torch.ones_like(target, dtype=torch.bool),
        }

        metrics = evaluate(
            FixedPredictionModel(prediction),
            [batch],
            torch.device("cpu"),
        )

        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(0.5))
        self.assertAlmostEqual(metrics["abs_rel"], 0.1875)
        self.assertAlmostEqual(metrics["delta1"], 0.5)
        for metric in ("mae", "rmse", "abs_rel", "delta1"):
            self.assertAlmostEqual(metrics[f"image_{metric}"], metrics[metric])

        self.assertAlmostEqual(metrics["mae_inside_tof"], 0.5)
        self.assertAlmostEqual(metrics["rmse_inside_tof"], math.sqrt(0.5))
        self.assertAlmostEqual(metrics["abs_rel_inside_tof"], 0.25)
        self.assertAlmostEqual(metrics["delta1_inside_tof"], 0.5)
        self.assertAlmostEqual(metrics["mae_outside_tof"], 0.5)
        self.assertAlmostEqual(metrics["rmse_outside_tof"], math.sqrt(0.5))
        self.assertAlmostEqual(metrics["abs_rel_outside_tof"], 0.125)
        self.assertAlmostEqual(metrics["delta1_outside_tof"], 0.5)

        self.assertAlmostEqual(metrics["rmse_0_2m"], 0.0)
        self.assertAlmostEqual(metrics["mae_2_4m"], 0.5)
        self.assertAlmostEqual(metrics["abs_rel_2_4m"], 0.25)
        self.assertAlmostEqual(metrics["delta1_2_4m"], 0.5)
        self.assertAlmostEqual(metrics["rmse_4_6m"], 1.0)
        self.assertAlmostEqual(metrics["boundary_accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["boundary_rmse"], math.sqrt(0.5))


if __name__ == "__main__":
    unittest.main()
