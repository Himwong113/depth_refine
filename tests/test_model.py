from __future__ import annotations

import unittest

import torch

from model import DepthRefinementUNet


class DepthRefinementUNetTests(unittest.TestCase):
    def test_calibrated_tof_forward_and_backward(self) -> None:
        model = DepthRefinementUNet(base_channels=8, positive_output=True)
        image = torch.randn(2, 3, 31, 37)
        tof = torch.zeros(2, 3, 31, 37)
        tof[:, 0, 4:27, 5:32] = 2.5
        tof[:, 1, 4:27, 5:32] = 0.1
        tof[:, 2, 4:27, 5:32] = 1.0

        output = model(image, tof)

        self.assertEqual(output.shape, (2, 1, 31, 37))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((output > 0).all())
        output.square().mean().backward()
        self.assertIsNotNone(model.stem[0].weight.grad)
        self.assertIsNotNone(model.depth_head.weight.grad)
        self.assertIsNotNone(model.confidence_head.weight.grad)

    def test_legacy_low_resolution_depth_remains_supported(self) -> None:
        model = DepthRefinementUNet(base_channels=8)
        image = torch.randn(1, 3, 33, 41)
        sparse_depth = torch.zeros(1, 1, 8, 8)
        sparse_depth[:, :, 2:6, 2:6] = 3.0

        output = model(image, sparse_depth)

        self.assertEqual(output.shape, (1, 1, 33, 41))
        self.assertTrue(torch.isfinite(output).all())

    def test_default_model_stays_below_parameter_budget(self) -> None:
        model = DepthRefinementUNet()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(parameter_count, 1_000_000)


if __name__ == "__main__":
    unittest.main()
