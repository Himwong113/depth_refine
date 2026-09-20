from __future__ import annotations

import unittest

import torch

from model import DepthRefinementUNet, GeometryAwareToFAttention


class DepthRefinementUNetTests(unittest.TestCase):
    def test_calibrated_tof_forward_and_backward(self) -> None:
        model = DepthRefinementUNet(base_channels=8, positive_output=True)
        image = torch.randn(2, 3, 31, 37)
        tof = torch.zeros(2, 3, 31, 37)
        tof[:, 0, 4:27, 5:32] = 2.5
        tof[:, 1, 4:27, 5:32] = 0.1
        tof[:, 2, 4:27, 5:32] = 1.0
        tokens = torch.zeros(2, 64, 7)
        tokens[:, 0] = torch.tensor([2.5, 0.1, 1.0, 0.5, 0.5, 0.75, 0.75])

        output = model(image, tof, tokens)

        self.assertEqual(output.shape, (2, 1, 31, 37))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((output > 0).all())
        output.square().mean().backward()
        self.assertIsNotNone(model.stem[0].weight.grad)
        self.assertIsNotNone(model.depth_head.weight.grad)
        self.assertIsNotNone(model.tof_conditioner.query_projection.weight.grad)
        self.assertIsNotNone(model.tof_conditioner.token_projection.weight.grad)
        self.assertIsNotNone(model.tof_conditioner.output_projection.weight.grad)
        self.assertGreater(
            model.tof_conditioner.query_projection.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            model.tof_conditioner.token_projection.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_invalid_tokens_cannot_change_attention_output(self) -> None:
        attention = GeometryAwareToFAttention(8, attention_dim=4, num_heads=2)
        rgb_features = torch.randn(1, 8, 3, 5)
        tokens = torch.zeros(1, 64, 7)
        tokens[:, 0] = torch.tensor([2.0, 0.1, 1.0, 0.5, 0.5, 0.5, 0.5])

        changed_invalid_tokens = tokens.clone()
        changed_invalid_tokens[:, 1:, 0:2] = 1000.0
        changed_invalid_tokens[:, 1:, 3:7] = torch.rand(1, 63, 4)

        output = attention(rgb_features, tokens)
        changed_output = attention(rgb_features, changed_invalid_tokens)
        torch.testing.assert_close(output, changed_output)

    def test_all_invalid_tokens_use_safe_null_condition(self) -> None:
        attention = GeometryAwareToFAttention(8, attention_dim=4, num_heads=2)
        rgb_features = torch.randn(2, 8, 3, 5)
        tokens = torch.zeros(2, 64, 7)
        tokens[0, 0] = torch.tensor(
            [float("inf"), 0.1, 1.0, 0.5, 0.5, 0.5, 0.5]
        )
        tokens[1, 0] = torch.tensor(
            [2.0, float("nan"), 1.0, 0.5, 0.5, 0.5, 0.5]
        )

        output = attention(rgb_features, tokens)

        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, rgb_features)

    def test_geometry_bias_prefers_nearby_rectangles(self) -> None:
        attention = GeometryAwareToFAttention(8, attention_dim=2, num_heads=1)
        rgb_features = torch.zeros(1, 8, 1, 2)
        tokens = torch.zeros(1, 64, 7)
        tokens[:, 0] = torch.tensor([2.0, 0.1, 1.0, 0.5, 0.25, 1.0, 0.25])
        tokens[:, 1] = torch.tensor([2.0, 0.1, 1.0, 0.5, 0.75, 1.0, 0.25])
        with torch.no_grad():
            attention.query_projection.weight.zero_()

        _, weights = attention._attention(rgb_features, tokens)

        self.assertGreater(weights[0, 0, 0, 0], weights[0, 0, 0, 1])
        self.assertGreater(weights[0, 0, 1, 1], weights[0, 0, 1, 0])

    def test_output_has_no_direct_rectangular_tof_blend(self) -> None:
        model = DepthRefinementUNet(base_channels=8).eval()
        image = torch.zeros(1, 3, 32, 48)
        tof = torch.zeros(1, 3, 32, 48)
        tof[:, 0, :, :24] = 1.0
        tof[:, 0, :, 24:] = 3.0
        tof[:, 2] = 1.0
        with torch.no_grad():
            model.depth_head.weight.zero_()
            model.depth_head.bias.zero_()

        with torch.inference_mode():
            output = model(image, tof)

        # The zero-initialized residual head starts from one global scale value;
        # it must not copy the 1 m / 3 m ToF zone boundary into the output.
        left = output[..., :24].mean()
        right = output[..., 24:].mean()
        torch.testing.assert_close(left, right)

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
        self.assertLess(parameter_count, 130_000)


if __name__ == "__main__":
    unittest.main()
