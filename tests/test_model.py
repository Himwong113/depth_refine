from __future__ import annotations

import unittest

import torch

from model import DepthRefinementUNet, FeaturePairQKAttention


class FeaturePairQKAttentionTests(unittest.TestCase):
    def test_different_channels_and_resolutions_produce_channel_attention(self) -> None:
        layer = FeaturePairQKAttention(
            query_channels=4,
            key_channels=8,
            attention_channels=6,
            max_tokens=16,
        )
        query_features = torch.randn(2, 4, 31, 37, requires_grad=True)
        key_features = torch.randn(2, 8, 15, 18, requires_grad=True)

        attention = layer(query_features, key_features)

        self.assertEqual(attention.shape, (2, 6, 6))
        self.assertTrue(torch.isfinite(attention).all())
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones(2, 6),
            rtol=1e-5,
            atol=1e-6,
        )
        attention.square().mean().backward()
        self.assertIsNotNone(query_features.grad)
        self.assertIsNotNone(key_features.grad)


class DepthRefinementUNetTests(unittest.TestCase):
    def test_adjacent_encoder_attention_reaches_all_left_decoder_nodes(self) -> None:
        model = DepthRefinementUNet(
            base_channels=4,
            attention_channels=4,
            max_attention_tokens=32,
            positive_output=True,
        )
        image = torch.randn(1, 3, 31, 37)
        sparse_depth = torch.rand(1, 1, 8, 8)

        output = model(image, sparse_depth)

        self.assertEqual(output.shape, (1, 1, 31, 37))
        self.assertTrue(torch.isfinite(output).all())
        output.mean().backward()
        for module in (
            model.encoder_qk_x21,
            model.encoder_qk_x32,
            model.encoder_qk_x43,
        ):
            self.assertIsNotNone(module.query_encoder[0].weight.grad)
            self.assertIsNotNone(module.key_encoder[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
