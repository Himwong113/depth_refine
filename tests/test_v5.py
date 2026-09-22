from __future__ import annotations

import unittest
import warnings

import torch

from distillation import (
    FeatureDistillationAdapters,
    _confidence_weighted_mean,
    student_distillation_loss_v5,
)
from main import build_model
from model import DepthOutput, EfficientFormerDepthStudent


def calibrated_tokens(batch_size: int) -> torch.Tensor:
    tokens = torch.zeros(batch_size, 64, 7)
    tokens[..., 0] = 2.0
    tokens[..., 1] = 0.1
    tokens[..., 2] = 1.0
    coordinate = (torch.arange(8) + 0.5) / 8.0
    center_y, center_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    tokens[..., 3] = center_y.flatten()
    tokens[..., 4] = center_x.flatten()
    tokens[..., 5:7] = 1.0 / 8.0
    return tokens


class V5ArchitectureTests(unittest.TestCase):
    def _model(self) -> EfficientFormerDepthStudent:
        return EfficientFormerDepthStudent(encoder_input_size=(64, 96))

    def test_scratch_encoder_and_output_contract(self) -> None:
        model = self._model().eval()
        image = torch.randn(1, 3, 64, 96)
        with torch.inference_mode():
            output = model(image, calibrated_tokens(1), return_aux=True)

        self.assertEqual(model.architecture, "student_v5")
        self.assertFalse(model.encoder.pretrained)
        self.assertEqual(output.depth.shape, (1, 1, 64, 96))
        self.assertEqual(output.feature_1_8.shape, (1, 64, 8, 12))
        self.assertEqual(output.feature_1_16.shape, (1, 96, 4, 6))
        self.assertTrue(torch.isfinite(output.depth).all())
        self.assertGreater(sum(p.numel() for p in model.parameters()), 3_000_000)

    def test_factory_builds_v5_without_pretrained_weights(self) -> None:
        model = build_model(
            {
                "model": {
                    "architecture": "student_v5",
                    "encoder_input_size": [64, 96],
                    "encoder_pretrained": False,
                }
            }
        )
        self.assertIsInstance(model, EfficientFormerDepthStudent)
        self.assertFalse(model.encoder.pretrained)

    def test_all_invalid_tokens_are_safe(self) -> None:
        model = self._model().eval()
        image = torch.randn(1, 3, 64, 96)
        tokens = torch.full((1, 64, 7), float("nan"))
        with torch.inference_mode():
            prediction = model(image, tokens)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_invalid_token_contents_cannot_change_prediction(self) -> None:
        model = self._model().eval()
        image = torch.randn(1, 3, 64, 96)
        first = calibrated_tokens(1)
        second = calibrated_tokens(1)
        first[..., 2] = 0.0
        second[..., 2] = 0.0
        second[..., :2] = torch.rand_like(second[..., :2]) * 20.0
        second[..., 3:] = torch.rand_like(second[..., 3:])
        with torch.inference_mode():
            first_prediction = model(image, first)
            second_prediction = model(image, second)
        torch.testing.assert_close(first_prediction, second_prediction)

    def test_scratch_encoder_trains_from_first_step(self) -> None:
        model = self._model().train()
        batch_norm = next(
            module
            for module in model.encoder.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        )
        original_running_mean = batch_norm.running_mean.detach().clone()
        output = model(
            torch.randn(1, 3, 64, 96),
            calibrated_tokens(1),
            return_aux=True,
        )
        output.depth.mean().backward()
        encoder_gradients = [
            parameter.grad
            for parameter in model.encoder.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in encoder_gradients))
        self.assertFalse(
            torch.equal(original_running_mean, batch_norm.running_mean)
        )

    def test_torchscript_matches_pytorch(self) -> None:
        model = self._model().eval()
        image = torch.randn(1, 3, 64, 96)
        tokens = calibrated_tokens(1)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            exported = torch.jit.freeze(
                torch.jit.trace(model, (image, tokens), strict=True)
            )
            reference = model(image, tokens)
            candidate = exported(image, tokens)
        torch.testing.assert_close(reference, candidate, rtol=1e-5, atol=1e-5)


class V5DistillationTests(unittest.TestCase):
    def test_confidence_scales_teacher_loss_and_gradient(self) -> None:
        values = torch.ones(1, 1, 2, 2, requires_grad=True)
        valid = torch.ones_like(values, dtype=torch.bool)
        pixel_weights = torch.ones_like(values)

        full = _confidence_weighted_mean(
            values, valid, pixel_weights, torch.ones_like(values)
        )
        full_gradient = torch.autograd.grad(full, values, retain_graph=True)[0]
        tenth = _confidence_weighted_mean(
            values, valid, pixel_weights, torch.full_like(values, 0.1)
        )
        tenth_gradient = torch.autograd.grad(tenth, values)[0]

        torch.testing.assert_close(tenth, full * 0.1)
        torch.testing.assert_close(tenth_gradient, full_gradient * 0.1)

    def test_empty_valid_mask_is_finite(self) -> None:
        student_depth = torch.ones(1, 1, 8, 8, requires_grad=True)
        teacher_depth = torch.ones(1, 1, 8, 8)
        student = DepthOutput(
            student_depth,
            torch.randn(1, 64, 1, 1, requires_grad=True),
            torch.randn(1, 96, 1, 1, requires_grad=True),
        )
        teacher = DepthOutput(
            teacher_depth,
            torch.randn(1, 128, 1, 1),
            torch.randn(1, 192, 1, 1),
        )
        target = torch.full((1, 1, 8, 8), float("nan"))
        valid = torch.zeros_like(target, dtype=torch.bool)
        loss, components = student_distillation_loss_v5(
            student,
            teacher,
            target,
            valid,
            calibrated_tokens(1),
            FeatureDistillationAdapters(),
            distillation_strength=1.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))


if __name__ == "__main__":
    unittest.main()
