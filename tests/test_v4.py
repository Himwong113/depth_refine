from __future__ import annotations

import unittest
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from distillation import (
    FeatureDistillationAdapters,
    distillation_ramp,
    student_distillation_loss,
    teacher_supervision_loss,
)
from main import build_model
from model import MobileDepthStudent, RGBToFTeacher


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


class FakeTeacherPyramid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.projection_scale = nn.Parameter(torch.ones(()))

    def forward(self, image: torch.Tensor):
        batch, _, height, width = image.shape
        base = image.mean(dim=1, keepdim=True)
        return tuple(
            F.interpolate(
                base,
                size=(height // divisor, width // divisor),
                mode="bilinear",
                align_corners=False,
            ).expand(batch, channels, -1, -1)
            * self.frozen_scale
            * self.projection_scale
            for channels, divisor in zip((64, 96, 128, 192), (2, 4, 8, 16))
        )


class V4ArchitectureTests(unittest.TestCase):
    def test_student_contract_compatibility_and_budget(self) -> None:
        model = MobileDepthStudent().eval()
        image = torch.randn(2, 3, 64, 80)
        tokens = calibrated_tokens(2)
        raster = torch.zeros(2, 3, 64, 80)
        with torch.inference_mode():
            output = model(image, tokens, return_aux=True)
            compatibility_depth = model(image, raster, tokens)

        self.assertEqual(output.depth.shape, (2, 1, 64, 80))
        self.assertEqual(output.feature_1_8.shape, (2, 64, 8, 10))
        self.assertEqual(output.feature_1_16.shape, (2, 96, 4, 5))
        torch.testing.assert_close(output.depth, compatibility_depth)
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 500_000)

    def test_all_invalid_tokens_are_safe(self) -> None:
        model = MobileDepthStudent().eval()
        image = torch.randn(1, 3, 64, 64)
        tokens = torch.full((1, 64, 7), float("nan"))
        with torch.inference_mode():
            prediction = model(image, tokens)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_invalid_token_contents_cannot_change_prediction(self) -> None:
        model = MobileDepthStudent().eval()
        image = torch.randn(1, 3, 64, 64)
        first = calibrated_tokens(1)
        second = calibrated_tokens(1)
        first[..., 2] = 0.0
        second[..., 2] = 0.0
        second[..., 0:2] = torch.rand_like(second[..., 0:2]) * 20.0
        second[..., 3:7] = torch.rand_like(second[..., 3:7])
        with torch.inference_mode():
            first_prediction = model(image, first)
            second_prediction = model(image, second)
        torch.testing.assert_close(first_prediction, second_prediction)

    def test_fixed_resolution_torchscript_matches_pytorch(self) -> None:
        model = MobileDepthStudent().eval()
        image = torch.randn(1, 3, 64, 80)
        tokens = calibrated_tokens(1)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            exported = torch.jit.freeze(
                torch.jit.trace(model, (image, tokens), strict=True)
            )
            reference = model(image, tokens)
            candidate = exported(image, tokens)
        torch.testing.assert_close(reference, candidate, rtol=1e-5, atol=1e-5)
        self.assertFalse(
            any("teacher" in name or "adapter" in name for name, _ in model.named_parameters())
        )

    def test_teacher_with_injected_pyramid_has_auxiliary_contract(self) -> None:
        model = RGBToFTeacher(rgb_extractor=FakeTeacherPyramid()).eval()
        image = torch.randn(1, 3, 64, 80)
        with torch.inference_mode():
            output = model(image, calibrated_tokens(1), return_aux=True)
        self.assertEqual(output.depth.shape, (1, 1, 64, 80))
        self.assertEqual(output.feature_1_8.shape, (1, 128, 8, 10))
        self.assertEqual(output.feature_1_16.shape, (1, 192, 4, 5))

    def test_factory_selects_student_without_breaking_legacy_default(self) -> None:
        student = build_model({"model": {"architecture": "student_v4"}})
        legacy = build_model({"model": {"base_channels": 8}})
        self.assertEqual(getattr(student, "architecture"), "student_v4")
        self.assertEqual(legacy.__class__.__name__, "DepthRefinementUNet")

    def test_distillation_ramp_and_adapter_gradients(self) -> None:
        student = MobileDepthStudent()
        teacher = RGBToFTeacher(rgb_extractor=FakeTeacherPyramid()).eval()
        adapters = FeatureDistillationAdapters()
        image = torch.randn(1, 3, 64, 80)
        tokens = calibrated_tokens(1)
        target = torch.rand(1, 1, 64, 80) * 4.0 + 0.2
        valid = torch.ones_like(target, dtype=torch.bool)
        student_output = student(image, tokens, return_aux=True)
        with torch.no_grad():
            teacher_output = teacher(image, tokens, return_aux=True)
        loss, components = student_distillation_loss(
            student_output,
            teacher_output,
            target,
            valid,
            tokens,
            adapters,
            distillation_strength=1.0,
        )
        loss.backward()

        self.assertEqual(distillation_ramp(5), 0.0)
        self.assertAlmostEqual(distillation_ramp(7), 0.4)
        self.assertEqual(distillation_ramp(10), 1.0)
        self.assertEqual(
            set(components), {"metric", "gradient", "teacher_depth", "feature"}
        )
        self.assertTrue(
            all(parameter.grad is not None for parameter in adapters.parameters())
        )
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in student.parameters()
                if parameter.requires_grad
            )
        )

    def test_teacher_frozen_extractor_and_trainable_path_gradients(self) -> None:
        extractor = FakeTeacherPyramid()
        teacher = RGBToFTeacher(rgb_extractor=extractor).train()
        image = torch.randn(1, 3, 64, 80)
        target = torch.rand(1, 1, 64, 80) * 4.0 + 0.2
        valid = torch.ones_like(target, dtype=torch.bool)

        output = teacher(image, calibrated_tokens(1), return_aux=True)
        loss, _ = teacher_supervision_loss(output, target, valid)
        loss.backward()

        self.assertIsNone(extractor.frozen_scale.grad)
        self.assertIsNotNone(extractor.projection_scale.grad)
        self.assertTrue(torch.isfinite(extractor.projection_scale.grad).all())
        for prefix in ("fusion_1_16", "fusion_1_8", "decoder_1_8", "depth_head"):
            gradients = [
                parameter.grad
                for name, parameter in teacher.named_parameters()
                if name.startswith(prefix) and parameter.requires_grad
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(
                all(
                    gradient is not None and torch.isfinite(gradient).all()
                    for gradient in gradients
                ),
                prefix,
            )


if __name__ == "__main__":
    unittest.main()
