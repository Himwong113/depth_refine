"""Training-only losses and adapters for the v4 teacher/student workflow."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.outputs import DepthOutput
from model.tof import tof_coverage_mask

__all__ = [
    "FeatureDistillationAdapters",
    "distillation_ramp",
    "student_ground_truth_loss",
    "student_distillation_loss",
    "teacher_supervision_loss",
]


def _valid_depth(target: Tensor, valid_mask: Tensor) -> Tensor:
    valid = valid_mask.bool() & torch.isfinite(target) & (target > 0)
    if not torch.any(valid):
        raise RuntimeError("batch contains no valid target-depth pixels")
    return valid


def _weighted_mean(values: Tensor, valid: Tensor, weights: Tensor | None = None) -> Tensor:
    if weights is None:
        weights = torch.ones_like(values)
    weights = weights.to(dtype=values.dtype) * valid.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def log_depth_gradient_loss(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    offsets: tuple[int, ...] = (1, 2, 4),
    pixel_weights: Tensor | None = None,
) -> Tensor:
    """Multi-scale L1 difference between predicted and target log gradients."""

    valid = _valid_depth(target, valid_mask)
    prediction = prediction.float().clamp_min(1e-3).log()
    target = target.float().clamp_min(1e-3).log()
    losses: list[Tensor] = []
    for offset in offsets:
        if prediction.shape[-1] > offset:
            pair_valid = valid[..., :, offset:] & valid[..., :, :-offset]
            difference = (
                (prediction[..., :, offset:] - prediction[..., :, :-offset])
                - (target[..., :, offset:] - target[..., :, :-offset])
            ).abs()
            weights = None
            if pixel_weights is not None:
                weights = 0.5 * (
                    pixel_weights[..., :, offset:] + pixel_weights[..., :, :-offset]
                )
            if torch.any(pair_valid):
                losses.append(_weighted_mean(difference, pair_valid, weights))
        if prediction.shape[-2] > offset:
            pair_valid = valid[..., offset:, :] & valid[..., :-offset, :]
            difference = (
                (prediction[..., offset:, :] - prediction[..., :-offset, :])
                - (target[..., offset:, :] - target[..., :-offset, :])
            ).abs()
            weights = None
            if pixel_weights is not None:
                weights = 0.5 * (
                    pixel_weights[..., offset:, :] + pixel_weights[..., :-offset, :]
                )
            if torch.any(pair_valid):
                losses.append(_weighted_mean(difference, pair_valid, weights))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def teacher_supervision_loss(
    output: Tensor | DepthOutput,
    target: Tensor,
    valid_mask: Tensor,
    metric_weight: float = 1.0,
    gradient_weight: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Stage-A metric MSE plus multi-scale log-depth gradient supervision."""

    prediction = output.depth if isinstance(output, DepthOutput) else output
    valid = _valid_depth(target, valid_mask)
    metric = _weighted_mean((prediction.float() - target.float()).square(), valid)
    gradient = log_depth_gradient_loss(prediction, target, valid)
    total = float(metric_weight) * metric + float(gradient_weight) * gradient
    return total, {"metric": metric, "gradient": gradient}


class FeatureDistillationAdapters(nn.Module):
    """Train-only 1x1 projections from student channels to teacher channels."""

    def __init__(
        self,
        student_channels: tuple[int, int] = (64, 96),
        teacher_channels: tuple[int, int] = (128, 192),
    ) -> None:
        super().__init__()
        self.feature_1_8 = nn.Conv2d(student_channels[0], teacher_channels[0], 1)
        self.feature_1_16 = nn.Conv2d(student_channels[1], teacher_channels[1], 1)

    def forward(self, output: DepthOutput) -> tuple[Tensor, Tensor]:
        return self.feature_1_8(output.feature_1_8), self.feature_1_16(
            output.feature_1_16
        )


def distillation_ramp(
    epoch: int, ground_truth_epochs: int = 5, ramp_end_epoch: int = 10
) -> float:
    """Return 0 for GT warmup, linearly rise, then remain at one."""

    if ramp_end_epoch <= ground_truth_epochs:
        raise ValueError("ramp_end_epoch must be greater than ground_truth_epochs")
    if epoch <= ground_truth_epochs:
        return 0.0
    if epoch >= ramp_end_epoch:
        return 1.0
    return (epoch - ground_truth_epochs) / float(
        ramp_end_epoch - ground_truth_epochs
    )


def _feature_cosine_loss(
    student: Tensor,
    teacher: Tensor,
    valid_mask: Tensor,
    confidence: Tensor,
    pixel_weights: Tensor,
) -> Tensor:
    if student.shape[-2:] != teacher.shape[-2:]:
        teacher = F.interpolate(
            teacher, size=student.shape[-2:], mode="bilinear", align_corners=False
        )
    valid = F.interpolate(valid_mask.float(), size=student.shape[-2:], mode="nearest") > 0.5
    confidence = F.interpolate(
        confidence, size=student.shape[-2:], mode="bilinear", align_corners=False
    )
    weights = F.interpolate(
        pixel_weights, size=student.shape[-2:], mode="nearest"
    ) * confidence
    cosine_error = 1.0 - F.cosine_similarity(
        student.float(), teacher.detach().float(), dim=1
    ).unsqueeze(1)
    return _weighted_mean(cosine_error, valid, weights)


def student_ground_truth_loss(
    student_depth: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    tof_tokens: Tensor,
    metric_weight: float = 1.0,
    gradient_weight: float = 0.1,
    outside_coverage_weight: float = 2.0,
    inside_coverage_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
    """Coverage-balanced student supervision shared by warmup and distillation."""

    valid = _valid_depth(target, valid_mask)
    coverage = tof_coverage_mask(tof_tokens, target.shape[-2:])
    pixel_weights = torch.where(
        coverage,
        target.new_tensor(float(inside_coverage_weight)),
        target.new_tensor(float(outside_coverage_weight)),
    )
    pixel_weights = pixel_weights / pixel_weights[valid].mean().clamp_min(1e-6)
    metric = _weighted_mean(
        (student_depth.float() - target.float()).square(), valid, pixel_weights
    )
    gradient = log_depth_gradient_loss(
        student_depth, target, valid, pixel_weights=pixel_weights
    )
    total = float(metric_weight) * metric + float(gradient_weight) * gradient
    return total, {"metric": metric, "gradient": gradient}, valid, pixel_weights


def student_distillation_loss(
    student: DepthOutput,
    teacher: DepthOutput,
    target: Tensor,
    valid_mask: Tensor,
    tof_tokens: Tensor,
    adapters: FeatureDistillationAdapters,
    distillation_strength: float,
    metric_weight: float = 1.0,
    gradient_weight: float = 0.1,
    teacher_depth_weight: float = 0.5,
    feature_weight: float = 0.05,
    outside_coverage_weight: float = 2.0,
    inside_coverage_weight: float = 1.0,
    confidence_temperature: float = 0.25,
    smooth_l1_beta: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Stage-B ground-truth, depth, and feature distillation objective."""

    supervised, supervised_components, valid, pixel_weights = (
        student_ground_truth_loss(
            student.depth,
            target,
            valid_mask,
            tof_tokens,
            metric_weight=metric_weight,
            gradient_weight=gradient_weight,
            outside_coverage_weight=outside_coverage_weight,
            inside_coverage_weight=inside_coverage_weight,
        )
    )
    metric = supervised_components["metric"]
    gradient = supervised_components["gradient"]

    teacher_error = (teacher.depth.detach().float() - target.float()).abs()
    confidence = torch.exp(-teacher_error / float(confidence_temperature))
    teacher_depth_error = F.smooth_l1_loss(
        student.depth.float(),
        teacher.depth.detach().float(),
        beta=float(smooth_l1_beta),
        reduction="none",
    )
    teacher_depth = _weighted_mean(
        teacher_depth_error, valid, pixel_weights * confidence
    )

    student_1_8, student_1_16 = adapters(student)
    feature_1_8 = _feature_cosine_loss(
        student_1_8,
        teacher.feature_1_8,
        valid,
        confidence,
        pixel_weights,
    )
    feature_1_16 = _feature_cosine_loss(
        student_1_16,
        teacher.feature_1_16,
        valid,
        confidence,
        pixel_weights,
    )
    feature = 0.5 * (feature_1_8 + feature_1_16)

    distilled = float(teacher_depth_weight) * teacher_depth + float(feature_weight) * feature
    total = supervised + float(distillation_strength) * distilled
    return total, {
        "metric": metric,
        "gradient": gradient,
        "teacher_depth": teacher_depth,
        "feature": feature,
    }
