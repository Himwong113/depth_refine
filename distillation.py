"""Training-only losses and adapters for teacher/student workflows."""

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
    "student_distillation_loss_v5",
    "teacher_supervision_loss",
]


def _valid_depth(target: Tensor, valid_mask: Tensor) -> Tensor:
    return valid_mask.bool() & torch.isfinite(target) & (target > 0)


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
    prediction = torch.where(
        valid, prediction.float(), torch.ones_like(prediction, dtype=torch.float32)
    ).clamp_min(1e-3).log()
    target = torch.where(
        valid, target.float(), torch.ones_like(target, dtype=torch.float32)
    ).clamp_min(1e-3).log()
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
    metric_error = torch.where(
        valid,
        (prediction.float() - target.float()).square(),
        torch.zeros_like(prediction, dtype=torch.float32),
    )
    metric = _weighted_mean(metric_error, valid)
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


def _confidence_weighted_mean(
    values: Tensor,
    valid: Tensor,
    pixel_weights: Tensor,
    confidence: Tensor,
) -> Tensor:
    """Weight by confidence without normalizing its absolute effect away."""

    base_weights = (
        valid.to(dtype=values.dtype) * pixel_weights.to(dtype=values.dtype)
    )
    confident_weights = base_weights * confidence.to(dtype=values.dtype)
    return (values * confident_weights).sum() / base_weights.sum().clamp_min(1.0)


def _feature_cosine_loss_v5(
    student: Tensor,
    teacher: Tensor,
    valid_mask: Tensor,
    confidence: Tensor,
    pixel_weights: Tensor,
) -> Tensor:
    """Area-weighted feature transfer with absolute confidence attenuation."""

    if student.shape[-2:] != teacher.shape[-2:]:
        teacher = F.interpolate(
            teacher, size=student.shape[-2:], mode="bilinear", align_corners=False
        )
    base = valid_mask.float() * pixel_weights.float()
    confident = base * confidence.float()
    base = F.adaptive_avg_pool2d(base, student.shape[-2:])
    confident = F.adaptive_avg_pool2d(confident, student.shape[-2:])
    cosine_error = 1.0 - F.cosine_similarity(
        student.float(), teacher.detach().float(), dim=1
    ).unsqueeze(1)
    return (cosine_error * confident).sum() / base.sum().clamp_min(1.0)


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
    valid_weights = pixel_weights[valid]
    normalization = (
        valid_weights.mean().clamp_min(1e-6)
        if valid_weights.numel()
        else pixel_weights.new_tensor(1.0)
    )
    pixel_weights = pixel_weights / normalization
    metric_error = torch.where(
        valid,
        (student_depth.float() - target.float()).square(),
        torch.zeros_like(student_depth, dtype=torch.float32),
    )
    metric = _weighted_mean(metric_error, valid, pixel_weights)
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
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
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


def student_distillation_loss_v5(
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
    """Corrected confidence-aware objective for scratch student_v5 training."""

    if confidence_temperature <= 0:
        raise ValueError("confidence_temperature must be positive")
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
    teacher_error = (teacher.depth.detach().float() - target.float()).abs()
    confidence = torch.exp(-teacher_error / float(confidence_temperature))
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    teacher_depth_error = F.smooth_l1_loss(
        student.depth.float(),
        teacher.depth.detach().float(),
        beta=float(smooth_l1_beta),
        reduction="none",
    )
    teacher_depth = _confidence_weighted_mean(
        teacher_depth_error,
        valid,
        pixel_weights,
        confidence,
    )

    student_1_8, student_1_16 = adapters(student)
    feature_1_8 = _feature_cosine_loss_v5(
        student_1_8,
        teacher.feature_1_8,
        valid,
        confidence,
        pixel_weights,
    )
    feature_1_16 = _feature_cosine_loss_v5(
        student_1_16,
        teacher.feature_1_16,
        valid,
        confidence,
        pixel_weights,
    )
    feature = 0.5 * (feature_1_8 + feature_1_16)
    distilled = (
        float(teacher_depth_weight) * teacher_depth
        + float(feature_weight) * feature
    )
    total = supervised + float(distillation_strength) * distilled
    valid_count = valid.sum().clamp_min(1).to(dtype=confidence.dtype)
    confidence_mean = confidence.sum() / valid_count
    coverage = tof_coverage_mask(tof_tokens, target.shape[-2:])
    inside_valid = valid & coverage
    outside_valid = valid & ~coverage
    inside_count = inside_valid.sum().clamp_min(1).to(dtype=confidence.dtype)
    outside_count = outside_valid.sum().clamp_min(1).to(dtype=confidence.dtype)
    return total, {
        "metric": supervised_components["metric"],
        "gradient": supervised_components["gradient"],
        "teacher_depth": teacher_depth,
        "feature": feature,
        "teacher_confidence": confidence_mean,
        "teacher_confidence_inside_tof": (
            confidence * inside_valid.to(dtype=confidence.dtype)
        ).sum()
        / inside_count,
        "teacher_confidence_outside_tof": (
            confidence * outside_valid.to(dtype=confidence.dtype)
        ).sum()
        / outside_count,
        "distillation_strength": confidence.new_tensor(
            float(distillation_strength)
        ),
    }
