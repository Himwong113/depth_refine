"""RGB feature pyramids for the mobile student and frozen teacher."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import ConvNormAct, DepthwiseSeparableBlock

__all__ = [
    "EfficientFormerV2RGBEncoder",
    "FrozenDepthAnythingPyramid",
    "MobileRGBEncoder",
]


class EfficientFormerV2RGBEncoder(nn.Module):
    """Scratch-initialized EfficientFormerV2-S0 feature pyramid.

    The official S0 attention path requires stage sizes that remain even across
    its internal stride-two attention.  The default 256x320 working resolution
    satisfies that constraint and keeps the deployment graph fixed.
    """

    output_channels = (32, 48, 96, 176)
    output_reductions = (4, 8, 16, 32)

    def __init__(
        self,
        image_channels: int = 3,
        model_name: str = "efficientformerv2_s0",
        input_size: tuple[int, int] | list[int] = (256, 320),
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if image_channels != 3:
            raise ValueError("EfficientFormerV2-S0 requires three-channel RGB input")
        if model_name != "efficientformerv2_s0":
            raise ValueError("student_v5 currently supports efficientformerv2_s0 only")
        if pretrained:
            raise ValueError(
                "student_v5 is intentionally scratch initialized; "
                "encoder_pretrained must be false"
            )
        if len(input_size) != 2:
            raise ValueError("encoder_input_size must contain height and width")
        self.input_size = tuple(int(value) for value in input_size)
        if min(self.input_size) < 32 or any(value % 32 for value in self.input_size):
            raise ValueError(
                "encoder_input_size dimensions must be at least 32 and divisible by 32"
            )
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency error
            raise ImportError(
                "student_v5 requires timm==1.0.29; install requirements.txt"
            ) from exc

        self.model_name = model_name
        self.pretrained = False
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            img_size=self.input_size,
        )
        channels = tuple(int(value) for value in self.backbone.feature_info.channels())
        reductions = tuple(
            int(value) for value in self.backbone.feature_info.reduction()
        )
        if channels != self.output_channels or reductions != self.output_reductions:
            raise RuntimeError(
                "unexpected EfficientFormerV2-S0 feature contract: "
                f"channels={channels}, reductions={reductions}"
            )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        working = F.interpolate(
            image,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )
        features = self.backbone(working)
        if len(features) != 4:
            raise RuntimeError("EfficientFormerV2-S0 must return four feature maps")
        return tuple(features)  # type: ignore[return-value]


class MobileRGBEncoder(nn.Module):
    """Four-stage encoder operating on an internally half-size RGB image."""

    def __init__(
        self,
        image_channels: int = 3,
        widths: tuple[int, int, int, int] = (32, 64, 96, 128),
    ) -> None:
        super().__init__()
        if len(widths) != 4 or min(widths) < 8:
            raise ValueError("widths must contain four channel counts >= 8")
        self.stem = ConvNormAct(image_channels, widths[0], kernel_size=3, stride=2)
        self.stage_1_4 = DepthwiseSeparableBlock(widths[0], widths[0])
        self.stage_1_8 = nn.Sequential(
            DepthwiseSeparableBlock(widths[0], widths[1], stride=2),
            DepthwiseSeparableBlock(widths[1], widths[1]),
        )
        self.stage_1_16 = nn.Sequential(
            DepthwiseSeparableBlock(widths[1], widths[2], stride=2),
            DepthwiseSeparableBlock(widths[2], widths[2]),
        )
        self.stage_1_32 = nn.Sequential(
            DepthwiseSeparableBlock(widths[2], widths[3], stride=2),
            DepthwiseSeparableBlock(widths[3], widths[3]),
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        working = F.interpolate(
            image,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        feature_1_4 = self.stage_1_4(self.stem(working))
        feature_1_8 = self.stage_1_8(feature_1_4)
        feature_1_16 = self.stage_1_16(feature_1_8)
        feature_1_32 = self.stage_1_32(feature_1_16)
        return feature_1_4, feature_1_8, feature_1_16, feature_1_32


class FrozenDepthAnythingPyramid(nn.Module):
    """Frozen Depth Anything V2 backbone/neck with trainable light projections.

    ``transformers`` is imported lazily so mobile student inference does not
    depend on that package.
    """

    output_channels = (64, 96, 128, 192)

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
        input_size: tuple[int, int] = (518, 686),
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "teacher_v4 requires transformers; install requirements-teacher.txt"
            ) from exc

        source = AutoModelForDepthEstimation.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.backbone = source.backbone
        self.neck = source.neck
        self.patch_size = int(getattr(source.config, "patch_size", 14))
        self.input_size = tuple(int(value) for value in input_size)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.neck.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.neck.eval()
        self.projections = nn.ModuleList(
            ConvNormAct(256, channels) for channels in self.output_channels
        )

    def train(self, mode: bool = True) -> "FrozenDepthAnythingPyramid":
        super().train(mode)
        self.backbone.eval()
        self.neck.eval()
        return self

    @staticmethod
    def _target_sizes(image: Tensor) -> tuple[tuple[int, int], ...]:
        height, width = image.shape[-2:]
        return tuple(
            (max(1, height // divisor), max(1, width // divisor))
            for divisor in (2, 4, 8, 16)
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        resized = F.interpolate(
            image, size=self.input_size, mode="bilinear", align_corners=False
        )
        frozen_dtype = next(self.backbone.parameters()).dtype
        resized = resized.to(dtype=frozen_dtype)
        with torch.no_grad():
            outputs = self.backbone(resized)
            feature_maps = outputs.feature_maps
            patch_height = resized.shape[-2] // self.patch_size
            patch_width = resized.shape[-1] // self.patch_size
            features = self.neck(feature_maps, patch_height, patch_width)

        projected: list[Tensor] = []
        train_dtype = next(self.projections.parameters()).dtype
        for feature, projection, size in zip(
            reversed(features), self.projections, self._target_sizes(image)
        ):
            feature = F.interpolate(
                feature.to(dtype=train_dtype),
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            projected.append(projection(feature))
        return tuple(projected)  # type: ignore[return-value]


def validate_pyramid(features: object) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normalize injectable teacher extractors to the canonical pyramid tuple."""

    if isinstance(features, Mapping):
        features = tuple(features[key] for key in ("1_2", "1_4", "1_8", "1_16"))
    if not isinstance(features, (tuple, list)) or len(features) != 4:
        raise ValueError("RGB extractor must return four pyramid tensors")
    return tuple(features)  # type: ignore[return-value]
