"""RGB feature pyramids for the mobile student and frozen teacher."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import ConvNormAct, DepthwiseSeparableBlock

__all__ = ["FrozenDepthAnythingPyramid", "MobileRGBEncoder"]


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
