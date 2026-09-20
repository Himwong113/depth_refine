"""Small additive decoders shared by the teacher and student."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import ConvNormAct, DepthwiseSeparableBlock

__all__ = ["FullResolutionDepthHead", "LiteDecoderBlock"]


class LiteDecoderBlock(nn.Module):
    """Project, resize, add an encoder skip, and refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.input_projection = ConvNormAct(in_channels, out_channels)
        self.skip_projection = ConvNormAct(skip_channels, out_channels)
        self.refine = DepthwiseSeparableBlock(out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(x + self.skip_projection(skip))


class FullResolutionDepthHead(nn.Module):
    """Delay expensive full-resolution work until a narrow final block."""

    def __init__(
        self,
        feature_channels: int,
        image_channels: int = 3,
        refine_channels: int = 8,
    ) -> None:
        super().__init__()
        self.feature_projection = ConvNormAct(feature_channels, refine_channels)
        self.refine = DepthwiseSeparableBlock(
            refine_channels + image_channels, refine_channels
        )
        self.depth = nn.Conv2d(refine_channels, 1, kernel_size=1)

    def forward(self, features: Tensor, image: Tensor) -> Tensor:
        features = self.feature_projection(features)
        features = F.interpolate(
            features, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.depth(self.refine(torch.cat((features, image), dim=1)))
