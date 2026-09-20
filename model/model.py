"""Compact RGB/ToF depth refinement network.

The network consumes normalized RGB and either calibrated three-channel ToF
features (mean depth, standard deviation, validity) or a legacy one-channel
low-resolution depth map. Spatial work is performed primarily below the input
resolution using depthwise-separable convolutions and additive skip paths.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = [
    "ConvNormAct",
    "DepthwiseSeparableBlock",
    "CoarseToFFusion",
    "LiteDecoderBlock",
    "DepthRefinementUNet",
    "DepthRefinementModel",
    "DepthRefineNet",
]


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """Convolution followed by GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    """Mobile-friendly spatial filtering with an optional residual path."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.block(x)
        return output + x if self.use_residual else output


class ChannelContext(nn.Module):
    """Cheap bottleneck channel context."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)


class CoarseToFFusion(nn.Module):
    """Use a learned coarse gate to inject ToF into RGB features."""

    def __init__(self, tof_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.tof_projection = ConvNormAct(tof_channels, feature_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_features: Tensor, tof_features: Tensor) -> Tensor:
        projected_tof = self.tof_projection(tof_features)
        gate = self.gate(torch.cat((rgb_features, projected_tof), dim=1))
        return rgb_features + gate * projected_tof


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


class DepthRefinementUNet(nn.Module):
    """Compact calibrated-ToF-guided encoder-decoder.

    tof_features should contain mean depth, standard deviation, and validity at
    RGB resolution. A one-channel low-resolution map remains supported and is
    converted using validity-normalized interpolation.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        depth_scale: float = 10.0,
        positive_output: bool = True,
    ) -> None:
        super().__init__()
        if image_channels <= 0:
            raise ValueError("image_channels must be positive")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if depth_scale <= 0:
            raise ValueError("depth_scale must be positive")

        self.image_channels = image_channels
        self.depth_scale = float(depth_scale)
        self.positive_output = positive_output
        widths = (
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4,
        )

        self.stem = ConvNormAct(image_channels, widths[0], kernel_size=3, stride=2)
        self.encoder1 = DepthwiseSeparableBlock(widths[0], widths[0])
        self.down2 = DepthwiseSeparableBlock(widths[0], widths[1], stride=2)
        self.encoder2 = DepthwiseSeparableBlock(widths[1], widths[1])
        self.down3 = DepthwiseSeparableBlock(widths[1], widths[2], stride=2)
        self.encoder3 = DepthwiseSeparableBlock(widths[2], widths[2])
        self.down4 = DepthwiseSeparableBlock(widths[2], widths[3], stride=2)
        self.tof_fusion = CoarseToFFusion(4, widths[3])
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableBlock(widths[3], widths[3]),
            ChannelContext(widths[3]),
        )

        self.decoder3 = LiteDecoderBlock(widths[3], widths[2], widths[2])
        self.decoder2 = LiteDecoderBlock(widths[2], widths[1], widths[1])
        self.decoder1 = LiteDecoderBlock(widths[1], widths[0], widths[0])
        self.full_refine = DepthwiseSeparableBlock(
            widths[0] + image_channels, widths[0]
        )
        self.depth_head = nn.Conv2d(widths[0], 1, kernel_size=1)
        nn.init.zeros_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)

    @staticmethod
    def _normalized_resize(
        depth: Tensor,
        valid: Tensor,
        size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        weighted_depth = F.interpolate(
            depth * valid, size=size, mode="bilinear", align_corners=False
        )
        support = F.interpolate(valid, size=size, mode="bilinear", align_corners=False)
        resized_depth = weighted_depth / support.clamp_min(1e-6)
        resized_valid = (support > 1e-3).to(dtype=depth.dtype)
        return resized_depth * resized_valid, resized_valid

    def _prepare_tof(self, tof_features: Tensor, size: tuple[int, int]) -> Tensor:
        if tof_features.ndim == 3:
            tof_features = tof_features.unsqueeze(1)
        if tof_features.ndim != 4 or tof_features.shape[1] not in {1, 3}:
            raise ValueError(
                "tof_features must have shape (B, 1, H, W), (B, 3, H, W), "
                f"or (B, H, W); got {tuple(tof_features.shape)}"
            )

        if tof_features.shape[1] == 1:
            mean = tof_features
            valid = (torch.isfinite(mean) & (mean > 0)).to(dtype=mean.dtype)
            mean = torch.nan_to_num(mean)
            mean, valid = self._normalized_resize(mean, valid, size)
            uncertainty = torch.zeros_like(mean)
        else:
            mean, uncertainty, valid = tof_features.split(1, dim=1)
            mean = torch.nan_to_num(mean)
            uncertainty = torch.nan_to_num(uncertainty).clamp_min(0)
            valid = torch.nan_to_num(valid).clamp(0, 1)
            if mean.shape[-2:] != size:
                mean, valid = self._normalized_resize(mean, valid, size)
                uncertainty = F.interpolate(
                    uncertainty, size=size, mode="nearest"
                ) * valid
            else:
                mean = mean * valid
                uncertainty = uncertainty * valid
        return torch.cat((mean, uncertainty, valid), dim=1)

    def _validate_inputs(self, image: Tensor, tof_features: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != self.image_channels:
            raise ValueError(
                f"image must have shape (B, {self.image_channels}, H, W); "
                f"got {tuple(image.shape)}"
            )
        if image.shape[-2] < 16 or image.shape[-1] < 16:
            raise ValueError("image height and width must both be at least 16")
        if not image.is_floating_point() or not tof_features.is_floating_point():
            raise TypeError("image and tof_features must be floating-point tensors")
        if image.shape[0] != tof_features.shape[0]:
            raise ValueError("image and tof_features batch sizes must match")
        if image.device != tof_features.device:
            raise ValueError("image and tof_features must be on the same device")
        return tof_features.to(dtype=image.dtype)

    def _coarse_tof_context(
        self,
        tof: Tensor,
        size: tuple[int, int],
        global_mean: Tensor,
    ) -> Tensor:
        mean, uncertainty, valid = tof.split(1, dim=1)
        support = F.adaptive_avg_pool2d(valid, size)
        coarse_mean = F.adaptive_avg_pool2d(mean * valid, size) / support.clamp_min(1e-6)
        coarse_uncertainty = (
            F.adaptive_avg_pool2d(uncertainty * valid, size)
            / support.clamp_min(1e-6)
        )
        coarse_valid = support.clamp(0, 1)
        coarse_global_mean = global_mean.expand(
            -1, -1, size[0], size[1]
        )
        return torch.cat(
            (
                coarse_mean / self.depth_scale,
                coarse_uncertainty / self.depth_scale,
                coarse_valid,
                coarse_global_mean / self.depth_scale,
            ),
            dim=1,
        )

    def forward(self, image: Tensor, tof_features: Tensor) -> Tensor:
        tof_features = self._validate_inputs(image, tof_features)
        tof = self._prepare_tof(tof_features, image.shape[-2:])
        mean, uncertainty, valid = tof.split(1, dim=1)
        valid_count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        global_mean = (mean * valid).sum(dim=(-2, -1), keepdim=True) / valid_count
        x1 = self.encoder1(self.stem(image))
        x2 = self.encoder2(self.down2(x1))
        x3 = self.encoder3(self.down3(x2))
        x4 = self.down4(x3)
        tof_context = self._coarse_tof_context(tof, x4.shape[-2:], global_mean)
        x4 = self.bottleneck(self.tof_fusion(x4, tof_context))
        decoded = self.decoder3(x4, x3)
        decoded = self.decoder2(decoded, x2)
        decoded = self.decoder1(decoded, x1)
        decoded = F.interpolate(
            decoded, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.full_refine(torch.cat((decoded, image), dim=1))

        # The sensor supplies only a global scale anchor at the output. Local
        # piecewise-constant ToF rectangles never enter this full-resolution
        # path, so their boundaries cannot be copied into the prediction.
        learned_depth = self.depth_head(decoded) + global_mean
        if self.positive_output:
            learned_depth = F.softplus(learned_depth)
        return learned_depth


DepthRefinementModel = DepthRefinementUNet
DepthRefineNet = DepthRefinementUNet
