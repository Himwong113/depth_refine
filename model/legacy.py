"""Legacy attention-v3 RGB/ToF depth refinement network.

The network consumes normalized RGB and calibrated ToF observations. RGB
bottleneck features query the 64 sensor zones through a very small geometric
cross-attention block before the lightweight decoder runs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = [
    "ConvNormAct",
    "DepthwiseSeparableBlock",
    "GeometryAwareToFAttention",
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


class GeometryAwareToFAttention(nn.Module):
    """Condition coarse RGB features on sparse calibrated ToF zone tokens."""

    token_dim = 7

    def __init__(
        self,
        feature_channels: int,
        attention_dim: int = 16,
        num_heads: int = 2,
        depth_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if num_heads <= 0 or attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        if depth_scale <= 0:
            raise ValueError("depth_scale must be positive")

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.depth_scale = float(depth_scale)
        self.feature_norm = nn.GroupNorm(
            _group_count(feature_channels), feature_channels
        )
        self.query_projection = nn.Conv2d(
            feature_channels, attention_dim, kernel_size=1, bias=False
        )
        self.token_projection = nn.Linear(
            self.token_dim, attention_dim * 2, bias=False
        )
        self.output_projection = nn.Conv2d(
            attention_dim, feature_channels, kernel_size=1, bias=False
        )

        # softplus keeps each head's geometric falloff positive. The small
        # residual scale starts training close to the RGB-only representation
        # while still allowing gradients into every attention projection.
        self.geometry_scale = nn.Parameter(torch.full((num_heads,), 4.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

    @staticmethod
    def _query_positions(
        height: int,
        width: int,
        reference: Tensor,
    ) -> Tensor:
        y = (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5)
        x = (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5)
        y = y / height
        x = x / width
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_y, grid_x), dim=-1).reshape(1, height * width, 1, 2)

    def _normalize_tokens(self, tof_tokens: Tensor) -> tuple[Tensor, Tensor]:
        measurements_are_valid = (
            torch.isfinite(tof_tokens[..., :2]).all(dim=-1)
            & (tof_tokens[..., 0] > 0)
            & (tof_tokens[..., 1] >= 0)
        )
        geometry_is_valid = (
            torch.isfinite(tof_tokens[..., 3:7]).all(dim=-1)
            & (tof_tokens[..., 5] > 0)
            & (tof_tokens[..., 6] > 0)
        )
        valid = (
            torch.isfinite(tof_tokens[..., 2])
            & (tof_tokens[..., 2] > 0.5)
            & measurements_are_valid
            & geometry_is_valid
        )
        clean_tokens = torch.nan_to_num(
            tof_tokens, nan=0.0, posinf=0.0, neginf=0.0
        )
        valid_value = valid.unsqueeze(-1).to(dtype=clean_tokens.dtype)
        measurements = clean_tokens[..., :2].clamp_min(0)
        measurements = measurements / self.depth_scale * valid_value
        geometry = clean_tokens[..., 3:7].clamp(0, 1)
        normalized = torch.cat((measurements, valid_value, geometry), dim=-1)
        return normalized, valid

    def _attention(
        self,
        rgb_features: Tensor,
        tof_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, _, height, width = rgb_features.shape
        normalized_tokens, valid = self._normalize_tokens(tof_tokens)

        queries = self.query_projection(self.feature_norm(rgb_features))
        queries = queries.flatten(2).transpose(1, 2)
        queries = queries.reshape(
            batch_size, height * width, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        key_values = self.token_projection(normalized_tokens)
        key_values = key_values.reshape(
            batch_size,
            tof_tokens.shape[1],
            2,
            self.num_heads,
            self.head_dim,
        )
        keys, values = key_values.unbind(dim=2)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # A zero-valued null token is always available. It makes an all-invalid
        # sensor frame safe and lets attention explicitly ignore poor ToF data.
        null_token = keys.new_zeros(batch_size, self.num_heads, 1, self.head_dim)
        keys = torch.cat((keys, null_token), dim=2)
        values = torch.cat((values, null_token), dim=2)

        logits = torch.matmul(queries, keys.transpose(-2, -1))
        logits = logits * (self.head_dim**-0.5)

        query_positions = self._query_positions(height, width, rgb_features)
        centers = normalized_tokens[:, None, :, 3:5]
        half_sizes = normalized_tokens[:, None, :, 5:7] * 0.5
        outside = (query_positions - centers).abs().sub(half_sizes).clamp_min(0)
        distance_squared = outside.square().sum(dim=-1)
        geometry_scale = F.softplus(self.geometry_scale).to(dtype=logits.dtype)
        geometry_bias = -geometry_scale[None, :, None, None] * distance_squared[:, None]
        geometry_bias = torch.cat(
            (geometry_bias, geometry_bias.new_zeros(*geometry_bias.shape[:-1], 1)),
            dim=-1,
        )
        logits = logits + geometry_bias

        valid_with_null = torch.cat(
            (valid, torch.ones(batch_size, 1, dtype=torch.bool, device=valid.device)),
            dim=1,
        )
        logits = logits.masked_fill(
            ~valid_with_null[:, None, None, :], torch.finfo(logits.dtype).min
        )
        weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(values.dtype)
        attended = torch.matmul(weights, values)
        return attended, weights

    def forward(self, rgb_features: Tensor, tof_tokens: Tensor) -> Tensor:
        attended, _ = self._attention(rgb_features, tof_tokens)
        batch_size, _, height, width = rgb_features.shape
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch_size, height * width, self.attention_dim
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, self.attention_dim, height, width
        )
        update = self.output_projection(attended)
        return rgb_features + self.residual_scale * update


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

    tof_features contains the raster used for the global metric-scale anchor;
    tof_tokens optionally supplies the exact 64 calibrated sensor zones for
    attention. A one-channel low-resolution map remains supported and is
    converted into approximate tokens for legacy two-input callers.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        depth_scale: float = 10.0,
        positive_output: bool = True,
        attention_dim: int = 16,
        attention_heads: int = 2,
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
        self.tof_conditioner = GeometryAwareToFAttention(
            widths[3],
            attention_dim=attention_dim,
            num_heads=attention_heads,
            depth_scale=self.depth_scale,
        )
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

    def _validate_inputs(
        self,
        image: Tensor,
        tof_features: Tensor,
        tof_tokens: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
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
        if tof_tokens is not None:
            if tof_tokens.ndim != 3 or tof_tokens.shape[1:] != (64, 7):
                raise ValueError(
                    "tof_tokens must have shape (B, 64, 7); "
                    f"got {tuple(tof_tokens.shape)}"
                )
            if not tof_tokens.is_floating_point():
                raise TypeError("tof_tokens must be a floating-point tensor")
            if image.shape[0] != tof_tokens.shape[0]:
                raise ValueError("image and tof_tokens batch sizes must match")
            if image.device != tof_tokens.device:
                raise ValueError("image and tof_tokens must be on the same device")
            tof_tokens = tof_tokens.to(dtype=image.dtype)
        return tof_features.to(dtype=image.dtype), tof_tokens

    @staticmethod
    def _tokens_from_raster(tof: Tensor) -> Tensor:
        """Build approximate 8x8 tokens for legacy two-input callers."""

        mean, uncertainty, valid = tof.split(1, dim=1)
        support = F.adaptive_avg_pool2d(valid, (8, 8))
        token_valid = support > 1e-3
        token_mean = F.adaptive_avg_pool2d(mean * valid, (8, 8))
        token_mean = token_mean / support.clamp_min(1e-6)
        token_uncertainty = F.adaptive_avg_pool2d(uncertainty * valid, (8, 8))
        token_uncertainty = token_uncertainty / support.clamp_min(1e-6)

        coordinate = (
            torch.arange(8, device=tof.device, dtype=tof.dtype) + 0.5
        ) / 8.0
        center_y, center_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        token_height = torch.full_like(center_y, 1.0 / 8.0)
        token_width = torch.full_like(center_x, 1.0 / 8.0)
        geometry = torch.stack(
            (center_y, center_x, token_height, token_width), dim=0
        )
        geometry = geometry.unsqueeze(0).expand(tof.shape[0], -1, -1, -1)

        token_grid = torch.cat(
            (
                token_mean,
                token_uncertainty,
                token_valid.to(dtype=tof.dtype),
                geometry,
            ),
            dim=1,
        )
        return token_grid.flatten(2).transpose(1, 2)

    def forward(
        self,
        image: Tensor,
        tof_features: Tensor,
        tof_tokens: Tensor | None = None,
    ) -> Tensor:
        tof_features, tof_tokens = self._validate_inputs(
            image, tof_features, tof_tokens
        )
        tof = self._prepare_tof(tof_features, image.shape[-2:])
        mean, _, valid = tof.split(1, dim=1)
        valid_count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        global_mean = (mean * valid).sum(dim=(-2, -1), keepdim=True) / valid_count
        if tof_tokens is None:
            tof_tokens = self._tokens_from_raster(tof)
        x1 = self.encoder1(self.stem(image))
        x2 = self.encoder2(self.down2(x1))
        x3 = self.encoder3(self.down3(x2))
        x4 = self.down4(x3)
        x4 = self.bottleneck(self.tof_conditioner(x4, tof_tokens))
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
