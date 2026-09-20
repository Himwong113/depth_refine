"""Calibrated ToF token validation, pooling, and lightweight fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import group_count

__all__ = [
    "FootprintAwareToFAttention",
    "normalize_tof_tokens",
    "pool_token_appearance",
    "tof_coverage_mask",
    "tof_depth_anchor",
]

TOKEN_DIM = 7


def normalize_tof_tokens(
    tof_tokens: Tensor, depth_scale: float = 10.0
) -> tuple[Tensor, Tensor]:
    """Sanitize tokens and return normalized values plus a strict valid mask."""

    if tof_tokens.ndim != 3 or tof_tokens.shape[-1] != TOKEN_DIM:
        raise ValueError(
            "tof_tokens must have shape (B, N, 7); "
            f"got {tuple(tof_tokens.shape)}"
        )
    if not tof_tokens.is_floating_point():
        raise TypeError("tof_tokens must be floating-point")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")

    measurement_valid = (
        torch.isfinite(tof_tokens[..., :2]).all(dim=-1)
        & (tof_tokens[..., 0] > 0)
        & (tof_tokens[..., 1] >= 0)
    )
    geometry_valid = (
        torch.isfinite(tof_tokens[..., 3:7]).all(dim=-1)
        & (tof_tokens[..., 3:5] >= 0).all(dim=-1)
        & (tof_tokens[..., 3:5] <= 1).all(dim=-1)
        & (tof_tokens[..., 5:7] > 0).all(dim=-1)
    )
    valid = (
        torch.isfinite(tof_tokens[..., 2])
        & (tof_tokens[..., 2] > 0.5)
        & measurement_valid
        & geometry_valid
    )
    clean = torch.nan_to_num(tof_tokens, nan=0.0, posinf=0.0, neginf=0.0)
    valid_value = valid.unsqueeze(-1).to(dtype=clean.dtype)
    measurements = clean[..., :2].clamp_min(0) / float(depth_scale)
    geometry = clean[..., 3:7].clamp(0, 1)
    normalized = torch.cat(
        (measurements * valid_value, valid_value, geometry), dim=-1
    )
    return normalized, valid


def tof_depth_anchor(tof_tokens: Tensor, fallback: float = 1.0) -> Tensor:
    """Compute one robust metric-scale anchor per image."""

    normalized, valid = normalize_tof_tokens(tof_tokens, depth_scale=1.0)
    depths = normalized[..., 0]
    count = valid.sum(dim=1, keepdim=True)
    mean = depths.sum(dim=1, keepdim=True) / count.clamp_min(1).to(depths.dtype)
    fallback_value = torch.full_like(mean, float(fallback))
    return torch.where(count > 0, mean, fallback_value).view(-1, 1, 1, 1)


def _normalized_grid(
    height: int, width: int, reference: Tensor
) -> tuple[Tensor, Tensor]:
    y = (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5)
    x = (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5)
    y = y / float(height)
    x = x / float(width)
    return torch.meshgrid(y, x, indexing="ij")


def pool_token_appearance(features: Tensor, normalized_tokens: Tensor) -> Tensor:
    """Average RGB features inside every calibrated rectangular footprint."""

    if features.ndim != 4:
        raise ValueError("features must have shape (B, C, H, W)")
    batch_size, _, height, width = features.shape
    if normalized_tokens.shape[0] != batch_size:
        raise ValueError("features and tokens must have the same batch size")

    grid_y, grid_x = _normalized_grid(height, width, features)
    center_y = normalized_tokens[..., 3, None, None]
    center_x = normalized_tokens[..., 4, None, None]
    half_h = normalized_tokens[..., 5, None, None] * 0.5
    half_w = normalized_tokens[..., 6, None, None] * 0.5
    inside = (
        (grid_y[None, None] - center_y).abs() <= half_h
    ) & ((grid_x[None, None] - center_x).abs() <= half_w)

    # Tiny footprints can fall between feature-cell centers. In that uncommon
    # case, a smooth nearest-cell fallback keeps the pooling well-defined.
    distance = (
        (grid_y[None, None] - center_y).square()
        + (grid_x[None, None] - center_x).square()
    )
    smooth_nearest = F.softmax(-64.0 * distance.flatten(2), dim=-1).view_as(distance)
    has_inside = inside.flatten(2).any(dim=-1)[..., None, None]
    weights = torch.where(has_inside, inside.to(features.dtype), smooth_nearest)
    valid = normalized_tokens[..., 2, None, None]
    weights = weights * valid
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return torch.einsum("bnhw,bchw->bnc", weights, features)


def tof_coverage_mask(tof_tokens: Tensor, size: tuple[int, int]) -> Tensor:
    """Rasterize valid calibrated footprints to a boolean coverage mask."""

    normalized, valid = normalize_tof_tokens(tof_tokens, depth_scale=1.0)
    height, width = size
    grid_y, grid_x = _normalized_grid(height, width, tof_tokens)
    center_y = normalized[..., 3, None, None]
    center_x = normalized[..., 4, None, None]
    half_h = normalized[..., 5, None, None] * 0.5
    half_w = normalized[..., 6, None, None] * 0.5
    inside = (
        ((grid_y[None, None] - center_y).abs() <= half_h)
        & ((grid_x[None, None] - center_x).abs() <= half_w)
        & valid[..., None, None]
    )
    return inside.any(dim=1, keepdim=True)


class FootprintAwareToFAttention(nn.Module):
    """Fuse token measurements, footprint appearance, and spatial geometry."""

    def __init__(
        self,
        feature_channels: int,
        appearance_channels: int,
        attention_dim: int = 32,
        num_heads: int = 2,
        geometry_heads: int = 1,
        depth_scale: float = 10.0,
        use_appearance_pooling: bool = True,
    ) -> None:
        super().__init__()
        if attention_dim <= 0 or num_heads <= 0 or attention_dim % num_heads:
            raise ValueError("attention_dim must be positive and divisible by num_heads")
        if not 0 <= geometry_heads <= num_heads:
            raise ValueError("geometry_heads must be between zero and num_heads")
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.geometry_heads = geometry_heads
        self.head_dim = attention_dim // num_heads
        self.depth_scale = float(depth_scale)
        self.use_appearance_pooling = bool(use_appearance_pooling)

        self.feature_norm = nn.GroupNorm(
            group_count(feature_channels), feature_channels
        )
        self.query_projection = nn.Conv2d(
            feature_channels, attention_dim, kernel_size=1, bias=False
        )
        self.token_projection = nn.Sequential(
            nn.Linear(TOKEN_DIM + appearance_channels, attention_dim),
            nn.SiLU(inplace=True),
            nn.Linear(attention_dim, attention_dim * 2, bias=False),
        )
        self.output_projection = nn.Conv2d(
            attention_dim, feature_channels, kernel_size=1, bias=False
        )
        self.null_key = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.null_value = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.geometry_scale = nn.Parameter(torch.full((geometry_heads,), 4.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        features: Tensor,
        tof_tokens: Tensor,
        appearance_features: Tensor,
    ) -> Tensor:
        batch_size, _, height, width = features.shape
        if tof_tokens.shape[0] != batch_size:
            raise ValueError("features and tof_tokens batch sizes must match")
        normalized, valid = normalize_tof_tokens(tof_tokens, self.depth_scale)
        if self.use_appearance_pooling:
            appearance = pool_token_appearance(appearance_features, normalized)
        else:
            appearance = appearance_features.new_zeros(
                batch_size, tof_tokens.shape[1], appearance_features.shape[1]
            )

        queries = self.query_projection(self.feature_norm(features))
        queries = queries.flatten(2).transpose(1, 2).reshape(
            batch_size, height * width, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        key_values = self.token_projection(torch.cat((normalized, appearance), dim=-1))
        key_values = key_values.reshape(
            batch_size, tof_tokens.shape[1], 2, self.num_heads, self.head_dim
        )
        keys, values = key_values.unbind(dim=2)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        keys = torch.cat((keys, self.null_key.expand(batch_size, -1, -1, -1)), dim=2)
        values = torch.cat(
            (values, self.null_value.expand(batch_size, -1, -1, -1)), dim=2
        )

        logits = torch.matmul(queries, keys.transpose(-2, -1)) * self.head_dim**-0.5
        if self.geometry_heads:
            grid_y, grid_x = _normalized_grid(height, width, features)
            query_positions = torch.stack((grid_y, grid_x), dim=-1).reshape(
                1, height * width, 1, 2
            )
            centers = normalized[:, None, :, 3:5]
            half_sizes = normalized[:, None, :, 5:7] * 0.5
            outside = (query_positions - centers).abs().sub(half_sizes).clamp_min(0)
            distance_squared = outside.square().sum(dim=-1)
            scales = F.softplus(self.geometry_scale).to(dtype=logits.dtype)
            bias = -scales[None, :, None, None] * distance_squared[:, None]
            remaining_heads = self.num_heads - self.geometry_heads
            token_bias = torch.cat(
                (
                    bias,
                    bias.new_zeros(
                        batch_size,
                        remaining_heads,
                        height * width,
                        tof_tokens.shape[1],
                    ),
                ),
                dim=1,
            )
            null_bias = token_bias.new_zeros(
                batch_size, self.num_heads, height * width, 1
            )
            logits = logits + torch.cat((token_bias, null_bias), dim=-1)

        valid_with_null = torch.cat(
            (valid, torch.ones(batch_size, 1, dtype=torch.bool, device=valid.device)),
            dim=1,
        )
        logits = logits.masked_fill(
            ~valid_with_null[:, None, None], torch.finfo(logits.dtype).min
        )
        weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(values.dtype)
        attended = torch.matmul(weights, values).permute(0, 2, 1, 3).reshape(
            batch_size, height * width, self.attention_dim
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, self.attention_dim, height, width
        )
        return features + self.residual_scale * self.output_projection(attended)
