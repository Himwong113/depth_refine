"""Depth refinement network.

The model accepts an RGB image and a (possibly lower-resolution) sparse depth
map.  Independent encoder and decoder channel-attention matrices are computed
from image queries and interpolated-depth keys. The encoder matrix is shared by
all encoder stages and the decoder matrix is shared by all decoder skip
concatenations. Every location has its own value projection.

Tensor shapes
-------------
``image``: ``(B, 3, H, W)``
``sparse_depth``: ``(B, 1, Hd, Wd)`` or ``(B, Hd, Wd)``
output: ``(B, 1, H, W)``
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


__all__ = [
    "DoubleConv",
    "DownBlock",
    "UpBlock",
    "SharedQKAttention",
    "AttentionValueFusion",
    "DepthRefinementUNet",
    "DepthRefinementModel",
    "DepthRefineNet",
]


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest useful GroupNorm group count for ``channels``."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with GroupNorm and SiLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Halve spatial resolution, then apply a :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Attend and refine an upsampled decoder/encoder skip concatenation."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention_channels: int | None = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        concatenated_channels = in_channels + skip_channels
        self.attention_fusion = (
            AttentionValueFusion(
                concatenated_channels,
                attention_channels,
                attention_dropout,
            )
            if attention_channels is not None
            else None
        )
        self.conv = DoubleConv(concatenated_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        skip: Tensor,
        shared_attention: Tensor | None = None,
    ) -> Tensor:
        # An explicit target size also handles odd input heights and widths.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        concatenated = torch.cat((x, skip), dim=1)
        if self.attention_fusion is not None:
            if shared_attention is None:
                raise ValueError(
                    "shared_attention is required when decoder attention is enabled"
                )
            concatenated = self.attention_fusion(concatenated, shared_attention)
        return self.conv(concatenated)


class SharedQKAttention(nn.Module):
    """Compute one RGB/depth channel-attention matrix.

    The image branch produces ``Q`` and the resized-depth branch produces
    ``K``.  After flattening the spatial embedding dimension ``E``, their
    shapes are ``(B, C_attn, E)`` and the returned ``QK^T`` attention has shape
    ``(B, C_attn, C_attn)``.  Pooling limits ``E`` without changing that shared
    attention shape.
    """

    def __init__(
        self,
        image_channels: int,
        attention_channels: int,
        max_tokens: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_channels <= 0 or attention_channels <= 0:
            raise ValueError("image_channels and attention_channels must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self.attention_channels = attention_channels
        self.max_tokens = max_tokens
        groups = _group_count(attention_channels)
        self.query_encoder = nn.Sequential(
            nn.Conv2d(image_channels, attention_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, attention_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(attention_channels, attention_channels, 1, bias=False),
        )
        self.key_encoder = nn.Sequential(
            nn.Conv2d(1, attention_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, attention_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(attention_channels, attention_channels, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def _pool(self, features: Tensor) -> Tensor:
        height, width = features.shape[-2:]
        if height * width <= self.max_tokens:
            return features

        scale = (self.max_tokens / float(height * width)) ** 0.5
        pooled_h = max(1, int(height * scale))
        pooled_w = max(1, int(width * scale))
        while pooled_h * pooled_w > self.max_tokens:
            if pooled_h >= pooled_w and pooled_h > 1:
                pooled_h -= 1
            elif pooled_w > 1:
                pooled_w -= 1
            else:
                break
        return F.adaptive_avg_pool2d(features, (pooled_h, pooled_w))

    def forward(self, image: Tensor, resized_depth: Tensor) -> Tensor:
        query = self._pool(self.query_encoder(image)).flatten(2)
        key = self._pool(self.key_encoder(resized_depth)).flatten(2)
        if query.shape[-1] != key.shape[-1]:
            raise RuntimeError("query and key embedding sizes must match")

        scale = query.shape[-1] ** -0.5
        scores = torch.bmm(query, key.transpose(1, 2)) * scale
        attention = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        return self.dropout(attention)


class AttentionValueFusion(nn.Module):
    """Apply shared ``QK^T`` attention to one feature tensor's private value.

    Every instance owns a separate value projection, so encoder and decoder
    locations compute ``V_i`` independently while reusing one attention matrix.
    """

    def __init__(
        self,
        stage_channels: int,
        attention_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if stage_channels <= 0 or attention_channels <= 0:
            raise ValueError("stage_channels and attention_channels must be positive")

        self.stage_channels = stage_channels
        self.attention_channels = attention_channels
        self.value_projection = nn.Conv2d(
            stage_channels, attention_channels, kernel_size=1, bias=False
        )
        self.output_projection = nn.Conv2d(
            attention_channels, stage_channels, kernel_size=1, bias=False
        )
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.GroupNorm(_group_count(stage_channels), stage_channels)

    def forward(self, features: Tensor, shared_attention: Tensor) -> Tensor:
        batch, _, height, width = features.shape
        expected = (batch, self.attention_channels, self.attention_channels)
        if tuple(shared_attention.shape) != expected:
            raise ValueError(
                f"shared_attention must have shape {expected}, "
                f"got {tuple(shared_attention.shape)}"
            )

        value = self.value_projection(features).flatten(2)
        attended = torch.bmm(shared_attention, value)
        attended = attended.reshape(batch, self.attention_channels, height, width)
        attended = self.dropout(self.output_projection(attended))
        return self.norm(features + attended)


class DepthRefinementUNet(nn.Module):
    """U-Net with separate shared attention matrices for encoder and decoder.

    Args:
        image_channels: Number of image channels, normally three.
        base_channels: Width of the first encoder stage.
        attention_channels: Channel size of the shared ``QK^T`` matrix. Defaults
            to ``base_channels``.
        max_attention_tokens: Maximum spatial embedding size ``E`` used for Q/K.
        attention_dropout: Dropout probability for attention and attended values.
        decoder_attention: Compute a second RGB/interpolated-depth ``QK^T``
            matrix with independent weights and apply it to decoder/encoder skip
            concatenations through independent value paths.
        residual_output: Add resized sparse depth to the predicted correction.
        positive_output: Apply ``softplus`` to the final prediction.
        depth_interpolation: Interpolation mode used to resize sparse depth.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        attention_channels: int | None = None,
        max_attention_tokens: int = 1024,
        attention_dropout: float = 0.0,
        decoder_attention: bool = True,
        residual_output: bool = True,
        positive_output: bool = False,
        depth_interpolation: str = "bilinear",
    ) -> None:
        super().__init__()
        if image_channels <= 0:
            raise ValueError("image_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if attention_channels is None:
            attention_channels = base_channels
        if attention_channels <= 0:
            raise ValueError("attention_channels must be positive")
        if depth_interpolation not in {"nearest", "nearest-exact", "bilinear", "bicubic"}:
            raise ValueError(f"unsupported depth_interpolation: {depth_interpolation!r}")

        self.image_channels = image_channels
        self.residual_output = residual_output
        self.positive_output = positive_output
        self.depth_interpolation = depth_interpolation

        widths = [base_channels * (2**stage) for stage in range(4)]

        self.shared_qk = SharedQKAttention(
            image_channels=image_channels,
            attention_channels=attention_channels,
            max_tokens=max_attention_tokens,
            dropout=attention_dropout,
        )

        self.image_stem = DoubleConv(image_channels, widths[0])
        self.image_down1 = DownBlock(widths[0], widths[1])
        self.image_down2 = DownBlock(widths[1], widths[2])
        self.image_down3 = DownBlock(widths[2], widths[3])

        # Each stage owns an independent V projection. Only QK^T is shared.
        self.fuse1 = AttentionValueFusion(widths[0], attention_channels, attention_dropout)
        self.fuse2 = AttentionValueFusion(widths[1], attention_channels, attention_dropout)
        self.fuse3 = AttentionValueFusion(widths[2], attention_channels, attention_dropout)
        self.fuse4 = AttentionValueFusion(widths[3], attention_channels, attention_dropout)

        decoder_attention_channels = attention_channels if decoder_attention else None
        self.decoder_shared_qk = (
            SharedQKAttention(
                image_channels=image_channels,
                attention_channels=attention_channels,
                max_tokens=max_attention_tokens,
                dropout=attention_dropout,
            )
            if decoder_attention
            else None
        )
        self.up3 = UpBlock(
            widths[3],
            widths[2],
            widths[2],
            decoder_attention_channels,
            attention_dropout,
        )
        self.up2 = UpBlock(
            widths[2],
            widths[1],
            widths[1],
            decoder_attention_channels,
            attention_dropout,
        )
        self.up1 = UpBlock(
            widths[1],
            widths[0],
            widths[0],
            decoder_attention_channels,
            attention_dropout,
        )
        self.output_head = nn.Conv2d(widths[0], 1, kernel_size=1)

    def _resize_depth(self, depth: Tensor, size: Tuple[int, int]) -> Tensor:
        if depth.shape[-2:] == size:
            return depth
        if self.depth_interpolation in {"bilinear", "bicubic"}:
            return F.interpolate(
                depth, size=size, mode=self.depth_interpolation, align_corners=False
            )
        return F.interpolate(depth, size=size, mode=self.depth_interpolation)

    def _validate_inputs(self, image: Tensor, sparse_depth: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")
        if image.shape[1] != self.image_channels:
            raise ValueError(
                f"image must have {self.image_channels} channels, got {image.shape[1]}"
            )
        if sparse_depth.ndim == 3:
            sparse_depth = sparse_depth.unsqueeze(1)
        if sparse_depth.ndim != 4 or sparse_depth.shape[1] != 1:
            raise ValueError(
                "sparse_depth must have shape (B, 1, Hd, Wd) or (B, Hd, Wd), "
                f"got {tuple(sparse_depth.shape)}"
            )
        if image.shape[0] != sparse_depth.shape[0]:
            raise ValueError("image and sparse_depth batch sizes must match")
        if image.shape[-2] < 8 or image.shape[-1] < 8:
            raise ValueError("image height and width must both be at least 8")
        if not image.is_floating_point() or not sparse_depth.is_floating_point():
            raise TypeError("image and sparse_depth must be floating-point tensors")
        if image.device != sparse_depth.device:
            raise ValueError("image and sparse_depth must be on the same device")
        if image.dtype != sparse_depth.dtype:
            sparse_depth = sparse_depth.to(dtype=image.dtype)
        return sparse_depth

    def forward(self, image: Tensor, sparse_depth: Tensor) -> Tensor:
        sparse_depth = self._validate_inputs(image, sparse_depth)
        resized_depth = self._resize_depth(sparse_depth, image.shape[-2:])
        shared_attention = self.shared_qk(image, resized_depth)

        image1 = self.fuse1(self.image_stem(image), shared_attention)
        image2 = self.fuse2(self.image_down1(image1), shared_attention)
        image3 = self.fuse3(self.image_down2(image2), shared_attention)
        image4 = self.fuse4(self.image_down3(image3), shared_attention)

        decoder_attention = (
            self.decoder_shared_qk(image, resized_depth)
            if self.decoder_shared_qk is not None
            else None
        )
        decoded = self.up3(image4, image3, decoder_attention)
        decoded = self.up2(decoded, image2, decoder_attention)
        decoded = self.up1(decoded, image1, decoder_attention)
        prediction = self.output_head(decoded)

        if self.residual_output:
            prediction = prediction + resized_depth
        if self.positive_output:
            prediction = F.softplus(prediction)
        return prediction


# Friendly names for callers that prefer "Model" or "Net" terminology.
DepthRefinementModel = DepthRefinementUNet
DepthRefineNet = DepthRefinementUNet
