"""Depth refinement network.

The model accepts an RGB image and a (possibly lower-resolution) sparse depth
map. Independent encoder and decoder channel-attention matrices are computed
from image queries and interpolated-depth keys. Encoder stages also produce
three adjacent-stage feature attention matrices for the left nested decoder
branch. The RGB/depth matrices are routed through the center and right branches.
Every location has its own value projection.

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
    "FeaturePairQKAttention",
    "AttentionValueFusion",
    "DualAttentionValueFusion",
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
    """Attend and refine an upsampled feature with one or more dense skips."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention_channels: int | None = None,
        attention_dropout: float = 0.0,
        dual_attention: bool = False,
    ) -> None:
        super().__init__()
        concatenated_channels = in_channels + skip_channels
        self.dual_attention = dual_attention and attention_channels is not None
        fusion_type = (
            DualAttentionValueFusion if self.dual_attention else AttentionValueFusion
        )
        self.attention_fusion = (
            fusion_type(
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
        skip: Tensor | tuple[Tensor, ...],
        shared_attention: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        skips = (skip,) if isinstance(skip, Tensor) else skip
        if not skips:
            raise ValueError("at least one skip tensor is required")
        target_size = skips[0].shape[-2:]
        if any(feature.shape[-2:] != target_size for feature in skips[1:]):
            raise ValueError("all skip tensors must have the same spatial size")

        # An explicit target size also handles odd input heights and widths.
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        concatenated = torch.cat((x, *skips), dim=1)
        if self.attention_fusion is not None:
            if shared_attention is None:
                raise ValueError("shared_attention is required for attention fusion")
            if self.dual_attention:
                if not isinstance(shared_attention, tuple) or len(shared_attention) != 2:
                    raise ValueError("dual attention requires (encoder, decoder) matrices")
            elif isinstance(shared_attention, tuple):
                raise ValueError("single attention requires one attention matrix")
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


class FeaturePairQKAttention(nn.Module):
    """Compute channel attention between two adjacent encoder feature stages.

    The shallower feature supplies ``Q`` and is spatially pooled to the deeper
    feature's resolution. The deeper feature supplies ``K``. Both projections
    are then pooled with the same spatial size so their flattened embedding
    dimension ``E`` matches. The result has shape ``(B, C_attn, C_attn)``.
    """

    def __init__(
        self,
        query_channels: int,
        key_channels: int,
        attention_channels: int,
        max_tokens: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if query_channels <= 0 or key_channels <= 0 or attention_channels <= 0:
            raise ValueError("query, key, and attention channels must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self.query_channels = query_channels
        self.key_channels = key_channels
        self.attention_channels = attention_channels
        self.max_tokens = max_tokens
        groups = _group_count(attention_channels)
        self.query_encoder = nn.Sequential(
            nn.Conv2d(query_channels, attention_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, attention_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(attention_channels, attention_channels, 1, bias=False),
        )
        self.key_encoder = nn.Sequential(
            nn.Conv2d(key_channels, attention_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, attention_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(attention_channels, attention_channels, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def _pooled_size(self, height: int, width: int) -> tuple[int, int]:
        if height * width <= self.max_tokens:
            return height, width

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
        return pooled_h, pooled_w

    def forward(self, query_features: Tensor, key_features: Tensor) -> Tensor:
        if query_features.ndim != 4 or key_features.ndim != 4:
            raise ValueError("query_features and key_features must be BCHW tensors")
        if query_features.shape[1] != self.query_channels:
            raise ValueError(
                f"query_features must have {self.query_channels} channels, "
                f"got {query_features.shape[1]}"
            )
        if key_features.shape[1] != self.key_channels:
            raise ValueError(
                f"key_features must have {self.key_channels} channels, "
                f"got {key_features.shape[1]}"
            )
        if query_features.shape[0] != key_features.shape[0]:
            raise ValueError("query and key batch sizes must match")
        if query_features.device != key_features.device:
            raise ValueError("query and key features must be on the same device")
        if not query_features.is_floating_point() or not key_features.is_floating_point():
            raise TypeError("query and key features must be floating-point tensors")
        if query_features.dtype != key_features.dtype:
            key_features = key_features.to(dtype=query_features.dtype)

        target_size = key_features.shape[-2:]
        if query_features.shape[-2:] != target_size:
            query_features = F.adaptive_avg_pool2d(query_features, target_size)

        query = self.query_encoder(query_features)
        key = self.key_encoder(key_features)
        pooled_size = self._pooled_size(*target_size)
        if pooled_size != target_size:
            query = F.adaptive_avg_pool2d(query, pooled_size)
            key = F.adaptive_avg_pool2d(key, pooled_size)

        query = query.flatten(2)
        key = key.flatten(2)
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


class DualAttentionValueFusion(nn.Module):
    """Merge encoder and decoder attention through parallel learned branches.

    Both branches own private value/output projections. A learned softmax gate,
    initialized to an equal mixture, combines their attended features before a
    single residual addition and normalization.
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
        self.encoder_value_projection = nn.Conv2d(
            stage_channels, attention_channels, kernel_size=1, bias=False
        )
        self.decoder_value_projection = nn.Conv2d(
            stage_channels, attention_channels, kernel_size=1, bias=False
        )
        self.encoder_output_projection = nn.Conv2d(
            attention_channels, stage_channels, kernel_size=1, bias=False
        )
        self.decoder_output_projection = nn.Conv2d(
            attention_channels, stage_channels, kernel_size=1, bias=False
        )
        self.mix_logits = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.GroupNorm(_group_count(stage_channels), stage_channels)

    def _attend(
        self,
        features: Tensor,
        attention: Tensor,
        value_projection: nn.Conv2d,
        output_projection: nn.Conv2d,
    ) -> Tensor:
        batch, _, height, width = features.shape
        expected = (batch, self.attention_channels, self.attention_channels)
        if tuple(attention.shape) != expected:
            raise ValueError(
                f"attention must have shape {expected}, got {tuple(attention.shape)}"
            )
        value = value_projection(features).flatten(2)
        attended = torch.bmm(attention, value).reshape(
            batch, self.attention_channels, height, width
        )
        return output_projection(attended)

    def forward(
        self,
        features: Tensor,
        shared_attention: tuple[Tensor, Tensor],
    ) -> Tensor:
        encoder_attention, decoder_attention = shared_attention
        encoder_features = self._attend(
            features,
            encoder_attention,
            self.encoder_value_projection,
            self.encoder_output_projection,
        )
        decoder_features = self._attend(
            features,
            decoder_attention,
            self.decoder_value_projection,
            self.decoder_output_projection,
        )
        mix = F.softmax(self.mix_logits, dim=0).to(dtype=features.dtype)
        attended = mix[0] * encoder_features + mix[1] * decoder_features
        return self.norm(features + self.dropout(attended))


class DepthRefinementUNet(nn.Module):
    """Nested U-Net (UNet++) with separate encoder and decoder attention.

    Args:
        image_channels: Number of image channels, normally three.
        base_channels: Width of the first encoder stage.
        attention_channels: Channel size of the shared ``QK^T`` matrix. Defaults
            to ``base_channels``.
        max_attention_tokens: Maximum spatial embedding size ``E`` used for Q/K.
        attention_dropout: Dropout probability for attention and attended values.
        decoder_attention: Compute a second RGB/interpolated-depth ``QK^T``
            matrix with independent weights. Adjacent encoder-pair attention is
            routed left, both RGB/depth matrices through the center, and decoder
            attention right. When disabled, center nodes use encoder attention
            and the right node has no attention fusion.
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
        residual_output: bool = False,
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
        # Adjacent encoder pairs produce stage-specific attention for the left
        # UNet++ column. Names follow the paired encoder levels: E2/E1, E3/E2,
        # and E4/E3.
        self.encoder_qk_x21 = FeaturePairQKAttention(
            widths[0],
            widths[1],
            attention_channels,
            max_attention_tokens,
            attention_dropout,
        )
        self.encoder_qk_x32 = FeaturePairQKAttention(
            widths[1],
            widths[2],
            attention_channels,
            max_attention_tokens,
            attention_dropout,
        )
        self.encoder_qk_x43 = FeaturePairQKAttention(
            widths[2],
            widths[3],
            attention_channels,
            max_attention_tokens,
            attention_dropout,
        )

        # Center nodes mix the RGB/depth encoder and decoder matrices. The
        # right node uses the decoder matrix.
        self.up2_1 = UpBlock(
            widths[3],
            widths[2],
            widths[2],
            attention_channels,
            attention_dropout,
        )
        self.up1_1 = UpBlock(
            widths[2],
            widths[1],
            widths[1],
            attention_channels,
            attention_dropout,
        )
        self.up0_1 = UpBlock(
            widths[1],
            widths[0],
            widths[0],
            attention_channels,
            attention_dropout,
        )
        self.up1_2 = UpBlock(
            widths[2],
            2 * widths[1],
            widths[1],
            attention_channels,
            attention_dropout,
            dual_attention=decoder_attention,
        )
        self.up0_2 = UpBlock(
            widths[1],
            2 * widths[0],
            widths[0],
            attention_channels,
            attention_dropout,
            dual_attention=decoder_attention,
        )
        self.up0_3 = UpBlock(
            widths[1],
            3 * widths[0],
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
        encoder_attention = self.shared_qk(image, resized_depth)
        input_image = image + resized_depth
        x0_0 = self.fuse1(self.image_stem(input_image), encoder_attention)
        x1_0 = self.fuse2(self.image_down1(x0_0), encoder_attention)
        x2_0 = self.fuse3(self.image_down2(x1_0), encoder_attention)
        x3_0 = self.fuse4(self.image_down3(x2_0), encoder_attention)

        decoder_attention = (
            self.decoder_shared_qk(image, resized_depth)
            if self.decoder_shared_qk is not None
            else None
        )
        center_attention = (
            (encoder_attention, decoder_attention)
            if decoder_attention is not None
            else encoder_attention
        )
        encoder_attention_x21 = self.encoder_qk_x21(x0_0, x1_0)
        encoder_attention_x32 = self.encoder_qk_x32(x1_0, x2_0)
        encoder_attention_x43 = self.encoder_qk_x43(x2_0, x3_0)

        # Match each decoder node to the encoder pair at the same resolution.
        x2_1 = self.up2_1(x3_0, x2_0, encoder_attention_x43)
        x1_1 = self.up1_1(x2_0, x1_0, encoder_attention_x32)
        x0_1 = self.up0_1(x1_0, x0_0, encoder_attention_x21)
        x1_2 = self.up1_2(x2_1, (x1_0, x1_1), center_attention)
        x0_2 = self.up0_2(x1_1, (x0_0, x0_1), center_attention)
        x0_3 = self.up0_3(x1_2, (x0_0, x0_1, x0_2), decoder_attention)
        prediction = self.output_head(x0_3)

        if self.residual_output:
            prediction = prediction + resized_depth
        if self.positive_output:
            prediction = F.softplus(prediction)
        return prediction


# Friendly names for callers that prefer "Model" or "Net" terminology.
DepthRefinementModel = DepthRefinementUNet
DepthRefineNet = DepthRefinementUNet
