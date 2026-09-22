"""Scratch EfficientFormerV2 student for real-time RGB-ToF depth refinement."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import ChannelContext, ConvNormAct, DepthwiseSeparableBlock
from .decoder import FullResolutionDepthHead, LiteDecoderBlock
from .outputs import DepthOutput
from .rgb import EfficientFormerV2RGBEncoder
from .tof import (
    FootprintAwareToFAttention,
    normalize_tof_tokens,
    pool_token_appearance,
    tof_depth_anchor,
)

__all__ = ["EfficientFormerDepthStudent"]


class EfficientFormerDepthStudent(nn.Module):
    """EfficientFormerV2-S0 RGB encoder with calibrated ToF cross-attention."""

    architecture = "student_v5"
    uses_tof_tokens_only = True

    def __init__(
        self,
        image_channels: int = 3,
        encoder_name: str = "efficientformerv2_s0",
        encoder_input_size: tuple[int, int] | list[int] = (256, 320),
        encoder_pretrained: bool = False,
        decoder_channels: tuple[int, int, int, int] | list[int] = (32, 64, 96, 128),
        detail_channels: int = 32,
        attention_dim: int = 32,
        attention_heads: int = 2,
        geometry_heads: int = 1,
        depth_scale: float = 10.0,
        positive_output: bool = True,
        fallback_depth: float = 1.0,
        refine_channels: int = 8,
        fusion_scales: tuple[str, ...] | list[str] = ("1_16", "1_8"),
        use_appearance_pooling: bool = True,
    ) -> None:
        super().__init__()
        if image_channels != 3:
            raise ValueError("student_v5 requires three-channel RGB input")
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four values")
        widths = tuple(int(value) for value in decoder_channels)
        if min(widths) < 8:
            raise ValueError("decoder channel values must be at least 8")
        if detail_channels != widths[0]:
            raise ValueError(
                "detail_channels must match decoder_channels[0] for the 1/4 skip"
            )
        unknown_scales = set(fusion_scales).difference({"1_16", "1_8"})
        if unknown_scales:
            raise ValueError(f"unknown fusion scales: {sorted(unknown_scales)}")

        self.image_channels = image_channels
        self.depth_scale = float(depth_scale)
        self.positive_output = bool(positive_output)
        self.fallback_depth = float(fallback_depth)
        self.fusion_scales = frozenset(fusion_scales)
        self.encoder = EfficientFormerV2RGBEncoder(
            image_channels=image_channels,
            model_name=encoder_name,
            input_size=encoder_input_size,
            pretrained=encoder_pretrained,
        )
        self.detail_encoder = nn.Sequential(
            ConvNormAct(image_channels, 16, kernel_size=3, stride=2),
            DepthwiseSeparableBlock(16, detail_channels, stride=2),
            DepthwiseSeparableBlock(detail_channels, detail_channels),
        )
        self.skip_1_8 = ConvNormAct(32, widths[1])
        self.skip_1_16 = ConvNormAct(48, widths[2])
        self.deep_1_16 = ConvNormAct(96, widths[3])
        self.deep_1_32 = ConvNormAct(176, widths[3])
        self.deep_refine = DepthwiseSeparableBlock(widths[3], widths[3])
        self.bottleneck = ChannelContext(widths[3])

        self.decoder_1_16 = LiteDecoderBlock(widths[3], widths[2], widths[2])
        self.fusion_1_16 = FootprintAwareToFAttention(
            widths[2],
            appearance_channels=widths[2],
            attention_dim=attention_dim,
            num_heads=attention_heads,
            geometry_heads=geometry_heads,
            depth_scale=depth_scale,
            use_appearance_pooling=use_appearance_pooling,
        )
        self.decoder_1_8 = LiteDecoderBlock(widths[2], widths[1], widths[1])
        self.fusion_1_8 = FootprintAwareToFAttention(
            widths[1],
            appearance_channels=widths[2],
            attention_dim=attention_dim,
            num_heads=attention_heads,
            geometry_heads=geometry_heads,
            depth_scale=depth_scale,
            use_appearance_pooling=use_appearance_pooling,
        )
        self.decoder_1_4 = LiteDecoderBlock(widths[1], widths[0], widths[0])
        self.depth_head = FullResolutionDepthHead(
            widths[0],
            image_channels=image_channels,
            refine_channels=refine_channels,
        )

    def _validate_inputs(self, image: Tensor, tof_tokens: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != self.image_channels:
            raise ValueError(
                f"image must have shape (B, {self.image_channels}, H, W); "
                f"got {tuple(image.shape)}"
            )
        if min(image.shape[-2:]) < 32:
            raise ValueError("image height and width must both be at least 32")
        if not image.is_floating_point():
            raise TypeError("image must be floating-point")
        if tof_tokens.ndim != 3 or tof_tokens.shape[1:] != (64, 7):
            raise ValueError(
                "tof_tokens must have shape (B, 64, 7); "
                f"got {tuple(tof_tokens.shape)}"
            )
        if image.shape[0] != tof_tokens.shape[0]:
            raise ValueError("image and tof_tokens batch sizes must match")
        if image.device != tof_tokens.device:
            raise ValueError("image and tof_tokens must be on the same device")
        return tof_tokens.to(dtype=image.dtype)

    @staticmethod
    def _resolve_tokens(
        tof_tokens_or_raster: Tensor, legacy_tof_tokens: Tensor | None
    ) -> Tensor:
        if legacy_tof_tokens is not None:
            return legacy_tof_tokens
        if tof_tokens_or_raster.ndim != 3:
            raise ValueError(
                "student_v5 requires calibrated tokens; call model(image, tof_tokens)"
            )
        return tof_tokens_or_raster

    @staticmethod
    def _size(image: Tensor, divisor: int) -> tuple[int, int]:
        return (
            max(1, image.shape[-2] // divisor),
            max(1, image.shape[-1] // divisor),
        )

    @staticmethod
    def _resize(feature: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(
            feature, size=size, mode="bilinear", align_corners=False
        )

    def forward(
        self,
        image: Tensor,
        tof_tokens_or_raster: Tensor,
        legacy_tof_tokens: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | DepthOutput:
        tof_tokens = self._resolve_tokens(tof_tokens_or_raster, legacy_tof_tokens)
        tof_tokens = self._validate_inputs(image, tof_tokens)
        stage_1_4, stage_1_8, stage_1_16, stage_1_32 = self.encoder(image)

        feature_1_4 = self.detail_encoder(image)
        feature_1_8 = self._resize(
            self.skip_1_8(stage_1_4), self._size(image, 8)
        )
        feature_1_16 = self._resize(
            self.skip_1_16(stage_1_8), self._size(image, 16)
        )
        deep_size = self._size(image, 32)
        feature_1_32 = self.deep_refine(
            self._resize(self.deep_1_16(stage_1_16), deep_size)
            + self._resize(self.deep_1_32(stage_1_32), deep_size)
        )

        decoded_1_16_rgb = self.decoder_1_16(
            self.bottleneck(feature_1_32), feature_1_16
        )
        normalized_tokens, _ = normalize_tof_tokens(tof_tokens, self.depth_scale)
        pooled_appearance = pool_token_appearance(
            decoded_1_16_rgb, normalized_tokens
        )
        decoded_1_16 = decoded_1_16_rgb
        if "1_16" in self.fusion_scales:
            decoded_1_16 = self.fusion_1_16(
                decoded_1_16, tof_tokens, pooled_appearance
            )
        decoded_1_8 = self.decoder_1_8(decoded_1_16, feature_1_8)
        if "1_8" in self.fusion_scales:
            decoded_1_8 = self.fusion_1_8(
                decoded_1_8, tof_tokens, pooled_appearance
            )
        decoded_1_4 = self.decoder_1_4(decoded_1_8, feature_1_4)

        anchor = tof_depth_anchor(tof_tokens, fallback=self.fallback_depth)
        depth = self.depth_head(decoded_1_4, image) + anchor
        if self.positive_output:
            depth = F.softplus(depth, beta=5.0)
        if return_aux:
            return DepthOutput(depth, decoded_1_8, decoded_1_16)
        return depth
