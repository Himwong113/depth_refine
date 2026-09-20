"""High-capacity RGB/ToF teacher used only while training the student."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import ChannelContext
from .decoder import FullResolutionDepthHead, LiteDecoderBlock
from .outputs import DepthOutput
from .rgb import FrozenDepthAnythingPyramid, validate_pyramid
from .tof import FootprintAwareToFAttention, tof_depth_anchor

__all__ = ["RGBToFTeacher"]


class RGBToFTeacher(nn.Module):
    """Frozen Depth Anything V2 features plus trainable calibrated ToF fusion."""

    architecture = "teacher_v4"
    uses_tof_tokens_only = True

    def __init__(
        self,
        image_channels: int = 3,
        depth_scale: float = 10.0,
        positive_output: bool = True,
        fallback_depth: float = 1.0,
        attention_dim: int = 64,
        attention_heads: int = 4,
        geometry_heads: int = 2,
        refine_channels: int = 16,
        backbone_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
        backbone_input_size: tuple[int, int] | list[int] = (518, 686),
        backbone_dtype: str = "float32",
        local_files_only: bool = False,
        rgb_extractor: nn.Module | None = None,
        fusion_scales: tuple[str, ...] | list[str] = ("1_16", "1_8"),
        use_appearance_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.positive_output = bool(positive_output)
        self.fallback_depth = float(fallback_depth)
        unknown_scales = set(fusion_scales).difference({"1_16", "1_8"})
        if unknown_scales:
            raise ValueError(f"unknown fusion scales: {sorted(unknown_scales)}")
        self.fusion_scales = frozenset(fusion_scales)
        if rgb_extractor is None:
            rgb_extractor = FrozenDepthAnythingPyramid(
                model_name=backbone_name,
                input_size=tuple(int(value) for value in backbone_input_size),
                local_files_only=local_files_only,
            )
            dtype = getattr(torch, backbone_dtype, None)
            if dtype not in {torch.float16, torch.float32, torch.bfloat16}:
                raise ValueError(
                    "backbone_dtype must be float16, float32, or bfloat16"
                )
            rgb_extractor.backbone.to(dtype=dtype)
            rgb_extractor.neck.to(dtype=dtype)
        self.rgb_extractor = rgb_extractor

        channels_1_2, channels_1_4, channels_1_8, channels_1_16 = (64, 96, 128, 192)
        self.bottleneck = ChannelContext(channels_1_16)
        self.fusion_1_16 = FootprintAwareToFAttention(
            channels_1_16,
            appearance_channels=channels_1_16,
            attention_dim=attention_dim,
            num_heads=attention_heads,
            geometry_heads=geometry_heads,
            depth_scale=depth_scale,
            use_appearance_pooling=use_appearance_pooling,
        )
        self.decoder_1_8 = LiteDecoderBlock(
            channels_1_16, channels_1_8, channels_1_8
        )
        self.fusion_1_8 = FootprintAwareToFAttention(
            channels_1_8,
            appearance_channels=channels_1_16,
            attention_dim=attention_dim,
            num_heads=attention_heads,
            geometry_heads=geometry_heads,
            depth_scale=depth_scale,
            use_appearance_pooling=use_appearance_pooling,
        )
        self.decoder_1_4 = LiteDecoderBlock(channels_1_8, channels_1_4, channels_1_4)
        self.decoder_1_2 = LiteDecoderBlock(channels_1_4, channels_1_2, channels_1_2)
        self.depth_head = FullResolutionDepthHead(
            channels_1_2,
            image_channels=image_channels,
            refine_channels=refine_channels,
        )

    @staticmethod
    def _resolve_tokens(
        tof_tokens_or_raster: Tensor, legacy_tof_tokens: Tensor | None
    ) -> Tensor:
        tokens = legacy_tof_tokens if legacy_tof_tokens is not None else tof_tokens_or_raster
        if tokens.ndim != 3 or tokens.shape[1:] != (64, 7):
            raise ValueError("teacher_v4 requires tof_tokens with shape (B, 64, 7)")
        return tokens

    def forward(
        self,
        image: Tensor,
        tof_tokens_or_raster: Tensor,
        legacy_tof_tokens: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | DepthOutput:
        if image.ndim != 4 or image.shape[1] != self.image_channels:
            raise ValueError(
                f"image must have shape (B, {self.image_channels}, H, W)"
            )
        tokens = self._resolve_tokens(tof_tokens_or_raster, legacy_tof_tokens)
        if tokens.shape[0] != image.shape[0] or tokens.device != image.device:
            raise ValueError("image and tof_tokens must share batch size and device")
        tokens = tokens.to(dtype=image.dtype)

        feature_1_2, feature_1_4, feature_1_8, feature_1_16 = validate_pyramid(
            self.rgb_extractor(image)
        )
        decoded_1_16 = self.bottleneck(feature_1_16)
        if "1_16" in self.fusion_scales:
            decoded_1_16 = self.fusion_1_16(
                decoded_1_16, tokens, feature_1_16
            )
        decoded_1_8 = self.decoder_1_8(decoded_1_16, feature_1_8)
        if "1_8" in self.fusion_scales:
            decoded_1_8 = self.fusion_1_8(decoded_1_8, tokens, feature_1_16)
        decoded_1_4 = self.decoder_1_4(decoded_1_8, feature_1_4)
        decoded_1_2 = self.decoder_1_2(decoded_1_4, feature_1_2)

        depth = self.depth_head(decoded_1_2, image)
        depth = depth + tof_depth_anchor(tokens, fallback=self.fallback_depth)
        if self.positive_output:
            depth = F.softplus(depth, beta=5.0)
        if return_aux:
            return DepthOutput(depth, decoded_1_8, decoded_1_16)
        return depth
