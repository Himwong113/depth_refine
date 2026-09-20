"""PyTorch models for RGB-guided depth refinement."""

from .legacy import (
    ConvNormAct,
    DepthRefineNet,
    DepthRefinementModel,
    DepthRefinementUNet,
    DepthwiseSeparableBlock,
    GeometryAwareToFAttention,
    LiteDecoderBlock,
)
from .model import DepthOutput, MobileDepthStudent
from .teacher import RGBToFTeacher
from .tof import FootprintAwareToFAttention, tof_coverage_mask

__all__ = [
    "ConvNormAct",
    "DepthRefineNet",
    "DepthRefinementModel",
    "DepthRefinementUNet",
    "DepthwiseSeparableBlock",
    "GeometryAwareToFAttention",
    "LiteDecoderBlock",
    "DepthOutput",
    "FootprintAwareToFAttention",
    "MobileDepthStudent",
    "RGBToFTeacher",
    "tof_coverage_mask",
]
