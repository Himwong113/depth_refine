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
from .student_v5 import EfficientFormerDepthStudent
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
    "EfficientFormerDepthStudent",
    "FootprintAwareToFAttention",
    "MobileDepthStudent",
    "RGBToFTeacher",
    "tof_coverage_mask",
]
