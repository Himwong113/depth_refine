"""Typed auxiliary outputs used only during teacher/student training."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class DepthOutput(NamedTuple):
    depth: Tensor
    feature_1_8: Tensor
    feature_1_16: Tensor
