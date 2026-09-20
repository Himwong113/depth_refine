"""Render the lightweight calibrated-ToF architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent / "docs"
INK = "#172033"
MUTED = "#64748b"
RGB = "#dbeafe"
RGB_EDGE = "#2563eb"
TOF = "#dcfce7"
TOF_EDGE = "#16a34a"
ENC = "#e0e7ff"
ENC_EDGE = "#4f46e5"
DEC = "#ffedd5"
DEC_EDGE = "#ea580c"
HEAD = "#f3e8ff"
HEAD_EDGE = "#9333ea"


def box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    face: str,
    edge: str,
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.09",
        linewidth=1.7,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        x + width / 2,
        y + height * 0.28,
        detail,
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )
    return x, y, width, height


def center_right(bounds):
    x, y, width, height = bounds
    return x + width, y + height / 2


def center_left(bounds):
    x, y, _, height = bounds
    return x, y + height / 2


def center_top(bounds):
    x, y, width, height = bounds
    return x + width / 2, y + height


def center_bottom(bounds):
    x, y, width, _ = bounds
    return x + width / 2, y


def arrow(axis, start, end, color=INK, dashed=False, curve=0.0):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.35,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=3,
            shrinkB=3,
        )
    )


def render() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(18, 8), facecolor="white")
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 8)
    axis.axis("off")

    axis.text(
        0.5,
        7.55,
        "Lightweight calibrated-ToF depth refinement",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.5,
        7.18,
        "125.9K parameters • ~1.40G convolution/attention MACs at 640×480 "
        "• geometric cross-attention",
        fontsize=9,
        color=MUTED,
    )

    rgb = box(axis, 0.55, 5.55, 1.8, 1.0, "RGB", "3 × H × W", RGB, RGB_EDGE)
    tof = box(
        axis,
        0.55,
        3.65,
        1.8,
        1.25,
        "Raw ToF",
        "mean + std + mask + fr\n64 calibrated zones",
        TOF,
        TOF_EDGE,
    )
    raster = box(
        axis,
        2.9,
        3.65,
        2.2,
        1.25,
        "ToF tokens",
        "mean/std/valid/box\n64 × 7",
        TOF,
        TOF_EDGE,
    )
    fusion = box(
        axis,
        5.65,
        4.6,
        2.0,
        1.25,
        "RGB stem",
        "RGB only • 1/2",
        RGB,
        RGB_EDGE,
    )

    e1 = box(axis, 8.15, 5.0, 1.6, 1.05, "Encoder 1", "32 ch • 1/2", ENC, ENC_EDGE)
    e2 = box(axis, 10.15, 5.0, 1.6, 1.05, "Encoder 2", "64 ch • 1/4", ENC, ENC_EDGE)
    e3 = box(axis, 12.15, 5.0, 1.6, 1.05, "Encoder 3", "96 ch • 1/8", ENC, ENC_EDGE)
    e4 = box(
        axis,
        14.15,
        5.0,
        1.8,
        1.05,
        "RGB Encoder 4",
        "128 ch • 1/16",
        ENC,
        ENC_EDGE,
    )
    tof_attention = box(
        axis,
        16.2,
        5.0,
        1.35,
        1.05,
        "Cross-attn",
        "2 heads × 8D\nsoft geometry",
        TOF,
        TOF_EDGE,
    )

    d3 = box(axis, 13.95, 2.75, 1.8, 1.05, "Decoder 3", "96 ch • 1/8", DEC, DEC_EDGE)
    d2 = box(axis, 11.6, 2.75, 1.8, 1.05, "Decoder 2", "64 ch • 1/4", DEC, DEC_EDGE)
    d1 = box(axis, 9.25, 2.75, 1.8, 1.05, "Decoder 1", "32 ch • 1/2", DEC, DEC_EDGE)
    refine = box(
        axis,
        6.55,
        1.15,
        2.1,
        1.15,
        "Full-res refine",
        "separable convolution\n+ RGB skip",
        DEC,
        DEC_EDGE,
    )
    heads = box(
        axis,
        3.65,
        1.15,
        2.25,
        1.15,
        "Residual + scale",
        "learned dense residual\n+ global ToF mean",
        HEAD,
        HEAD_EDGE,
    )
    output = box(
        axis,
        0.65,
        1.15,
        2.25,
        1.15,
        "Dense metric depth",
        "1 × H × W",
        HEAD,
        HEAD_EDGE,
    )

    arrow(axis, center_right(tof), center_left(raster), TOF_EDGE)
    arrow(axis, center_right(rgb), center_left(fusion), RGB_EDGE, curve=0.08)
    arrow(axis, center_right(fusion), center_left(e1), RGB_EDGE)
    arrow(axis, center_right(e1), center_left(e2), ENC_EDGE)
    arrow(axis, center_right(e2), center_left(e3), ENC_EDGE)
    arrow(axis, center_right(e3), center_left(e4), ENC_EDGE)
    arrow(axis, center_right(e4), center_left(tof_attention), ENC_EDGE)
    arrow(axis, center_bottom(tof_attention), center_top(d3), DEC_EDGE, curve=0.08)
    arrow(axis, center_left(d3), center_right(d2), DEC_EDGE)
    arrow(axis, center_left(d2), center_right(d1), DEC_EDGE)
    arrow(axis, center_bottom(d1), center_top(refine), DEC_EDGE, curve=0.1)
    arrow(axis, center_left(refine), center_right(heads), HEAD_EDGE)
    arrow(axis, center_left(heads), center_right(output), HEAD_EDGE)

    arrow(axis, center_bottom(e3), center_top(d3), ENC_EDGE, dashed=True)
    arrow(axis, center_bottom(e2), center_top(d2), ENC_EDGE, dashed=True)
    arrow(axis, center_bottom(e1), center_top(d1), ENC_EDGE, dashed=True)
    arrow(
        axis,
        center_bottom(fusion),
        center_top(refine),
        RGB_EDGE,
        dashed=True,
        curve=0.18,
    )
    arrow(
        axis,
        center_right(raster),
        center_left(tof_attention),
        TOF_EDGE,
        dashed=True,
        curve=-0.18,
    )
    arrow(
        axis,
        center_bottom(raster),
        center_top(heads),
        TOF_EDGE,
        dashed=True,
        curve=-0.22,
    )

    axis.text(
        9.1,
        0.45,
        "Solid: feature flow   •   Dashed: RGB skips, ToF token conditioning, "
        "and global scale only",
        ha="center",
        fontsize=8,
        color=MUTED,
    )

    figure.savefig(
        OUTPUT_DIR / "depth_refinement_architecture.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIR / "depth_refinement_architecture.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


if __name__ == "__main__":
    render()
