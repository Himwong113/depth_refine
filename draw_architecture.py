"""Render the RGB–ToF teacher/student architecture diagram."""

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
FROZEN = "#e2e8f0"
FROZEN_EDGE = "#475569"
LOSS = "#fce7f3"
LOSS_EDGE = "#db2777"


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
    figure, axis = plt.subplots(figsize=(18, 10), facecolor="white")
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 10)
    axis.axis("off")

    axis.text(
        0.45,
        9.55,
        "RGB–ToF teacher–student metric-depth refinement",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.45,
        9.18,
        "Stage A learns a strong frozen-backbone teacher • Stage B distills to "
        "a 141.7K-parameter mobile student • only the student deploys",
        fontsize=9,
        color=MUTED,
    )

    teacher_band = FancyBboxPatch(
        (2.65, 5.72),
        14.75,
        2.75,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.0,
        edgecolor="#bfdbfe",
        facecolor="#f8fbff",
        zorder=-2,
    )
    student_band = FancyBboxPatch(
        (2.65, 0.88),
        14.75,
        2.75,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.0,
        edgecolor="#c7d2fe",
        facecolor="#fafaff",
        zorder=-2,
    )
    axis.add_patch(teacher_band)
    axis.add_patch(student_band)
    axis.text(
        2.9,
        8.18,
        "STAGE A — TRAIN TEACHER",
        fontsize=8.5,
        fontweight="bold",
        color=RGB_EDGE,
    )
    axis.text(
        2.9,
        3.34,
        "STAGE B — DISTILL INTO STUDENT",
        fontsize=8.5,
        fontweight="bold",
        color=ENC_EDGE,
    )

    rgb = box(axis, 0.4, 7.05, 1.65, 0.95, "RGB", "3 × 480 × 640", RGB, RGB_EDGE)
    tof = box(
        axis,
        0.4,
        5.62,
        1.65,
        1.05,
        "Calibrated ToF",
        "64 × 7 tokens\nmean/std/valid/box",
        TOF,
        TOF_EDGE,
    )
    target = box(
        axis,
        0.4,
        4.15,
        1.65,
        0.95,
        "Ground truth",
        "valid metric depth",
        HEAD,
        HEAD_EDGE,
    )

    teacher_rgb = box(
        axis,
        3.0,
        6.43,
        2.15,
        1.15,
        "Depth Anything V2-L",
        "frozen backbone + neck\nno gradients • eval mode",
        FROZEN,
        FROZEN_EDGE,
    )
    teacher_pyramid = box(
        axis,
        5.75,
        6.43,
        2.05,
        1.15,
        "Trainable pyramid",
        "64/96/128/192 ch\n1/2 • 1/4 • 1/8 • 1/16",
        ENC,
        ENC_EDGE,
    )
    teacher_fusion = box(
        axis,
        8.4,
        6.43,
        2.15,
        1.15,
        "Teacher ToF fusion",
        "footprint appearance\n4 heads • 1/16 + 1/8",
        TOF,
        TOF_EDGE,
    )
    teacher_decoder = box(
        axis,
        11.15,
        6.43,
        2.15,
        1.15,
        "Teacher decoder",
        "additive RGB skips\n16-ch full-res refine",
        DEC,
        DEC_EDGE,
    )
    teacher_output = box(
        axis,
        13.9,
        6.43,
        2.95,
        1.15,
        "Teacher metric depth + features",
        "640 × 480 depth\nfused features at 1/8 + 1/16",
        HEAD,
        HEAD_EDGE,
    )

    distillation = box(
        axis,
        11.55,
        4.18,
        3.1,
        1.05,
        "Coverage-aware objectives",
        "GT MSE + gradients • teacher depth\nfeature cosine • 2× outside ToF",
        LOSS,
        LOSS_EDGE,
    )

    student_rgb = box(
        axis,
        3.0,
        1.58,
        2.15,
        1.15,
        "Mobile RGB encoder",
        "internal 320 × 240\n32/64/96/128 ch",
        RGB,
        RGB_EDGE,
    )
    student_fusion = box(
        axis,
        5.75,
        1.58,
        2.05,
        1.15,
        "Student ToF fusion",
        "footprint appearance\n2 heads • 1/16 + 1/8",
        TOF,
        TOF_EDGE,
    )
    student_decoder = box(
        axis,
        8.4,
        1.58,
        2.15,
        1.15,
        "Lightweight decoder",
        "additive skips\ndecode only to 1/4",
        DEC,
        DEC_EDGE,
    )
    student_head = box(
        axis,
        11.15,
        1.58,
        2.15,
        1.15,
        "Full-resolution head",
        "8-ch projection + RGB\npositive residual + ToF anchor",
        DEC,
        DEC_EDGE,
    )
    student_output = box(
        axis,
        13.9,
        1.58,
        2.95,
        1.15,
        "Deployed metric depth",
        "1 × 480 × 640\nteacher-free single pass",
        HEAD,
        HEAD_EDGE,
    )

    arrow(axis, center_right(rgb), center_left(teacher_rgb), RGB_EDGE)
    arrow(axis, center_right(teacher_rgb), center_left(teacher_pyramid), FROZEN_EDGE)
    arrow(axis, center_right(teacher_pyramid), center_left(teacher_fusion), ENC_EDGE)
    arrow(axis, center_right(teacher_fusion), center_left(teacher_decoder), TOF_EDGE)
    arrow(axis, center_right(teacher_decoder), center_left(teacher_output), DEC_EDGE)

    arrow(axis, center_right(rgb), center_left(student_rgb), RGB_EDGE, curve=0.32)
    arrow(axis, center_right(student_rgb), center_left(student_fusion), RGB_EDGE)
    arrow(axis, center_right(student_fusion), center_left(student_decoder), TOF_EDGE)
    arrow(axis, center_right(student_decoder), center_left(student_head), DEC_EDGE)
    arrow(axis, center_right(student_head), center_left(student_output), HEAD_EDGE)

    arrow(axis, center_right(tof), center_left(teacher_fusion), TOF_EDGE, curve=0.3)
    arrow(axis, center_right(tof), center_left(student_fusion), TOF_EDGE, curve=0.22)
    arrow(
        axis,
        center_bottom(teacher_output),
        center_top(distillation),
        LOSS_EDGE,
        dashed=True,
        curve=0.12,
    )
    arrow(
        axis,
        center_right(target),
        center_left(distillation),
        LOSS_EDGE,
        dashed=True,
        curve=-0.06,
    )
    arrow(
        axis,
        center_top(student_head),
        center_bottom(distillation),
        LOSS_EDGE,
        dashed=True,
        curve=-0.12,
    )

    axis.text(
        9.0,
        0.38,
        "Solid: inference feature flow   •   Dashed magenta: training-only "
        "supervision   •   teacher and adapters are absent from deployment",
        ha="center",
        fontsize=8,
        color=MUTED,
    )

    svg_path = OUTPUT_DIR / "depth_refinement_architecture.svg"
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
        encoding="utf-8",
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
