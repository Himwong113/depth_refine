"""Render the DepthRefinementUNet architecture as SVG and PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "docs"

INK = "#172033"
MUTED = "#5C667A"
GRID = "#D7DCE5"
RGB = "#DDEBFF"
RGB_EDGE = "#3977C3"
DEPTH = "#DCF5E8"
DEPTH_EDGE = "#24835A"
ENC = "#E8E1F8"
ENC_EDGE = "#6F55A8"
DEC = "#FFE6D5"
DEC_EDGE = "#C86A31"
ATTN_ENC = "#E6E0FF"
ATTN_DEC = "#FFE2CA"
VALUE = "#FFF3BF"
OUTPUT = "#E2F3F4"
WHITE = "#FFFFFF"


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    face: str = WHITE,
    edge: str = INK,
    fontsize: float = 8.0,
    weight: str = "normal",
    radius: float = 0.10,
    linewidth: float = 1.4,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.22,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 1.4,
    style: str = "-|>",
    dashed: bool = False,
    connection: str = "arc3",
    zorder: int = 2,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=linewidth,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def attention_generator(
    ax,
    x: float,
    y: float,
    title: str,
    symbol: str,
    face: str,
    edge: str,
):
    width, height = 5.6, 2.25
    box(ax, x, y, width, height, "", face="#FAFBFD", edge=edge, radius=0.16)
    ax.text(
        x + width / 2,
        y + height - 0.25,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=edge,
        zorder=5,
    )
    box(
        ax,
        x + 0.25,
        y + 0.45,
        1.25,
        0.85,
        "RGB Conv\nQ projection",
        face=RGB,
        edge=RGB_EDGE,
        fontsize=7.3,
    )
    box(
        ax,
        x + 1.75,
        y + 0.45,
        1.25,
        0.85,
        "Depth Conv\nK projection",
        face=DEPTH,
        edge=DEPTH_EDGE,
        fontsize=7.3,
    )
    box(
        ax,
        x + 3.28,
        y + 0.35,
        1.90,
        1.05,
        f"{symbol} = softmax\n(QKᵀ / √E)\nCₐ × Cₐ",
        face=face,
        edge=edge,
        fontsize=8.0,
        weight="bold",
    )
    arrow(ax, (x + 1.50, y + 0.88), (x + 3.28, y + 1.04), color=edge)
    arrow(ax, (x + 3.00, y + 0.88), (x + 3.28, y + 0.72), color=edge)
    ax.text(
        x + width / 2,
        y + 0.14,
        "spatial embedding pooled to E ≤ 1024",
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )


def stage(
    ax,
    x: float,
    y: float,
    title: str,
    shape: str,
    lines: list[str],
    *,
    face: str,
    edge: str,
):
    width, height = 1.78, 3.00
    box(ax, x, y, width, height, "", face=WHITE, edge=edge, radius=0.14, linewidth=1.6)
    box(
        ax,
        x + 0.05,
        y + height - 0.58,
        width - 0.10,
        0.48,
        title,
        face=face,
        edge=edge,
        fontsize=9.0,
        weight="bold",
        radius=0.08,
        linewidth=1.0,
    )
    ax.text(
        x + width / 2,
        y + height - 0.78,
        shape,
        ha="center",
        va="center",
        fontsize=7.0,
        color=MUTED,
    )
    line_y = y + height - 1.22
    for index, text in enumerate(lines):
        line_face = VALUE if "private V" in text else "#F6F7FA"
        box(
            ax,
            x + 0.13,
            line_y - index * 0.37,
            width - 0.26,
            0.33,
            text,
            face=line_face,
            edge=GRID,
            fontsize=6.7,
            radius=0.04,
            linewidth=0.8,
        )
    return (x, y, width, height)


def render() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(24, 13), facecolor="white")
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(
        10.5,
        11.63,
        "DepthRefinementUNet",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        10.5,
        11.25,
        "Dual shared channel attention with independent values at every U-Net level",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED,
    )

    # Inputs and resized depth used by both attention generators and output residual.
    box(ax, 0.35, 9.55, 1.65, 0.70, "RGB image\nB × 3 × H × W", face=RGB, edge=RGB_EDGE, fontsize=8.2, weight="bold")
    box(ax, 0.35, 8.35, 1.65, 0.70, "Sparse depth\nB × 1 × Hₐ × Wₐ", face=DEPTH, edge=DEPTH_EDGE, fontsize=8.0, weight="bold")
    box(ax, 0.35, 7.20, 1.65, 0.60, "Interpolate\nto H × W", face=DEPTH, edge=DEPTH_EDGE, fontsize=7.5)
    arrow(ax, (1.18, 8.35), (1.18, 7.80), color=DEPTH_EDGE)

    attention_generator(ax, 2.55, 8.35, "ENCODER SHARED ATTENTION", "A_enc", ATTN_ENC, ENC_EDGE)
    attention_generator(ax, 9.05, 8.35, "DECODER SHARED ATTENTION (independent Q/K)", "A_dec", ATTN_DEC, DEC_EDGE)

    # Input routing to both independent attention generators.
    for target_x in (2.80, 9.30):
        arrow(ax, (2.00, 9.90), (target_x, 9.42), color=RGB_EDGE, connection="arc3,rad=-0.08")
        arrow(ax, (2.00, 7.50), (target_x + 1.50, 8.80), color=DEPTH_EDGE, connection="arc3,rad=0.08")

    # U-Net stages.
    e1 = stage(ax, 1.10, 3.65, "ENCODER E1", "H × W × C", ["DoubleConv", "private V1", "A_enc · V1", "Add & GroupNorm"], face=ENC, edge=ENC_EDGE)
    e2 = stage(ax, 3.55, 3.00, "ENCODER E2", "H/2 × W/2 × 2C", ["MaxPool ↓2", "DoubleConv", "private V2", "A_enc · V2", "Add & GroupNorm"], face=ENC, edge=ENC_EDGE)
    e3 = stage(ax, 6.00, 2.35, "ENCODER E3", "H/4 × W/4 × 4C", ["MaxPool ↓2", "DoubleConv", "private V3", "A_enc · V3", "Add & GroupNorm"], face=ENC, edge=ENC_EDGE)
    e4 = stage(ax, 8.45, 1.70, "BOTTLENECK E4", "H/8 × W/8 × 8C", ["MaxPool ↓2", "DoubleConv", "private V4", "A_enc · V4", "Add & GroupNorm"], face=ENC, edge=ENC_EDGE)

    d3 = stage(ax, 10.90, 2.35, "DECODER D3", "H/4 × W/4 × 4C", ["↑2 + concat E3", "private Vd3", "A_dec · Vd3", "Add & GroupNorm", "DoubleConv"], face=DEC, edge=DEC_EDGE)
    d2 = stage(ax, 13.35, 3.00, "DECODER D2", "H/2 × W/2 × 2C", ["↑2 + concat E2", "private Vd2", "A_dec · Vd2", "Add & GroupNorm", "DoubleConv"], face=DEC, edge=DEC_EDGE)
    d1 = stage(ax, 15.80, 3.65, "DECODER D1", "H × W × C", ["↑2 + concat E1", "private Vd1", "A_dec · Vd1", "Add & GroupNorm", "DoubleConv"], face=DEC, edge=DEC_EDGE)

    stages = [e1, e2, e3, e4, d3, d2, d1]
    for left, right, edge in zip(stages[:-1], stages[1:], [ENC_EDGE] * 3 + [INK] + [DEC_EDGE] * 2):
        arrow(
            ax,
            (left[0] + left[2], left[1] + left[3] / 2),
            (right[0], right[1] + right[3] / 2),
            color=edge,
            linewidth=1.7,
        )

    # RGB enters E1.
    arrow(ax, (1.10, 9.55), (1.65, 6.50), color=RGB_EDGE, connection="arc3,rad=0.06")

    # U-Net skip connections.
    skip_specs = [
        (e1, d1, 7.05, "skip E1"),
        (e2, d2, 6.65, "skip E2"),
        (e3, d3, 6.25, "skip E3"),
    ]
    for source, target, apex, label in skip_specs:
        sx = source[0] + source[2] / 2
        tx = target[0] + target[2] / 2
        sy = source[1] + source[3]
        ty = target[1] + target[3]
        arrow(
            ax,
            (sx, sy),
            (tx, ty),
            color="#68758A",
            linewidth=1.15,
            connection=f"arc3,rad={-0.20 if label == 'skip E1' else -0.16}",
        )
        ax.text((sx + tx) / 2, apex, label, ha="center", va="center", fontsize=7.2, color=MUTED)

    # Shared attention buses (one matrix reused within each side).
    ax.plot([2.15, 9.15], [7.55, 7.55], color=ENC_EDGE, linewidth=1.4, linestyle="--", zorder=1)
    arrow(ax, (6.90, 8.35), (6.90, 7.55), color=ENC_EDGE, dashed=True)
    for stage_box in (e1, e2, e3, e4):
        cx = stage_box[0] + stage_box[2] / 2
        arrow(ax, (cx, 7.55), (cx, stage_box[1] + stage_box[3]), color=ENC_EDGE, dashed=True, linewidth=1.1)
    ax.text(2.25, 7.72, "same A_enc", fontsize=7.4, color=ENC_EDGE, fontweight="bold")

    ax.plot([11.75, 16.70], [7.35, 7.35], color=DEC_EDGE, linewidth=1.4, linestyle="--", zorder=1)
    arrow(ax, (13.40, 8.35), (13.40, 7.35), color=DEC_EDGE, dashed=True)
    for stage_box in (d3, d2, d1):
        cx = stage_box[0] + stage_box[2] / 2
        arrow(ax, (cx, 7.35), (cx, stage_box[1] + stage_box[3]), color=DEC_EDGE, dashed=True, linewidth=1.1)
    ax.text(15.70, 7.52, "same A_dec", fontsize=7.4, color=DEC_EDGE, fontweight="bold")

    # Output head and sparse-depth residual.
    box(ax, 18.05, 4.75, 1.05, 0.72, "1 × 1 Conv\ndepth Δ", face=OUTPUT, edge="#2A7C86", fontsize=7.5, weight="bold")
    arrow(ax, (17.58, 5.22), (18.05, 5.12), color="#2A7C86", linewidth=1.7)
    add = Circle((19.55, 5.10), 0.25, facecolor=WHITE, edgecolor="#2A7C86", linewidth=1.6, zorder=4)
    ax.add_patch(add)
    ax.text(19.55, 5.10, "+", ha="center", va="center", fontsize=14, color=INK, zorder=5)
    arrow(ax, (19.10, 5.10), (19.30, 5.10), color="#2A7C86", linewidth=1.7)
    box(ax, 19.95, 4.72, 0.80, 0.76, "Softplus*", face=OUTPUT, edge="#2A7C86", fontsize=7.4, weight="bold")
    arrow(ax, (19.80, 5.10), (19.95, 5.10), color="#2A7C86", linewidth=1.7)
    ax.text(20.35, 4.46, "dense depth\nB × 1 × H × W", ha="center", va="top", fontsize=7.1, color=MUTED)

    # Residual route from resized sparse depth.
    ax.plot(
        [1.20, 0.82, 0.82, 19.55],
        [7.20, 6.90, 1.18, 1.18],
        color=DEPTH_EDGE,
        linewidth=1.3,
        linestyle="--",
        zorder=1,
    )
    arrow(ax, (19.55, 1.18), (19.55, 4.85), color=DEPTH_EDGE, linewidth=1.3, dashed=True, zorder=1)
    ax.text(10.2, 1.34, "resized sparse-depth residual", ha="center", va="center", fontsize=7.5, color=DEPTH_EDGE)

    # Formula and legend strip.
    box(
        ax,
        0.40,
        0.18,
        6.15,
        0.58,
        "AttentionValueFusionᵢ:  Fᵢ = GroupNorm(Xᵢ + Wₒᵢ (A · Wᵥᵢ Xᵢ))",
        face="#FAF7E8",
        edge="#B89A2C",
        fontsize=7.7,
    )
    ax.text(7.0, 0.48, "Solid arrows: feature flow", fontsize=7.2, color=INK, va="center")
    ax.text(9.6, 0.48, "Dashed arrows: shared attention / residual", fontsize=7.2, color=MUTED, va="center")
    ax.text(14.35, 0.48, "Yellow: independent V projection", fontsize=7.2, color="#8A701A", va="center")
    ax.text(18.05, 0.48, "* enabled in config.yml", fontsize=7.2, color=MUTED, va="center")

    svg_path = OUTPUT_DIR / "depth_refinement_architecture.svg"
    png_path = OUTPUT_DIR / "depth_refinement_architecture.png"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.20, facecolor="white")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0.20, facecolor="white")
    plt.close(fig)
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    render()
