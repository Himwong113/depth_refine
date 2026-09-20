"""Render a publication figure with editable SVG and vector PDF companions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT_DIR = Path(__file__).resolve().parent / "docs"
INK, GRAY = "#263342", "#596675"
BLUE, TEAL, OCHRE = "#345f87", "#267b75", "#a16b29"


def label(ax, x, y, text, size=9, color=INK, weight="normal", ha="center"):
    ax.text(x, y, text, fontsize=size, color=color, weight=weight,
            ha=ha, va="center", linespacing=1.5, zorder=5)


def block(ax, x, y, w, h, title, detail="", color=BLUE, fill="#edf3f8"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=fill, edgecolor=color, linewidth=0.9, zorder=3))
    label(ax, x+w/2, y+h*(0.69 if detail else 0.5), title, 9, color, "bold")
    if detail:
        label(ax, x+w/2, y+h*0.30, detail, 7.6, GRAY)


def arrow(ax, points, color=INK, dashed=False):
    style = (0, (3, 2.5)) if dashed else "-"
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=color, linewidth=1,
                linestyle=style, zorder=2)
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=9,
        color=color, linewidth=1, linestyle=style,
        shrinkA=0, shrinkB=1, zorder=2))


def render():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "svg.hashsalt": "depth-v4",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(15.8, 8.5), facecolor="white")
    ax = fig.add_axes((0.02, 0.03, 0.96, 0.94))
    ax.set(xlim=(0, 16), ylim=(0, 8.5))
    ax.axis("off")
    label(ax, 0.1, 8.22, "(a)  RGB–ToF teacher–student framework", 10.5, weight="bold", ha="left")
    label(ax, 15.8, 8.22, "Input / output resolution: 640 × 480", 8, GRAY, ha="right")

    for y, teacher in ((6.75, True), (4.55, False)):
        color, fill = (BLUE, "#edf3f8") if teacher else (TEAL, "#eaf4f1")
        label(ax, 0.15, y+0.55, "Teacher" if teacher else "Student", 10, color, "bold", ha="left")
        label(ax, 0.15, y+0.22, "Stage A" if teacher else "Stage B", 8, GRAY, ha="left")
        block(ax, 1.35, y, 1.25, 0.9, "RGB", "3 × H × W", GRAY, "#f7f8fa")
        block(ax, 3.05, y, 2.55, 0.9,
              "Depth Anything V2-L" if teacher else "Mobile RGB encoder",
              "Frozen backbone + neck" if teacher else "Resize to 320 × 240",
              GRAY if teacher else color, "#f0f2f4" if teacher else fill)
        block(ax, 6.05, y, 2.3, 0.9,
              "Pyramid projections" if teacher else "Feature pyramid",
              "64 / 96 / 128 / 192 channels" if teacher else "32 / 64 / 96 / 128 channels", color, fill)
        block(ax, 8.85, y, 3.65, 0.9, "ToF fusion + additive decoder",
              "1/16 + 1/8 fusion  ·  " + ("4 heads" if teacher else "2 heads"), color, fill)
        block(ax, 13.0, y, 2.75, 0.9, "Metric-depth head",
              ("16-ch" if teacher else "8-ch") + " refinement → 1 × H × W", color, fill)
        for start, end in ((2.6, 3.05), (5.6, 6.05), (8.35, 8.85), (12.5, 13.0)):
            arrow(ax, [(start, y+0.45), (end, y+0.45)])

    block(ax, 6.05, 7.82, 2.3, 0.30, "64 calibrated ToF tokens", color=GRAY, fill="white")
    arrow(ax, [(8.35, 7.97), (10.65, 7.97), (10.65, 7.65)], TEAL)
    block(ax, 6.05, 3.98, 2.3, 0.30, "64 calibrated ToF tokens", color=GRAY, fill="white")
    arrow(ax, [(8.35, 4.13), (10.65, 4.13), (10.65, 4.55)], TEAL)
    label(ax, 1.35, 6.22, "Stage A: train projections, fusion and decoder.\nStage B: freeze the entire teacher.", 8, GRAY, ha="left")
    for x, w, title in ((8.88, 3.55, "Spatial-feature distillation"), (13.05, 2.65, "Depth distillation")):
        cx = x+w/2
        block(ax, x, 5.89, w, 0.42, title, color=OCHRE, fill="#fbf5ec")
        arrow(ax, [(cx, 6.75), (cx, 6.31)], OCHRE, True)
        arrow(ax, [(cx, 5.45), (cx, 5.89)], OCHRE, True)
    label(ax, 1.35, 4.12, "Deployment: student only · 141.7K parameters", 8, TEAL, "bold", ha="left")
    label(ax, 15.75, 3.75, "Both heads: residual + valid-token mean anchor → positive metric depth", 7.6, GRAY, ha="right")
    ax.plot([0.1, 15.85], [3.52, 3.52], color="#ccd3d9", linewidth=0.7)
    ax.plot([8.05, 8.05], [0.43, 3.28], color="#dce1e6", linewidth=0.7)
    label(ax, 0.1, 3.20, "(b)  Appearance-enriched sensor fusion", 10.5, weight="bold", ha="left")
    label(ax, 8.35, 3.20, "(c)  Coverage-aware knowledge transfer", 10.5, weight="bold", ha="left")

    block(ax, 0.15, 2.08, 1.8, 0.72, "RGB features", "1/16 resolution")
    block(ax, 0.15, 0.97, 1.8, 0.72, "ToF tokens", "mean, std, valid, box", TEAL, "#eaf4f1")
    block(ax, 2.42, 2.08, 2.0, 0.72, "Footprint pooling", "Appearance per zone", TEAL, "#eaf4f1")
    block(ax, 2.42, 0.97, 2.0, 0.72, "Concatenate + MLP", "Appearance + sensor", TEAL, "#eaf4f1")
    block(ax, 4.95, 0.97, 2.65, 0.72, "Cross-attention", "Geometry + global heads", TEAL, "#eaf4f1")
    arrow(ax, [(1.95, 2.44), (2.42, 2.44)])
    arrow(ax, [(1.95, 1.33), (2.42, 1.33)], TEAL)
    arrow(ax, [(3.42, 2.08), (3.42, 1.69)], TEAL)
    arrow(ax, [(4.42, 1.33), (4.95, 1.33)], TEAL)
    label(ax, 4.69, 1.57, "K,V", 7.2, TEAL)
    label(ax, 6.28, 2.72, "Decoder features at 1/16 or 1/8", 7.7, BLUE)
    arrow(ax, [(6.28, 2.48), (6.28, 1.69)], BLUE)
    label(ax, 6.48, 2.12, "Q", 8, BLUE)
    label(ax, 0.15, 0.55, "Separate teacher / student weights · invalid-token mask · learnable null token", 7.5, GRAY, ha="left")

    label(ax, 8.4, 2.75, "Teacher targets are detached; gradients update the student and adapters.", 7.8, GRAY, ha="left")
    label(ax, 8.4, 2.28,
          r"$\mathcal{L}=\mathcal{L}_{\mathrm{MSE}}+0.1\mathcal{L}_{\mathrm{grad}}+r(e)\,[0.5\mathcal{L}_{\mathrm{depth}}+0.05\mathcal{L}_{\mathrm{feat}}]$",
          12, ha="left")
    label(ax, 8.4, 1.80, "Depth: Smooth L1  ·  Features: cosine loss after 1 × 1 adapters", 8, GRAY, ha="left")
    label(ax, 8.4, 1.39, "Coverage weights: outside 2 / inside 1; normalize before confidence.", 8, GRAY, ha="left")
    label(ax, 8.4, 1.00, r"Confidence: $\exp(-|D_T-D_{GT}|/0.25)$; valid target pixels only.", 8, GRAY, ha="left")
    label(ax, 8.4, 0.55, "Schedule: GT only (1–5) → ramp (6–10) → full distillation", 8, OCHRE, ha="left")
    arrow(ax, [(0.2, 0.12), (0.7, 0.12)])
    label(ax, 0.82, 0.12, "Feature flow", 7.5, GRAY, ha="left")
    arrow(ax, [(2.5, 0.12), (3.0, 0.12)], OCHRE, True)
    label(ax, 3.12, 0.12, "Training supervision", 7.5, GRAY, ha="left")
    label(ax, 15.8, 0.12, "Fusion is interleaved with decoding; RGB skip connections are summarized.", 7, GRAY, ha="right")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        path = OUTPUT_DIR / f"depth_refinement_architecture.{ext}"
        metadata = {"Date": None} if ext == "svg" else None
        fig.savefig(path, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08, metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    render()
