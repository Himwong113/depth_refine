"""Evaluation visualizations for RGB-guided depth refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from torch import Tensor


def create_depth_comparison_figure(
    batch: dict[str, Any],
    prediction: Tensor,
    teacher_prediction: Tensor | None = None,
    sample_index: int = 0,
    normalize_image: bool = True,
    image_mean: Sequence[float] = (0.485, 0.456, 0.406),
    image_std: Sequence[float] = (0.229, 0.224, 0.225),
    depth_min: float = 0.0,
    depth_max: float | None = None,
) -> Figure:
    """Create aligned input, prediction, optional teacher, and target panels."""

    if not 0 <= sample_index < prediction.shape[0]:
        raise IndexError(f"sample_index {sample_index} is outside the prediction batch")
    if (
        teacher_prediction is not None
        and not 0 <= sample_index < teacher_prediction.shape[0]
    ):
        raise IndexError(
            f"sample_index {sample_index} is outside the teacher prediction batch"
        )
    if depth_max is not None and depth_max <= depth_min:
        raise ValueError("depth_max must be greater than depth_min")

    rgb = batch["image"][sample_index].detach().cpu().float().numpy()
    if normalize_image:
        mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        standard_deviation = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        rgb = rgb * standard_deviation + mean
    rgb = np.clip(rgb.transpose(1, 2, 0), 0.0, 1.0)

    if "tof_features" in batch:
        sparse = batch["tof_features"][sample_index, 0].cpu().float().numpy()
        sparse_valid = (
            batch["tof_features"][sample_index, 2].cpu().numpy() > 0.5
        )
        sparse_title = "Calibrated ToF depth"
    else:
        sparse = batch["sparse_depth"][sample_index, 0].cpu().float().numpy()
        sparse_valid = (
            batch["sparse_valid_mask"][sample_index, 0].cpu().numpy().astype(bool)
        )
        sparse_title = "Sparse ToF depth (8×8)"
    refined = prediction[sample_index, 0].detach().cpu().float().numpy()
    teacher = (
        teacher_prediction[sample_index, 0].detach().cpu().float().numpy()
        if teacher_prediction is not None
        else None
    )
    target = batch["target_depth"][sample_index, 0].cpu().float().numpy()
    target_valid = (
        batch["target_valid_mask"][sample_index, 0].cpu().numpy().astype(bool)
    )
    target_valid &= np.isfinite(target) & (target > 0)

    sparse_display = np.where(sparse_valid, sparse, np.nan)
    refined_display = np.where(np.isfinite(refined), refined, np.nan)
    teacher_display = (
        np.where(np.isfinite(teacher), teacher, np.nan)
        if teacher is not None
        else None
    )
    target_display = np.where(target_valid, target, np.nan)
    if depth_max is None:
        valid_values = target[target_valid]
        if valid_values.size == 0:
            display_depth_max = max(depth_min + 1.0, 1.0)
        else:
            robust_maximum = float(np.percentile(valid_values, 99.0))
            display_depth_max = max(robust_maximum, depth_min + 1e-3)
    else:
        display_depth_max = depth_max

    valid_error = target_valid & np.isfinite(refined)
    sample_mae = (
        float(np.mean(np.abs(refined[valid_error] - target[valid_error])))
        if np.any(valid_error)
        else float("nan")
    )
    sample_rmse = (
        float(np.sqrt(np.mean(np.square(refined[valid_error] - target[valid_error]))))
        if np.any(valid_error)
        else float("nan")
    )
    teacher_rmse = None
    if teacher is not None:
        valid_teacher_error = target_valid & np.isfinite(teacher)
        teacher_rmse = (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            teacher[valid_teacher_error] - target[valid_teacher_error]
                        )
                    )
                )
            )
            if np.any(valid_teacher_error)
            else float("nan")
        )

    colormap = plt.get_cmap("turbo").copy()
    colormap.set_bad(color="black")
    panel_count = 5 if teacher_display is not None else 4
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(22 if panel_count == 5 else 18, 5),
        constrained_layout=True,
    )
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(
        sparse_display,
        cmap=colormap,
        vmin=depth_min,
        vmax=display_depth_max,
        interpolation="nearest",
    )
    axes[1].set_title(sparse_title)
    prediction_axis = 3 if teacher_display is not None else 2
    target_axis = prediction_axis + 1
    if teacher_display is not None:
        axes[2].imshow(
            teacher_display,
            cmap=colormap,
            vmin=depth_min,
            vmax=display_depth_max,
        )
        axes[2].set_title(f"Teacher depth\nRMSE {teacher_rmse:.3f} m")
    axes[prediction_axis].imshow(
        refined_display,
        cmap=colormap,
        vmin=depth_min,
        vmax=display_depth_max,
    )
    prediction_title = (
        "Student depth" if teacher_display is not None else "Refined depth"
    )
    axes[prediction_axis].set_title(
        f"{prediction_title}\nRMSE {sample_rmse:.3f} m | MAE {sample_mae:.3f} m"
    )
    color_image = axes[target_axis].imshow(
        target_display,
        cmap=colormap,
        vmin=depth_min,
        vmax=display_depth_max,
    )
    axes[target_axis].set_title("Ground truth")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(
        color_image,
        ax=axes[1:],
        label="Depth (m)",
        fraction=0.025,
        pad=0.02,
    )
    return figure


class EvaluationVisualizer:
    """Save aligned teacher/student depth comparisons for evaluation samples."""

    def __init__(
        self,
        output_dir: str | Path,
        max_images: int,
        normalize_image: bool = True,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        depth_min: float = 0.0,
        depth_max: float | None = None,
    ) -> None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        if depth_max is not None and depth_max <= depth_min:
            raise ValueError("depth_max must be greater than depth_min")

        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_images = max_images
        self.normalize_image = normalize_image
        self.image_mean = image_mean
        self.image_std = image_std
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.saved_images = 0

    def __call__(
        self,
        batch: dict[str, Any],
        prediction: Tensor,
        batch_index: int,
    ) -> None:
        del batch_index
        self._save(batch, prediction)

    def compare(
        self,
        batch: dict[str, Any],
        student_prediction: Tensor,
        teacher_prediction: Tensor,
        batch_index: int,
    ) -> None:
        """Save teacher and student outputs generated from one evaluation batch."""

        del batch_index
        self._save(batch, student_prediction, teacher_prediction)

    def _save(
        self,
        batch: dict[str, Any],
        prediction: Tensor,
        teacher_prediction: Tensor | None = None,
    ) -> None:
        if self.saved_images >= self.max_images:
            return

        batch_size = prediction.shape[0]
        for sample_index in range(batch_size):
            if self.saved_images >= self.max_images:
                break

            figure = create_depth_comparison_figure(
                batch,
                prediction,
                teacher_prediction=teacher_prediction,
                sample_index=sample_index,
                normalize_image=self.normalize_image,
                image_mean=self.image_mean,
                image_std=self.image_std,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
            )

            relative_path = Path(batch["path"][sample_index])
            safe_name = "_".join(relative_path.with_suffix("").parts) + ".png"
            figure.savefig(self.output_dir / safe_name, dpi=140, bbox_inches="tight")
            plt.close(figure)
            self.saved_images += 1
