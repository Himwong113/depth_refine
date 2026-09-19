"""Evaluation visualizations for RGB-guided depth refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


class EvaluationVisualizer:
    """Save RGB, sparse depth, refined depth, and ground truth panels."""

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
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.saved_images = 0

        self.colormap = plt.get_cmap("turbo").copy()
        self.colormap.set_bad(color="black")

    def _rgb_image(self, image: Tensor) -> np.ndarray:
        rgb = image.detach().cpu().float().numpy()
        if self.normalize_image:
            rgb = rgb * self.image_std + self.image_mean
        return np.clip(rgb.transpose(1, 2, 0), 0.0, 1.0)

    def _depth_range(self, target: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
        if self.depth_max is not None:
            return self.depth_min, self.depth_max
        valid_values = target[valid_mask]
        if valid_values.size == 0:
            return self.depth_min, max(self.depth_min + 1.0, 1.0)
        robust_maximum = float(np.percentile(valid_values, 99.0))
        return self.depth_min, max(robust_maximum, self.depth_min + 1e-3)

    def __call__(
        self,
        batch: dict[str, Any],
        prediction: Tensor,
        batch_index: int,
    ) -> None:
        del batch_index
        if self.saved_images >= self.max_images:
            return

        batch_size = prediction.shape[0]
        for sample_index in range(batch_size):
            if self.saved_images >= self.max_images:
                break

            rgb = self._rgb_image(batch["image"][sample_index])
            sparse = batch["sparse_depth"][sample_index, 0].cpu().float().numpy()
            sparse_valid = batch["sparse_valid_mask"][sample_index, 0].cpu().numpy().astype(bool)
            refined = prediction[sample_index, 0].detach().cpu().float().numpy()
            target = batch["target_depth"][sample_index, 0].cpu().float().numpy()
            target_valid = (
                batch["target_valid_mask"][sample_index, 0].cpu().numpy().astype(bool)
            )

            sparse_display = np.where(sparse_valid, sparse, np.nan)
            refined_display = np.where(np.isfinite(refined), refined, np.nan)
            target_display = np.where(target_valid, target, np.nan)
            depth_min, depth_max = self._depth_range(target, target_valid)

            valid_error = target_valid & np.isfinite(refined)
            sample_mae = float(np.mean(np.abs(refined[valid_error] - target[valid_error])))

            figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[1].imshow(
                sparse_display,
                cmap=self.colormap,
                vmin=depth_min,
                vmax=depth_max,
                interpolation="nearest",
            )
            axes[1].set_title("Sparse ToF depth (8×8)")
            axes[2].imshow(
                refined_display,
                cmap=self.colormap,
                vmin=depth_min,
                vmax=depth_max,
            )
            axes[2].set_title(f"Refined depth\nMAE {sample_mae:.3f} m")
            color_image = axes[3].imshow(
                target_display,
                cmap=self.colormap,
                vmin=depth_min,
                vmax=depth_max,
            )
            axes[3].set_title("Ground truth")
            for axis in axes:
                axis.axis("off")
            figure.colorbar(
                color_image,
                ax=axes[1:],
                label="Depth (m)",
                fraction=0.025,
                pad=0.02,
            )

            relative_path = Path(batch["path"][sample_index])
            safe_name = "_".join(relative_path.with_suffix("").parts) + ".png"
            figure.savefig(self.output_dir / safe_name, dpi=140, bbox_inches="tight")
            plt.close(figure)
            self.saved_images += 1

