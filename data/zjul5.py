"""PyTorch input pipeline for the ZJU-L5 RGB/ToF/depth dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def rasterize_calibrated_tof(
    hist_data: np.ndarray,
    rect_data: np.ndarray,
    sparse_mask: np.ndarray,
    image_size: tuple[int, int],
) -> Tensor:
    """Rasterize each calibrated ToF zone into RGB pixel coordinates.

    The returned channels are mean depth, distribution standard deviation, and
    validity. Rectangles may extend outside the camera frame and are clipped.
    """

    if hist_data.shape != (64, 2):
        raise ValueError(f"expected hist_data shape (64, 2), got {hist_data.shape}")
    if rect_data.shape != (64, 4):
        raise ValueError(f"expected fr shape (64, 4), got {rect_data.shape}")
    if sparse_mask.shape != (64,):
        raise ValueError(f"expected mask shape (64,), got {sparse_mask.shape}")

    height, width = image_size
    features = np.zeros((3, height, width), dtype=np.float32)
    for zone_index, (top, left, bottom, right) in enumerate(rect_data):
        mean, standard_deviation = hist_data[zone_index]
        valid = (
            bool(sparse_mask[zone_index])
            and np.isfinite(mean)
            and mean > 0
            and np.isfinite(standard_deviation)
            and standard_deviation >= 0
        )
        if not valid:
            continue
        y0 = int(np.clip(top, 0, height))
        x0 = int(np.clip(left, 0, width))
        y1 = int(np.clip(bottom, 0, height))
        x1 = int(np.clip(right, 0, width))
        if y1 <= y0 or x1 <= x0:
            continue
        features[0, y0:y1, x0:x1] = mean
        features[1, y0:y1, x0:x1] = standard_deviation
        features[2, y0:y1, x0:x1] = 1.0
    return torch.from_numpy(features)


def build_calibrated_tof_tokens(
    hist_data: np.ndarray,
    rect_data: np.ndarray,
    sparse_mask: np.ndarray,
    image_size: tuple[int, int],
) -> Tensor:
    """Encode 64 ToF zones as metric observations plus normalized geometry."""

    if hist_data.shape != (64, 2):
        raise ValueError(f"expected hist_data shape (64, 2), got {hist_data.shape}")
    if rect_data.shape != (64, 4):
        raise ValueError(f"expected fr shape (64, 4), got {rect_data.shape}")
    if sparse_mask.shape != (64,):
        raise ValueError(f"expected mask shape (64,), got {sparse_mask.shape}")

    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")

    tokens = np.zeros((64, 7), dtype=np.float32)
    top = np.clip(rect_data[:, 0], 0, height).astype(np.float32)
    left = np.clip(rect_data[:, 1], 0, width).astype(np.float32)
    bottom = np.clip(rect_data[:, 2], 0, height).astype(np.float32)
    right = np.clip(rect_data[:, 3], 0, width).astype(np.float32)

    rect_height = bottom - top
    rect_width = right - left
    mean = hist_data[:, 0]
    standard_deviation = hist_data[:, 1]
    valid = (
        sparse_mask
        & np.isfinite(mean)
        & (mean > 0)
        & np.isfinite(standard_deviation)
        & (standard_deviation >= 0)
        & (rect_height > 0)
        & (rect_width > 0)
    )

    tokens[:, 0] = np.where(valid, mean, 0.0)
    tokens[:, 1] = np.where(valid, standard_deviation, 0.0)
    tokens[:, 2] = valid.astype(np.float32)
    tokens[:, 3] = (top + bottom) * (0.5 / height)
    tokens[:, 4] = (left + right) * (0.5 / width)
    tokens[:, 5] = rect_height / height
    tokens[:, 6] = rect_width / width
    return torch.from_numpy(tokens)


class ZJUL5Dataset(Dataset[dict[str, Any]]):
    """Load paired RGB, 8x8 ToF depth, and 480x640 target depth.

    The root ``data.json`` manifest is used instead of globbing files so train,
    validation, and test membership remains explicit and reproducible.

    Returned sample keys:
        ``image``: normalized ``float32`` tensor shaped ``(3, 480, 640)``.
        ``sparse_depth``: raw ToF mean depth shaped ``(1, 8, 8)``.
        ``tof_features``: calibrated mean/std/validity shaped ``(3, H, W)``.
        ``tof_tokens``: 64 mean/std/validity/rectangle tokens shaped ``(64, 7)``.
        ``target_depth``: high-resolution depth shaped ``(1, 480, 640)``;
        values outside the configured metric range are invalid and zeroed.
        ``target_valid_mask``: valid target pixels shaped ``(1, 480, 640)``.
        ``sparse_valid_mask``: valid ToF zones shaped ``(1, 8, 8)``.
        ``path``: path of the source HDF5 file relative to the dataset root.
    """

    REQUIRED_KEYS = {"rgb", "depth", "hist_data", "fr", "mask"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        manifest: str | Path = "data.json",
        normalize_image: bool = True,
        image_mean: tuple[float, float, float] | list[float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] | list[float] = (0.229, 0.224, 0.225),
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        legacy_clamp_out_of_range: bool = False,
        augment: bool = False,
        horizontal_flip_probability: float = 0.5,
        photometric_probability: float = 0.5,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"ZJU-L5 root directory not found: {self.root}")

        manifest_path = Path(manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ZJU-L5 manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest_data = json.load(manifest_file)
        if not isinstance(manifest_data, dict):
            raise ValueError("ZJU-L5 manifest root must be a mapping")

        if split == "all":
            entries = [
                entry
                for split_entries in manifest_data.values()
                for entry in split_entries
            ]
        else:
            if split not in manifest_data:
                available = ", ".join(sorted(manifest_data))
                raise ValueError(f"unknown split {split!r}; available splits: {available}, all")
            entries = manifest_data[split]

        if not isinstance(entries, list) or not entries:
            raise ValueError(f"ZJU-L5 split {split!r} contains no samples")

        self.sample_paths: list[Path] = []
        self.relative_paths: list[str] = []
        for entry in entries:
            relative_path = entry.get("filename") if isinstance(entry, dict) else entry
            if not isinstance(relative_path, str):
                raise ValueError("each manifest entry must contain a string 'filename'")
            sample_path = self.root / relative_path
            if not sample_path.is_file():
                raise FileNotFoundError(f"manifest sample not found: {sample_path}")
            self.sample_paths.append(sample_path)
            self.relative_paths.append(relative_path)

        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("image_mean and image_std must each contain three values")
        if any(value <= 0 for value in image_std):
            raise ValueError("all image_std values must be positive")
        if min_depth <= 0:
            raise ValueError("min_depth must be positive")
        if max_depth <= min_depth:
            raise ValueError("max_depth must be greater than min_depth")
        for name, probability in (
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("photometric_probability", photometric_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        self.split = split
        self.normalize_image = normalize_image
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.legacy_clamp_out_of_range = bool(legacy_clamp_out_of_range)
        self.augment = augment and split == "train"
        self.horizontal_flip_probability = horizontal_flip_probability
        self.photometric_probability = photometric_probability

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_path = self.sample_paths[index]
        with h5py.File(sample_path, "r") as sample_file:
            missing_keys = self.REQUIRED_KEYS.difference(sample_file.keys())
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise KeyError(f"{sample_path} is missing required datasets: {missing}")

            rgb = np.asarray(sample_file["rgb"], dtype=np.uint8)
            target_depth = np.asarray(sample_file["depth"], dtype=np.float32)
            hist_data = np.asarray(sample_file["hist_data"], dtype=np.float32)
            rect_data = np.asarray(sample_file["fr"], dtype=np.int32)
            sparse_mask = np.asarray(sample_file["mask"], dtype=np.bool_)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected RGB shape (H, W, 3), got {rgb.shape} in {sample_path}")
        if target_depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"RGB/depth size mismatch in {sample_path}: {rgb.shape[:2]} vs "
                f"{target_depth.shape}"
            )
        if (
            hist_data.shape != (64, 2)
            or rect_data.shape != (64, 4)
            or sparse_mask.shape != (64,)
        ):
            raise ValueError(
                f"expected hist_data (64, 2), fr (64, 4), and mask (64,), got "
                f"{hist_data.shape}, {rect_data.shape}, and {sparse_mask.shape} "
                f"in {sample_path}"
            )

        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        image = image.div_(255.0)

        sparse_depth_array = np.where(sparse_mask, hist_data[:, 0], 0.0)
        sparse_depth = torch.from_numpy(
            np.ascontiguousarray(sparse_depth_array.reshape(1, 8, 8))
        ).float()
        sparse_valid_mask = torch.from_numpy(
            np.ascontiguousarray(sparse_mask.reshape(1, 8, 8))
        )
        tof_features = rasterize_calibrated_tof(
            hist_data,
            rect_data,
            sparse_mask,
            rgb.shape[:2],
        )
        tof_tokens = build_calibrated_tof_tokens(
            hist_data,
            rect_data,
            sparse_mask,
            rgb.shape[:2],
        )

        finite_positive = np.isfinite(target_depth) & (target_depth > 0)
        if self.legacy_clamp_out_of_range:
            target_valid_mask_array = finite_positive
            target_depth = np.where(
                target_valid_mask_array,
                np.clip(target_depth, self.min_depth, self.max_depth),
                0.0,
            )
        else:
            target_valid_mask_array = (
                finite_positive
                & (target_depth >= self.min_depth)
                & (target_depth <= self.max_depth)
            )
            target_depth = np.where(target_valid_mask_array, target_depth, 0.0)

        target_depth_tensor = torch.from_numpy(
            np.ascontiguousarray(target_depth[None, ...])
        ).float()
        target_valid_mask = torch.from_numpy(
            np.ascontiguousarray(target_valid_mask_array[None, ...])
        )

        if self.augment and random.random() < self.photometric_probability:
            gamma = random.uniform(0.9, 1.1)
            brightness = random.uniform(0.75, 1.25)
            color = image.new_tensor(
                [random.uniform(0.9, 1.1) for _ in range(3)]
            ).view(3, 1, 1)
            image = (image.pow(gamma) * brightness * color).clamp(0, 1)
        if self.augment and random.random() < self.horizontal_flip_probability:
            image = image.flip(-1)
            tof_features = tof_features.flip(-1)
            target_depth_tensor = target_depth_tensor.flip(-1)
            target_valid_mask = target_valid_mask.flip(-1)
            sparse_depth = sparse_depth.flip(-1)
            sparse_valid_mask = sparse_valid_mask.flip(-1)
            tof_tokens[:, 4] = 1.0 - tof_tokens[:, 4]
        if self.normalize_image:
            image = (image - self.image_mean) / self.image_std

        return {
            "image": image,
            "sparse_depth": sparse_depth,
            "tof_features": tof_features,
            "tof_tokens": tof_tokens,
            "target_depth": target_depth_tensor,
            "target_valid_mask": target_valid_mask,
            "sparse_valid_mask": sparse_valid_mask,
            "path": self.relative_paths[index],
        }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_zjul5_manifest_splits(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the explicitly defined splits in the configured manifest."""

    root = Path(config["root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ZJU-L5 root directory not found: {root}")
    manifest_path = Path(config.get("manifest", "data.json")).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ZJU-L5 manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest_data = json.load(manifest_file)
    if not isinstance(manifest_data, dict):
        raise ValueError("ZJU-L5 manifest root must be a mapping")
    return tuple(manifest_data)


def build_zjul5_dataloader(config: dict[str, Any]) -> DataLoader[dict[str, Any]]:
    """Construct a ZJU-L5 DataLoader from the ``data`` YAML section."""

    dataset_options = {
        "root": config["root"],
        "split": config.get("split", "train"),
        "manifest": config.get("manifest", "data.json"),
        "normalize_image": config.get("normalize_image", True),
        "image_mean": config.get("image_mean", (0.485, 0.456, 0.406)),
        "image_std": config.get("image_std", (0.229, 0.224, 0.225)),
        "min_depth": config.get("min_depth", 0.1),
        "max_depth": config.get("max_depth", 10.0),
        "legacy_clamp_out_of_range": config.get(
            "legacy_clamp_out_of_range", False
        ),
        "augment": config.get("augment", False),
        "horizontal_flip_probability": config.get(
            "horizontal_flip_probability", 0.5
        ),
        "photometric_probability": config.get("photometric_probability", 0.5),
    }
    dataset = ZJUL5Dataset(**dataset_options)

    batch_size = int(config.get("batch_size", 1))
    num_workers = int(config.get("num_workers", 0))
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if num_workers < 0:
        raise ValueError("data.num_workers cannot be negative")

    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 0)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(config.get("shuffle", dataset.split == "train")),
        num_workers=num_workers,
        pin_memory=bool(config.get("pin_memory", False)),
        drop_last=bool(config.get("drop_last", False)),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )
