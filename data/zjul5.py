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


class ZJUL5Dataset(Dataset[dict[str, Any]]):
    """Load paired RGB, 8x8 ToF depth, and 480x640 target depth.

    The official root ``data.json`` manifest is used instead of globbing files.
    This intentionally excludes legacy samples whose HDF5 schema differs from
    the train/test samples.

    Returned sample keys:
        ``image``: normalized ``float32`` tensor shaped ``(3, 480, 640)``.
        ``sparse_depth``: ToF mean depth shaped ``(1, 8, 8)``; invalid zones are 0.
        ``target_depth``: high-resolution depth shaped ``(1, 480, 640)``.
        ``target_valid_mask``: valid target pixels shaped ``(1, 480, 640)``.
        ``sparse_valid_mask``: valid ToF zones shaped ``(1, 8, 8)``.
        ``path``: path of the source HDF5 file relative to the dataset root.
    """

    REQUIRED_KEYS = {"rgb", "depth", "hist_data", "mask"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        manifest: str | Path = "data.json",
        normalize_image: bool = True,
        image_mean: tuple[float, float, float] | list[float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] | list[float] = (0.229, 0.224, 0.225),
        max_depth: float | None = None,
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
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be positive or null")

        self.split = split
        self.normalize_image = normalize_image
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.max_depth = max_depth

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
            sparse_mask = np.asarray(sample_file["mask"], dtype=np.bool_)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected RGB shape (H, W, 3), got {rgb.shape} in {sample_path}")
        if target_depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"RGB/depth size mismatch in {sample_path}: {rgb.shape[:2]} vs "
                f"{target_depth.shape}"
            )
        if hist_data.shape != (64, 2) or sparse_mask.shape != (64,):
            raise ValueError(
                f"expected hist_data (64, 2) and mask (64,), got "
                f"{hist_data.shape} and {sparse_mask.shape} in {sample_path}"
            )

        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        image = image.div_(255.0)
        if self.normalize_image:
            image = (image - self.image_mean) / self.image_std

        sparse_depth_array = np.where(sparse_mask, hist_data[:, 0], 0.0)
        sparse_depth = torch.from_numpy(
            np.ascontiguousarray(sparse_depth_array.reshape(1, 8, 8))
        ).float()
        sparse_valid_mask = torch.from_numpy(
            np.ascontiguousarray(sparse_mask.reshape(1, 8, 8))
        )

        target_valid_mask_array = np.isfinite(target_depth) & (target_depth > 0)
        target_depth = np.where(target_valid_mask_array, target_depth, 0.0)
        if self.max_depth is not None:
            target_valid_mask_array &= target_depth <= self.max_depth
            target_depth = np.where(target_valid_mask_array, target_depth, 0.0)

        target_depth_tensor = torch.from_numpy(
            np.ascontiguousarray(target_depth[None, ...])
        ).float()
        target_valid_mask = torch.from_numpy(
            np.ascontiguousarray(target_valid_mask_array[None, ...])
        )

        return {
            "image": image,
            "sparse_depth": sparse_depth,
            "target_depth": target_depth_tensor,
            "target_valid_mask": target_valid_mask,
            "sparse_valid_mask": sparse_valid_mask,
            "path": self.relative_paths[index],
        }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_zjul5_dataloader(config: dict[str, Any]) -> DataLoader[dict[str, Any]]:
    """Construct a ZJU-L5 DataLoader from the ``data`` YAML section."""

    dataset_options = {
        "root": config["root"],
        "split": config.get("split", "train"),
        "manifest": config.get("manifest", "data.json"),
        "normalize_image": config.get("normalize_image", True),
        "image_mean": config.get("image_mean", (0.485, 0.456, 0.406)),
        "image_std": config.get("image_std", (0.229, 0.224, 0.225)),
        "max_depth": config.get("max_depth"),
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

