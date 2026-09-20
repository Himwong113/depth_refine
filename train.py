"""Train the depth-refinement network from ``config.yml``."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import torch

from data import build_zjul5_dataloader, get_zjul5_manifest_splits
from main import (
    DEFAULT_CONFIG_PATH,
    build_model,
    load_config,
    print_batch_summary,
    print_model_summary,
)
from training import train_model


def _build_distillation_teacher(training_options: dict) -> torch.nn.Module | None:
    """Build and strictly restore the frozen Stage-A teacher when requested."""

    if training_options.get("objective", "legacy") != "student_distillation_v4":
        return None
    distillation_options = training_options.get("distillation", {})
    teacher_options = distillation_options.get("teacher_model")
    checkpoint_value = distillation_options.get("teacher_checkpoint")
    if not isinstance(teacher_options, dict) or checkpoint_value is None:
        raise ValueError(
            "student distillation requires distillation.teacher_model and "
            "distillation.teacher_checkpoint"
        )
    teacher = build_model({"model": teacher_options})
    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("teacher checkpoint has no model state dictionary")
    checkpoint_architecture = checkpoint.get("architecture")
    if checkpoint_architecture not in {None, "teacher_v4"}:
        raise ValueError(
            f"expected a teacher_v4 checkpoint, got {checkpoint_architecture!r}"
        )
    teacher.load_state_dict(state, strict=True)
    return teacher


def _resolve_validation_split(
    requested_split: str | None,
    available_splits: tuple[str, ...],
) -> tuple[str | None, bool]:
    """Select validation data, falling back from a missing val split to test."""

    if requested_split is None:
        return None, False
    if requested_split in available_splits:
        return requested_split, False
    if requested_split == "val" and "test" in available_splits:
        return "test", True
    available = ", ".join(available_splits)
    raise ValueError(
        f"validation split {requested_split!r} is unavailable; manifest splits: "
        f"{available}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the depth-refinement model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume model, optimizer, scheduler, scaler, and epoch from a checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="override the total target epoch count from config.yml",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    if "data" not in config or "training" not in config:
        raise ValueError("training requires 'data' and 'training' config sections")

    model = build_model(config)
    print_model_summary(
        model,
        arguments.config,
        config["model"],
        config.get("summary", {}),
    )

    available_splits = get_zjul5_manifest_splits(config["data"])
    if "train" not in available_splits:
        raise ValueError("the ZJU-L5 manifest must define a 'train' split")

    training_data_options = dict(config["data"])
    training_data_options["split"] = "train"
    training_data_options["shuffle"] = True
    train_loader = build_zjul5_dataloader(training_data_options)
    if training_data_options.get("inspect_first_batch", True):
        print_batch_summary(train_loader)

    requested_validation_split = config["training"].get("validation_split", "val")
    validation_split, using_test_fallback = _resolve_validation_split(
        requested_validation_split,
        available_splits,
    )
    validation_loader = None
    if validation_split is not None:
        if using_test_fallback:
            warnings.warn(
                "manifest has no 'val' split; using 'test' for validation. "
                "Because test data now influences model selection, its final "
                "metrics are not an unbiased performance estimate.",
                RuntimeWarning,
                stacklevel=2,
            )
        validation_data_options = dict(training_data_options)
        validation_data_options["split"] = validation_split
        validation_data_options["shuffle"] = False
        validation_data_options["inspect_first_batch"] = False
        validation_loader = build_zjul5_dataloader(validation_data_options)

    training_options = dict(config["training"])
    training_options["validation_split"] = validation_split
    training_options["validation_uses_test_fallback"] = using_test_fallback
    if arguments.resume is not None:
        training_options["resume_from"] = str(arguments.resume)
    if arguments.epochs is not None:
        if arguments.epochs <= 0:
            raise ValueError("--epochs must be positive")
        training_options["epochs"] = arguments.epochs

    teacher_model = _build_distillation_teacher(training_options)

    train_model(
        model,
        train_loader,
        validation_loader,
        training_options,
        training_data_options,
        teacher_model,
    )


if __name__ == "__main__":
    main()
