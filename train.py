"""Train the depth-refinement network from ``config.yml``."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

from data import build_zjul5_dataloader, get_zjul5_manifest_splits
from main import (
    DEFAULT_CONFIG_PATH,
    build_model,
    load_config,
    print_batch_summary,
    print_model_summary,
)
from training import train_model


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

    train_model(
        model,
        train_loader,
        validation_loader,
        training_options,
        training_data_options,
    )


if __name__ == "__main__":
    main()
