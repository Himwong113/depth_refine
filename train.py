"""Train the depth-refinement network from ``config.yml``."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import build_zjul5_dataloader
from main import (
    DEFAULT_CONFIG_PATH,
    build_model,
    load_config,
    print_batch_summary,
    print_model_summary,
)
from training import train_model


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

    train_loader = build_zjul5_dataloader(config["data"])
    if config["data"].get("inspect_first_batch", True):
        print_batch_summary(train_loader)

    validation_split = config["training"].get("validation_split", "test")
    validation_loader = None
    if validation_split is not None:
        validation_data_options = dict(config["data"])
        validation_data_options["split"] = validation_split
        validation_data_options["shuffle"] = False
        validation_data_options["inspect_first_batch"] = False
        validation_loader = build_zjul5_dataloader(validation_data_options)

    training_options = dict(config["training"])
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
        config["data"],
    )


if __name__ == "__main__":
    main()
