"""Build the depth-refinement model from ``config.yml``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from data import build_zjul5_dataloader
from model import DepthRefinementUNet


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yml"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML file before any model is constructed."""

    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("the YAML root must be a mapping")
    if "model" not in config:
        raise ValueError("config.yml must contain a 'model' section")
    if not isinstance(config["model"], dict):
        raise ValueError("the 'model' section must be a mapping")
    for section in ("summary", "data", "training", "evaluation"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"the {section!r} section must be a mapping")
    return config


def build_model(config: dict[str, Any]) -> DepthRefinementUNet:
    """Pass all values under ``model`` to the PyTorch model constructor."""

    return DepthRefinementUNet(**config["model"])


def format_memory_size(size_bytes: int) -> str:
    """Format a byte count using binary memory units."""

    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    raise RuntimeError("unreachable")


def print_model_summary(
    model: DepthRefinementUNet,
    config_path: Path,
    model_options: dict[str, Any],
    summary_options: dict[str, Any],
) -> None:
    """Apply summary settings and print parameter and memory statistics."""

    dtype_name = summary_options.get("parameter_dtype", "float32")
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtypes:
        raise ValueError(
            f"unsupported summary.parameter_dtype {dtype_name!r}; "
            f"choose one of {', '.join(dtypes)}"
        )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    reported_element_size = torch.empty((), dtype=dtypes[dtype_name]).element_size()
    parameter_bytes = total_parameters * reported_element_size
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    print(f"Config loaded first: {config_path.resolve()}")
    print(f"Model: {model.__class__.__name__}")
    print("Model constructor parameters:")
    for name, value in model_options.items():
        print(f"  {name}: {value}")
    print(f"Parameter dtype: {dtype_name}")
    print(f"Total parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Parameter memory:     {format_memory_size(parameter_bytes)}")
    print(f"Buffer memory:        {format_memory_size(buffer_bytes)}")
    print(f"Estimated model size: {format_memory_size(parameter_bytes + buffer_bytes)}")
    print("Note: model size excludes gradients, optimizer state, and activations.")

    if summary_options.get("show_architecture", False):
        print("\nArchitecture:\n")
        print(model)


def print_batch_summary(data_loader: Any) -> None:
    """Load one real dataset batch and report its tensor contract."""

    batch = next(iter(data_loader))
    dataset = data_loader.dataset
    print(f"\nDataset: {dataset.__class__.__name__}")
    print(f"Split: {dataset.split} ({len(dataset):,} samples)")
    print(f"Image batch:              {tuple(batch['image'].shape)}")
    print(f"Sparse depth batch:       {tuple(batch['sparse_depth'].shape)}")
    print(f"High-resolution depth:    {tuple(batch['target_depth'].shape)}")
    print(f"High-resolution validity: {tuple(batch['target_valid_mask'].shape)}")
    print(f"Sparse validity:          {tuple(batch['sparse_valid_mask'].shape)}")
    print(f"First sample: {batch['path'][0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load config.yml, construct the model, and show its statistics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration path (default: {DEFAULT_CONFIG_PATH})",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    model = build_model(config)
    print_model_summary(
        model,
        arguments.config,
        config["model"],
        config.get("summary", {}),
    )

    if "data" in config:
        data_loader = build_zjul5_dataloader(config["data"])
        if config["data"].get("inspect_first_batch", True):
            print_batch_summary(data_loader)



if __name__ == "__main__":
    main()
