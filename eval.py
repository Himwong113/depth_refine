"""Evaluate a trained depth-refinement checkpoint on ZJU-L5."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_zjul5_dataloader
from main import DEFAULT_CONFIG_PATH, build_model, load_config
from training import evaluate, resolve_device
from visualization import EvaluationVisualizer


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> int | None:
    """Load either a training checkpoint or a plain model state dictionary."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        epoch = checkpoint.get("epoch")
        return int(epoch) if epoch is not None else None
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=True)
        return None
    raise ValueError("checkpoint must contain a model state dictionary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint produced by train.py",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--split", choices=("train", "val", "test", "all"))
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save RGB, sparse depth, refined depth, and ground-truth panels",
    )
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--num-visualizations", type=int)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    if "data" not in config:
        raise ValueError("evaluation requires a 'data' config section")
    evaluation_options = config.get("evaluation", {})

    device = resolve_device(arguments.device or evaluation_options.get("device", "auto"))
    model = build_model(config).to(device=device, dtype=torch.float32)
    epoch = load_checkpoint(model, arguments.checkpoint, device)

    data_options = dict(config["data"])
    data_options["split"] = arguments.split or evaluation_options.get("split", "test")
    data_options["shuffle"] = False
    data_options["inspect_first_batch"] = False
    data_loader = build_zjul5_dataloader(data_options)

    max_batches = arguments.max_batches
    if max_batches is None:
        max_batches = evaluation_options.get("max_batches")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    visualize = arguments.visualize or bool(evaluation_options.get("visualize", False))
    visualizer = None
    if visualize:
        visualization_dir = arguments.visualization_dir or Path(
            evaluation_options.get("visualization_dir", "evaluation_outputs")
        )
        num_visualizations = arguments.num_visualizations
        if num_visualizations is None:
            num_visualizations = int(evaluation_options.get("num_visualizations", 8))
        visualizer = EvaluationVisualizer(
            output_dir=visualization_dir,
            max_images=num_visualizations,
            normalize_image=bool(data_options.get("normalize_image", True)),
            image_mean=data_options.get("image_mean", (0.485, 0.456, 0.406)),
            image_std=data_options.get("image_std", (0.229, 0.224, 0.225)),
            depth_min=float(evaluation_options.get("depth_min", 0.0)),
            depth_max=evaluation_options.get("depth_max"),
        )

    metrics = evaluate(
        model,
        data_loader,
        device,
        max_batches=max_batches,
        prediction_callback=visualizer,
    )
    print(f"Checkpoint: {arguments.checkpoint.resolve()}")
    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")
    print(f"Device: {device}")
    print(f"Split: {data_options['split']} ({len(data_loader.dataset):,} samples)")
    print(f"Masked MAE:  {metrics['mae']:.6f} m")
    print(f"Masked RMSE: {metrics['rmse']:.6f} m")
    for label, key in (
        ("0–2 m", "rmse_0_2m"),
        ("2–4 m", "rmse_2_4m"),
        ("4–6 m", "rmse_4_6m"),
        ("6+ m", "rmse_6_infm"),
    ):
        if key in metrics:
            print(f"RMSE {label:>5}: {metrics[key]:.6f} m")
    if visualizer is not None:
        print(
            f"Visualizations: {visualizer.saved_images} saved to "
            f"{visualizer.output_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
