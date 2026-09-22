"""Export a fixed-resolution v4 or v5 student TorchScript artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import torch

from main import build_model, load_config


def _load_student_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a state dictionary or training checkpoint")
    architecture = checkpoint.get("architecture")
    expected_architecture = getattr(model, "architecture", None)
    if architecture not in {None, expected_architecture}:
        raise ValueError(
            f"expected {expected_architecture} checkpoint, got {architecture!r}"
        )
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model state dictionary")
    if any("teacher" in key or "adapter" in key for key in state):
        raise ValueError("deployable student state unexpectedly contains training modules")
    model.load_state_dict(state, strict=True)


def _example_tokens() -> torch.Tensor:
    tokens = torch.zeros(1, 64, 7)
    tokens[..., 0] = 1.0
    tokens[..., 1] = 0.1
    tokens[..., 2] = 1.0
    coordinate = (torch.arange(8) + 0.5) / 8.0
    center_y, center_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    tokens[..., 3] = center_y.flatten()
    tokens[..., 4] = center_x.flatten()
    tokens[..., 5:7] = 1.0 / 8.0
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/student_v4.yml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("student_v4_480x640.pt"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    arguments = parser.parse_args()
    if arguments.height < 32 or arguments.width < 32:
        raise ValueError("export height and width must both be at least 32")

    config = load_config(arguments.config)
    model = build_model(config).cpu().eval()
    if getattr(model, "architecture", None) not in {"student_v4", "student_v5"}:
        raise ValueError("export requires a student_v4 or student_v5 architecture")
    _load_student_checkpoint(model, arguments.checkpoint)
    image = torch.randn(1, 3, arguments.height, arguments.width)
    tokens = _example_tokens()
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        exported = torch.jit.trace(model, (image, tokens), strict=True)
        exported = torch.jit.freeze(exported)
        reference = model(image, tokens)
        candidate = exported(image, tokens)
    maximum_error = (reference - candidate).abs().max().item()
    if maximum_error > arguments.tolerance:
        raise RuntimeError(
            f"export parity error {maximum_error:.6g} exceeds "
            f"tolerance {arguments.tolerance:.6g}"
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(exported, arguments.output)
    print(f"Exported student: {arguments.output.resolve()}")
    print(f"Input:  image (1, 3, {arguments.height}, {arguments.width})")
    print("Input:  tof_tokens (1, 64, 7)")
    print(f"Output: depth (1, 1, {arguments.height}, {arguments.width})")
    print(f"Maximum PyTorch/TorchScript error: {maximum_error:.6g}")


if __name__ == "__main__":
    main()
