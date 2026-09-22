"""Benchmark fixed-resolution student inference on the selected local device."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from eval import load_checkpoint
from export_student import _example_tokens
from main import build_model, format_memory_size, load_config
from training import resolve_device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/student_v5.yml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    arguments = parser.parse_args()
    if min(arguments.height, arguments.width) < 32:
        raise ValueError("height and width must both be at least 32")
    if arguments.warmup < 0 or arguments.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")

    config = load_config(arguments.config)
    device = resolve_device(arguments.device)
    model = build_model(config).to(device).eval()
    if getattr(model, "architecture", None) not in {"student_v4", "student_v5"}:
        raise ValueError("benchmark requires a student_v4 or student_v5 model")
    if arguments.checkpoint is not None:
        load_checkpoint(model, arguments.checkpoint, device)

    image = torch.randn(
        1, 3, arguments.height, arguments.width, device=device
    )
    tokens = _example_tokens().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    fp16_size = parameters * torch.empty((), dtype=torch.float16).element_size()

    with torch.inference_mode():
        for _ in range(arguments.warmup):
            model(image, tokens)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elapsed_ms: list[float] = []
        for _ in range(arguments.iterations):
            start = time.perf_counter()
            model(image, tokens)
            _synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    timings = torch.tensor(elapsed_ms)
    p50 = timings.quantile(0.50).item()
    p95 = timings.quantile(0.95).item()
    print(f"Architecture: {getattr(model, 'architecture')}")
    print(f"Device: {device}")
    print(f"Input/output: {arguments.width}×{arguments.height}")
    print(f"Parameters: {parameters:,}")
    print(f"FP16 parameter size: {format_memory_size(fp16_size)}")
    print(f"Warmup/iterations: {arguments.warmup}/{arguments.iterations}")
    print(f"Model-only latency p50: {p50:.3f} ms")
    print(f"Model-only latency p95: {p95:.3f} ms")
    print(f"Throughput from p50: {1000.0 / p50:.2f} FPS")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        print(f"Peak allocated CUDA memory: {format_memory_size(peak)}")


if __name__ == "__main__":
    main()
