"""Training and validation utilities for depth refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` or validate an explicitly requested training device."""

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA, but CUDA is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("training.device requests MPS, but MPS is unavailable")
    return device


def masked_depth_loss(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    loss_name: str,
    smooth_l1_beta: float = 0.1,
    scale_invariant_lambda: float = 0.85,
    scale_invariant_alpha: float = 10.0,
    minimum_depth: float = 1e-3,
) -> Tensor:
    """Calculate loss using only valid high-resolution target pixels."""

    valid_mask = valid_mask.bool() & torch.isfinite(target) & (target > 0)
    if not torch.any(valid_mask):
        raise RuntimeError("batch contains no valid target-depth pixels")

    # Loss calculations stay in fp32 even during mixed-precision training.
    prediction_values = prediction[valid_mask].float()
    target_values = target[valid_mask].float()
    if not torch.isfinite(prediction_values).all():
        raise FloatingPointError("prediction contains non-finite values at valid pixels")

    if loss_name == "l1":
        return F.l1_loss(prediction_values, target_values)
    if loss_name == "mse":
        return F.mse_loss(prediction_values, target_values)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(
            prediction_values,
            target_values,
            beta=smooth_l1_beta,
        )
    if loss_name == "scale_invariant":
        if not 0.0 <= scale_invariant_lambda <= 1.0:
            raise ValueError("training.scale_invariant_lambda must be between 0 and 1")
        if scale_invariant_alpha <= 0:
            raise ValueError("training.scale_invariant_alpha must be positive")
        if minimum_depth <= 0:
            raise ValueError("training.minimum_depth must be positive")

        prediction_values = prediction_values.clamp_min(minimum_depth)
        target_values = target_values.clamp_min(minimum_depth)
        log_difference = prediction_values.log() - target_values.log()
        loss_squared = log_difference.square().mean()
        scale_correction = scale_invariant_lambda * log_difference.mean().square()
        radicand = (loss_squared - scale_correction).clamp_min(0.0)

        # The subtraction keeps an exact zero loss while stabilizing sqrt at 0.
        epsilon = radicand.new_tensor(1e-12)
        return scale_invariant_alpha * (
            torch.sqrt(radicand + epsilon) - torch.sqrt(epsilon)
        )

    raise ValueError(
        "training.loss must be one of: scale_invariant, l1, mse, smooth_l1"
    )


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    prediction_callback: Callable[[dict[str, Any], Tensor, int], None] | None = None,
) -> dict[str, float]:
    """Evaluate masked MAE and RMSE on a DataLoader."""

    model.eval()
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    valid_pixel_count = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            image = batch["image"].to(device, non_blocking=True)
            sparse_depth = batch["sparse_depth"].to(device, non_blocking=True)
            target = batch["target_depth"].to(device, non_blocking=True)
            valid_mask = batch["target_valid_mask"].to(device, non_blocking=True).bool()

            prediction = model(image, sparse_depth)
            if prediction_callback is not None:
                prediction_callback(batch, prediction, batch_index)
            valid_mask &= torch.isfinite(prediction) & torch.isfinite(target)
            error = prediction[valid_mask] - target[valid_mask]
            absolute_error_sum += error.abs().sum().item()
            squared_error_sum += error.square().sum().item()
            valid_pixel_count += error.numel()

    if valid_pixel_count == 0:
        raise RuntimeError("validation produced no valid target-depth pixels")
    return {
        "mae": absolute_error_sum / valid_pixel_count,
        "rmse": (squared_error_sum / valid_pixel_count) ** 0.5,
    }


def _save_checkpoint(checkpoint_path: Path, state: dict[str, Any]) -> None:
    """Atomically write a checkpoint so interrupted saves do not corrupt it."""

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(checkpoint_path)


def _checkpoint_state(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    best_validation_rmse: float,
    best_epoch: int | None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_config": config,
        "best_validation_rmse": best_validation_rmse,
        "best_epoch": best_epoch,
    }


def _load_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int | None]:
    """Restore training state and return next epoch and best-metric state."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required_keys = {"epoch", "model_state_dict", "optimizer_state_dict"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"resume checkpoint is missing: {missing}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    completed_epoch = int(checkpoint["epoch"])
    best_validation_rmse = float(checkpoint.get("best_validation_rmse", float("inf")))
    best_epoch_value = checkpoint.get("best_epoch")
    best_epoch = int(best_epoch_value) if best_epoch_value is not None else None
    return completed_epoch + 1, best_validation_rmse, best_epoch


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    config: dict[str, Any],
) -> None:
    """Train the model and periodically save resumable checkpoints."""

    epochs = int(config.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")

    device = resolve_device(str(config.get("device", "auto")))
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_name = str(config.get("loss", "scale_invariant"))
    smooth_l1_beta = float(config.get("smooth_l1_beta", 0.1))
    scale_invariant_lambda = float(config.get("scale_invariant_lambda", 0.85))
    scale_invariant_alpha = float(config.get("scale_invariant_alpha", 10.0))
    minimum_depth = float(config.get("minimum_depth", 1e-3))
    gradient_clip_norm = config.get("gradient_clip_norm", 1.0)
    log_every = int(config.get("log_every", 10))
    save_every = int(config.get("save_every", 1))
    validation_every = int(config.get("validation_every", 1))
    max_train_batches = config.get("max_train_batches")
    max_validation_batches = config.get("max_validation_batches")
    if max_train_batches is not None:
        max_train_batches = int(max_train_batches)
    if max_validation_batches is not None:
        max_validation_batches = int(max_validation_batches)

    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_validation_rmse = float("inf")
    best_epoch = None
    resume_from = config.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from).expanduser()
        if not resume_path.is_absolute() and not resume_path.is_file():
            resume_path = checkpoint_dir / resume_path
        start_epoch, best_validation_rmse, best_epoch = _load_resume_checkpoint(
            resume_path,
            model,
            optimizer,
            scaler,
            device,
        )

        # Older epoch checkpoints predate best-metric metadata. Recover it from
        # best.pt when available so a resumed run does not forget prior results.
        best_path = checkpoint_dir / "best.pt"
        if best_validation_rmse == float("inf") and best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_validation_rmse = float(
                best_checkpoint.get("best_validation_rmse", float("inf"))
            )
            best_epoch_value = best_checkpoint.get("best_epoch")
            best_epoch = int(best_epoch_value) if best_epoch_value is not None else None

        print(f"Resumed checkpoint: {resume_path.resolve()}")
        print(f"Continuing from epoch: {start_epoch}")

    if start_epoch > epochs:
        raise ValueError(
            f"checkpoint already completed epoch {start_epoch - 1}, but "
            f"training.epochs is {epochs}; increase the total epoch count"
        )

    print(f"\nTraining device: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Loss: {loss_name}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        completed_batches = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_index > max_train_batches:
                break

            image = batch["image"].to(device, non_blocking=True)
            sparse_depth = batch["sparse_depth"].to(device, non_blocking=True)
            target = batch["target_depth"].to(device, non_blocking=True)
            valid_mask = batch["target_valid_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(image, sparse_depth)
                loss = masked_depth_loss(
                    prediction,
                    target,
                    valid_mask,
                    loss_name,
                    smooth_l1_beta,
                    scale_invariant_lambda,
                    scale_invariant_alpha,
                    minimum_depth,
                )

            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(gradient_clip_norm)
                )
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            completed_batches += 1
            if log_every > 0 and batch_index % log_every == 0:
                print(
                    f"Epoch {epoch}/{epochs} · batch {batch_index}/{len(train_loader)} "
                    f"· loss {epoch_loss / completed_batches:.6f}"
                )

        if completed_batches == 0:
            raise RuntimeError("training completed no batches")
        mean_loss = epoch_loss / completed_batches
        message = f"Epoch {epoch}/{epochs} complete · train loss {mean_loss:.6f}"

        validation_metrics = None
        if (
            validation_loader is not None
            and validation_every > 0
            and epoch % validation_every == 0
        ):
            validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                max_batches=max_validation_batches,
            )
            message += (
                f" · val MAE {validation_metrics['mae']:.6f} "
                f"· val RMSE {validation_metrics['rmse']:.6f}"
            )
        print(message)

        if (
            validation_metrics is not None
            and validation_metrics["rmse"] < best_validation_rmse
        ):
            best_validation_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            best_path = checkpoint_dir / "best.pt"
            _save_checkpoint(
                best_path,
                _checkpoint_state(
                    epoch,
                    model,
                    optimizer,
                    scaler,
                    config,
                    best_validation_rmse,
                    best_epoch,
                ),
            )
            print(
                f"Saved new best checkpoint: {best_path.resolve()} "
                f"(epoch {epoch}, RMSE {best_validation_rmse:.6f})"
            )

        should_save = save_every > 0 and (
            epoch % save_every == 0 or epoch == epochs
        )
        if should_save:
            checkpoint_path = checkpoint_dir / f"depth_refinement_epoch_{epoch:03d}.pt"
            _save_checkpoint(
                checkpoint_path,
                _checkpoint_state(
                    epoch,
                    model,
                    optimizer,
                    scaler,
                    config,
                    best_validation_rmse,
                    best_epoch,
                ),
            )
            print(f"Saved checkpoint: {checkpoint_path.resolve()}")
