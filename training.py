"""Training and validation utilities for depth refinement."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from visualization import create_depth_comparison_figure


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


class _TensorBoardSampleLogger:
    """Write eval-style inference comparison figures to TensorBoard."""

    def __init__(
        self,
        writer: SummaryWriter,
        epoch: int,
        max_images: int,
        data_config: dict[str, Any],
        tensorboard_config: dict[str, Any],
    ) -> None:
        self.writer = writer
        self.epoch = epoch
        self.max_images = max_images
        self.logged_images = 0
        self.normalize_image = bool(data_config.get("normalize_image", True))
        self.image_mean = data_config.get("image_mean", (0.485, 0.456, 0.406))
        self.image_std = data_config.get("image_std", (0.229, 0.224, 0.225))
        self.depth_min = float(tensorboard_config.get("depth_min", 0.0))
        self.depth_max = tensorboard_config.get("depth_max")
        if self.depth_max is not None:
            self.depth_max = float(self.depth_max)

    def __call__(
        self,
        batch: dict[str, Any],
        prediction: Tensor,
        batch_index: int,
    ) -> None:
        del batch_index
        for sample_index in range(prediction.shape[0]):
            if self.logged_images >= self.max_images:
                return
            figure = create_depth_comparison_figure(
                batch,
                prediction,
                sample_index=sample_index,
                normalize_image=self.normalize_image,
                image_mean=self.image_mean,
                image_std=self.image_std,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
            )
            self.writer.add_figure(
                f"Samples/validation_{self.logged_images + 1}",
                figure,
                global_step=self.epoch,
                close=True,
            )
            self.logged_images += 1


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
    loss_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate masked loss, MAE, and RMSE on a DataLoader."""

    model.eval()
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    valid_pixel_count = 0
    loss_sum = 0.0
    completed_batches = 0

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
            if loss_config is not None:
                loss_sum += masked_depth_loss(
                    prediction,
                    target,
                    valid_mask,
                    str(loss_config.get("loss", "scale_invariant")),
                    float(loss_config.get("smooth_l1_beta", 0.1)),
                    float(loss_config.get("scale_invariant_lambda", 0.85)),
                    float(loss_config.get("scale_invariant_alpha", 10.0)),
                    float(loss_config.get("minimum_depth", 1e-3)),
                ).item()
            completed_batches += 1
            valid_mask &= torch.isfinite(prediction) & torch.isfinite(target)
            error = prediction[valid_mask] - target[valid_mask]
            absolute_error_sum += error.abs().sum().item()
            squared_error_sum += error.square().sum().item()
            valid_pixel_count += error.numel()

    if valid_pixel_count == 0:
        raise RuntimeError("validation produced no valid target-depth pixels")
    metrics = {
        "mae": absolute_error_sum / valid_pixel_count,
        "rmse": (squared_error_sum / valid_pixel_count) ** 0.5,
    }
    if loss_config is not None:
        metrics["loss"] = loss_sum / completed_batches
    return metrics


def _save_checkpoint(checkpoint_path: Path, state: dict[str, Any]) -> None:
    """Atomically write a checkpoint so interrupted saves do not corrupt it."""

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(checkpoint_path)


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the configured epoch-level learning-rate scheduler."""

    scheduler_name = str(config.get("lr_scheduler", "cosine")).lower()
    if scheduler_name in {"none", "constant"}:
        return None
    if scheduler_name != "cosine":
        raise ValueError("training.lr_scheduler must be one of: cosine, constant, none")

    warmup_epochs = int(config.get("warmup_epochs", 0))
    if not 0 <= warmup_epochs < epochs:
        raise ValueError(
            "training.warmup_epochs must be non-negative and smaller than "
            "training.epochs"
        )

    minimum_learning_rate = float(config.get("minimum_learning_rate", 1e-6))
    learning_rate = float(config.get("learning_rate", 3e-4))
    if not 0.0 <= minimum_learning_rate <= learning_rate:
        raise ValueError(
            "training.minimum_learning_rate must be between zero and "
            "training.learning_rate"
        )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=minimum_learning_rate,
    )
    if warmup_epochs == 0:
        return cosine

    warmup_start_factor = float(config.get("warmup_start_factor", 0.1))
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError(
            "training.warmup_start_factor must be greater than zero and at most one"
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _checkpoint_state(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    best_validation_rmse: float,
    best_epoch: int | None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "training_config": config,
        "best_validation_rmse": best_validation_rmse,
        "best_epoch": best_epoch,
    }


def _load_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
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

    initial_scheduler_lrs = [group["lr"] for group in optimizer.param_groups]
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    completed_epoch = int(checkpoint["epoch"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    elif scheduler is not None:
        # Older checkpoints have no scheduler state. Reconstruct the learning
        # rate at the next epoch instead of restarting the schedule.
        for parameter_group, initial_lr in zip(
            optimizer.param_groups, initial_scheduler_lrs, strict=True
        ):
            parameter_group["lr"] = initial_lr
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler.step\(\)` before",
            )
            for _ in range(completed_epoch):
                scheduler.step()

    best_validation_rmse = float(checkpoint.get("best_validation_rmse", float("inf")))
    best_epoch_value = checkpoint.get("best_epoch")
    best_epoch = int(best_epoch_value) if best_epoch_value is not None else None
    return completed_epoch + 1, best_validation_rmse, best_epoch


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    config: dict[str, Any],
    data_config: dict[str, Any] | None = None,
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
    scheduler = _build_lr_scheduler(optimizer, config, epochs)

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

    tensorboard_config = dict(config.get("tensorboard", {}))
    tensorboard_enabled = bool(tensorboard_config.get("enabled", True))
    tensorboard_image_every = int(tensorboard_config.get("image_every", 1))
    tensorboard_num_images = int(tensorboard_config.get("num_images", 2))
    if tensorboard_image_every <= 0:
        raise ValueError("training.tensorboard.image_every must be positive")
    if tensorboard_num_images <= 0:
        raise ValueError("training.tensorboard.num_images must be positive")
    if data_config is None:
        data_config = {}

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
            scheduler,
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

    writer = None
    if tensorboard_enabled:
        tensorboard_dir = Path(
            tensorboard_config.get("log_dir", "runs/depth_refinement")
        ).expanduser()
        writer = SummaryWriter(
            log_dir=str(tensorboard_dir),
            purge_step=start_epoch if resume_from else None,
        )

    print(f"\nTraining device: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Loss: {loss_name}")
    print(f"LR scheduler: {config.get('lr_scheduler', 'cosine')}")
    if writer is not None:
        print(f"TensorBoard log: {Path(writer.log_dir).resolve()}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        current_learning_rate = optimizer.param_groups[0]["lr"]
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
        message = (
            f"Epoch {epoch}/{epochs} complete · train loss {mean_loss:.6f} "
            f"· lr {current_learning_rate:.3e}"
        )

        validation_metrics = None
        if (
            validation_loader is not None
            and validation_every > 0
            and epoch % validation_every == 0
        ):
            prediction_callback = None
            if writer is not None and epoch % tensorboard_image_every == 0:
                prediction_callback = _TensorBoardSampleLogger(
                    writer,
                    epoch,
                    tensorboard_num_images,
                    data_config,
                    tensorboard_config,
                )
            validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                max_batches=max_validation_batches,
                prediction_callback=prediction_callback,
                loss_config=config,
            )
            message += (
                f" · val loss {validation_metrics['loss']:.6f} "
                f"· val MAE {validation_metrics['mae']:.6f} "
                f"· val RMSE {validation_metrics['rmse']:.6f}"
            )
        print(message)

        if writer is not None:
            writer.add_scalar("Loss/train", mean_loss, epoch)
            writer.add_scalar("Learning_rate/epoch", current_learning_rate, epoch)
            if validation_metrics is not None:
                writer.add_scalar("Loss/validation", validation_metrics["loss"], epoch)
                writer.add_scalar(
                    "Metrics/validation_MAE", validation_metrics["mae"], epoch
                )
                writer.add_scalar(
                    "Metrics/validation_RMSE", validation_metrics["rmse"], epoch
                )
            writer.flush()

        if scheduler is not None:
            scheduler.step()

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
                    scheduler,
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
                    scheduler,
                    scaler,
                    config,
                    best_validation_rmse,
                    best_epoch,
                ),
            )
            print(f"Saved checkpoint: {checkpoint_path.resolve()}")

    if writer is not None:
        writer.close()
