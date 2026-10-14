import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data.dataset import HoleLayoutDataset
from .losses import HoleLayoutLoss
from .models import ConditionalHoleUNet


LOSS_NAMES = (
    "total",
    "heatmap",
    "offset",
    "mask",
    "radius",
    "density",
    "count",
    "geometry",
    "region",
)
HEATMAP_METRIC_NAMES = (
    "heatmap_positive",
    "heatmap_negative",
    "num_pos",
    "pred_heatmap_max",
    "pred_heatmap_mean",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: Dict[str, Any]) -> ConditionalHoleUNet:
    cfg = config["model"]
    return ConditionalHoleUNet(
        input_channels=int(cfg["input_channels"]),
        base_channels=int(cfg["base_channels"]),
        tabular_dim=int(cfg["tabular_dim"]),
        mlp_hidden=int(cfg["mlp_hidden"]),
        dropout=float(cfg["dropout"]),
    )


def _move_targets(targets: Dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: HoleLayoutLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    channels_last: bool = False,
    max_grad_norm: Optional[float] = 1.0,
    tracker: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    log_interval: int = 10,
    max_consecutive_nonfinite_gradients: int = 8,
) -> Tuple[Dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    loss_totals: Dict[str, float] = {}
    batches = 0
    skipped_gradient_steps = 0
    consecutive_nonfinite_gradients = 0
    context = torch.enable_grad if training else torch.inference_mode
    if training:
        optimizer.zero_grad(set_to_none=True)
    with context():
        for batch_index, batch in enumerate(tqdm(loader, leave=False), start=1):
            image = batch["image"].to(device, non_blocking=True)
            if channels_last:
                image = image.contiguous(memory_format=torch.channels_last)
            tabular = batch["tabular"].to(device, non_blocking=True)
            targets = _move_targets(batch["targets"], device)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                predictions = model(image, tabular)
                losses = criterion(predictions, targets)
            if not torch.isfinite(losses["total"]).all():
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_index}: "
                    f"{ {name: float(value.detach()) for name, value in losses.items()} }"
                )
            if training:
                assert scaler is not None
                group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
                group_size = min(accumulation_steps, len(loader) - group_start + 1)
                scaler.scale(losses["total"] / group_size).backward()
                should_step = batch_index % accumulation_steps == 0 or batch_index == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    clip_norm = (
                        max_grad_norm
                        if max_grad_norm is not None and max_grad_norm > 0
                        else float("inf")
                    )
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_norm, error_if_nonfinite=False
                    )
                    gradients_are_finite = bool(torch.isfinite(gradient_norm).item())
                    scale_before_update = float(scaler.get_scale())
                    if gradients_are_finite:
                        scaler.step(optimizer)
                        consecutive_nonfinite_gradients = 0
                        global_step += 1
                    else:
                        skipped_gradient_steps += 1
                        consecutive_nonfinite_gradients += 1
                    bad_gradient_names = []
                    if not gradients_are_finite:
                        bad_gradient_names = [
                            name
                            for name, parameter in model.named_parameters()
                            if parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                        ]
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if not gradients_are_finite:
                        diagnostic = {
                            "warning": "non_finite_gradients_skipped",
                            "epoch": epoch,
                            "batch": batch_index,
                            "accumulation_group_start": group_start,
                            "amp_scale_before": scale_before_update,
                            "amp_scale_after": float(scaler.get_scale()),
                            "bad_parameters": bad_gradient_names[:12],
                            "losses": {
                                name: float(losses[name].detach().item())
                                for name in LOSS_NAMES
                            },
                        }
                        tqdm.write(json.dumps(diagnostic, ensure_ascii=False))
                        if (
                            consecutive_nonfinite_gradients
                            >= max_consecutive_nonfinite_gradients
                        ):
                            raise FloatingPointError(
                                "Non-finite gradients persisted for %d consecutive "
                                "optimizer steps; last diagnostic: %s"
                                % (
                                    consecutive_nonfinite_gradients,
                                    json.dumps(diagnostic, ensure_ascii=False),
                                )
                            )

            batch_losses = {
                name: float(losses[name].detach().item()) for name in LOSS_NAMES
            }
            batch_metrics = {
                name: float(losses[name].detach().item()) for name in HEATMAP_METRIC_NAMES
            }
            for name, value in {**batch_losses, **batch_metrics}.items():
                loss_totals[name] = loss_totals.get(name, 0.0) + value
            batches += 1

            if (
                training
                and tracker is not None
                and should_step
                and global_step % log_interval == 0
            ):
                tracker.log(
                    {
                        "epoch": epoch,
                        "train/global_step": global_step,
                        **{
                            f"train/batch_{name}_loss": value
                            for name, value in batch_losses.items()
                        },
                        **{
                            f"train/batch_{name}": value
                            for name, value in batch_metrics.items()
                        },
                    }
                )

    averages = {
        name: value / max(batches, 1) for name, value in loss_totals.items()
    }
    if training:
        averages["skipped_gradient_steps"] = float(skipped_gradient_steps)
    return averages, global_step


def _init_wandb(config: Dict[str, Any]) -> Optional[Any]:
    tracking = config.get("wandb", {})
    if not bool(tracking.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled, but wandb is not installed. "
            "Run `python -m pip install wandb`."
        ) from exc

    run = wandb.init(
        project=str(tracking.get("project", "bridge-hole-layout")),
        entity=tracking.get("entity"),
        name=tracking.get("run_name"),
        mode=str(tracking.get("mode", "online")),
        config=config,
    )
    run.define_metric("train/global_step")
    run.define_metric("train/batch_*", step_metric="train/global_step")
    run.define_metric("epoch")
    run.define_metric("train/epoch_*", step_metric="epoch")
    run.define_metric("val/epoch_*", step_metric="epoch")
    return run


def train_stage(config: Dict[str, Any]) -> None:
    set_seed(int(config["seed"]))
    requested_device = str(config["train"].get("device", "auto")).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment has no CUDA-enabled PyTorch. "
            f"Detected torch={torch.__version__}, torch.version.cuda={torch.version.cuda}. "
            "Install the CUDA 12.8 wheel before training."
        )
    device = torch.device(
        "cuda" if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["train"].get("cudnn_benchmark", True))

    stage = config["train"]["stage"]
    # Explicit train.domains takes precedence. Preserve the historical defaults
    # for existing configs that only define a stage.
    domains = config["train"].get("domains")
    if domains is None:
        domains = ["B"] if stage == "pretrain_b" else ["A"]
    train_dataset = HoleLayoutDataset(config, "train", domain=domains)
    val_dataset = HoleLayoutDataset(config, "val", domain=domains)
    print(json.dumps({"dataset_selection": {
        "train": train_dataset.selection_summary,
        "val": val_dataset.selection_summary,
    }}, ensure_ascii=False))
    num_workers = int(config["data"]["num_workers"])
    loader_args = {
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_args["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    if bool(config["train"].get("domain_balanced_sampling", False)):
        domain_group_counts = {}
        for record in train_dataset.records:
            key = (record["domain"].upper(), record["group_id"])
            domain_group_counts[key] = domain_group_counts.get(key, 0) + 1
        groups_per_domain = {}
        for domain_name, _ in domain_group_counts:
            groups_per_domain[domain_name] = groups_per_domain.get(domain_name, 0) + 1
        domain_count = len(groups_per_domain)
        weights = [
            1.0
            / (
                domain_count
                * groups_per_domain[record["domain"].upper()]
                * domain_group_counts[(record["domain"].upper(), record["group_id"])]
            )
            for record in train_dataset.records
        ]
        train_loader_args = dict(loader_args)
        train_loader_args["sampler"] = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, shuffle=False, **train_loader_args)
    elif bool(config["train"].get("group_balanced_sampling", True)):
        group_counts = {}
        for record in train_dataset.records:
            group_counts[record["group_id"]] = group_counts.get(record["group_id"], 0) + 1
        weights = [1.0 / group_counts[record["group_id"]] for record in train_dataset.records]
        train_loader_args = dict(loader_args)
        train_loader_args["sampler"] = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, shuffle=False, **train_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    model = build_model(config).to(device)
    channels_last = bool(config["train"].get("channels_last", False)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "amp": bool(config["train"]["amp"]) and device.type == "cuda",
                "batch_size": int(config["train"]["batch_size"]),
                "gradient_accumulation_steps": int(
                    config["train"].get("gradient_accumulation_steps", 1)
                ),
            },
            ensure_ascii=False,
        )
    )
    checkpoint = config["train"].get("checkpoint")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
    criterion = HoleLayoutLoss(
        float(config["loss"]["heatmap_weight"]),
        float(config["loss"]["offset_weight"]),
        float(config["loss"]["radius_weight"]),
        float(config["loss"].get("mask_weight", 0.5)),
        float(config["loss"].get("density_weight", 0.1)),
        float(config["loss"].get("count_weight", 0.1)),
        float(config["loss"].get("geometry_weight", 0.5)),
        float(config["loss"].get("region_weight", 0.2)),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["train"].get("lr_reduce_factor", 0.5)),
        patience=int(config["train"].get("lr_reduce_patience", 3)),
        min_lr=float(config["train"].get("min_learning_rate", 1e-7)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(config["train"]["amp"]) and device.type == "cuda",
        init_scale=float(config["train"].get("amp_init_scale", 16.0)),
        growth_interval=int(config["train"].get("amp_growth_interval", 2000)),
    )
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    bad_epochs = 0
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    freeze_epochs = int(config["train"].get("freeze_encoder_epochs", 0))
    tracker = _init_wandb(config)
    log_interval = max(1, int(config.get("wandb", {}).get("log_interval", 10)))
    accumulation_steps = max(
        1, int(config["train"].get("gradient_accumulation_steps", 1))
    )
    max_grad_norm = config["train"].get("max_grad_norm", 1.0)
    max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
    global_step = 0
    try:
        for epoch in range(int(config["train"]["epochs"])):
            epoch_number = epoch + 1
            model.freeze_encoder(epoch < freeze_epochs)
            train_losses, global_step = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler,
                bool(config["train"]["amp"]),
                accumulation_steps=accumulation_steps,
                channels_last=channels_last,
                max_grad_norm=max_grad_norm,
                tracker=tracker,
                epoch=epoch_number,
                global_step=global_step,
                log_interval=log_interval,
                max_consecutive_nonfinite_gradients=int(
                    config["train"].get(
                        "max_consecutive_nonfinite_gradients", 8
                    )
                ),
            )
            val_losses, _ = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_amp=bool(config["train"]["amp"]),
                channels_last=channels_last,
                epoch=epoch_number,
                global_step=global_step,
            )
            scheduler.step(val_losses["total"])
            payload = {
                "epoch": epoch_number,
                "train_loss": train_losses["total"],
                "val_loss": val_losses["total"],
                "train_skipped_gradient_steps": train_losses.get(
                    "skipped_gradient_steps", 0.0
                ),
                "stage": stage,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{f"train_{name}": train_losses[name] for name in HEATMAP_METRIC_NAMES},
                **{f"val_{name}": val_losses[name] for name in HEATMAP_METRIC_NAMES},
            }
            print(json.dumps(payload, ensure_ascii=False))
            if tracker is not None:
                tracker.log(
                    {
                        "epoch": epoch_number,
                        "train/global_step": global_step,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        **{
                            f"train/epoch_{name}_loss": value
                            for name, value in train_losses.items()
                            if name in LOSS_NAMES
                        },
                        **{
                            f"val/epoch_{name}_loss": value
                            for name, value in val_losses.items()
                            if name in LOSS_NAMES
                        },
                        "train/epoch_skipped_gradient_steps": train_losses.get(
                            "skipped_gradient_steps", 0.0
                        ),
                        **{
                            f"train/epoch_{name}": train_losses[name]
                            for name in HEATMAP_METRIC_NAMES
                        },
                        **{
                            f"val/epoch_{name}": val_losses[name]
                            for name in HEATMAP_METRIC_NAMES
                        },
                    }
                )
            state = {"model": model.state_dict(), "epoch": epoch_number, "config": config}
            torch.save(state, output_dir / "last.pt")
            if val_losses["total"] < best_loss - early_stopping_min_delta:
                best_loss = val_losses["total"]
                bad_epochs = 0
                torch.save(state, output_dir / "best.pt")
            else:
                bad_epochs += 1
                if early_stopping_patience > 0 and bad_epochs >= early_stopping_patience:
                    print(
                        json.dumps(
                            {
                                "early_stopping": True,
                                "epoch": epoch_number,
                                "best_val_loss": best_loss,
                                "patience": early_stopping_patience,
                            },
                            ensure_ascii=False,
                        )
                    )
                    break
    finally:
        if tracker is not None:
            tracker.finish()
