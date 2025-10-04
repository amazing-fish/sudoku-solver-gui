from __future__ import annotations

import os
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_confusion_matrix,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)
from torch.optim.swa_utils import AveragedModel
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast

from .dataset import SyntheticDigitConfig, SyntheticDigitDataset
from .model import create_model


logger = logging.getLogger(__name__)


def _prepare_dataloaders(
    batch_size: int,
    *,
    eval_batch_size: int | None = None,
    preprocess_backend: str = "cpu",
    preprocess_device: str | None = None,
    synthesis_batch_size: int = 256,
    progress_interval: float = 2.0,
) -> tuple[DataLoader, DataLoader]:
    device = preprocess_device or "auto"
    train_config = SyntheticDigitConfig(
        samples_per_digit=500,
        seed=1,
        preprocess_backend=preprocess_backend,
        device=device,
        synthesis_batch_size=synthesis_batch_size,
        progress_interval=progress_interval,
    )
    test_config = SyntheticDigitConfig(
        samples_per_digit=100,
        seed=2023,
        preprocess_backend=preprocess_backend,
        device=device,
        synthesis_batch_size=synthesis_batch_size,
        progress_interval=progress_interval,
    )
    logger.info(
        "构建数据集: train_samples_per_digit=%s, test_samples_per_digit=%s, 包含空白=%s",
        train_config.samples_per_digit,
        test_config.samples_per_digit,
        train_config.include_blank,
    )
    train_dataset = SyntheticDigitDataset(train_config)
    test_dataset = SyntheticDigitDataset(test_config)

    eval_batch = eval_batch_size or batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch, shuffle=False)
    logger.info(
        "数据加载完成: train_size=%s, test_size=%s, train_batch=%s, eval_batch=%s, 合成预处理后端=%s",
        len(train_dataset),
        len(test_dataset),
        batch_size,
        eval_batch,
        preprocess_backend,
    )
    return train_loader, test_loader


def _compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if logits.numel() == 0:
        return 0.0
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return float(correct.double().mean().item())


def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().tolist()]


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    criterion: nn.Module,
    amp_enabled: bool,
    num_classes: int,
    desc: str = "验证",
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    samples = 0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []

    progress = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)
            with autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * batch_size
            samples += batch_size
            preds.append(outputs.argmax(dim=1).cpu())
            targets.append(labels.detach().cpu())
            logits.append(outputs.detach().cpu())

    if samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "top3_accuracy": 0.0,
            "per_class_accuracy": [],
            "confusion_matrix": [],
            "support": [],
        }

    preds_tensor = torch.cat(preds)
    targets_tensor = torch.cat(targets)
    logits_tensor = torch.cat(logits)

    avg_loss = total_loss / samples
    accuracy = multiclass_accuracy(preds_tensor, targets_tensor, num_classes=num_classes, average="micro")
    macro_precision = multiclass_precision(
        preds_tensor,
        targets_tensor,
        num_classes=num_classes,
        average="macro",
        zero_division=0.0,
    )
    macro_recall = multiclass_recall(
        preds_tensor,
        targets_tensor,
        num_classes=num_classes,
        average="macro",
        zero_division=0.0,
    )
    macro_f1 = multiclass_f1_score(
        preds_tensor,
        targets_tensor,
        num_classes=num_classes,
        average="macro",
        zero_division=0.0,
    )
    confusion = multiclass_confusion_matrix(
        preds_tensor,
        targets_tensor,
        num_classes=num_classes,
    ).to(torch.float64)
    per_class_accuracy = torch.where(
        confusion.sum(dim=1) > 0,
        torch.diag(confusion) / confusion.sum(dim=1),
        torch.zeros(num_classes, dtype=torch.float64),
    )
    top3_accuracy = _compute_topk_accuracy(logits_tensor, targets_tensor, k=min(3, num_classes))

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy.detach().cpu().item()),
        "macro_precision": float(macro_precision.detach().cpu().item()),
        "macro_recall": float(macro_recall.detach().cpu().item()),
        "macro_f1": float(macro_f1.detach().cpu().item()),
        "top3_accuracy": float(top3_accuracy),
        "per_class_accuracy": _tensor_to_list(per_class_accuracy),
        "confusion_matrix": confusion.tolist(),
        "support": _tensor_to_list(confusion.sum(dim=1)),
    }


def _build_optimizer(
    model: nn.Module,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name_lower = name.lower()
    if name_lower == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name_lower == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name_lower == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"不支持的优化器: {name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str | None,
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
) -> tuple[torch.optim.lr_scheduler._LRScheduler | None, bool]:
    if not name:
        return None, False

    name_lower = name.lower()
    if name_lower in {"none", "null"}:
        return None, False
    if name_lower == "onecycle":
        if steps_per_epoch <= 0:
            logger.warning("OneCycleLR 需要正数的 steps_per_epoch，当前为 0，跳过调度器。")
            return None, False
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return scheduler, True
    if name_lower == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=learning_rate * 0.01,
        )
        return scheduler, False
    raise ValueError(f"不支持的学习率调度策略: {name}")


def _save_checkpoint(
    path: Path,
    *,
    model_state: Dict[str, torch.Tensor],
    epoch: int,
    metrics: Dict[str, Any],
    history: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model_state,
        "epoch": epoch,
        "metrics": metrics,
        "history": list(history),
        "config": config,
    }
    torch.save(payload, path)
    logger.info(
        "已保存检查点: %s (epoch=%s, macro_f1=%.4f, accuracy=%.4f)",
        path.resolve(),
        epoch,
        metrics.get("macro_f1", float("nan")),
        metrics.get("accuracy", float("nan")),
    )


def train_model(
    model_path: str | os.PathLike[str] = "models/digit_cnn.pt",
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
    *,
    eval_batch_size: int | None = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    scheduler: str | None = None,
    label_smoothing: float = 0.0,
    max_grad_norm: float | None = None,
    patience: int | None = None,
    best_model_path: str | os.PathLike[str] | None = None,
    use_amp: bool | None = None,
    ema_decay: float = 0.0,
    synthetic_backend: str = "cpu",
    synthetic_device: str | None = None,
    synthetic_batch_size: int = 256,
    synthetic_progress_interval: float = 2.0,
) -> nn.Module:
    """使用合成数据训练数字分类器，并保存最佳模型与训练日志。"""

    torch.manual_seed(42)

    requested_device = device
    if isinstance(requested_device, torch.device):
        device_obj = requested_device
    else:
        device_str = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(device_str).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("请求使用 CUDA 设备，但当前环境不可用，自动切换到 CPU。")
            device_str = "cpu"
        device_obj = torch.device(device_str)

    amp_enabled = False
    if use_amp is None:
        amp_enabled = device_obj.type == "cuda"
    else:
        if use_amp and device_obj.type != "cuda":
            logger.warning("AMP 仅在 CUDA 设备上可用，已自动禁用。")
            amp_enabled = False
        else:
            amp_enabled = bool(use_amp and device_obj.type == "cuda")

    if amp_enabled:
        logger.info("启用 CUDA 自动混合精度训练。")

    model_path = Path(model_path)
    if best_model_path is None:
        best_model_path = model_path.with_name(f"{model_path.stem}_best{model_path.suffix}")
    best_model_path = Path(best_model_path)

    if ema_decay < 0.0 or ema_decay >= 1.0:
        if ema_decay != 0.0:
            raise ValueError("ema_decay 必须位于 [0, 1) 区间，0 表示禁用")
        ema_decay = 0.0

    logger.info(
        "训练参数: epochs=%s, batch_size=%s, eval_batch_size=%s, learning_rate=%s, optimizer=%s, weight_decay=%s, scheduler=%s, label_smoothing=%s, max_grad_norm=%s, patience=%s, ema_decay=%s, device=%s, synthetic_backend=%s, synthetic_device=%s, synthesis_batch=%s, synthetic_progress_interval=%ss",
        epochs,
        batch_size,
        eval_batch_size or batch_size,
        learning_rate,
        optimizer_name,
        weight_decay,
        scheduler or "none",
        label_smoothing,
        max_grad_norm,
        patience,
        ema_decay,
        device_obj,
        synthetic_backend,
        synthetic_device or "auto",
        synthetic_batch_size,
        synthetic_progress_interval,
    )

    train_loader, test_loader = _prepare_dataloaders(
        batch_size,
        eval_batch_size=eval_batch_size,
        preprocess_backend=synthetic_backend,
        preprocess_device=synthetic_device or "auto",
        synthesis_batch_size=synthetic_batch_size,
        progress_interval=synthetic_progress_interval,
    )

    num_classes = 10
    model = create_model(num_classes=num_classes).to(device_obj)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("模型创建完成，共有参数量: %s", total_params)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, optimizer_name, learning_rate, weight_decay)
    scheduler_obj, scheduler_batch_step = _build_scheduler(
        optimizer,
        scheduler,
        epochs,
        len(train_loader),
        learning_rate,
    )
    scaler = GradScaler(enabled=amp_enabled)

    ema_model: AveragedModel | None = None
    if 0.0 < ema_decay < 1.0:
        avg_fn = lambda avg_param, model_param, num_avg: ema_decay * avg_param + (1.0 - ema_decay) * model_param
        ema_model = AveragedModel(model, avg_fn=avg_fn)
        ema_model.to(device_obj)
        logger.info("启用参数 EMA: decay=%.4f", ema_decay)

    history: list[Dict[str, Any]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_metrics: Dict[str, Any] | None = None
    best_epoch = 0
    best_score = float("-inf")
    no_improve_epochs = 0
    monitor_metric = "macro_f1"

    training_config: Dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size or batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "scheduler": scheduler or "none",
        "label_smoothing": label_smoothing,
        "max_grad_norm": max_grad_norm,
        "patience": patience,
        "ema_decay": ema_decay,
        "device": str(device_obj),
        "synthetic_backend": synthetic_backend,
        "synthetic_device": synthetic_device or "auto",
        "synthetic_batch_size": synthetic_batch_size,
        "synthetic_progress_interval": synthetic_progress_interval,
        "amp_enabled": amp_enabled,
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_top3_correct = 0
        seen_samples = 0

        train_progress = tqdm(
            train_loader,
            desc=f"训练 Epoch {epoch + 1}/{epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for images, labels in train_progress:
            images = images.to(device_obj)
            labels = labels.to(device_obj)
            batch_size_curr = images.size(0)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                ema_model.update_parameters(model)
            if scheduler_obj is not None and scheduler_batch_step:
                scheduler_obj.step()

            loss_value = float(loss.detach().item())
            predictions = outputs.detach().argmax(dim=1)
            topk_indices = outputs.detach().topk(min(3, num_classes), dim=1).indices
            running_loss += loss_value * batch_size_curr
            train_correct += int((predictions == labels).sum().item())
            train_top3_correct += int(topk_indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item())
            seen_samples += batch_size_curr

            current_lr = optimizer.param_groups[0]["lr"]
            train_progress.set_postfix(
                loss=f"{loss_value:.4f}",
                lr=f"{current_lr:.2e}",
            )

        if scheduler_obj is not None and not scheduler_batch_step:
            scheduler_obj.step()

        if seen_samples == 0:
            logger.warning("训练数据为空，提前结束。")
            break

        train_loss = running_loss / seen_samples
        train_accuracy = train_correct / seen_samples
        train_top3 = train_top3_correct / seen_samples

        eval_target_model = ema_model.module if ema_model is not None else model
        val_metrics = evaluate(
            eval_target_model,
            test_loader,
            device_obj,
            criterion=eval_criterion,
            amp_enabled=amp_enabled,
            num_classes=num_classes,
            desc=f"验证 Epoch {epoch + 1}",
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "train_top3_accuracy": float(train_top3),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_top3_accuracy": float(val_metrics["top3_accuracy"]),
            }
        )

        logger.info(
            "Epoch %s/%s 完成: train_loss=%.4f, train_acc=%.4f, train_top3=%.4f, val_loss=%.4f, val_acc=%.4f, val_macro_f1=%.4f, val_precision=%.4f, val_recall=%.4f, val_top3=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            train_accuracy,
            train_top3,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["macro_precision"],
            val_metrics["macro_recall"],
            val_metrics["top3_accuracy"],
        )

        score = val_metrics.get(monitor_metric, 0.0)
        improved = score > best_score + 1e-6
        if improved or best_state is None:
            best_score = score
            best_state = deepcopy(eval_target_model.state_dict())
            best_metrics = {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in val_metrics.items()
            }
            best_metrics["epoch"] = epoch + 1
            best_metrics["train_loss"] = float(train_loss)
            best_metrics["train_accuracy"] = float(train_accuracy)
            best_metrics["train_top3_accuracy"] = float(train_top3)
            best_epoch = epoch + 1
            no_improve_epochs = 0
            per_class_str = ", ".join(
                f"{digit}:{acc:.4f}" for digit, acc in enumerate(val_metrics["per_class_accuracy"])
            )
            logger.info(
                "检测到新的最佳模型: epoch=%s, macro_f1=%.4f, accuracy=%.4f, top3=%.4f",
                best_epoch,
                val_metrics["macro_f1"],
                val_metrics["accuracy"],
                val_metrics["top3_accuracy"],
            )
            logger.info("验证集每类准确率: %s", per_class_str)
        else:
            no_improve_epochs += 1
            if patience and patience > 0 and no_improve_epochs >= patience:
                logger.info(
                    "提前停止: 监控指标 %s 连续 %s 轮未提升。",
                    monitor_metric,
                    patience,
                )
                break

    if best_state is None:
        best_state = deepcopy(model.state_dict())
        best_metrics = {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "loss": 0.0,
            "top3_accuracy": 0.0,
            "per_class_accuracy": [0.0] * num_classes,
            "confusion_matrix": [[0.0] * num_classes for _ in range(num_classes)],
            "support": [0.0] * num_classes,
            "epoch": 0,
        }

    model.load_state_dict(best_state)
    if ema_model is not None:
        ema_model.module.load_state_dict(best_state)

    if best_metrics is None:
        best_metrics = {}
    best_metrics.setdefault("epoch", best_epoch)

    if best_model_path.resolve() == model_path.resolve():
        _save_checkpoint(
            model_path,
            model_state=best_state,
            epoch=best_metrics.get("epoch", best_epoch),
            metrics=best_metrics,
            history=history,
            config=training_config,
        )
    else:
        _save_checkpoint(
            best_model_path,
            model_state=best_state,
            epoch=best_metrics.get("epoch", best_epoch),
            metrics=best_metrics,
            history=history,
            config=training_config,
        )

        _save_checkpoint(
            model_path,
            model_state=best_state,
            epoch=best_metrics.get("epoch", best_epoch),
            metrics=best_metrics,
            history=history,
            config=training_config,
        )

    logger.info(
        "训练完成: 最佳 epoch=%s, macro_f1=%.4f, accuracy=%.4f, best_path=%s, final_path=%s",
        best_metrics.get("epoch", best_epoch),
        best_metrics.get("macro_f1", 0.0),
        best_metrics.get("accuracy", 0.0),
        best_model_path.resolve(),
        model_path.resolve(),
    )

    return model
