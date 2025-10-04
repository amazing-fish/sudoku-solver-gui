from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from .dataset import SyntheticDigitConfig, SyntheticDigitDataset
from .model import create_model


def _display_synthetic_preview(dataset: SyntheticDigitDataset, output_path: Path) -> None:
    if len(dataset) == 0:
        return
    rng = np.random.default_rng(dataset.config.seed + 2024)
    index = int(rng.integers(0, len(dataset)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_pil(index).save(output_path)
    print(
        "示例合成样本 (标签={}):".format(dataset.labels[index])
    )
    print(dataset.render_ascii(index))
    print(f"合成样本图片已保存至: {output_path.resolve()}")


def _prepare_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_config = SyntheticDigitConfig(samples_per_digit=1500, seed=1)
    test_config = SyntheticDigitConfig(samples_per_digit=300, seed=2023)
    train_dataset = SyntheticDigitDataset(train_config)
    test_dataset = SyntheticDigitDataset(test_config)

    _display_synthetic_preview(train_dataset, Path("data/synthetic_preview.png"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _compute_classification_metrics(
    labels: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    labels_np = labels.cpu().numpy()
    preds_np = predictions.cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels_np, preds_np), 1)

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    true_positive = np.diag(cm)
    total = support.sum()

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=float),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive, dtype=float),
        where=(precision + recall) > 0,
    )

    accuracy = float(true_positive.sum() / total) if total else 0.0
    macro_precision = float(np.mean(precision)) if precision.size else 0.0
    macro_recall = float(np.mean(recall)) if recall.size else 0.0
    macro_f1 = float(np.mean(f1)) if f1.size else 0.0
    weighted_precision = float(np.average(precision, weights=support)) if total else 0.0
    weighted_recall = float(np.average(recall, weights=support)) if total else 0.0
    weighted_f1 = float(np.average(f1, weights=support)) if total else 0.0

    per_class_accuracy = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=float),
        where=support > 0,
    )

    metrics: Dict[str, float] = {
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
    }

    for idx, value in enumerate(per_class_accuracy):
        metrics[f"class_{idx}_acc"] = float(value)

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    num_classes: int = 10,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels: List[torch.Tensor] = []
    all_predictions: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            all_labels.append(labels.detach().cpu())
            all_predictions.append(predictions.detach().cpu())

    if not all_labels:
        metrics = {"loss": 0.0}
        metrics.update({"accuracy": 0.0, "f1_macro": 0.0})
        return metrics

    labels_tensor = torch.cat(all_labels)
    predictions_tensor = torch.cat(all_predictions)
    metrics = _compute_classification_metrics(labels_tensor, predictions_tensor, num_classes)
    metrics["loss"] = total_loss / total if total else 0.0
    return metrics


def train_model(
    model_path: str | os.PathLike[str] = "models/digit_cnn.pt",
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
    weight_decay: float = 1e-4,
    scheduler_type: str = "cosine",
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-4,
    target_metric: str = "f1_macro",
    grad_clip_norm: float | None = 1.0,
    use_amp: bool = True,
    save_last: bool = True,
) -> nn.Module:
    """Train the digit classifier on synthetic data and save the model.

    新增功能包括：
    * 更丰富的评估指标（准确率、宏/加权精确率、召回率、F1 等）。
    * 自动混合精度、梯度裁剪、学习率调度等现代训练策略。
    * 早停与最佳模型保存。"""

    torch.manual_seed(42)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader = _prepare_dataloaders(batch_size)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler: _LRScheduler | ReduceLROnPlateau | None = None
    if scheduler_type == "cosine" and epochs > 1:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.1)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(1, early_stopping_patience // 2),
            verbose=False,
        )
    elif scheduler_type not in {"none", ""}:
        raise ValueError(
            "scheduler_type 必须是 'cosine'、'reduce_on_plateau' 或 'none'"
        )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    history: List[Dict[str, float]] = []
    best_metric = float("-inf")
    best_epoch = -1
    best_state_dict: Dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    print(
        "开始训练: "
        f"epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, device={device}, "
        f"weight_decay={weight_decay}, scheduler={scheduler_type}, target_metric={target_metric}"
    )

    for epoch in range(epochs):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            running_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = running_loss / total_samples if total_samples else 0.0
        train_acc = running_correct / total_samples if total_samples else 0.0

        val_metrics = evaluate(model, test_loader, device, criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.perf_counter() - epoch_start
        log_message = (
            f"Epoch {epoch + 1:03d}/{epochs} | {epoch_time:.2f}s | "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.6f} "
            f"val_f1_macro={val_metrics.get('f1_macro', 0.0):.6f} | lr={current_lr:.8f}"
        )
        print(log_message)

        metrics_record = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc}
        metrics_record.update(val_metrics)
        history.append(metrics_record)

        monitored_metric = val_metrics.get(target_metric)
        if monitored_metric is None:
            raise KeyError(f"未找到目标评估指标: {target_metric}")

        if monitored_metric > best_metric + early_stopping_min_delta:
            best_metric = monitored_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

            torch.save(best_state_dict, model_path)
            print(
                f"  ↳ 新最佳模型已在第 {best_epoch} 轮取得 {target_metric}={best_metric:.6f}，"
                f"已保存至 {model_path.resolve()}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  ↳ 已连续 {epochs_without_improvement} 轮未提升 {target_metric}。"
            )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(monitored_metric)
        elif isinstance(scheduler, _LRScheduler):
            scheduler.step()

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"早停触发：在 {early_stopping_patience} 轮内 {target_metric} 未取得显著提升。"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if save_last:
        last_model_path = Path(model_path)
        last_model_path = last_model_path.with_name(
            f"{last_model_path.stem}_last{last_model_path.suffix}"
        )
        last_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), last_model_path)
        print(f"最后一轮模型权重已另存为 {last_model_path.resolve()}")

    if best_epoch > 0:
        print(
            f"最佳模型来自第 {best_epoch} 轮，{target_metric}={best_metric:.6f}。"
        )

    final_metrics = evaluate(model, test_loader, device, criterion)
    print("最佳模型在验证集上的最终指标：")
    for key, value in final_metrics.items():
        if key.startswith("class_"):
            continue
        print(f"  - {key}: {value:.6f}")

    class_metrics = {k: v for k, v in final_metrics.items() if k.startswith("class_")}
    if class_metrics:
        print("  - per_class_accuracy:")
        for key in sorted(class_metrics):
            print(f"      {key}: {class_metrics[key]:.6f}")

    return model
