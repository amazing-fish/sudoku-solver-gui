from __future__ import annotations

import os
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SyntheticDigitConfig, SyntheticDigitDataset
from .model import create_model
from .utils import format_config, select_device


logger = logging.getLogger(__name__)


def _prepare_dataloaders(
    batch_size: int,
    *,
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(
        "数据加载完成: train_size=%s, test_size=%s, batch_size=%s, 合成预处理后端=%s",
        len(train_dataset),
        len(test_dataset),
        batch_size,
        preprocess_backend,
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def train_model(
    model_path: str | os.PathLike[str] = "models/digit_cnn.pt",
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
    *,
    synthetic_backend: str = "cpu",
    synthetic_device: str | None = None,
    synthetic_batch_size: int = 256,
    synthetic_progress_interval: float = 2.0,
) -> nn.Module:
    """Train the digit classifier on synthetic data and save the model."""

    torch.manual_seed(42)

    device_obj = select_device(device)

    logger.info(
        "训练参数: %s",
        format_config(
            {
                "batch_size": batch_size,
                "device": device_obj,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "synthetic_backend": synthetic_backend,
                "synthetic_device": synthetic_device or "auto",
                "synthetic_progress_interval": synthetic_progress_interval,
                "synthesis_batch": synthetic_batch_size,
            }
        ),
    )

    train_loader, test_loader = _prepare_dataloaders(
        batch_size,
        preprocess_backend=synthetic_backend,
        preprocess_device=synthetic_device or "auto",
        synthesis_batch_size=synthetic_batch_size,
        progress_interval=synthetic_progress_interval,
    )

    model = create_model().to(device_obj)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("模型创建完成，共有参数量: %s", total_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device_obj)
            labels = labels.to(device_obj)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_accuracy = evaluate(model, test_loader, device_obj)
        logger.info(
            "Epoch %s/%s 完成: train_loss=%.4f, test_acc=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            test_accuracy,
        )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info("模型已保存到 %s", model_path.resolve())
    return model
