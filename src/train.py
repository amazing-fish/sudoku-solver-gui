from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SyntheticDigitConfig, SyntheticDigitDataset
from .model import create_model


def _prepare_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataset = SyntheticDigitDataset(SyntheticDigitConfig(samples_per_digit=500, seed=1))
    test_dataset = SyntheticDigitDataset(SyntheticDigitConfig(samples_per_digit=100, seed=2023))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
) -> nn.Module:
    """Train the digit classifier on synthetic data and save the model."""

    torch.manual_seed(42)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader = _prepare_dataloaders(batch_size)

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, test_acc={test_accuracy:.4f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path.resolve()}")
    return model
