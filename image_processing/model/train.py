"""训练数独识别模型。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

try:  # pragma: no cover - 帮助使用者定位依赖问题
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("运行训练脚本需要安装 torch 与 torchvision。") from exc

from . import DEFAULT_MODEL_PATH
from .dataset import SudokuDataset
from .sudoku_resnet import create_torch_model


def _build_dataloaders(
    dataset: SudokuDataset,
    *,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader | None]:
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split 必须位于 [0, 1) 区间")

    if val_split == 0.0 or len(dataset) == 1:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, None

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def _train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.view(-1, 10), labels.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs.view(-1, 10), labels.view(-1))
        running_loss += float(loss.item())

        predictions = outputs.argmax(dim=-1)
        correct += torch.eq(predictions, labels).sum().item()
        total += labels.numel()

    avg_loss = running_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练数独识别模型")
    parser.add_argument("dataset", type=Path, help="npz 格式的数据集路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集划分比例")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="训练完成后保存模型权重的路径",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 的工作进程数")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练所用设备 (cpu/cuda)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device)
    dataset = SudokuDataset(args.dataset)
    train_loader, val_loader = _build_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    model = create_torch_model(pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        log_message = f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}"

        if val_loader is not None:
            val_loss, val_accuracy = _evaluate(model, val_loader, criterion, device)
            log_message += f", val_loss: {val_loss:.4f}, val_acc: {val_accuracy:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(log_message)

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"模型已保存至 {output_path}")


if __name__ == "__main__":
    main()
