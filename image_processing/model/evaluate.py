"""评估训练好的数独识别模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - 帮助使用者定位依赖问题
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("运行评估脚本需要安装 torch 与 torchvision。") from exc

from .dataset import SudokuDataset
from .sudoku_resnet import create_torch_model


@torch.no_grad()
def _evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    model.eval()
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
    parser = argparse.ArgumentParser(description="评估数独识别模型的精度")
    parser.add_argument("dataset", type=Path, help="npz 格式的数据集路径")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("image_processing/model/sudoku_model.pth"),
        help="待评估的模型权重路径",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="评估时的批大小")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="评估所用设备 (cpu/cuda)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 的工作进程数")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset = SudokuDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = create_torch_model(pretrained=False)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    loss, accuracy = _evaluate(model, dataloader, device)
    print(f"loss: {loss:.4f}, accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
