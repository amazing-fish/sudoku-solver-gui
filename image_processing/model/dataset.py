"""面向 PyTorch 的数独数据集实现。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:  # pragma: no cover - 运行环境可能缺失 torch
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - 帮助使用者定位依赖问题
    raise RuntimeError(
        "未安装 torch，无法使用 SudokuDataset。请先安装 PyTorch 后再重试。"
    ) from exc


@dataclass
class DatasetInfo:
    samples: int
    grid_size: int
    cell_size: int
    description: str


def _load_metadata(metadata_path: Path) -> DatasetInfo | None:
    if not metadata_path.exists():
        return None

    try:
        payload: Dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    try:
        return DatasetInfo(**payload)
    except TypeError:
        return None


class SudokuDataset(Dataset):
    """从 `.npz` 文件中读取图像与标签。"""

    def __init__(self, data_path: Path | str) -> None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到数据集文件: {path}")

        with np.load(path) as data:
            self._images = data["images"].astype(np.float32)
            self._labels = data["labels"].astype(np.int64)

        if self._images.ndim != 4 or self._images.shape[1:3] != (252, 252):
            raise ValueError("数据集中图像尺寸不符合 252x252 的预期")

        if self._labels.shape != (self._images.shape[0], 9, 9):
            raise ValueError("标签形状必须为 (样本数, 9, 9)")

        self._metadata = _load_metadata(path.with_suffix(".json"))

    @property
    def metadata(self) -> DatasetInfo | None:
        return self._metadata

    def __len__(self) -> int:  # pragma: no cover - 委托给 numpy
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        image = self._images[index] / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)

        label = self._labels[index]
        label_tensor = torch.from_numpy(label)

        return image_tensor, label_tensor


__all__ = ["SudokuDataset", "DatasetInfo"]
