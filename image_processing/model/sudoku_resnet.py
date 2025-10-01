"""数独识别模型及其占位实现。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

PLACEHOLDER_IDENTIFIER = "sudoku-solver-gui-placeholder"

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 运行环境可能缺失 torch
    import torch
    import torch.nn as nn
    from torchvision.models import resnet101
except ModuleNotFoundError:  # pragma: no cover - 运行环境可能缺失 torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    resnet101 = None  # type: ignore[assignment]


class _PlaceholderSudokuModel:
    """在无法加载真实权重时使用的占位模型。"""

    def eval(self) -> "_PlaceholderSudokuModel":
        return self

    def __call__(self, _tensor: np.ndarray) -> np.ndarray:
        return np.zeros((1, 9, 9, 10), dtype=np.float32)


if nn is not None and resnet101 is not None:  # pragma: no branch - 简化运行时判断

    class _TorchSudokuModel(nn.Module):  # type: ignore[misc]
        """与历史版本兼容的 ResNet 模型定义。"""

        def __init__(self, pretrained: bool = True) -> None:
            super().__init__()
            base_model = resnet101(pretrained=pretrained)
            layers = list(base_model.children())[:-1]
            self.features = nn.Sequential(*layers)
            self.fc = nn.Linear(2048, 9 * 9 * 10)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # pragma: no cover - 需 torch 环境
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = x.view(x.size(0), 9, 9, 10)
            return x

else:

    class _TorchSudokuModel:  # type: ignore[too-many-ancestors]
        """占位定义，用于在缺少 torch 时给出更友好的错误提示。"""

        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("无法导入 torch/torchvision，无法创建真实模型")


def create_torch_model(*, pretrained: bool = True) -> "_TorchSudokuModel":
    """创建用于训练或推理的原生 torch 模型。"""

    if torch is None or nn is None or resnet101 is None:  # pragma: no cover - 需 torch 环境
        raise RuntimeError("缺少 torch/torchvision 依赖，无法构建真实模型")

    return _TorchSudokuModel(pretrained=pretrained)


class SudokuResNet:
    """统一封装的数独识别模型调用接口。"""

    def __init__(self, torch_model: Optional[_TorchSudokuModel] = None) -> None:
        self._torch_model = torch_model
        self._placeholder = _PlaceholderSudokuModel() if torch_model is None else None

    def eval(self) -> "SudokuResNet":
        if self._torch_model is not None:
            self._torch_model.eval()
        return self

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        if self._torch_model is None:
            assert self._placeholder is not None  # 帮助类型检查
            return self._placeholder(tensor)

        if torch is None:  # pragma: no cover - 安全守卫
            raise RuntimeError("缺少 torch 依赖，无法运行真实模型")

        with torch.no_grad():  # pragma: no cover - 需 torch 环境
            input_tensor = torch.from_numpy(tensor).permute(0, 3, 1, 2).float()
            output = self._torch_model(input_tensor)
            return output.cpu().numpy()


def _looks_like_placeholder(model_path: Path) -> bool:
    try:
        content = model_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return PLACEHOLDER_IDENTIFIER in content


def load_model(model_path: Path | str, *, map_location: str | "torch.device" = "cpu") -> SudokuResNet:
    """加载数独识别模型，若失败则返回占位模型。"""

    path = Path(model_path)

    if _looks_like_placeholder(path):
        logger.info("检测到占位模型文件，使用占位实现。")
        return SudokuResNet()

    if torch is None or nn is None or resnet101 is None:
        logger.warning("未安装 torch/torchvision，使用占位模型进行推理。")
        return SudokuResNet()

    try:
        state_dict = torch.load(path, map_location=map_location)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - 需捕获 torch.load 的多种异常
        logger.warning("加载模型权重失败，将使用占位模型: %s", exc)
        return SudokuResNet()

    torch_model = create_torch_model(pretrained=False)
    torch_model.load_state_dict(state_dict)
    return SudokuResNet(torch_model=torch_model)


__all__ = ["SudokuResNet", "create_torch_model", "load_model"]
