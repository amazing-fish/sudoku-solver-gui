"""数独识别占位模型实现。"""

from __future__ import annotations

import numpy as np


class SudokuResNet:
    """占位模型，用于在未提供真实权重时保持接口兼容。"""

    def eval(self) -> "SudokuResNet":  # noqa: D401 - 与 torch 模型接口兼容
        """返回自身，模仿 PyTorch `eval()` 接口。"""

        return self

    # 与旧实现保持一致的接口
    def load_state_dict(self, *_args, **_kwargs) -> None:  # noqa: D401
        """占位方法，兼容旧的加载流程。"""

    def predict(self, _image: np.ndarray) -> np.ndarray:
        """返回全零的预测结果。"""

        return np.zeros((1, 9, 9, 10), dtype=np.float32)

    # 为了兼容旧的 torch-style 调用
    def __call__(self, _tensor: np.ndarray) -> np.ndarray:
        return self.predict(_tensor)


__all__ = ["SudokuResNet"]
