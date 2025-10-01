"""数独识别模型工具模块。"""

from __future__ import annotations

from pathlib import Path

MODEL_FILENAME = "sudoku_model.pth"
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def ensure_model_file(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """确保默认模型文件存在并返回其路径。"""

    from .build_model import build_model

    model_path = Path(model_path)
    if not model_path.exists():
        build_model(model_path)
    return model_path


__all__ = ["MODEL_FILENAME", "MODEL_DIR", "DEFAULT_MODEL_PATH", "ensure_model_file"]
