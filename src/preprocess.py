from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple

__all__ = [
    "PREPROCESS_MEAN",
    "PREPROCESS_STD",
    "PREPROCESS_VERSION",
    "PreprocessResult",
    "preprocess_cell",
    "preprocess_cell_batch",
]

PREPROCESS_MEAN: float = 0.1307
PREPROCESS_STD: float = 0.3081
PREPROCESS_VERSION: str = "20241004"
_TARGET_SIZE: int = 28


@dataclass(frozen=True)
class PreprocessResult:
    data: np.ndarray
    is_blank: bool


def _is_blank(binary_inv: np.ndarray) -> bool:
    non_zero = cv2.countNonZero(binary_inv)
    total = binary_inv.shape[0] * binary_inv.shape[1]
    if non_zero == 0:
        return True

    ratio = non_zero / total
    if ratio < 0.015:
        return True

    coords = cv2.findNonZero(binary_inv)
    if coords is None:
        return True

    x, y, w, h = cv2.boundingRect(coords)
    if w * h < 0.02 * total:
        return True

    return False


def _normalize_cleaned(cleaned: np.ndarray, blank: bool) -> PreprocessResult:
    if blank:
        normalized = np.zeros((_TARGET_SIZE, _TARGET_SIZE), dtype=np.float32)
        normalized = (normalized - PREPROCESS_MEAN) / PREPROCESS_STD
        return PreprocessResult(normalized, True)

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        normalized = np.zeros((_TARGET_SIZE, _TARGET_SIZE), dtype=np.float32)
        normalized = (normalized - PREPROCESS_MEAN) / PREPROCESS_STD
        return PreprocessResult(normalized, True)

    x, y, w, h = cv2.boundingRect(coords)
    digit = cleaned[y : y + h, x : x + w]
    side = max(w, h)
    padded = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - w) // 2
    y_offset = (side - h) // 2
    padded[y_offset : y_offset + h, x_offset : x_offset + w] = digit
    resized = cv2.resize(padded, (_TARGET_SIZE, _TARGET_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - PREPROCESS_MEAN) / PREPROCESS_STD
    return PreprocessResult(normalized, False)


def preprocess_cell(cell: np.ndarray) -> PreprocessResult:
    """Preprocess a Sudoku cell and indicate whether it is blank."""

    blur = cv2.GaussianBlur(cell, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    blank = _is_blank(cleaned)
    return _normalize_cleaned(cleaned, blank)


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32)
    coords -= (size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, size, size)


def _local_contrast_enhance(tensor: torch.Tensor) -> torch.Tensor:
    # 近似 CLAHE 的局部对比度增强，结果限制在 [0, 1]
    # 使用奇数核保证与输入张量保持一致的空间尺寸，避免广播失败
    max_pool = F.max_pool2d(tensor, kernel_size=9, stride=1, padding=4)
    min_pool = -F.max_pool2d(-tensor, kernel_size=9, stride=1, padding=4)
    enhanced = (tensor - min_pool)
    denom = (max_pool - min_pool).clamp_min(1e-3)
    enhanced = enhanced / denom
    return enhanced.clamp(0.0, 1.0)


def preprocess_cell_batch(
    cells: np.ndarray,
    *,
    backend: str = "cpu",
    device: str | torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """批量预处理单元格，返回归一化结果与空白标记。"""

    if cells.ndim != 3:
        raise ValueError("cells 应为 (N, H, W) 形状的数组")

    backend = backend.lower()
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"不支持的预处理后端: {backend}")

    if backend == "cpu":
        results = [preprocess_cell(cell) for cell in cells]
        data = np.stack([item.data for item in results], axis=0)
        blanks = np.array([item.is_blank for item in results], dtype=bool)
        return data, blanks

    if isinstance(device, str) or device is None:
        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_obj = torch.device(device_str)
    elif isinstance(device, torch.device):
        device_obj = device
    else:
        raise TypeError("device 参数类型不受支持")

    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 GPU 预处理但当前环境未检测到可用的 CUDA 设备")

    tensor = torch.from_numpy(cells).to(device=device_obj, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(1)

    kernel = _gaussian_kernel(5, 1.0, device_obj)
    blurred = F.conv2d(tensor, kernel, padding=2)

    enhanced = _local_contrast_enhance(blurred)

    mean = F.avg_pool2d(enhanced, kernel_size=11, stride=1, padding=5)
    binary = (enhanced > (mean - (2.0 / 255.0))).float()
    binary_inv = 1.0 - binary

    erosion = 1.0 - F.max_pool2d(1.0 - binary_inv, kernel_size=3, stride=1, padding=1)
    opened = F.max_pool2d(erosion, kernel_size=3, stride=1, padding=1)
    opened = (opened > 0.5).float()

    cleaned = (opened.squeeze(1) * 255.0).to(dtype=torch.uint8).cpu().numpy()

    results = [_normalize_cleaned(image, _is_blank(image)) for image in cleaned]
    data = np.stack([item.data for item in results], axis=0)
    blanks = np.array([item.is_blank for item in results], dtype=bool)
    return data, blanks
