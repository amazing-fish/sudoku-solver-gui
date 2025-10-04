from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "PREPROCESS_MEAN",
    "PREPROCESS_STD",
    "PREPROCESS_VERSION",
    "PreprocessResult",
    "preprocess_cell",
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
